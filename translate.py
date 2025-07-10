#!/usr/bin/env python3

import sys
import requests
import re
import torch

from aux import CmdlineArgs, log
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM
from data import do_list_in_batches, lang_bin_mapping, make_gen_text
from modelops import to_cpl_spec, load_module_config, hf_tok, is_gen_ai
from collections import defaultdict
from langconv import is_nllb, is_madlad, any_to_mdl_type, get_mdl_type, any_to_neurotolge, is_dec_only_llm
from tokops import load_tokenizer, tokenizeit, detokenizemany


def prepare_for_translation(provided_inputs, toktup, input_language, output_language=None, device=None):
    if is_nllb(toktup[0]):
        toktup[0].src_lang = input_language
        inputs_to_process = provided_inputs
    elif is_madlad(toktup[0]):
        madlad_tgt_lang = output_language
        inputs_to_process = [f"{madlad_tgt_lang} {inp}" for inp in provided_inputs]
    else:
        raise NotImplementedError("Model type not supported")

    prepared_inputs = tokenizeit(toktup, inputs_to_process, 1024, False) #tokenizer(inputs_to_process, return_tensors="pt", padding=True, truncation=True, max_length=512)

    if device is not None:
        prepared_inputs.to(device)

    frc_bos = toktup[0].get_lang_id(output_language) if output_language is not None else None

    return prepared_inputs, frc_bos


def finalize_translation(outputs, toktup):
    result = detokenizemany(toktup, outputs) # tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return result


def loadmodel(mdlname="facebook/m2m100_418M", accelerator=None):
    cl = AutoModelForCausalLM if is_gen_ai(mdlname) else AutoModelForSeq2SeqLM

    if accelerator is not None:
        model = cl.from_pretrained(mdlname, token=hf_tok, torch_dtype=torch.bfloat16)
        model = accelerator.prepare(model)
    else:
        model = cl.from_pretrained(mdlname, token=hf_tok, torch_dtype=torch.bfloat16, device_map="auto")

    return model


def encode(model, input_batch):
    model = model.module if hasattr(model, "module") else model

    if is_nllb(model):
        enc = model.model.encoder
    elif is_madlad(model):
        enc = model.base_model.encoder
    else:
        raise NotImplementedError(f"Model {model} is not supported yet.")

    inputs_without_labels = { k: input_batch[k] for k in input_batch if k != "labels" }

    return enc(**inputs_without_labels)


def coupled_encode(coupling_specs, lang_to_bin, input_lang, input_texts, debug=False):

    mdl_type = get_mdl_type(coupling_specs[0].model)
    conv_input_lang = any_to_mdl_type(mdl_type, input_lang)

    this = coupling_specs[lang_to_bin[conv_input_lang]]

    # 0. input text --> input token IDs
    these_inputs, _ = prepare_for_translation(input_texts, (this.tokenizer, this.postokenizer), conv_input_lang, device=this.model.device)
    attention_mask = these_inputs["attention_mask"]
    if debug:
        for iii in range(len(input_texts)):
            toklist = []
            for tok_idx in these_inputs['input_ids'][iii]:
                try:
                    tok = this.tokenizer.convert_ids_to_tokens([tok_idx])[0]
                except IndexError:
                    tok = this.postokenizer['idx2tok'][str(tok_idx.item())]
                toklist.append(tok)
            print(these_inputs['input_ids'][iii])
            print(toklist)

    # 1. input token IDs --> encoder vectors
    #embeddings = this.model.model.encoder(**these_inputs)
    return encode(this.model, these_inputs), attention_mask


def postproc_llm_output(raw_outputs, tok):
    eos_id = tok.convert_tokens_to_ids(tok.eos_token)

    for i, _ in enumerate(raw_outputs):
        repl = None
        for ii, t in enumerate(raw_outputs[i]):
            if t.item() == eos_id:
                repl = eos_id
            if repl is not None:
                raw_outputs[i][ii] = repl

    return raw_outputs


def llm_generate(coupling_specs, input_language, output_language, input_texts, debug=False):
    mdl_type = get_mdl_type(coupling_specs[0].model)
    conv_input_lang = any_to_mdl_type(mdl_type, input_language)
    conv_output_lang = any_to_mdl_type(mdl_type, output_language)

    tokenizer = coupling_specs[0].tokenizer

    prep_texts = [make_gen_text(conv_input_lang, conv_output_lang, input_txt, None) for input_txt in input_texts]

    tokenized = tokenizeit((tokenizer, None), prep_texts, 1024, is_target=False, is_llm=True)

    obj = coupling_specs[0].model
    obj = obj.module if hasattr(obj, "module") else obj

    tokenized['input_ids'] = tokenized['input_ids'].to(obj.device)
    tokenized['attention_mask'] = tokenized['attention_mask'].to(obj.device)

    raw_outputs = obj.generate(**tokenized, max_length)

    # 3. output token IDs --> output text
    pre_result = tokenizer.batch_decode(postproc_llm_output(raw_outputs, tokenizer), skip_special_tokens=True)

    result = [raw_out[len(prep_texts[i]):].split("\n")[0] for i, raw_out in enumerate(pre_result)]
    """
    for i, raw_out in enumerate(pre_result):
        print("====")
        print(i, raw_out)
        print("%%%%")
        print(raw_out[len(prep_texts[i])-3:])
        print("----")
    """

    return result

def coupled_generate(coupling_specs, lang_to_bin, output_lang, encoder_embeddings, att_mask, debug=False):
    mdl_type = get_mdl_type(coupling_specs[0].model)
    conv_output_lang = any_to_mdl_type(mdl_type, output_lang)

    dec_idx = lang_to_bin[conv_output_lang]

    tokenizer = coupling_specs[dec_idx].tokenizer

    # 2. encoder vectors --> output token IDs
    frc_bos = tokenizer.convert_tokens_to_ids(conv_output_lang)
    obj = coupling_specs[dec_idx].model
    obj = obj.module if hasattr(obj, "module") else obj

    raw_outputs = obj.generate(forced_bos_token_id=frc_bos, encoder_outputs=encoder_embeddings, attention_mask=att_mask)
    if debug:
        for rwout in raw_outputs:
            print(rwout)
            print(tokenizer.convert_ids_to_tokens(rwout))

    # 3. output token IDs --> output text
    result = finalize_translation(raw_outputs, (tokenizer, coupling_specs[dec_idx].postokenizer))

    return result


def make_uniq(lang_to_bin):
    result = defaultdict(lambda: 0)

    for lang in lang_to_bin:
        bin_set = lang_to_bin[lang]
        result[lang] = 0 if 0 in bin_set else list(bin_set)[0]

    return result


def translate_with_neurotolge(translation_input: str, src_lang: str, tgt_lang: str) -> dict:
    url = "https://api.tartunlp.ai/translation/v2"

    payload = {
        "text": translation_input,
        "src": any_to_neurotolge(src_lang),
        "tgt": any_to_neurotolge(tgt_lang),
        "domain": "general",
        "application": "benchmarking"
    }

    error = None

    for i in range(5):
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()  # Raise an error for bad status codes
            return response.json()['result']
        except requests.exceptions.RequestException as e:
            error = {"error": str(e)}

    return error


def remove_dia(snt):
    if ">" in snt:
        return re.sub(r'^<[^>]+> ', '', snt)
    else:
        return snt


def neurotolge_in_batches(input_texts, src_lang, tgt_lang):
    neurotolge_langs = {'eng', 'est', 'ger', 'lit', 'lav', 'lvs', 'fin', 'rus', 'ukr', 'kca', 'koi', 'kpv', 'krl', 'lud', 'mdf', 'mhr', 'mns', 'mrj', 'myv', 'olo', 'udm', 'vep', 'liv', 'vro', 'sma', 'sme', 'smn', 'sms', 'smj', 'nor', 'hun'}

    if src_lang in neurotolge_langs and tgt_lang in neurotolge_langs:
        all_outputs = list()

        for inp_batch in do_list_in_batches(input_texts, 8):
            inp_batch_no_dia = [remove_dia(s) for s in inp_batch]
            these_outputs = translate_with_neurotolge(inp_batch_no_dia, src_lang, tgt_lang)
            if len(these_outputs) != len(inp_batch_no_dia):
                raise Exception(f"Something went wrong.: {src_lang}/{tgt_lang}/{these_outputs}")
            all_outputs += these_outputs
            log(f"Translated {len(all_outputs)}/{len(input_texts)} sentences")

        return all_outputs
    else:
        return None


def coupled_translate(coupling_specs, input_texts, input_language, output_language, debug=False):
    lang_to_bin = make_uniq(lang_bin_mapping(coupling_specs))

    all_outputs = list()

    for inp_batch in do_list_in_batches(input_texts, 32):
        if is_dec_only_llm(coupling_specs[0].tokenizer):
            these_outputs = llm_generate(coupling_specs, input_language, output_language, input_texts, debug=debug)
        else:
            encoder_embeddings, att_mask = coupled_encode(coupling_specs, lang_to_bin, input_language, inp_batch, debug=debug)
            these_outputs = coupled_generate(coupling_specs, lang_to_bin, output_language, encoder_embeddings, att_mask, debug=debug)

        all_outputs += these_outputs

    return all_outputs


def load_and_init_module_config(model_id, accelerator=None):
    config = load_module_config(model_id)

    coupling_specs = list()

    main_model = None

    for i, entry in enumerate(config):
        lang_set = entry["lang_set"]
        model_id = entry["model_id"] if i > 0 else model_id

        log(f"Loading model and tokenizer from '{model_id}'")
        model = loadmodel(model_id, accelerator)
        tokenizer, postok = load_tokenizer(model_id)

        if i == 0:
            main_model = model

        #(langs, model, tokenizer, location):
        coupling_specs += to_cpl_spec(lang_set, model, tokenizer, postok, model_id)

    return main_model, coupling_specs


def _cmdline_args(input_values):
    description = """Translate STDIN text with a translation model - TODO"""

    pos_args = ["mdl_id", "from_lang", "to_lang"]

    #post-process the arguments
    args = CmdlineArgs(description, pos_args, input_args=input_values, kw_arg_dict={"debug": False})

    log(f"Launched as {args}")

    return args


def and_i_called_this_function_do_main_too(iv):
    args = _cmdline_args(iv)

    inputs = [line.strip() for line in sys.stdin]
    # inputs = ["See on ikka tore uudis.", "Ma ikka katsetaks ka täpitähtedega tõlkimist.", "Mis tähed on täpitähed?"]

    log(f"Inputs: {inputs}")

    main_model, module_config = load_and_init_module_config(args.mdl_id)
    log("Model loaded, starting to translate")
    outputs = coupled_translate(module_config, inputs, args.from_lang, args.to_lang, debug=args.debug)

    print("\n".join(outputs))

    log("Done...")


if __name__ == "__main__":
    input_values = sys.argv[1:] if len(sys.argv) > 1 \
        else ["models/nllb", "et", "en"]

    and_i_called_this_function_do_main_too(input_values)
