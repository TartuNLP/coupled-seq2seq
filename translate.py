#!/usr/bin/env python3

import sys

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from data import do_list_in_batches, lang_bin_mapping
from coupling import load_module_config, to_cpl_spec
from collections import defaultdict
from langconv import is_nllb, is_madlad, any_to_mdl_type, get_mdl_type

hf_tok = "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"

host_remote = True


def prepare_for_translation(provided_inputs, tokenizer, input_language, output_language=None, device=None):
    if is_nllb(tokenizer):
        tokenizer.src_lang = input_language
        inputs_to_process = provided_inputs
    elif is_madlad(tokenizer):
        madlad_tgt_lang = output_language
        inputs_to_process = [f"{madlad_tgt_lang} {inp}" for inp in provided_inputs]
    else:
        raise NotImplementedError("Model type not supported")

    prepared_inputs = tokenizer(inputs_to_process, return_tensors="pt", padding=True, truncation=True, max_length=512)

    if host_remote and device is not None:
        prepared_inputs.to(device)

    frc_bos = tokenizer.get_lang_id(output_language) if output_language is not None else None

    return prepared_inputs, frc_bos


def finalize_translation(outputs, tokenizer, output_language):
    result = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return result


def loadtokenizer(mdlname="facebook/m2m100_418M"):
    tokenizer = AutoTokenizer.from_pretrained(mdlname, token=hf_tok)
    return tokenizer


def loadmodel(mdlname="facebook/m2m100_418M"):
    if host_remote:
        model = AutoModelForSeq2SeqLM.from_pretrained(mdlname, token=hf_tok, device_map="auto")
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(mdlname, token=hf_tok)
    return model


def encode(model, input_batch):
    if is_nllb(model):
        enc = model.model.encoder
    elif is_madlad(model):
        enc = model.base_model.encoder
    else:
        raise NotImplementedError(f"Model {model} is not supported yet.")

    inputs_without_labels = { k: input_batch[k] for k in input_batch if k != "labels" }

    return enc(**inputs_without_labels)


def coupled_encode(coupling_specs, lang_to_bin, input_lang, input_texts):

    mdl_type = get_mdl_type(coupling_specs[0].model)
    conv_input_lang = any_to_mdl_type(mdl_type, input_lang)

    this = coupling_specs[lang_to_bin[conv_input_lang]]

    # 0. input text --> input token IDs
    these_inputs, _ = prepare_for_translation(input_texts, this.tokenizer, conv_input_lang, device=this.model.device)
    attention_mask = these_inputs["attention_mask"]

    # 1. input token IDs --> encoder vectors
    #embeddings = this.model.model.encoder(**these_inputs)
    return encode(this.model, these_inputs), attention_mask


def coupled_generate(coupling_specs, lang_to_bin, output_lang, encoder_embeddings, att_mask):
    mdl_type = get_mdl_type(coupling_specs[0].model)
    conv_output_lang = any_to_mdl_type(mdl_type, output_lang)

    dec_idx = lang_to_bin[conv_output_lang]

    tokenizer = coupling_specs[dec_idx].tokenizer

    # 2. encoder vectors --> output token IDs
    frc_bos = tokenizer.convert_tokens_to_ids(conv_output_lang)
    raw_outputs = coupling_specs[dec_idx].model.generate(forced_bos_token_id=frc_bos, encoder_outputs=encoder_embeddings, attention_mask=att_mask)

    # 3. output token IDs --> output text
    result = finalize_translation(raw_outputs, tokenizer, conv_output_lang)

    return result


def make_uniq(lang_to_bin):
    result = defaultdict(lambda: 0)

    for lang in lang_to_bin:
        bin_set = lang_to_bin[lang]
        result[lang] = 0 if 0 in bin_set else list(bin_set)[0]

    return result


def coupled_translate(coupling_specs, input_texts, input_language, output_language):
    lang_to_bin = make_uniq(lang_bin_mapping(coupling_specs))

    all_outputs = list()

    for inp_batch in do_list_in_batches(input_texts, 64):
        encoder_embeddings, att_mask = coupled_encode(coupling_specs, lang_to_bin, input_language, inp_batch)

        these_outputs = coupled_generate(coupling_specs, lang_to_bin, output_language, encoder_embeddings, att_mask)

        all_outputs += these_outputs

    return all_outputs


def load_and_init_module_config(model_id):
    config = load_module_config(model_id)

    coupling_specs = list()

    main_model = None

    for i, entry in enumerate(config):
        lang_set = entry["lang_set"]
        model_id = entry["model_id"] if i > 0 else model_id

        model = loadmodel(model_id)
        tokenizer = loadtokenizer(model_id)

        if i == 0:
            main_model = model

        #(langs, model, tokenizer, location):
        coupling_specs += to_cpl_spec(lang_set, model, tokenizer, model_id)

    return main_model, coupling_specs


if __name__ == "__main__":
    if len(sys.argv) > 1:
        host_remote = True
    else:
        host_remote = False

    if not host_remote:
        sys.argv = ("X", "models/nllb", "et", "en")

    mdl_id = sys.argv[1]

    from_lang = sys.argv[2]
    to_lang = sys.argv[3]

    main_model, module_config = load_and_init_module_config(mdl_id)

    if host_remote:
        inputs = [line.strip() for line in sys.stdin]
    else:
        inputs = ["See on ikka tore uudis."]

    outputs = coupled_translate(module_config, inputs, from_lang, to_lang)

    print("\n".join(outputs))

