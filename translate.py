#!/usr/bin/env python3

import sys

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from data import do_list_in_batches, lang_bin_mapping
from vivisect import load_module_config, to_cpl_spec, switch_modules
from collections import defaultdict
from langconv import any_to_madlad, any_to_nllb, is_nllb, is_madlad

hf_tok = "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"

host_remote = True

SMUGRI_LOW = "fkv,izh,kca,koi,kpv,krl,liv,lud,mdf,mhr,mns,mrj,myv,olo,sjd,sje,sju,sma,sme,smj,smn,sms,udm,vep,vot,vro"
SMUGRI_HIGH = "deu,eng,est,fin,hun,lvs,nor,rus,swe"


def maybe_smugri(lang_def):
    if lang_def == "smugri-low":
        return SMUGRI_LOW
    elif lang_def == "smugri-high":
        return SMUGRI_HIGH
    elif lang_def == "smugri":
        return SMUGRI_LOW + "," + SMUGRI_HIGH
    else:
        return lang_def


def prepare_for_translation(inputs, tokenizer, input_language, output_language=None):
    if is_nllb(tokenizer):
        tokenizer.src_lang = any_to_nllb(input_language)
        inputs_to_process = inputs
    elif is_madlad(tokenizer):
        madlad_tgt_lang = any_to_madlad(output_language)
        inputs_to_process = [f"{madlad_tgt_lang} {inp}" for inp in inputs]

    prepared_inputs = tokenizer(inputs_to_process, return_tensors="pt", padding=True, truncation=True, max_length=512)

    if host_remote:
        inputs.to("cuda:0")

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


def coupled_encode(coupling_specs, lang_to_bin, input_language, input_texts):
    this = coupling_specs[lang_to_bin[input_language]]

    # 0. input text --> input token IDs
    these_inputs, _ = prepare_for_translation(input_texts, this.tokenizer, input_language)

    # 1. input token IDs --> encoder vectors
    embeddings = this.encoder(**these_inputs)

    return embeddings


def coupled_generate(model, coupling_specs, lang_to_bin, output_language, encoder_embeddings):
    dec_idx = lang_to_bin[output_language]

    switch_modules(model, coupling_specs, None, dec_idx)

    tokenizer = coupling_specs[dec_idx].tokenizer

    # 2. encoder vectors --> output token IDs
    frc_bos = tokenizer.get_lang_id(output_language)
    raw_outputs = model.generate(forced_bos_token_id=frc_bos, encoder_outputs=encoder_embeddings)

    # 3. output token IDs --> output text
    result = finalize_translation(raw_outputs, tokenizer, output_language)

    return result


def make_uniq(lang_to_bin):
    result = defaultdict(lambda: 0)

    for lang in lang_to_bin:
        bin_set = lang_to_bin[lang]
        result[lang] = 0 if 0 in bin_set else list(bin_set)[0]

    return result


def coupled_translate(model, coupling_specs, input_texts, input_language, output_language):
    lang_to_bin = make_uniq(lang_bin_mapping(coupling_specs))

    all_outputs = list()

    for inp_batch in do_list_in_batches(input_texts, 16):
        encoder_embeddings = coupled_encode(coupling_specs, lang_to_bin, input_language, inp_batch)

        these_outputs = coupled_generate(model, coupling_specs, lang_to_bin, output_language, encoder_embeddings)

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
        sys.argv = ("X", "models/m2m_orig", "et", "de")

    mdl_id = sys.argv[1]

    from_lang = sys.argv[2]
    to_lang = sys.argv[3]

    main_model, module_config = load_and_init_module_config(mdl_id)

    if host_remote:
        inputs = [line.strip() for line in sys.stdin]
    else:
        inputs = ["See on ikka tore uudis."]

    outputs = coupled_translate(main_model, module_config, inputs, from_lang, to_lang)

    print("\n".join(outputs))
