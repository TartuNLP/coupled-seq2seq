#!/usr/bin/env python3

import sys
import json

from collections import defaultdict

from benchmark import load_or_translate
from translate import load_and_init_module_config
from langconv import get_high_set
from accelerate import Accelerator
from aux import log


def load_raw_data(path):
    with open(path, 'r') as f:
        return json.load(f)


def save_raw_data(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def apply_func_to_hires_snts(snt_set, func):
    high_set = get_high_set()

    for tupl in snt_set:
        langs = [k for k in tupl if not "-dia" in k and k in high_set]

        if langs:
            revlangs = high_set - set(langs)

            for revlang in revlangs:
                for lang in langs:
                    # translate sentences tupl[lang] from lang to revlang
                    # OR
                    # add the result as tupl[revlang]
                    func(tupl, lang, revlang)


def translate_cache_dict(tr_dict, model_id, module_config, corpus_path, accelerator):
    for idx, lp in enumerate(sorted(tr_dict.keys())):
        if idx % accelerator.num_processes == accelerator.process_index:
            log(f"Process {accelerator.process_index} translating {lp}")

            inputs = sorted(tr_dict[lp])

            #hypos = coupled_translate(module_config, inputs, src_lang, tgt_lang)

            hypos = load_or_translate(module_config, inputs, lp, model_id, corpus_path)

            for i, o in zip(inputs, hypos):
                tr_dict[lp][i] = o

    accelerator.wait_for_everyone()


def add_hires_synth_data(mdl_id, corpus_in, corpus_out):
    log("Loading data")
    data = load_raw_data(corpus_in)

    accelerator = Accelerator()

    log("Loading model")
    main_model, module_config = load_and_init_module_config(mdl_id, accelerator)

    for part in data:
        tr_dict = defaultdict(lambda: defaultdict(lambda: None))

        log(f"Preparing {part}")
        def _transfer(tup, src, tgt):
            lp = f"{src}-{tgt}"
            inp_snt = tup[src]

            # this "touches" the value: if it was not there, now it is None
            # and if it was there, then we use it
            if tr_dict[lp][inp_snt] is not None:
                tup[tgt] = tr_dict[lp][inp_snt]

        # collect sentences to translate
        apply_func_to_hires_snts(part, _transfer)

        log(f"Translating {part}")
        translate_cache_dict(tr_dict, mdl_id, module_config, corpus_in, accelerator)

        # put translations back into data structure
        log(f"Integrating {part}")
        apply_func_to_hires_snts(part, _transfer)

    log("Saving data")
    save_raw_data(corpus_out, data)

if __name__ == '__main__':
    mdl_id_param = sys.argv[1]
    corpus_param = sys.argv[2]
    corpus_output_param = sys.argv[3]

    add_hires_synth_data(mdl_id_param, corpus_param, corpus_output_param)
