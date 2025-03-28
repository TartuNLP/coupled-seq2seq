#!/usr/bin/env python3

import sys
import json

from collections import defaultdict

from benchmark import get_hyp_cache_dir, translate_all_hyps
from translate import load_and_init_module_config
from langconv import get_high_set, any_to_mdl_type, get_mdl_type
from accelerate import Accelerator
from aux import log as logg


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


"""
def translate_cache_dict(tr_dict, model_id, module_config, corpus_path, accelerator):
    if accelerator is None:
        for idx, lp in enumerate(sorted(tr_dict.keys())):
            if idx % accelerator.num_processes == accelerator.process_index:
                logg(f"Process {accelerator.process_index} translating {lp}", accelerator)

                inputs = sorted(tr_dict[lp].items())

                hypos = load_or_translate(module_config, inputs, lp, model_id, corpus_path)

                for i, o in zip(inputs, hypos):
                    tr_dict[lp][i] = o
    else:
        for idx, lp in enumerate(sorted(tr_dict.keys())):
            if idx % accelerator.num_processes == accelerator.process_index:
                logg(f"Process {accelerator.process_index} translating {lp}", accelerator)
    
                inputs = sorted(tr_dict[lp].items())
    
                hypos = load_or_translate(module_config, inputs, lp, model_id, corpus_path)
    
                for i, o in zip(inputs, hypos):
                    tr_dict[lp][i] = o
    
        accelerator.wait_for_everyone()
"""


def add_hires_synth_data(mdl_id, corpus_in, corpus_out):
    accelerator = Accelerator()

    logg("Loading data", accelerator)
    data = load_raw_data(corpus_in)


    logg("Loading model", accelerator)
    main_model, module_config = load_and_init_module_config(mdl_id, accelerator)

    mdl_type = get_mdl_type(main_model)

    if accelerator.is_main_process:
        _ = get_hyp_cache_dir(mdl_id, create=True)
    l = len(data)

    for i, part in enumerate(data):
        tr_dict = defaultdict(lambda: defaultdict(lambda: None))

        logg(f"Preparing part {i+1}/{l}", accelerator)
        def _transfer(tup, src, tgt):
            srcm = any_to_mdl_type(mdl_type, src)
            tgtm = any_to_mdl_type(mdl_type, tgt)

            lp = f"{srcm}-{tgtm}"
            inp_snt = tup[src]

            # this "touches" the value: if it was not there, now it is None
            # and if it was there, then we use it
            if tr_dict[lp][inp_snt] is not None:
                tup[tgt] = tr_dict[lp][inp_snt]

        # collect sentences to translate
        apply_func_to_hires_snts(part['sentences'], _transfer)

        in_tr_dict_list = { lp: sorted(tr_dict[lp].items()) for lp in tr_dict }

        logg(f"Translating part {i+1}/{l}", accelerator)
        #translate_cache_dict(tr_dict, mdl_id, module_config, corpus_in, accelerator)
        translate_all_hyps(in_tr_dict_list, module_config, mdl_id, corpus_in, accelerator)

        logg(f"Collecting part {i+1}/{l}", accelerator)
        out_tr_dict_list = translate_all_hyps(in_tr_dict_list, module_config, mdl_id, corpus_in)

        for lp in out_tr_dict_list:
            for inp, outp in out_tr_dict_list[lp]:
                tr_dict[lp][inp] = outp

        # put translations back into data structure
        logg(f"Integrating part {i+1}/{l}", accelerator)
        apply_func_to_hires_snts(part['sentences'], _transfer)

    logg("Saving data", accelerator)
    save_raw_data(corpus_out, data)

if __name__ == '__main__':
    try:
        mdl_id_param = sys.argv[1]
        corpus_param = sys.argv[2]
        corpus_output_param = sys.argv[3]
    except IndexError:
        mdl_id_param = "models/nllb600m"
        corpus_param = "data/flt.json"
        corpus_output_param = "data/fltout.json"

    add_hires_synth_data(mdl_id_param, corpus_param, corpus_output_param)
