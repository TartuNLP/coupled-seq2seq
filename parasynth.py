#!/usr/bin/env python3

import sys
import json

from collections import defaultdict

from benchmark import get_hyp_cache_dir, translate_all_hyps
from translate import load_and_init_module_config
from langconv import get_high_set, any_to_mdl_type, get_mdl_type
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


def report_part_stats(part, part_index, num_parts):
    hi_set = get_high_set()

    num_snts = len(part['sentences'])
    hires_langs = {k for k in part['sentences'][0] if "dia" not in k and k in hi_set}
    num_hires_langs = len(hires_langs)
    langs_to_do = hi_set - hires_langs
    num_to_translate = num_hires_langs * len(langs_to_do)

    log(f"Part {part_index + 1}/{num_parts}; {num_snts} sentences, num hires: {num_hires_langs}, to translate: {num_to_translate}")

    return num_snts * num_hires_langs, num_snts * num_to_translate


def add_hires_synth_data(mdl_id, corpus_in, corpus_out, dry=False):
    accelerator = Accelerator()

    log("Loading data", accelerator)
    data = load_raw_data(corpus_in)


    log("Loading model", accelerator)
    if dry:
        main_model, module_config = None, None
        mdl_type = None
    else:
        main_model, module_config = load_and_init_module_config(mdl_id, accelerator)
        mdl_type = get_mdl_type(main_model)

    if accelerator.is_main_process:
        _ = get_hyp_cache_dir(mdl_id, create=True)
    l = len(data)

    tot_snt = 0
    tot_tr = 0

    for i, part in enumerate(data):
        tr_dict = defaultdict(lambda: defaultdict(lambda: None))

        num_snt, num_tr = report_part_stats(part, i, l)
        tot_snt += num_snt
        tot_tr += num_tr

        if not dry:
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

            log(f"Translating part {i+1}/{l}", accelerator)
            #translate_cache_dict(tr_dict, mdl_id, module_config, corpus_in, accelerator)
            translate_all_hyps(in_tr_dict_list, module_config, mdl_id, f"{corpus_in}-{i}", accelerator)

            log(f"Collecting part {i+1}/{l}", accelerator)
            out_tr_dict_list = translate_all_hyps(in_tr_dict_list, module_config, mdl_id, corpus_in)

            for lp in out_tr_dict_list:
                for inp, outp in out_tr_dict_list[lp]:
                    tr_dict[lp][inp] = outp

            # put translations back into data structure
            log(f"Integrating part {i+1}/{l}", accelerator)
            apply_func_to_hires_snts(part['sentences'], _transfer)

    log(f"Total sentences: {tot_snt}, total to generate: {tot_tr}", accelerator)
    if not dry:
        log("Saving data", accelerator)
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

    try:
        _ = sys.argv[4]
        dry_run = True
    except IndexError:
        dry_run = False

    add_hires_synth_data(mdl_id_param, corpus_param, corpus_output_param, dry_run)
