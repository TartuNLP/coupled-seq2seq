#!/usr/bin/env python3

import json
import sys

from random import choices
from collections import defaultdict

from accelerate import Accelerator

from data import LazyTokenizingInferenceDataset
from inference import predict
from metrics import SMUGRI_RES
from aux import log, load_tokenizer, load_model
from promptops import PF_TR_FLT

# hi-res languages and how likely we should be to translate into them from other hi-res langs
HI_RES_WITH_WEIGHTS = {"English": 13, "Estonian": 11, "Finnish": 8, "Hungarian": 3, "Latvian": 2,
                       "Russian": 4, "Swedish": 2, "Norwegian": 2, "German": 0, "French": 0}


def nest():
    return defaultdict(nest)


def get_gen_lang(lang):
    return lang.replace(", dictionary", "").replace(", speech", "")


def is_hi(lang):
    return lang in HI_RES_WITH_WEIGHTS or lang in {'est', 'eng', 'fin', 'hun', 'lvs', 'nor', 'rus'}


def sample_and_count_pivot_entries(input_data):
    result = nest()

    for entry in input_data:
        src_lang = entry['src_lang']

        gen_src_lang = get_gen_lang(src_lang)

        if entry['task'] != 'translate' or 'bible' in src_lang or not is_hi(gen_src_lang):
            continue

        this_dict = result[ entry['tgt_lang'] ][ entry['tgt_segm'] ][ gen_src_lang ]
        src_segm = entry['src_segm']

        if src_segm in this_dict:
            this_dict[src_segm] += 1
        else:
            this_dict[src_segm] = 1

    return result


def get_out_langs_with_weights(exclude):
    output_langs = { k: v for k, v in HI_RES_WITH_WEIGHTS.items() if k not in exclude }

    population, raw_weights = zip(*output_langs.items())
    norm_sum = float(sum(raw_weights))
    weights = [w / norm_sum for w in raw_weights]

    return population, weights


def do_something_without_global_ctx():
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    log(f"Reading input from {input_file}")
    with open(input_file, 'r') as fh_in:
        data = json.load(fh_in)

    augm_data = list()

    stats = sample_and_count_pivot_entries(data)

    log(f"Generating pairs")
    for lo_res_lang, dict1 in stats.items():
        for lo_res_segm, dict2 in dict1.items():
            # this segm in this lo_res_lang has M hi-res translations
            # we need to translate these translations into other hi-res langs

            out_lang_candidates, weights = get_out_langs_with_weights(dict2)

            for hi_res_lang, dict3 in dict2.items():
                for hi_res_segm, cnt in dict3.items():
                    gen_lo_res_lang = lo_res_lang.split(',')[0]
                    if gen_lo_res_lang in SMUGRI_RES['xlow']:
                        ccnt = cnt * 5
                    elif gen_lo_res_lang in SMUGRI_RES['low']:
                        ccnt = cnt * 3
                    else:
                        ccnt = cnt

                    repl_hi_res_langs = set(choices(out_lang_candidates, weights=weights, k=ccnt))

                    for new_hi_res_lang in repl_hi_res_langs:
                        augm_data.append({
                            'lo_lang': lo_res_lang,
                            'lo_segm': lo_res_segm,
                            'hi_lang': hi_res_lang,
                            'hi_segm': hi_res_segm,
                            'new_hi_res_lang': new_hi_res_lang,
                        })

    log(f"Saving output to {output_file}")
    with open(output_file, 'w') as fh_out:
        json.dump(augm_data, fh_out, indent=2)

    log(f"Done")


def do_something_else_without_global_ctx():
    acc = Accelerator()
    device = acc.device
    mdl_id = "utter-project/EuroLLM-9B-Instruct"

    with open(f"{sys.argv[1]}{acc.process_index}", 'r') as fh_in:
        data = json.load(fh_in)

    for entry in data:
        entry['hyp-translation'] = entry['hyp-output']
        del (entry['hyp-output'])

    tok = load_tokenizer(mdl_id, acc)

    dl = LazyTokenizingInferenceDataset(data, tok, PF_TR_FLT, debug=False)

    mdl = load_model(mdl_id, device, acc, attention="flash_attention_2")
    mdl.eval()

    outputs = predict(mdl, tok, dl, acc,
                      multi=True,
                      debug=False,
                      max_len=2500,
                      sync=False)

    with open(f"{sys.argv[1]}{acc.process_index}-flt", 'w') as fh_out:
        json.dump(outputs, fh_out, indent=2)


if __name__ == "__main__":
    #do_something_without_global_ctx()
    do_something_else_without_global_ctx()
