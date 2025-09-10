#!/usr/bin/env python3

import json
import sys

from random import choices, shuffle
from collections import defaultdict

from accelerate import Accelerator

from data import LazyTokenizingInferenceDataset
from inference import filter_tr_pair
from metrics import SMUGRI_RES
from aux import log, load_tokenizer, load_model, env_stuff
from promptops import PF_TR_FLT

# hi-res languages and how likely we should be to translate into them from other hi-res langs
HI_RES_WITH_WEIGHTS = {"English": 13, "Estonian": 11, "Finnish": 8, "Hungarian": 3, "Latvian": 2,
                       "Russian": 4, "Swedish": 2, "Norwegian": 2, "German": 0, "French": 0}


def nest():
    return defaultdict(nest)


def get_gen_lang(lang):
    if lang.startswith("Estonian"):
        return lang.replace(", dictionary", "").replace(", speech", "").replace(", ocr", "")
    else:
        return lang.split(',')[0]


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


def get_multiplier(lang):
    gen_lang = get_gen_lang(lang)

    if gen_lang in SMUGRI_RES['xlow']:
        return 6
    elif gen_lang in SMUGRI_RES['low']:
        return 4
    else:
        return 2


def do_pre_bt_generation():
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    out_lang_candidates, weights = get_out_langs_with_weights({})

    log(f"Reading input from {input_file}")
    with open(input_file, 'r') as fh_in:
        data = json.load(fh_in)

    augm_data = list()

    for entry in data:
        gen_src_lang = get_gen_lang(entry['src_lang'])

        if entry['task'] == 'generate' and not is_hi(gen_src_lang):
            mul = get_multiplier(entry['src_lang'])
            log(f"Generating for {entry['src_lang']}: {mul}")
            repl_hi_res_langs = set(choices(out_lang_candidates, weights=weights, k=mul))

            for tgt_l in repl_hi_res_langs:
                augm_data.append({
                    'src_lang': entry['src_lang'],
                    'tgt_lang': tgt_l,
                    'src_segm': entry['src_segm'],
                    'task': 'translate'
                })
        else:
            log(f"Skipping {entry['src_lang']}")


    log(f"Saving output to {output_file}")
    with open(output_file, 'w') as fh_out:
        json.dump(augm_data, fh_out, indent=2)

    log(f"Done")


def do_pre_pivot_generation():
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
                    ccnt = cnt * get_multiplier(lo_res_lang)

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


def lets_do_some_filtering():
    acc = Accelerator()
    if not acc.is_main_process:
        sys.exit(0)
    res = []
    for f in sys.argv[2:]:
        log(f"Processing {f}")
        with open(f, 'r') as fh_in:
            data = json.load(fh_in)

        log(f"Filtering")
        for entry in data:
            entry['flt'] = filter_tr_pair(entry['hi_segm'],
                                          entry['hyp-output'],
                                          entry['hi_lang'],
                                          entry['new_hi_res_lang'])
            res.append(entry)
    log(f"Saving")

    with open(sys.argv[1], 'w') as fh_out:
        json.dump(res, fh_out, indent=2)

"""
  {
    "lo_lang": "Livonian, Standard",
    "lo_segm": "Izā um tämpõ kuonnõ.",
    "hi_lang": "Estonian",
    "hi_segm": "Isa on täna kodus.",
    "new_hi_res_lang": "Latvian",
    "hyp-output": "Tēvs šodien ir mājās.",
    "hyp-index": 96,
    "flt": "ok"
  }
  {
    "src_segm": "Elettih ukko da akka. A akka oli ilman paha akka, nagoli judai, i judai, i judai, i judai. Nagoli, midä mužikka, ukko, ruadau, hänellä nagoli pahoin.",
    "tgt_segm": "Жили старик и старуха. А старуха была очень плохая старуха, всегда шумела, и шумела, и шумела, и шумела. Всегда, что муж, старик, делает, ей всегда плохо.",
    "src_lang": "Proper Karelian, Tolmachi",
    "tgt_lang": "Russian",
    "task": "translate"
  },

"""

def do_conversion():
    out_fn = sys.argv[1]

    aug_data = []

    for in_fn in sys.argv[2:]:
        log(f"Processing {in_fn}")
        with open(in_fn, 'r') as fh_in:
            data = json.load(fh_in)

        for entry in data:
            if entry['flt'] == 'ok':
                aug_data.append({
                    'src_segm': entry['hyp-output'],
                    'src_lang': entry['new_hi_res_lang'] + ", synth",
                    'tgt_segm': entry['lo_segm'],
                    'tgt_lang': entry['lo_lang'],
                    'task': 'approx-translate'
                })

    log(f"Saving {out_fn}")
    with open(out_fn, 'w') as fh_out:
        json.dump(aug_data, fh_out, indent=2)

if __name__ == "__main__":
    #do_pre_pivot_generation()
    do_pre_bt_generation()

    #do_something_else_without_global_ctx()
    #lets_do_some_filtering()
    #do_conversion()
