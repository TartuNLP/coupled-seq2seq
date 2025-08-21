#!/usr/bin/env python3

from data import read_input
from aux import log

import sys
import json
from collections import defaultdict
from evaluate import load as load_metric

SMUGRI_RES = {
    'high': set("Estonian,English,Russian,Finnish,Hungarian,Latvian,German,Swedish,Norwegian,French".split(",")),
    'mid': set("Komi,Komi-Zyrian,Northern Sami,Meadow Mari".split(",")),
    'low': set("Udmurt,Proper Karelian,Southern Sami,Livvi,Veps,Moksha,Erzya,Lule Sami,Võro,Hill Mari,"
               "Komi-Permyak,Inari Sami".split(",")),
    'xlow': set("Ludian,Livonian,Izhorian,Votic,Shur Khanty,Skolt Sami,Meänkieli,"
                "Sred Khanty,Surgut Khanty,Priur Khanty,Vakh Khanty,Unk Khanty,"
                "Pite Sami,Mansi,Kazym Khanty,Kven,Ume Sami,Kildin Sami".split(","))
}

def _gen_lang(lang):
    return lang.split(",")[0]


def _hi_or_lo_lang(lang):
    gen_lang = _gen_lang(lang)

    for k, v in SMUGRI_RES.items():
        if gen_lang in v:
            return k

    log(f"Unrecognized language: {lang} / {gen_lang}")
    return '?'


def _collect_lp_pairs(json_inputs, str_outputs):
    sets_by_lp = defaultdict(list)

    for i, o in zip(json_inputs, str_outputs):
        ref = i["tgt_segm"]
        hyp = o
        det_lp = 'detailed: ' + i["src_lang"] + " -> " + i["tgt_lang"]
        gen_lp = 'general: ' + _gen_lang(i["src_lang"]) + " -> " + _gen_lang(i["tgt_lang"])
        hilo_lp = 'classes: ' + _hi_or_lo_lang(i["src_lang"]) + " -> " + _hi_or_lo_lang(i["tgt_lang"])

        sets_by_lp[det_lp].append((hyp, ref))
        sets_by_lp[gen_lp].append((hyp, ref))
        sets_by_lp[hilo_lp].append((hyp, ref))

    return sets_by_lp


def compute_metrics(json_inputs, str_outputs):
    sets_by_lp = _collect_lp_pairs(json_inputs, str_outputs)

    metric = load_metric("chrf")

    result = []

    for lp in sets_by_lp:
        preds, outputs = zip(*sets_by_lp[lp])
        metric_value = metric.compute(predictions=preds, references=outputs)

        result.append((lp, metric_value, len(preds)))

    return result


def sort_and_cut(json_outputs):
    outputs = [x[1] for x in sorted(json_outputs, key=lambda x: x[0])]
    return outputs


def read_json_output(path, req_len):
    try:
        result = read_input(path, "json")
        log(f"Read the file from {path}")
    except FileNotFoundError:
        result = []

        try:
            i = 0
            while True:
                res_i = read_input(f"{path}.{i}", "json")
                result += res_i
                i += 1
        except FileNotFoundError:
            pass
        log(f"Read {i} files from {path}.idx")

    assert len(result) == req_len, "something went wrong with the outputs"

    return sort_and_cut(result)


def avoid_global_scope():
    json_inputs = read_input(sys.argv[1], "json")
    str_outputs = read_json_output(sys.argv[2], len(json_inputs))

    lp_metric_dict = compute_metrics(json_inputs, str_outputs)

    for lp, metric, size in lp_metric_dict:
        print(f"{lp}: {metric['score']:.2f} ({size})")

if __name__ == "__main__":
    avoid_global_scope()