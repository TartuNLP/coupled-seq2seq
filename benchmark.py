#!/usr/bin/env python3

import sys
import os
import json
from collections import defaultdict

from data import split_by_lang, make_path_compatible, get_tr_pairs
from translate import coupled_translate, load_and_init_module_config
from evaluate import load as load_metric
from langconv import get_mdl_type, get_joshi_class
from datetime import datetime

from aux import log


def get_hyp_cache_filename(model_location, benchmark_corpus, src_lang, tgt_lang):
    hyp_location = os.path.join(model_location, "hyp_cache")

    if not os.path.exists(hyp_location):
        os.makedirs(hyp_location)

    corpus_base = os.path.basename(benchmark_corpus)
    basename = f"{corpus_base}-{src_lang}-to-{tgt_lang}"

    hyp_file = os.path.join(hyp_location, f"{basename}.hyp")
    src_file = os.path.join(hyp_location, f"{basename}.src")

    return hyp_file, src_file


def get_benchmark_filename(model_location, benchmark_corpus):
    corpus_base = os.path.basename(benchmark_corpus)
    hyp_file = f"{corpus_base}-scores.json"
    return os.path.join(model_location, hyp_file)


def load_hyps_from_file(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return [line.strip() for line in f]


def save_hyps_to_file(hypos, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for hyp in hypos:
            f.write(hyp + "\n")


def load_or_translate(mod_config, input_list, src_lang, tgt_lang, model_location, benchmark_corpus):
    cache_filename, src_filename = get_hyp_cache_filename(model_location, benchmark_corpus, src_lang, tgt_lang)

    try:
        hypos = load_hyps_from_file(cache_filename)
    except FileNotFoundError:
        hypos = coupled_translate(mod_config, input_list, src_lang, tgt_lang)

        save_hyps_to_file(hypos, cache_filename)
        save_hyps_to_file(input_list, src_filename)

    return hypos


def do_main():
    mdl_id = sys.argv[1]
    corpus = sys.argv[2]

    log("Loading model")
    main_model, module_config = load_and_init_module_config(mdl_id)

    log("Loading data")
    # lp_test_sets = split_by_lang(filename=corpus)
    lp_test_sets = split_by_lang(filename=corpus, model_type=get_mdl_type(main_model))

    log("Loading metrics")
    exp_id = make_path_compatible(mdl_id) + "---" + make_path_compatible(corpus)
    metric_bleu = load_metric("sacrebleu", experiment_id=exp_id)
    metric_chrf = load_metric("chrf", experiment_id=exp_id)

    scores = dict()

    log("Starting benchmarking")

    avgs = defaultdict(list)

    for ii, lp in enumerate(lp_test_sets):
        start_time = datetime.now()
        from_lang, to_lang = lp.split("-")

        from_joshi = get_joshi_class(from_lang)
        to_joshi = get_joshi_class(to_lang)

        jlp = f"{from_joshi}-{to_joshi}"

        inputs, outputs = zip(*lp_test_sets[lp])

        hyps = load_or_translate(module_config, inputs, from_lang, to_lang, mdl_id, corpus)

        result1 = metric_bleu.compute(predictions=hyps, references=outputs)
        result2 = metric_chrf.compute(predictions=hyps, references=outputs, word_order=2)

        scores[lp + "-bleu"] = result1['score']
        scores[lp + "-chrf"] = result2['score']

        avgs[jlp + "-bleu"].append(result1['score'])
        avgs[jlp + "-chrf"].append(result2['score'])

        end_time = datetime.now()

        time_per_sample = (end_time - start_time) / len(hyps)

        log(f"{ii+1}/{len(lp_test_sets)} LP: {lp}, BLEU: {result1['score']:03f}, " +
            f"chrf++: {result2['score']:03f}, num translated: {len(hyps)}, time per sample: {time_per_sample}")

    for avg_k in avgs:
        scores[avg_k] = sum(avgs[avg_k]) / len(avgs[avg_k])

    filename = get_benchmark_filename(mdl_id, corpus)
    with open(filename, "w") as ofh:
        json.dump(scores, ofh, indent=2, sort_keys=True)

if __name__ == '__main__':
    do_main()