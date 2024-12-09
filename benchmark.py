#!/usr/bin/env python3

import sys
import os
import json

from data import split_by_lang
from translate import coupled_translate, load_and_init_module_config
from evaluate import load as load_metric

from datetime import datetime

from aux import log


def get_hyp_cache_filename(model_location, benchmark_corpus, src_lang, tgt_lang):
    corpus_base = os.path.basename(benchmark_corpus)
    hyp_file = f"{corpus_base}-{src_lang}-to-{tgt_lang}.hyp"
    return os.path.join(model_location, hyp_file)


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


def load_or_translate(model, mod_config, input_list, src_lang, tgt_lang, model_location, benchmark_corpus):
    cache_filename = get_hyp_cache_filename(model_location, benchmark_corpus, src_lang, tgt_lang)

    try:
        hypos = load_hyps_from_file(cache_filename)
    except FileNotFoundError:
        hypos = coupled_translate(model, mod_config, input_list, src_lang, tgt_lang)

        save_hyps_to_file(hypos, cache_filename)

    return hypos


if __name__ == '__main__':
    mdl_id = sys.argv[1]
    corpus = sys.argv[2]

    log("Loading data")
    lp_test_sets = split_by_lang(filename=corpus)

    log("Loading metrics")
    metric_bleu = load_metric("sacrebleu")
    metric_chrf = load_metric("chrf")

    log("Loading model")
    main_model, module_config = load_and_init_module_config(mdl_id)

    scores = dict()

    log("Starting benchmarking")

    for lp in lp_test_sets:
        start_time = datetime.now()
        from_lang, to_lang = lp.split("-")

        inputs, outputs = zip(*lp_test_sets[lp])

        hyps = load_or_translate(main_model, module_config, inputs, from_lang, to_lang, mdl_id, corpus)

        result1 = metric_bleu.compute(predictions=hyps, references=outputs)
        result2 = metric_chrf.compute(predictions=hyps, references=outputs, word_order=2)

        scores[lp + "-bleu"] = result1['score']
        scores[lp + "-chrf"] = result2['score']

        end_time = datetime.now()

        time_per_sample = (end_time - start_time) / len(hyps)

        log(f"LP: {lp}, BLEU: {result1['score']}, chrf++: {result2['score']}, num translated: {len(hyps)}, time per sample: {time_per_sample}")

    filename = get_benchmark_filename(mdl_id, corpus)
    with open(filename, "w") as ofh:
        json.dump(scores, ofh, indent=2, sort_keys=True)
