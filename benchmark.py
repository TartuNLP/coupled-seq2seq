#!/usr/bin/env python3

import sys
import os
import json
from collections import defaultdict

from data import split_by_lang, make_path_compatible, get_tr_pairs
from translate import coupled_translate, load_and_init_module_config
from evaluate import load as load_metric
from langconv import get_mdl_type, get_joshi_class
from accelerate import Accelerator

from aux import log


def get_hyp_cache_dir(model_location):
    hyp_location = os.path.join(model_location, "hyp_cache")

    if not os.path.exists(hyp_location):
        os.makedirs(hyp_location)

    return hyp_location


def get_hyp_cache_filename(model_location, benchmark_corpus, src_lang, tgt_lang):
    hyp_location = get_hyp_cache_dir(model_location)

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


def load_or_translate(mod_config, input_output_list, lp, model_location, benchmark_corpus):
    src_lang, tgt_lang = lp.split("-")

    inputs, _ = zip(*input_output_list)

    cache_filename, src_filename = get_hyp_cache_filename(model_location, benchmark_corpus, src_lang, tgt_lang)

    try:
        hypos = load_hyps_from_file(cache_filename)
    except FileNotFoundError:
        hypos = coupled_translate(mod_config, inputs, src_lang, tgt_lang)

        save_hyps_to_file(hypos, cache_filename)
        save_hyps_to_file(inputs, src_filename)

    return hypos


def translate_all_hyps(lp_test_set_dict, module_conf, model_id, corpus_id, accelerator):
    result = dict()

    key_list = sorted(lp_test_set_dict.keys())

    for idx, lp in enumerate(key_list):
        if idx % accelerator.num_processes == accelerator.process_index:
            log(f"Process {accelerator.process_index} translating {lp}")
            these_hyps = load_or_translate(module_conf, lp_test_set_dict[lp], lp, model_id, corpus_id)
            result[lp] = these_hyps

    accelerator.wait_for_everyone()

    return result


def get_joshi_lp(from_lang, to_lang):
    from_joshi = get_joshi_class(from_lang)
    to_joshi = get_joshi_class(to_lang)

    return f"{from_joshi}-{to_joshi}"


def get_all_scores(hyps_dict, lp_test_sets, metric_dict):
    scores = dict()
    avgs = defaultdict(list)

    for lp in lp_test_sets:
        from_lang, to_lang = lp.split("-")
        jlp = get_joshi_lp(from_lang, to_lang)

        _, outputs = zip(*lp_test_sets[lp])

        for metric_name in metric_dict:
            metric_func = metric_dict[metric_name]

            metric_value = metric_func.compute(predictions=hyps_dict[lp], references=outputs)

            scores[lp + "-" + metric_name] = metric_value['score']

            avgs[jlp + "-" + metric_name].append(metric_value['score'])

    for avg_k in avgs:
        scores[avg_k] = sum(avgs[avg_k]) / len(avgs[avg_k])

    return scores


def save_scores(scores, mdl_id, corpus):
    filename = get_benchmark_filename(mdl_id, corpus)
    with open(filename, "w") as ofh:
        json.dump(scores, ofh, indent=2, sort_keys=True)


def do_main():
    mdl_id = sys.argv[1]
    corpus = sys.argv[2]

    accelerator = Accelerator()

    log("Loading model")
    main_model, module_config = load_and_init_module_config(mdl_id, accelerator)

    log("Loading data")
    lp_test_sets = split_by_lang(filename=corpus, model_type=get_mdl_type(main_model))

    log("Loading metrics")
    exp_id = make_path_compatible(mdl_id) + "---" + make_path_compatible(corpus)

    metric_dict = {
        'bleu': load_metric("sacrebleu", experiment_id=exp_id),
        'chrf': load_metric("chrf", experiment_id=exp_id) }

    log("Starting benchmarking")

    _ = get_hyp_cache_dir(mdl_id)
    hyps_dict = translate_all_hyps(lp_test_sets, module_config, mdl_id, corpus, accelerator)

    scores = get_all_scores(hyps_dict, lp_test_sets, metric_dict)

    save_scores(scores, mdl_id, corpus)


if __name__ == '__main__':
    do_main()