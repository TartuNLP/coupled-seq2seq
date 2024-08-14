#!/usr/bin/env python3

import sys

from data import split_by_lang
from translate import coupled_translate, load_and_init_module_config
from evaluate import load as load_metric

if __name__ == '__main__':
    mdl_id = sys.argv[1]
    corpus = sys.argv[2]

    module_config = load_and_init_module_config(mdl_id)

    metric = load_metric("sacrebleu")

    lp_test_sets = split_by_lang(filename=corpus)

    for lp in lp_test_sets:
        from_lang, to_lang = lp.split("-")

        inputs, outputs = zip(*lp_test_sets[lp])

        hyps = coupled_translate(module_config, inputs, from_lang, to_lang)

        result = metric.compute(predictions=hyps, references=outputs)

        print(f"LP: {lp}, BLEU: {result['score']}")


