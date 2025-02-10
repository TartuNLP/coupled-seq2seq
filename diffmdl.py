#!/usr/bin/env python3

import sys
import torch

from datetime import datetime
from transformers import AutoModelForSeq2SeqLM


def get_mdl_param_dict(mdl):
    return {n: p for n, p in mdl.named_parameters()}


def log(msg):
    sys.stderr.write(str(datetime.now()) + ": " + msg + '\n')


def _avg_diff(pd1, pd2, skip_emb):
    result = 0
    count = 0

    raw_count = 0

    for k in pd1.keys():
        # log(k)
        if not (skip_emb and "shared" in k):
            delta = pd1[k] - pd2[k]

            raw_count += 1

            if len(delta.shape) == 1:
                thiscount = delta.shape[0]
            elif len(delta.shape) == 2:
                thiscount = delta.shape[0] * delta.shape[1]
            else:
                raise Exception("Unexpected shape")
            count += thiscount
            deltasum = torch.sum(delta)
            #log(f"DETDIFF {k}: {deltasum/thiscount}")
            result += deltasum
    # print(f"Count {count}, raw count {raw_count}")

    return result / count


def avg_mdl_diff(m1, m2, skip_emb=False):
    pd1 = get_mdl_param_dict(m1)
    pd2 = get_mdl_param_dict(m2)

    assert (pd1.keys() == pd2.keys())

    return _avg_diff(pd1, pd2, skip_emb)


if __name__ == "__main__":
    mdl1_id = sys.argv[1]
    mdl2_id = sys.argv[2]

    log(f"Load mdl 1: {mdl1_id}")
    model1 = AutoModelForSeq2SeqLM.from_pretrained(mdl1_id)

    log(f"Load mdl 2: {mdl2_id}")
    model2 = AutoModelForSeq2SeqLM.from_pretrained(mdl2_id)

    log(f"Full diff: {avg_mdl_diff(model1, model2)}")

    #log(f"Encoder diff: {avg_mdl_diff(model1.get_encoder(), model2.get_encoder(), True)}")
    #log(f"Decoder diff: {avg_mdl_diff(model1.get_decoder(), model2.get_decoder(), True)}")
    #log(f"Embedding diff: {avg_mdl_diff(model1.get_input_embeddings(), model2.get_input_embeddings())}")
