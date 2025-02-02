#!/usr/bin/env python3

import sys
import os
import torch

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_scheduler
from accelerate import Accelerator
from torch.utils.data import DataLoader, DistributedSampler

from translate import hf_tok, encode
from data import MultilingualBatchingCachingDataset
from aux import log, maybe_smugri, to_kwargs, SameLineLogger
from collections import namedtuple
from coupling import to_cpl_spec, save_all_models
from initmodel import mdl_param_count

from trainmodel import cmdline_args, get_lps_from_specs


def load_hf_tok(mdl_id, tok_id=None, verbose=False):
    if tok_id is None:
        tok_id = mdl_id

    tokenizer = AutoTokenizer.from_pretrained(tok_id, token=hf_tok)

    return tokenizer



def do_main():
    # if not host_remote:
    #     #sys.argv = ["X", "models/smol", "data/smugri4a-dev.json", "smugri", "facebook/nllb-200-distilled-600m", "smugri-high"]
    #     sys.argv = ["X", "models/smol", "data/smugri4a-dev.json", "smugri", "skip_training=yes"]

    args, train_kwargs = cmdline_args()

    log(f"Launched as {args}")

    log("loading coupled model and tokenizer")
    coupled_tokenizer = load_hf_tok(args.coupled_mdl_id, verbose=True)

    coupling_specs = to_cpl_spec(args.coupled_langs, None, coupled_tokenizer, args.save_location)

    if args.anchor_mdl_id is not None:
        log("loading anchor model and tokenizer")
        anchor_tokenizer = load_hf_tok(args.anchor_mdl_id, verbose=True)

        coupling_specs += to_cpl_spec(args.anchor_langs, None, anchor_tokenizer, args.anchor_mdl_id)

    lp_set = set(get_lps_from_specs(coupling_specs))

    batch_size = int(train_kwargs['batch']) if 'batch' in train_kwargs else 16

    shard_size = int(train_kwargs['shard_size']) if 'shard_size' in train_kwargs else 1000000

    mbd = MultilingualBatchingCachingDataset(args.train_data_file, coupling_specs, batch_size,
                                             verbose=True, leave_only=lp_set, shard_size=shard_size)
    mbd.cache_data()


if __name__ == "__main__":
    do_main()