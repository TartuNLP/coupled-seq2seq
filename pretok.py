#!/usr/bin/env python3

import os

from transformers import AutoTokenizer

from translate import hf_tok
from data import MultilingualBatchingCachingDataset
from aux import log, CmdlineArgs
from langconv import lang_set_maybe_smugri
from modelops import to_cpl_spec


def load_hf_tok(mdl_id, tok_id=None, verbose=False):
    if tok_id is None:
        tok_id = mdl_id

    tokenizer = AutoTokenizer.from_pretrained(tok_id, token=hf_tok)

    return tokenizer


def _cmdline_args():
    description = """Pre-tokenize data and cache the results"""

    pos_args = ["mdl_id", "train_file", "langs", "cache_path"]
    pos_types = [str, str, lang_set_maybe_smugri, str]

    kw_args = { "anchor_mdl_id": None, "anchor_langs": None, "batch_size": 16, "shard_size": 100000 }

    #post-process the arguments
    args = CmdlineArgs(description, pos_arg_list=pos_args, pos_arg_types=pos_types, kw_arg_dict=kw_args)

    if args.anchor_langs is not None:
        args.anchor_langs = lang_set_maybe_smugri(args.anchor_langs)

    # if the directory args.save_location already exists, raise an exception:
    if os.path.exists(args.cache_path):
        raise Exception(f"Save location '{args.cache_path}' already exists, don't want to overwrite")

    log(f"Launched as {args}")

    return args


def oh_look_another_do_main_function():
    args = _cmdline_args()

    log("loading tokenizer")
    main_tokenizer = load_hf_tok(args.mdl_id, verbose=True)

    coupling_specs = to_cpl_spec(args.langs, None, main_tokenizer, None)

    if args.anchor_mdl_id is not None:
        log("loading anchor model tokenizer")
        anchor_tokenizer = load_hf_tok(args.anchor_mdl_id, verbose=True)

        coupling_specs += to_cpl_spec(args.anchor_langs, None, anchor_tokenizer, None)

    mbd = MultilingualBatchingCachingDataset(args.train_file, coupling_specs, args)
    mbd.load_and_cache_data(args.cache_path)


if __name__ == "__main__":
    oh_look_another_do_main_function()
