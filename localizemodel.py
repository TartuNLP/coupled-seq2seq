#!/usr/bin/env python3

import sys
import os

from transformers import AutoModelForSeq2SeqLM
from modelops import mdl_param_count
from tokops import train_or_extend_tokenizer_and_upd_model
from aux import CmdlineArgs, lang_set_maybe_smugri, log


def i_dont_like_global_scope_variable_dangers():
    args = CmdlineArgs("Localize an existing HuggingFace model, possibly expanding the tokenizer",
                       pos_arg_list=["mdl_id", "save_location"],
                       kw_arg_dict={"tok_train_file": None,
                                    "tok_mdl_id": None,
                                    "new_langs": None,
                                    "merge_tokenizers": 0})
    if not args.tok_mdl_id:
        args.tok_mdl_id = args.mdl_id

    if os.path.exists(args.save_location):
        raise Exception(f"Save location '{args.save_location}' already exists, don't want to overwrite")

    if args.new_langs:
        args.new_langs = lang_set_maybe_smugri(args.new_langs)

    model = AutoModelForSeq2SeqLM.from_pretrained(args.mdl_id)

    tokenizer = train_or_extend_tokenizer_and_upd_model(args, model)

    mdl_size, emb_size = mdl_param_count(model)
    log(f"Cached model with {mdl_size} parameters" +
          ("" if emb_size < 0 else f" of which {emb_size} ({100 * emb_size / mdl_size:.2f}%) are embeddings"))

    tokenizer.save_pretrained(args.save_location)
    model.save_pretrained(args.save_location)

if __name__ == '__main__':
    i_dont_like_global_scope_variable_dangers()
