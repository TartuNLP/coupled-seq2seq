#!/usr/bin/env python3

import sys

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from modelops import mdl_param_count
from traintok import get_stupid_correction, get_unk_toks, extend_tok_langs
from aux import CmdlineArgs, lang_set_maybe_smugri


def maybe_update_tokenizer(tok, tok_corpus, tok_new_langs, mdl_id):
    updated = False

    if tok_corpus is not None:
        unk_toks = get_unk_toks(tok, tok_corpus, verbose=True)

        old_len = len(tok)

        tok.add_tokens(unk_toks)

        updated = True

    if tok_new_langs is not None:
        extend_tok_langs(tok, tok_new_langs)
        updated = True

    if updated:
        upd_amt = get_stupid_correction(mdl_id)
        new_len = len(tok)
        model.resize_token_embeddings(new_len + upd_amt)

        print(f"Increased tokens from {old_len} to {new_len}")


if __name__ == '__main__':
    args = CmdlineArgs("Localize an existing HuggingFace model, possibly expanding the tokenizer",
                       pos_arg_list=["mdl_id", "save_location"],
                       kw_arg_dict={"tok_train_file": None,
                                    "new_langs": None})

    model = AutoModelForSeq2SeqLM.from_pretrained(args.mdl_id)

    tokenizer = AutoTokenizer.from_pretrained(args.mdl_id)

    if args.new_langs:
        args.new_langs = lang_set_maybe_smugri(args.new_langs)

    maybe_update_tokenizer(tokenizer, args.tok_train_file, args.new_langs, args.mdl_id)

    mdl_size, emb_size = mdl_param_count(model)
    print(f"Cached model with {mdl_size} parameters" +
          ("" if emb_size < 0 else f" of which {emb_size} ({100 * emb_size / mdl_size:.2f}%) are embeddings"))

    tokenizer.save_pretrained(args.save_location)

    model.save_pretrained(args.save_location)
