#!/usr/bin/env python3

import sys

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from initmodel import mdl_param_count
from traintok import get_stupid_correction, get_unk_toks, extend_tok_langs
from aux import maybe_smugri


def maybe_update_tokenizer(tok, tok_corpus, tok_new_langs = None):
    unk_toks = get_unk_toks(tok, tok_corpus, verbose=True)

    old_len = len(tok)

    tok.add_tokens(unk_toks)

    if tok_new_langs is not None:
        extend_tok_langs(tok, tok_new_langs)

    upd_amt = get_stupid_correction(mdl_id)
    new_len = len(tok)
    model.resize_token_embeddings(new_len + upd_amt)

    print(f"Increased tokens from {old_len} to {new_len}")


if __name__ == '__main__':
    try:
        mdl_id = sys.argv[1]
        mdl_new_name = sys.argv[2]

        model = AutoModelForSeq2SeqLM.from_pretrained(mdl_id)

        tokenizer = AutoTokenizer.from_pretrained(mdl_id)

        if len(sys.argv) > 3:
            tok_corp_file = sys.argv[3]

            try:
                tok_new_langs = maybe_smugri(sys.argv[4]).split(",")
            except IndexError:
                tok_new_langs = None

            maybe_update_tokenizer(tokenizer, tok_corp_file, tok_new_langs)

        mdl_size, emb_size = mdl_param_count(model)
        print(f"Cached model with {mdl_size} parameters" +
              ("" if emb_size < 0 else f" of which {emb_size} ({100 * emb_size / mdl_size:.2f}%) are embeddings"))

        tokenizer.save_pretrained(mdl_new_name)

        model.save_pretrained(mdl_new_name)
    except IndexError:
        print("Usage: localizemodel.py  <model_id>  <model_new_name>  [<tok_corpus> <new_langs>]")
