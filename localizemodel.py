#!/usr/bin/env python3

import sys

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from initmodel import mdl_param_count
from traintok import get_stupid_correction, get_unk_toks

if __name__ == '__main__':
    try:
        mdl_id = sys.argv[1]
        mdl_new_name = sys.argv[2]

        model = AutoModelForSeq2SeqLM.from_pretrained(mdl_id)

        tokenizer = AutoTokenizer.from_pretrained(mdl_id)

        try:
            tok_corpus = sys.argv[3]

            unk_toks = get_unk_toks(tokenizer, tok_corpus, verbose=True)
            old_len = len(tokenizer)
            tokenizer.add_tokens(unk_toks)

            upd_amt = get_stupid_correction(mdl_id)
            new_len = len(tokenizer)
            model.resize_token_embeddings(new_len + upd_amt)

            print(f"Increased tokens from {old_len} to {new_len}")

        except IndexError:
            pass

        mdl_size, emb_size = mdl_param_count(model)
        print(f"Cached model with {mdl_size} parameters" +
              ("" if emb_size < 0 else f" of which {emb_size} ({100 * emb_size / mdl_size:.2f}%) are embeddings"))

        tokenizer.save_pretrained(mdl_new_name)

        model.save_pretrained(mdl_new_name)
    except IndexError:
        print("Usage: localizemodel.py  <model_id>  <model_new_name>  [<tok_corpus>]")
