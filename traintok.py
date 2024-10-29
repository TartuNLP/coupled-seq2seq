#!/usr/bin/env python3
import os
import sys

import sentencepiece as spm
import json

from transformers import AutoTokenizer
from transformers.models.nllb import NllbTokenizer
from collections import defaultdict

def test_tok(tok, snt, lang):
    tok.src_lang = lang
    out = tok(text = snt)
    print(out['input_ids'])
    print(tok.tokenize(snt))
    print(tok.convert_ids_to_tokens(out['input_ids']))
    print("-")


def get_stupid_correction(mdl_id):
    l_mdl_id = mdl_id.lower()

    if "m2m" in l_mdl_id:
        correction = 108
    elif "nllb" in l_mdl_id:
        correction = 2
    else:
        correction = 0

    return correction


def tsv_to_json_vocab(location):
    new_location = location + ".json"

    with open(location, "r") as f, open(new_location, "w") as w:
        idx_dict = { "<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3 }

        for line in f:
            tok, _ = line.strip().split("\t")
            if tok not in idx_dict:
                idx_dict[tok] = len(idx_dict)

        json.dump(idx_dict, w)

    return new_location


def x_report_tok_corpus_overlap(tokenizer, filename):
    all_toks = set()

    with open(filename, "r", encoding='utf-8') as f:
        for line in f:
            toks = tokenizer(line.strip(), return_tensors="pt")
            for tok in toks["input_ids"][0]:
                all_toks.add(int(tok))

        print(f"Tokenizer vocab_size: {tokenizer.vocab_size}, used tokens: {len(all_toks)}")


def get_unk_toks(tokenizer, corpus, verbose=False):
    unk_id = tokenizer.unk_token_id
    unk_toks = defaultdict(int)

    all_toks = set()

    total_count = 0
    unk_count = 0

    with open(corpus, "r", encoding='utf-8') as f:
        for snt in f:
            toks = tokenizer.tokenize(snt.strip())
            ids = tokenizer.convert_tokens_to_ids(toks)

            for t, i in zip(toks, ids):
                if i == unk_id:
                    unk_toks[t] += 1
                    unk_count += 1
                total_count += 1

                all_toks.add(t)

    if verbose:
        print(f"Tokenizer vocab size: {tokenizer.vocab_size}, nr of actually used tokens: {len(all_toks)}")
        print(f"Corpus token count: {total_count}, UNK token percentage: {100*unk_count/total_count:.2f}%")

    return list(unk_toks)


def test_existing_toks(test_snt = "Pǟgiņ vȯȯnnõ mäd kolēgõn", lang = "fi", mdl_list = ["facebook/m2m100_418M", "facebook/seamless-m4t-v2-large", "facebook/nllb-200-1.3B", "google/madlad400-3b-mt", "google/gemma-7b", "google/mt5-base", "facebook/mbart-large-50"]):
    for mdl_id in mdl_list:
       print(mdl_id)
       try:
           m2mtok = AutoTokenizer.from_pretrained(mdl_id)
           test_tok(m2mtok, test_snt, lang)
       except Exception as e:
           print("Failed because:", e)


def extend_tok_langs(tokenizer, lang_set):
    if 'additional_special_tokens' in tokenizer.special_tokens_map:
        orig_langs = tokenizer.special_tokens_map['additional_special_tokens']
        orig_lang_set = set(orig_langs)

        addable_langs = list(set(lang_set) - orig_lang_set)
    else:
        addable_langs = lang_set

    tokenizer.add_special_tokens({'additional_special_tokens': orig_langs + addable_langs})


def learn_spm_tokenizer(corpus, model_dir, vocab_size, lang_set = None):
    tmp_location = os.path.join(model_dir, "sentencepiece.bpe.tmp")
    os.makedirs(model_dir, exist_ok=True)

    spm.SentencePieceTrainer.train(input=corpus, model_prefix=tmp_location, vocab_size=vocab_size)

    tok = NllbTokenizer(tmp_location + ".model", additional_special_tokens=lang_set)

    for tmp_file in (".vocab", ".model"):
        os.remove(tmp_location + tmp_file)

    return tok


if __name__ == '__main__':
    # sys.argv: either 1 or 4 arguments; 1 is for loading, 4 is for training new
    # 1: tokenizer location (to load or save)
    # 2: text corpus for training a new SentencePiece model
    # 3: size of the new model's vocabulary
    # 4: comma-separated list of languages for the new tokenizer

    try:
        tok = learn_spm_tokenizer(sys.argv[2], sys.argv[1], int(sys.argv[3]), lang_set=sys.argv[4].split(","))
        tok.save_pretrained(sys.argv[1])
    except IndexError:
        tok = AutoTokenizer.from_pretrained(sys.argv[1])

    new_lang = "liv_Latn"

    snts = ["Pǟgiņ vȯnnõ", "see on jama"]

    print(sys.argv[1])
    for snt in snts:
        test_tok(tok, snt, new_lang)

    test_existing_toks(snts[0], lang="liv_Latn")
