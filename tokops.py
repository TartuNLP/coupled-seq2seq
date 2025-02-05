#!/usr/bin/env python3
import os
import tempfile

import sentencepiece as spm
import json

from transformers import AutoTokenizer
from transformers.models.nllb import NllbTokenizer
from transformers.models.t5 import T5Tokenizer
from collections import defaultdict

from aux import CmdlineArgs, lang_set_maybe_smugri, log
from langconv import langs_to_madlad, langs_to_nllb, is_nllb, is_madlad
from translate import hf_tok


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


def get_top_toks(tokenizer, corpus, num_top_toks):
    freq_count = defaultdict(int)

    with open(corpus, "r", encoding='utf-8') as f:
        for snt in f:
            toks = tokenizer.tokenize(snt.strip())

            for t in toks:
                freq_count[t] += 1

    sorted_freq_count = sorted(freq_count.keys(), key=lambda x: -freq_count[x])

    return sorted_freq_count[:num_top_toks]


def test_existing_toks(test_snt="Pǟgiņ vȯȯnnõ mäd kolēgõn", lang="fi", mdl_list=["facebook/m2m100_418M", "facebook/seamless-m4t-v2-large", "facebook/nllb-200-1.3B", "google/madlad400-3b-mt", "google/gemma-7b", "google/mt5-base", "facebook/mbart-large-50"]):
    for mdl_id in mdl_list:
       print(mdl_id)
       try:
           m2mtok = AutoTokenizer.from_pretrained(mdl_id)
           test_tok(m2mtok, test_snt, lang)
       except Exception as e:
           print("Failed because:", e)


def extend_tok_langs(tokenizer, lang_set_raw):
    if is_nllb(tokenizer):
        lang_set = langs_to_nllb(lang_set_raw)
    elif is_madlad(tokenizer):
        lang_set = langs_to_madlad(lang_set_raw)
    else:
        raise NotImplementedError

    if 'additional_special_tokens' in tokenizer.special_tokens_map:
        orig_langs = tokenizer.special_tokens_map['additional_special_tokens']
        orig_lang_set = set(orig_langs)

        addable_langs = list(set(lang_set) - orig_lang_set)
    else:
        orig_langs = []
        addable_langs = lang_set

    tokenizer.add_special_tokens({'additional_special_tokens': orig_langs + addable_langs})


def wrap_tok_in_correct_class(location, base_model_id, lang_set):
    l_base_mdl_id = base_model_id.lower()

    if "nllb" in l_base_mdl_id:
        nllb_lang_set = langs_to_nllb(lang_set)
        return NllbTokenizer(location + ".model", additional_special_tokens=nllb_lang_set)

    elif "madlad" in l_base_mdl_id or "t5" in l_base_mdl_id:
        madlad_lang_set = langs_to_madlad(lang_set)
        return T5Tokenizer(location + ".model", additional_special_tokens=madlad_lang_set)
    else:
        raise ValueError("Incompatible model type for tokenizer")


def remove_tmp_spm_files(location):
    for tmp_file in (".vocab", ".model"):
        os.remove(location + tmp_file)


def learn_spm_tokenizer(corpus, save_location, base_model_id, vocab_size, lang_set=None):
    tmp_location = os.path.join(save_location, "sentencepiece.bpe.tmp")
    os.makedirs(save_location, exist_ok=True)

    spm.SentencePieceTrainer.train(input=corpus, model_prefix=tmp_location, vocab_size=vocab_size)

    tok = wrap_tok_in_correct_class(tmp_location, base_model_id, lang_set)

    remove_tmp_spm_files(tmp_location)

    return tok


def do_new_tok(tokargs):
    # if args.vocab_size is set then we're actually creating a new tokenizer for saving
    # if not, then it is only for merging with top N tokens
    if not hasattr(tokargs, "vocab_size") or tokargs.vocab_size is None:
        # for a merging-oriented tokenizer let's use 16000 as some default number
        voc_size = 16000

        # and generate a tmp location for saving the files:
        with tempfile.TemporaryDirectory() as tmpdirname:
            location = tmpdirname
    else:
        correction = get_stupid_correction(tokargs.mdl_id)
        voc_size = tokargs.vocab_size - correction
        location = tokargs.save_location

    return learn_spm_tokenizer(tokargs.tok_train_file, location, base_model_id=tokargs.mdl_id,
                                    vocab_size=voc_size, lang_set=tokargs.new_langs)


def remove_known_toks(toks, tokenizer):
    return [t for t in toks if not t in tokenizer.vocab]

def train_or_extend_tokenizer_and_upd_model(args, model):
    tokenizer_changed = False

    # train a new sentence-piece tokenizer
    if hasattr(args, "vocab_size") and args.vocab_size:
        assert args.new_langs is not None, "lang_set must be provided"
        assert args.tok_train_file is not None, "tok_train_file must be provided"
        args.vocab_size = int(args.vocab_size)

        log("Training new tokenizer")
        tokenizer_changed = True
        tokenizer = do_new_tok(args)
        old_len = len(tokenizer)

    # save the pre-trained model's tokenizer,
    # possibly adding new languages and tokens
    else:
        log("Reusing existing tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(args.mdl_id, token=hf_tok)
        old_len = len(tokenizer)

        if args.new_langs is not None:
            log("Extending existing tokenizer with languages")
            tokenizer_changed = True
            extend_tok_langs(tokenizer, args.new_langs)

        if args.tok_train_file:
            tokenizer_changed = True

            if args.merge_tokenizers:
                log("Extending existing tokenizer with top tokens from corpus-specific tokenizer, training new tok")
                new_tok = do_new_tok(args)
                log("And extracting top tokens from it")
                toks_to_maybe_add = get_top_toks(new_tok, args.tok_train_file, int(args.merge_tokenizers))
            else:
                log("Extending existing tokenizer with UNK tokens from corpus")
                toks_to_maybe_add = get_unk_toks(tokenizer, args.tok_train_file, verbose=True)

            toks_to_add = remove_known_toks(toks_to_maybe_add, tokenizer)

            new_tok_num = tokenizer.add_tokens(toks_to_add)
            log(f"Added {new_tok_num} tokens")

    if tokenizer_changed:
        upd_amt = get_stupid_correction(args.mdl_id)
        new_len = len(tokenizer)

        model.resize_token_embeddings(new_len + upd_amt)

        log(f"Increased tokens from {old_len} to {new_len}")

    return tokenizer
