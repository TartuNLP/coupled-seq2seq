#!/usr/bin/env python3
import os
import sentencepiece as spm
import json
import sys

from transformers import AutoTokenizer
from transformers.models.nllb import NllbTokenizer
from transformers.models.t5 import T5Tokenizer
from collections import defaultdict

from aux import log, CmdlineArgs
from langconv import langs_to_madlad, langs_to_nllb, is_nllb, is_madlad, is_dec_only_llm
from modelops import hf_tok


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


def extend_tok_langs(tokenizer, lang_set_raw):
    if is_nllb(tokenizer):
        lang_set = langs_to_nllb(lang_set_raw)
    elif is_madlad(tokenizer):
        lang_set = langs_to_madlad(lang_set_raw)
    elif is_dec_only_llm(tokenizer):
        return
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
    correction = get_stupid_correction(tokargs.mdl_id)
    voc_size = tokargs.vocab_size - correction
    location = tokargs.save_location

    return learn_spm_tokenizer(tokargs.tok_train_file, location, base_model_id=tokargs.tok_mdl_id,
                               vocab_size=voc_size, lang_set=tokargs.new_langs)


def remove_known_toks(toks, tokenizer):
    return [t for t in toks if not t in tokenizer.get_vocab()]


def _handle_new_tokenizer(args):
    assert args.new_langs is not None, "lang_set must be provided"
    assert args.tok_train_file is not None, "tok_train_file must be provided"
    args.vocab_size = int(args.vocab_size)

    log("Training new tokenizer")
    tokenizer = do_new_tok(args)

    return tokenizer


def get_postoken_filename(save_location):
    return os.path.join(save_location, "postokens.json")


def save_postokens(added_tokens, location):
    if added_tokens is not None:
        os.makedirs(location, exist_ok=True)
        with open(get_postoken_filename(location), "w") as f:
            json.dump(added_tokens, f)


def _handle_adding_tokens(tokenizer, toks_to_add, args):
    if len(toks_to_add) == 0:
        return None

    log(f"Adding tokens: {toks_to_add}")

    base_idx = len(tokenizer)

    added_tok_dict = { t: (base_idx + i) for i, t in enumerate(toks_to_add) }
    added_tok_rev_dict = { int(i): t for t, i in added_tok_dict.items() }

    comb_dict = { 'tok2idx': added_tok_dict, 'idx2tok': added_tok_rev_dict }

    save_postokens(comb_dict, args.save_location)

    return comb_dict


def _handle_existing_tokenizer(args):
    log("Reusing existing tokenizer")
    tokenizer, added_tokens = load_tokenizer(args.tok_mdl_id)

    if args.new_langs is not None:
        log("Extending existing tokenizer with languages")
        extend_tok_langs(tokenizer, args.new_langs)

    if args.merge_tokenizers or args.merge_tok_mdl_id:
        """
        assert args.tok_train_file is not None, "For merging tokenizers a text file must be provided" \
                                                + " to find the top N tokens to merge"
        assert args.merge_tokenizers is not None and args.merge_tok_mdl_id is not None, \
            "Both merge_tokenizers and merge_tok_mdl_id must be provided"
        """
        raise NotImplementedError("Merging is currently not supported")

    added_tok_count = 0

    if args.tok_train_file:
        if args.merge_tokenizers:
            """
            merge_tok_max = int(args.merge_tokenizers)
            log(f"Extending existing tokenizer ({args.merge_tok_mdl_id}) with up to {merge_tok_max} top tokens" +
                f" from another tokenizer and corpus ({args.tok_train_file})")
            new_tok = AutoTokenizer.from_pretrained(args.merge_tok_mdl_id, token=hf_tok)
            toks_to_maybe_add = get_top_toks(new_tok, args.tok_train_file, merge_tok_max)
            """
            raise NotImplementedError("Merging is currently not supported")

        else:
            log(f"Extending existing tokenizer with UNK tokens from corpus ({args.tok_train_file})")
            toks_to_maybe_add = get_unk_toks(tokenizer, args.tok_train_file, verbose=True)

        toks_to_add = remove_known_toks(toks_to_maybe_add, tokenizer)
        added_tok_count = len(toks_to_add)
        added_tokens = _handle_adding_tokens(tokenizer, toks_to_add, args)

    return tokenizer, added_tok_count, added_tokens


def train_or_extend_tokenizer_and_upd_model(args, model):
    if hasattr(args, "vocab_size") and args.vocab_size:
        # train a new sentence-piece tokenizer
        tokenizer = _handle_new_tokenizer(args)
        added_tok_count = 0
        added_dict = None
    else:
        # save the pre-trained model's tokenizer, possibly adding new languages and tokens
        tokenizer, added_tok_count, added_dict = _handle_existing_tokenizer(args)

    upd_amt = get_stupid_correction(args.mdl_id)
    new_len = len(tokenizer) + added_tok_count

    model.resize_token_embeddings(new_len + upd_amt)

    return tokenizer, added_dict


def load_tokenizer(tok_mdl_id):
    orig_tokenizer = AutoTokenizer.from_pretrained(tok_mdl_id, token=hf_tok, use_fast=False)

    postoken_file = get_postoken_filename(tok_mdl_id)
    if os.path.exists(postoken_file):
        with open(postoken_file, "r") as f:
            postokens = json.load(f)
    else:
        postokens = None

    return orig_tokenizer, postokens


def detokenizeit(toktup, tok_ids):
    #return toktup[0].decode(tok_ids, skip_special_tokens=True)

    toks = []

    for tok_id_tensor in tok_ids:
        tok_id = tok_id_tensor.item()
        try:
            if tok_id not in toktup[0].added_tokens_decoder:
                toks.append(toktup[0].convert_ids_to_tokens(tok_id))
        except IndexError:
            toks.append(toktup[1]['idx2tok'][str(tok_id)])

    result = "".join(toks).replace("▁", " ")[1:]

    return result, toks


def detokenizemany(toktup, tok_mtx):
    result = [detokenizeit(toktup, tok_ids)[0] for tok_ids in tok_mtx]

    return result


def tokenizeit(toktup, sntlist, maxlen, is_target, preset_toks=None):
    tokenizer, postokens = toktup

    if preset_toks is None:
        if is_target:
            orig_toks = tokenizer(text_target=sntlist, return_tensors="pt",
                                  padding="longest", truncation=True, max_length=maxlen)
        else:
            orig_toks = tokenizer(text=sntlist, return_tensors="pt",
                                  padding="longest", truncation=True, max_length=maxlen)
    else:
        orig_toks = preset_toks

    if postokens is not None and tokenizer.unk_token_id in orig_toks['input_ids']:

        #specials_without_unk = set(tokenizer.all_special_ids) - set([tokenizer.unk_token_id])
        for idx, snt in enumerate(sntlist):
            if tokenizer.unk_token_id in orig_toks['input_ids'][idx]:
                true_toks = tokenizer.tokenize(snt)
                for ord_idx, tok_idx in enumerate(orig_toks['input_ids'][idx]):
                    if ord_idx > 0 and tok_idx == tokenizer.unk_token_id and postokens is not None and true_toks[ord_idx - 1] in postokens['tok2idx']:
                        orig_toks['input_ids'][idx][ord_idx] = postokens['tok2idx'][true_toks[ord_idx - 1]]

    return orig_toks


def run_tokenizer_testing():
    args = CmdlineArgs("Test a tokenizer: tokenize & de-tokenize some text and check if these match",
                       pos_arg_list=["tok_mdl_id", "txt_file"])

    #tokenizer = AutoTokenizer.fromm_pretrained(args.tok_mdl_id, token=hf_tok)    if os.path.exists()
    toktup = load_tokenizer(args.tok_mdl_id)

    success = 0
    failure = 0

    with open(args.txt_file, "r", encoding="utf-8") as f:
        snts = f.read().split("\n")

        toks = tokenizeit(toktup, snts, 1024, False)

        for i, snt in enumerate(snts):
            tok_ids = toks['input_ids'][i]

            #detoks = toktup[0].decode(tok_ids, skip_special_tokens=True)
            detoks, tok_strs = detokenizeit(toktup, tok_ids)

            if detoks != snt:
                failure += 1
                #log(f"Tokens:   {toktup[0].convert_ids_to_tokens(tok_ids)}")
                log(f"Tokens:   {tok_strs}")
                log(f"Test failed:\n{snt} !=\n{detoks}")
            else:
                success += 1
            i += 1

    log(f"Test result: {success} successful / {failure} failed")


if __name__ == "__main__":
    sys.argv = ['', 'models/nllbxt', 'data/tok-test.txt']
    run_tokenizer_testing()
