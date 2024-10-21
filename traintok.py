#!/usr/bin/env python3
import os

import sentencepiece as spm
import json

from transformers import AutoTokenizer
from transformers.models.m2m_100 import M2M100Tokenizer
from collections import defaultdict

def test_tok(tok, snt, lang):
    tok.src_lang = lang
    out = tok(text = snt)
    print(out['input_ids'])
    print(tok.tokenize(snt))
    print(tok.convert_ids_to_tokens(out['input_ids']))
    print("-")


def get_stupid_correction(mdl_id):
    if "m2m" in mdl_id.lower():
        correction = 108
    elif "nllb" in mdl_id.lower():
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


#def extend_tokenizer(tokenizer, new_tokens):
#    #with open(corpus, 'r', encoding='utf-8') as f:
#    #    lines = [l.strip() for l in f]
#    #    unk_toks, ratio, _ = get_unk_toks(tokenizer, lines)
#
#        print(f"Added {len(unk_toks)} tokens to the tokenizer: {' '.join(unk_toks)}; " +
#              f"UNK tokens account for {100*ratio:.2f}% of the corpus")
#
#        tokenizer.add_tokens(list(unk_toks))


def test_existing_toks(test_snt = "Pǟgiņ vȯȯnnõ mäd kolēgõn", lang = "fi", mdl_list = ["facebook/m2m100_418M", "facebook/seamless-m4t-v2-large", "facebook/nllb-200-1.3B", "google/madlad400-3b-mt", "google/gemma-7b", "google/mt5-base", "facebook/mbart-large-50"]):
    for mdl_id in mdl_list:
       print(mdl_id)
       try:
           m2mtok = AutoTokenizer.from_pretrained(mdl_id)
           test_tok(m2mtok, test_snt, lang)
       except Exception as e:
           print("Failed because:", e)


def learn_spm_tokenizer(corpus, model_dir, vocab_size, lang_set = None):
    tmp_location = os.path.join(model_dir, "sentencepiece.bpe.tmp")
    os.makedirs(model_dir, exist_ok=True)

    spm.SentencePieceTrainer.train(input=corpus, model_prefix=tmp_location, vocab_size=vocab_size)

    json_vocab_location = tsv_to_json_vocab(tmp_location + ".vocab")

    m2m_tok = M2M100Tokenizer(json_vocab_location, tmp_location + ".model",
                              language_codes = "m2m100", num_madeup_words = 0)

    for tmp_file in (".vocab", ".model", ".vocab.json"):
        os.remove(tmp_location + tmp_file)

    return m2m_tok


"""
    tok = AutoTokenizer.from_pretrained(sys.argv[1])
    
    test_tok(tok, "Pǟgiņ vȯnnõ mäd kolēgõn", "fi")
    s = "Amā tǟdzi nēšti um LI Līvõd institūt jag"
    raw_data = load_raw_liv_data("train.json")
    data_iter = data_iter_for_tok_train(raw_data, {"fi", "et"})
    unk_toks = get_unk_toks(tok, data_iter)
    for t in unk_toks:
        tok.add_tokens(t)
        print(t)
    
    test_tok(tok, "Pǟgiņ vȯȯnnõ mäd kolēgõn", "fi")
     
    tok.save_pretrained("upd-m2m-tok-liv-et")
     

    #tmp_tok = learn_spm_tokenizer(data_iter, 16000, )
    #tok.save_pretrained("newtok.tokmdl")

    #tok = AutoTokenizer.from_pretrained(output_name)
    #test_tok(tok, "see on õige яма, räägime uuesti", "fi")

    #tokenizer = AutoTokenizer.from_pretrained("facebook/m2m100_418M")
    #snts = ["Pǟgiņ vȯȯnnõ mäd kolēgõn", "see on jama"]
    #test_tok(tokenizer, snt, "fi")

    #unk_toks = get_unk_toks(tokenizer, snts)
    #print(unk_toks)

    #tokenizer.add_tokens(unk_toks)

    #tokenizer.save_pretrained("upd.tok")

"""


if __name__ == '__main__':
    raise Exception("Just testing stuff")

    raw_data = load_raw_liv_data("data/train.json")
    data_iter = data_iter_for_tok_train(raw_data, {"fi", "et"})

    tmp_tok = learn_spm_tokenizer(data_iter, 16000, )
    #tok.save_pretrained("newtok.tokmdl")

    #tok = AutoTokenizer.from_pretrained(output_name)
    #test_tok(tok, "see on õige яма, räägime uuesti", "fi")

    tokenizer = AutoTokenizer.from_pretrained("facebook/m2m100_418M")
    snts = ["Pǟgiņ vȯȯnnõ mäd kolēgõn", "see on jama"]
    #test_tok(tokenizer, snt, "fi")

    unk_toks = get_unk_toks(tokenizer, snts)
    print(unk_toks)

    tokenizer.add_tokens(unk_toks)

    tokenizer.save_pretrained("upd.tok")

