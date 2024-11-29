#!/usr/bin/env python3

import sys

from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
from translate import hf_tok
from traintok import learn_spm_tokenizer, get_stupid_correction, get_unk_toks, extend_tok_langs

from aux import log

def maybe_convert(value):
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def get_changed_config(model_id, **kw):
    conf = AutoConfig.from_pretrained(model_id)

    if "tok_train_set" in kw:
        del kw["tok_train_set"]

    for kwarg in kw:
        if kwarg in conf.__dict__:
            conf.__dict__[kwarg] = maybe_convert(kwargs[kwarg])
        else:
            raise KeyError(f'key "{kwarg}" is not in model config')

    return conf


def to_kwargs(raw_kwargs):
    return dict(raw_entry.split("=") for raw_entry in raw_kwargs)


def mdl_param_count(model):
    result = 0
    embedding_size = -1

    for n, p in model.named_parameters():
        this_count = 1

        for s in p.shape:
            this_count *= s

        result += this_count

        if n == "model.shared.weight":
            embedding_size = this_count

    return result, embedding_size


def handle_tokenizers(mdl_id, mdl_new_name, kwargs):
    lang_set = kwargs["lang_set"].split(",") if "lang_set" in kwargs else None

    tokenizer_changed = False

    # train a new sentence-piece tokenizer
    if "tok_train_set" in kwargs and "vocab_size" in kwargs:
        assert lang_set is not None, "lang_set must be provided"
        tokenizer_changed = True
        correction = get_stupid_correction(mdl_id)

        tokenizer = learn_spm_tokenizer(kwargs["tok_train_set"], mdl_new_name,
                                        vocab_size=int(kwargs["vocab_size"]) - correction, lang_set=lang_set)

    # save the pre-trained model's tokenizer,
    # possibly adding new languages and tokens
    else:
        tokenizer = AutoTokenizer.from_pretrained(mdl_id, token=hf_tok)

        if lang_set is not None:
            tokenizer_changed = True
            extend_tok_langs(tokenizer, lang_set)

        if "tok_train_set" in kwargs:
            tokenizer_changed = True
            unk_toks = get_unk_toks(tokenizer, kwargs["tok_train_set"], verbose=True)
            tokenizer.add_tokens(unk_toks)

    tokenizer.save_pretrained(mdl_new_name)

    return tokenizer, tokenizer_changed


if __name__ == '__main__':
    sys.argv = ["X", "facebook/m2m100_418M", "new_tok", "decoder_layers=2", "dropout=0.01"]

    try:
        mdl_id = sys.argv[1]
        mdl_new_name = sys.argv[2]
        kwargs = to_kwargs(sys.argv[3:])

        tok, it_changed = handle_tokenizers(mdl_id, mdl_new_name, kwargs)

        config = get_changed_config(mdl_id, **kwargs)

        model = AutoModelForSeq2SeqLM.from_config(config)
        if it_changed:
            log("Yes, it did change")
            model.resize_token_embeddings(len(tok) + get_stupid_correction(mdl_id))
        model.save_pretrained(mdl_new_name)

        mdl_size, emb_size = mdl_param_count(model)
        print(f"Created model with {mdl_size} parameters" +
              ("" if emb_size < 0 else f" of which {emb_size} ({100*emb_size/mdl_size:.2f}%) are embeddings"))

    except IndexError:
        sys.stderr.write("""Usage: initmodel.py  <model_id>  <model_new_name>  [param=value]+""")
