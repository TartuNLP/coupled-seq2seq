#!/usr/bin/env python3

import sys

from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer

from modelops import mdl_param_count
from translate import hf_tok
from traintok import learn_spm_tokenizer, get_stupid_correction, get_unk_toks, extend_tok_langs

from aux import get_changed_config, lang_set_maybe_smugri, CmdlineArgs


def handle_tokenizers(mdl_id, mdl_new_name, args):
    tokenizer_changed = False

    # train a new sentence-piece tokenizer
    if args.tok_train_file and args.vocab_size:
        assert args.new_langs is not None, "lang_set must be provided"

        tokenizer_changed = True
        correction = get_stupid_correction(mdl_id)

        tokenizer = learn_spm_tokenizer(args.tok_train_file, mdl_new_name, base_model_id=mdl_id,
                                        vocab_size=int(args.vocab_size) - correction, lang_set=args.new_langs)

    # save the pre-trained model's tokenizer,
    # possibly adding new languages and tokens
    else:
        tokenizer = AutoTokenizer.from_pretrained(mdl_id, token=hf_tok)

        if args.new_langs is not None:
            tokenizer_changed = True
            extend_tok_langs(tokenizer, args.new_langs)

        if args.tok_train_file:
            tokenizer_changed = True
            unk_toks = get_unk_toks(tokenizer, args.tok_train_file, verbose=True)
            tokenizer.add_tokens(unk_toks)

    tokenizer.save_pretrained(mdl_new_name)

    return tokenizer, tokenizer_changed

def just_do_main_stuff_and_avoid_global_variable_ctx():
    args = CmdlineArgs("Initialize a new HuggingFace model randomly, off of an existing configuration, with possible changes",
                       pos_arg_list=["mdl_id", "save_location"],
                       kw_arg_dict={ k: None for k in ["tok_train_file", "new_langs", "vocab_size",
                                    "activation_dropout", "activation_function", "d_model",
                                    "decoder_attention_heads", "decoder_ffn_dim", "decoder_layerdrop", "decoder_layers",
                                    "encoder_attention_heads", "encoder_ffn_dim", "encoder_layerdrop", "encoder_layers",
                                    "num_hidden_layers"] })

    if args.new_langs:
        args.new_langs = lang_set_maybe_smugri(args.new_langs)

    tok, it_changed = handle_tokenizers(args.mdl_id, args.save_location, args)

    config = get_changed_config(AutoConfig.from_pretrained(args.mdl_id), args)

    model = AutoModelForSeq2SeqLM.from_config(config)
    if it_changed:
        model.resize_token_embeddings(len(tok) + get_stupid_correction(args.mdl_id))
    model.save_pretrained(args.save_location)

    mdl_size, emb_size = mdl_param_count(model)
    print(f"Created model with {mdl_size} parameters" +
          ("" if emb_size < 0 else f" of which {emb_size} ({100*emb_size/mdl_size:.2f}%) are embeddings"))

if __name__ == '__main__':
    just_do_main_stuff_and_avoid_global_variable_ctx()
