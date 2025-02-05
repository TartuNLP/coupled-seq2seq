#!/usr/bin/env python3

import os

from transformers import AutoConfig, AutoModelForSeq2SeqLM

from modelops import mdl_param_count
from tokops import get_stupid_correction, train_or_extend_tokenizer_and_upd_model

from aux import get_changed_config, lang_set_maybe_smugri, CmdlineArgs


def just_do_main_stuff_and_avoid_global_ctx_variables():
    args = CmdlineArgs("Initialize a new HuggingFace model randomly, off of an existing configuration, with possible changes",
                       pos_arg_list=["mdl_id", "save_location"],
                       kw_arg_dict={ k: None for k in ["tok_train_file", "new_langs", "vocab_size", "merge_tokenizers",
                                    "tok_mdl_id", "activation_dropout", "activation_function", "d_model",
                                    "decoder_attention_heads", "decoder_ffn_dim", "decoder_layerdrop", "decoder_layers",
                                    "encoder_attention_heads", "encoder_ffn_dim", "encoder_layerdrop", "encoder_layers",
                                    "num_hidden_layers"] })
    if not args.tok_mdl_id:
        args.tok_mdl_id = args.mdl_id

    if args.new_langs:
        args.new_langs = lang_set_maybe_smugri(args.new_langs)

    if os.path.exists(args.save_location):
        raise Exception(f"Save location '{args.save_location}' already exists, don't want to overwrite")

    config = get_changed_config(AutoConfig.from_pretrained(args.mdl_id), args)

    model = AutoModelForSeq2SeqLM.from_config(config)

    tokenizer = train_or_extend_tokenizer_and_upd_model(args, model)

    tokenizer.save_pretrained(args.save_location)
    model.save_pretrained(args.save_location)

    mdl_size, emb_size = mdl_param_count(model)
    print(f"Created model with {mdl_size} parameters" +
          ("" if emb_size < 0 else f" of which {emb_size} ({100 * emb_size / mdl_size:.2f}%) are embeddings"))

if __name__ == '__main__':
    just_do_main_stuff_and_avoid_global_ctx_variables()
