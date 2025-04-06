#!/usr/bin/env python3

import os
import torch

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM

from accel import SwitchingAccelerator
from accelerate import Accelerator
from data import MultilingualDatasetIterator
from aux import log, CmdlineArgs
from langconv import lang_set_maybe_smugri, is_dec_only_llm
from modelops import mdl_param_count, to_cpl_spec, save_all_models, hf_tok
from tokops import load_tokenizer


def freeze_model(model):
    for n, p in model.named_parameters():
        p.requires_grad = False


def load_hf_mdl_and_tok(mdl_id, tok_id=None, verbose=False):
    if tok_id is None:
        tok_id = mdl_id

    tokenizer = load_tokenizer(tok_id) # AutoTokenizer.fromm_pretrained(tok_id, token=hf_tok)

    if is_dec_only_llm(tokenizer[0]):
        model = AutoModelForCausalLM.from_pretrained(mdl_id, token=hf_tok, torch_dtype=torch.bfloat16)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(mdl_id, token=hf_tok, torch_dtype=torch.bfloat16)

    if verbose:
        mdl_size, _ = mdl_param_count(model)
        log(f"Loaded {mdl_id} with {mdl_size} params, voc size {model.config.vocab_size}")

    return model, tokenizer


def _cmdline_args():
    description = """Train or tune models"""

    pos_args = ["mdl_id", "save_location", "train_pretok_file", "langs"]
    pos_types = [str, str, str, lang_set_maybe_smugri]

    kw_args = { "anchor_mdl_id": None, "anchor_langs": None, "batch_size": 16, "continue_training": False,
                "save_steps": 100000, "lr": 1.5e-5, "accum_steps": 1, "log_steps": 100, "epochs": 4  }

    #post-process the arguments
    args = CmdlineArgs(description, pos_arg_list=pos_args, pos_arg_types=pos_types, kw_arg_dict=kw_args)

    if args.anchor_langs is not None:
        args.anchor_langs = lang_set_maybe_smugri(args.anchor_langs)

    # if the directory args.save_location already exists, raise an exception:
    if not args.continue_training and os.path.exists(args.save_location):
        raise Exception(f"Save location '{args.save_location}' already exists, don't want to overwrite")

    return args


def yes_i_called_this_function_do_main():
    args = _cmdline_args()
    tmp_acc = Accelerator()

    log("loading coupled model and tokenizer", accelerator=tmp_acc)
    main_model, main_tokenizer = load_hf_mdl_and_tok(args.mdl_id, verbose=True)

    coupling_specs = to_cpl_spec(args.langs, main_model, main_tokenizer[0], main_tokenizer[1], args.save_location)

    if args.anchor_mdl_id:
        log("loading anchor model and tokenizer", accelerator=tmp_acc)
        anchor_model, anchor_tokenizer = load_hf_mdl_and_tok(args.anchor_mdl_id, verbose=True)
        freeze_model(anchor_model)

        coupling_specs += to_cpl_spec(args.anchor_langs, anchor_model, anchor_tokenizer[0], anchor_tokenizer[1], args.anchor_mdl_id)

    train_set = MultilingualDatasetIterator(args.train_pretok_file)

    acc_trainer = SwitchingAccelerator(coupling_specs, train_set, args)

    upd_model, loss_list = acc_trainer.train()

    #save_all_models(args.save_location, upd_model, main_tokenizer, coupling_specs, loss_list, trainer=acc_trainer.accelerator)


if __name__ == "__main__":
    #sys.argv = ". models/smol models/smol_next data/smugri4a-dev.json-tokcache/thiscache.json smugri log_steps=1 lr=1e-5".split()
    #sys.argv = ". models/llama3.2-1b models/llama-tuned data/smugri4a-dev.json-tokcache/llama.json smugri".split()
    yes_i_called_this_function_do_main()
