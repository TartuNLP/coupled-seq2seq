#!/usr/bin/env python3

import os
import json
import torch
import sys

from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer

from accel import SwitchingAccelerator
from modelops import hf_tok, save_all_models

from aux import log, CmdlineArgs
from data import do_list_in_batches


def _cmdline_args():
    description = """Train or tune decoder models"""

    result = CmdlineArgs(description,
                         pos_arg_list=["mdl_id", "save_location", "train_file"],
                         pos_arg_types=[str, str, str],
                         kw_arg_dict={ "continue_training": False, "save_steps": 100000, "lr": 1.5e-5,
                            "batch_size": 8, "nr_sents_per_gpu": 0, "log_steps": 100, "epochs": 4 })

    # if the directory args.save_location already exists, raise an exception:
    if not result.continue_training and os.path.exists(result.save_location):
        raise Exception(f"Save location '{result.save_location}' already exists, don't want to overwrite.")

    if result.nr_sents_per_gpu == 0:
        result.nr_sents_per_gpu = result.batch_size

    return result


def load_json_list(json_file):
    with open(json_file, "r") as f:
        return json.load(f)


def load_hf_model(mdl_id):
    model = AutoModelForCausalLM.from_pretrained(mdl_id, token=hf_tok, torch_dtype=torch.bfloat16)
    return model


def load_hf_tokenizer(mdl_id):
    tokenizer = AutoTokenizer.from_pretrained(mdl_id, token=hf_tok)
    return tokenizer


def _no_globals_main():
    args = _cmdline_args()
    tmp_acc = Accelerator()

    try:
        log(f"Num proc: {tmp_acc.num_processes}, proc ID: {tmp_acc.process_index}")
        log("loading model", accelerator=tmp_acc)
        mdl = load_hf_model(args.mdl_id)

        log("loading tokenizer", accelerator=tmp_acc)
        tok = load_hf_tokenizer(args.mdl_id)

        log("loading data", accelerator=tmp_acc)
        train_set = load_json_list(args.train_file)

        log("training", accelerator=tmp_acc)

        acc_trainer = SwitchingAccelerator(train_set, args, mdl, tok)
        upd_model = acc_trainer.train()

        log("saving", accelerator=tmp_acc)
        save_all_models(args.save_location, upd_model, tok)
    except Exception as e:
        # in multiprocess scenarios it is hard to read the stack trace, so just show one:
        if tmp_acc.is_main_process:
            raise e


if __name__ == "__main__":
    #sys.argv = "_ models/llama3.2-1b models/newmdl tmp.json".split()

    _no_globals_main()
