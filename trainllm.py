#!/usr/bin/env python3

import promptops
from aux import log, CmdlineArgs


import json
import os, socket, torch
import sys

from torch.utils.data import Dataset as TorchDataset
from datetime import datetime

from accelerate import Accelerator
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    logging,
    TrainerCallback
)

"""
1/4 This simply reads in command-line arguments 
"""

def _cmdline_args():
    description = """Train or tune decoder models"""

    result = CmdlineArgs(description,
                         pos_arg_list=["mdl_id", "save_location", "train_file"],
                         pos_arg_types=[str, str, str],
                         kw_arg_dict={ "continue_training": False, "save_steps": 100, "lr": 1.5e-5,
                            "batch_size": 1024, "nr_sents_per_gpu": 4, "log_steps": 1, "epochs": 4,
                            "max_length": 3000, "prompt_format": promptops.PF_SMUGRI_MT,
                            "deepspeed": "none"})

    # if the directory args.save_location already exists, raise an exception:
    if not result.continue_training and os.path.exists(result.save_location):
        raise Exception(f"Save location '{result.save_location}' already exists, don't want to overwrite.")

    if result.nr_sents_per_gpu == 0:
        result.nr_sents_per_gpu = result.batch_size

    if result.deepspeed == "none":
        result.deepspeed = None

    return result

"""
2/4 This here is used in training in order to report timing and predictions 
"""

class StepTimerCallback(TrainerCallback):
    def __init__(self):
        self._step_start = None
        self.lengths = []
        self.abs_start = datetime.now()

        self.actual_first_step = None

        self.zero = self.abs_start - self.abs_start

    def on_step_begin(self, args, state, control, **kwargs):
        # called right before each training step
        self._step_start = datetime.now()

    def on_step_end(self, args, state, control, **kwargs):
        if self.actual_first_step is None:
            self.actual_first_step = state.global_step - 1

        # called right after each training step
        now = datetime.now()
        elapsed = now - self._step_start
        tot_elapsed = now - self.abs_start
        self.lengths.append(elapsed)

        avg = sum(self.lengths, start=self.zero) / len(self.lengths)

        remaining = state.max_steps - self.actual_first_step - state.global_step
        prediction = (tot_elapsed/(state.global_step - self.actual_first_step)) * remaining

        # you can use logging.get_logger(...) instead of print
        print(f"[step {state.global_step}/{state.max_steps}] took {elapsed}, avg {avg}; approx {prediction} remaining")

"""
3/4 This here is a dataset which reads in raw string files and only when asked for a sample it tokenizes it 
"""


class LazyTokenizingDataset(TorchDataset):
    def __init__(self, texts, tokenizer, max_length=512, prompt_format="raw"):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_format = prompt_format

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # Return plain Python lists; let the collator pad & build labels.
        entry = self.texts[idx]

        prompt = promptops.prep_prompt(entry, self.prompt_format)

        return promptops.tokenize_str(self.tokenizer, prompt)

def load_training_data(path, tokenizer, cmd_args):
    with open(path, "r") as f:
        data = json.load(f)

    train_set_iter = LazyTokenizingDataset(data, tokenizer, cmd_args.max_length, cmd_args.prompt_format)

    return train_set_iter

"""
4/4 Finally, the filling of TrainingArguments and the launching of Trainer:
"""

def get_training_args(cmdline_args, acc, testing_on_mac=False):
    world_size = acc.num_processes

    assert cmdline_args.batch_size % (cmdline_args.nr_sents_per_gpu * world_size) == 0, \
        "Batch size must be divisible by the number of GPUs and nr of sents per GPU"

    accum_steps = cmdline_args.batch_size // (cmdline_args.nr_sents_per_gpu * world_size)

    log(f"Nr of processes (GPUs): {world_size}, per-device batch: {cmdline_args.nr_sents_per_gpu}, accum. steps: {accum_steps}")

    if cmdline_args.deepspeed is not None:
        with open(cmdline_args.deepspeed, "r") as f:
            dpspd = json.load(f)

            #correct the dictionary with current values, so that we wouldn't need to update the JSON every time
            dpspd['train_batch_size'] = cmdline_args.batch_size
            dpspd['train_micro_batch_size_per_gpu'] = cmdline_args.nr_sents_per_gpu
            dpspd['gradient_accumulation_steps'] = accum_steps
    else:
        dpspd = None

    tr_args = TrainingArguments(
        output_dir=cmdline_args.save_location,
        per_device_train_batch_size=cmdline_args.nr_sents_per_gpu,
        gradient_accumulation_steps=accum_steps,
        num_train_epochs=cmdline_args.epochs,
        save_steps=cmdline_args.save_steps,
        save_total_limit=10,
        logging_steps=cmdline_args.log_steps,
        deepspeed=dpspd,
        learning_rate=cmdline_args.lr,
        disable_tqdm=True,
        report_to="none",
        # Optional but often helpful on LUMI/ROCm if you enable it in your args:
        bf16=not testing_on_mac, #True,
        ddp_find_unused_parameters=False,
        #dataloader_num_workers=1,
        #group_by_length=True,
        log_level="debug",
        #gradient_checkpointing=True,
        #dataloader_persistent_workers=True
    )

    return tr_args


def simple_train(testing_on_mac=False):
    cmd_args = _cmdline_args()
    acc = Accelerator()
    training_args = get_training_args(cmd_args, acc, testing_on_mac)

    log(f"Load tokenizer", accelerator=acc)

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cmd_args.mdl_id)

    # LLaMA 3.x: no pad token by default — use EOS for padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<|reserved_special_token_100|>"

    log(f"Load model", accelerator=acc)
    model = AutoModelForCausalLM.from_pretrained(cmd_args.mdl_id,
                                                 low_cpu_mem_usage=False,
                                                 torch_dtype=torch.bfloat16,
                                                 attn_implementation=("eager" if testing_on_mac else "flash_attention_2"))
    model.config.use_cache = False
    model = model.to(acc.device)

    log(f"attention implementation used: { model.model.layers[0].self_attn.__class__.__name__ }.", accelerator=acc)
    log(f"device: {model.device}.", accelerator=acc)

    # Make sure the model knows the pad id (avoids warnings/edge-cases)
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    log(f"Load data", accelerator=acc)
    tokenized_train_data = load_training_data(cmd_args.train_file, tokenizer, cmd_args)

    # Dynamic padding + proper causal labels with pads masked to -100
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,  # helps performance; set None if you prefer exact lengths
    )

    log(f"Preparing to train", accelerator=acc)

    clbks = [StepTimerCallback] if acc.is_main_process else []

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_data,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=clbks,
    )

    logging.set_verbosity_debug()

    log(f"Starting training", accelerator=acc)
    trainer.train(resume_from_checkpoint=cmd_args.continue_training)

    log(f"Done, saving model", accelerator=acc)
    trainer.save_model()


def env_stuff():
    os.environ.setdefault("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "---"))
    os.environ.setdefault("RANK", os.environ.get("SLURM_PROCID", "0"))
    os.environ.setdefault("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1"))
    os.environ.setdefault("MASTER_ADDR", os.environ.get("SLURM_LAUNCH_NODE_IPADDR", "127.0.0.1"))
    os.environ.setdefault("MASTER_PORT", "29500")  # pick an open port

    # Optional: make sure each process selects its own GPU
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    try:
        log(
            f"host={socket.gethostname()} "
            f"RANK={os.environ['RANK']}/{os.environ['WORLD_SIZE']} "
            f"LOCAL_RANK={os.environ['LOCAL_RANK']} "
            f"HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES')} "
            f"ROCR_VISIBLE_DEVICES={os.environ.get('ROCR_VISIBLE_DEVICES')} "
            f"cuda_count={torch.cuda.device_count()} curr_dev={torch.cuda.current_device()}"
        )
    except AssertionError:
        log(
            f"host={socket.gethostname()} "
            f"RANK={os.environ['RANK']}/{os.environ['WORLD_SIZE']} "
            f"LOCAL_RANK={os.environ['LOCAL_RANK']} "
            f"HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES')} "
            f"ROCR_VISIBLE_DEVICES={os.environ.get('ROCR_VISIBLE_DEVICES')} "
            f"no cuda"
        )

class LoggingKillingTrainer(Trainer):
    def compute_loss(self, model, inputs, **kwargs):
        log(f"Here is the batch for training: {inputs}")
        raise NotImplementedError
        #return super().compute_loss(model, inputs, **kwargs)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.argv = "_ models/llama3.2-1b models/tmp1 small-structured-data.json batch_size=16 nr_sents_per_gpu=1 log_steps=5 save_steps=100 epochs=1 lr=1e-4".split()
        testing_on_mac = True
    else:
        env_stuff()
        testing_on_mac = False
    simple_train(testing_on_mac)
