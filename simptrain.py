#!/usr/bin/env python3

import json
import torch

from torch.utils.data import Dataset as TorchDataset
from trainllm import _cmdline_args
from aux import log

from accelerate import Accelerator
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,  # NEW
    logging
)


import os, socket, torch



class LazyTokenizingDataset(TorchDataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # Return plain Python lists; let the collator pad & build labels.
        text = self.texts[idx]
        tokens = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            # no padding here – dynamic padding happens in the collator
            return_attention_mask=True,
        )
        # Do NOT add "labels" here; the collator will create them and mask pads to -100.
        return tokens


def load_training_data(path, tokenizer, cmd_args):
    with open(path, "r") as f:
        data = json.load(f)
    #train_set_iter = BatchingIterator(data, cmd_args.batch_size, tokenizer, cmd_args.max_length)
    train_set_iter = LazyTokenizingDataset(data, tokenizer, cmd_args.max_length)
    return train_set_iter


def get_training_args(cmdline_args, acc):
    world_size = acc.num_processes

    assert cmdline_args.batch_size % (cmdline_args.nr_sents_per_gpu * world_size) == 0, \
        "Batch size must be divisible by the number of GPUs and nr of sents per GPU"

    accum_steps = cmdline_args.batch_size // (cmdline_args.nr_sents_per_gpu * world_size)

    log(f"Nr of processes (GPUs): {world_size}, per-device batch: {cmdline_args.nr_sents_per_gpu}, accum. steps: {accum_steps}")

    tr_args = TrainingArguments(
        output_dir=cmdline_args.save_location,
        per_device_train_batch_size=cmdline_args.nr_sents_per_gpu,
        gradient_accumulation_steps=accum_steps,
        num_train_epochs=cmdline_args.epochs,
        save_steps=cmdline_args.save_steps,
        save_total_limit=3,
        logging_steps=cmdline_args.log_steps,
        learning_rate=cmdline_args.lr,
        report_to="none",
        # Optional but often helpful on LUMI/ROCm if you enable it in your args:
        bf16=True,
        ddp_find_unused_parameters=False,
        dataloader_num_workers=1,
        group_by_length=True,
        log_level="debug",
        gradient_checkpointing=True,
        dataloader_persistent_workers=True
    )

    return tr_args


def simple_train():
    cmd_args = _cmdline_args()
    acc = Accelerator()

    log(f"Load tokenizer", accelerator=acc)
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cmd_args.mdl_id)
    # LLaMA 3.x: no pad token by default — use EOS for padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log(f"Load model", accelerator=acc)
    model = AutoModelForCausalLM.from_pretrained(cmd_args.mdl_id,
                                                 low_cpu_mem_usage=False,
                                                 torch_dtype=torch.bfloat16,
                                                 attn_implementation="flash_attention_2")
    model.config.use_cache = False
    model = model.to('cuda')
    log(f"attention implementation used: { model.model.layers[0].self_attn.__class__.__name__ }.", accelerator=acc)
    log(f"device: {model.device}.", accelerator=acc)

    # Make sure the model knows the pad id (avoids warnings/edge-cases)
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    log(f"Load data", accelerator=acc)
    tokenized_train_data = load_training_data(cmd_args.train_file, tokenizer, cmd_args)
    training_args = get_training_args(cmd_args, acc)

    # Dynamic padding + proper causal labels with pads masked to -100
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,  # helps performance; set None if you prefer exact lengths
    )

    log(f"Preparing to train", accelerator=acc)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_data,
        tokenizer=tokenizer,
        data_collator=data_collator,  # NEW
    )

    logging.set_verbosity_debug()

    log(f"Starting training", accelerator=acc)
    trainer.train()

    log(f"Done, saving model", accelerator=acc)
    trainer.save_model()


def env_stuff():
    os.environ.setdefault("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "---"))
    os.environ.setdefault("RANK", os.environ.get("SLURM_PROCID", "0"))
    os.environ.setdefault("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1"))
    os.environ.setdefault("MASTER_ADDR", os.environ.get("SLURM_LAUNCH_NODE_IPADDR", "127.0.0.1"))
    os.environ.setdefault("MASTER_PORT", "29500")  # pick an open port

    # Optional: make sure each process selects its own GPU
    #torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


    log(
        f"host={socket.gethostname()} "
        f"RANK={os.environ['RANK']}/{os.environ['WORLD_SIZE']} "
        f"LOCAL_RANK={os.environ['LOCAL_RANK']} "
        f"HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES')} "
        f"ROCR_VISIBLE_DEVICES={os.environ.get('ROCR_VISIBLE_DEVICES')} "
        f"cuda_count={torch.cuda.device_count()} curr_dev={torch.cuda.current_device()}"
    )

if __name__ == "__main__":
    env_stuff()
    simple_train()
