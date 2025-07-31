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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_data,
        tokenizer=tokenizer,
        data_collator=data_collator,  # NEW
    )

    log(f"Start training", accelerator=acc)
    trainer.train()

    trainer.save_model()


def env_stuff():
    os.environ.setdefault("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0"))
    os.environ.setdefault("RANK", os.environ.get("SLURM_PROCID", "0"))
    os.environ.setdefault("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1"))
    os.environ.setdefault("MASTER_ADDR", os.environ.get("SLURM_LAUNCH_NODE_IPADDR", "127.0.0.1"))
    os.environ.setdefault("MASTER_PORT", "29500")  # pick an open port

    # Optional: make sure each process selects its own GPU
    try:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    except Exception:
        log("Well that did not work")
        pass

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


"""
import os
import json
import argparse
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)

# ---------------------------
# Data loading (raw strings)
# ---------------------------

def load_texts(path: str) -> List[str]:
    "Expect a JSON list of strings. If you have JSONL (one text per line), uncomment the JSONL branch."
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, list) and all(isinstance(x, str) for x in data), \
            "data.json must be a JSON list of strings"
        return data

    # JSONL fallback (uncomment if needed)
    # if path.endswith(".jsonl"):
    #     texts = []
    #     with open(path, "r", encoding="utf-8") as f:
    #         for line in f:
    #             if line.strip():
    #                 texts.append(json.loads(line))
    #     # If JSONL lines are dicts with "text", map them:
    #     if texts and isinstance(texts[0], dict) and "text" in texts[0]:
    #         texts = [ex["text"] for ex in texts]
    #     return texts

    raise ValueError(f"Unsupported data file: {path}. Use a JSON list of strings.")


# ----------------------------------------
# Dataset that returns raw text only
# ----------------------------------------

class RawTextDataset(Dataset):
    def __init__(self, texts: List[str]):
        self.texts = texts

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Return raw text; tokenization is done in the collator for efficiency.
        return {"text": self.texts[idx]}


# ----------------------------------------------------------
# Collator: tokenize per-batch, create labels, mask padding
# ----------------------------------------------------------

class CausalLMCollator:
    def __init__(self, tokenizer, max_length: int = None, pad_to_multiple_of: int = 8):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of

        # LLaMA 3.x has no pad token by default -> use eos as pad
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts = [ex["text"] for ex in batch]
        enc = self.tokenizer(
            texts,
            truncation=True,
            padding=True,                 # dynamic padding to longest in the batch
            max_length=self.max_length,   # keep None to use model default context if you wish
            return_tensors="pt",
            pad_to_multiple_of=self.pad_to_multiple_of,  # good for tensor cores
        )
        # labels = input_ids, but ignore padded positions in loss
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        enc["labels"] = labels
        return enc


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", type=str, required=True,
                   help="e.g., meta-llama/Meta-Llama-3-8B or a local path")
    p.add_argument("--data_path", type=str, default="data.json",
                   help="JSON list of strings")
    p.add_argument("--output_dir", type=str, default="./output")
    p.add_argument("--max_length", type=int, default=None,
                   help="Optional truncation length. Leave None to rely on model defaults.")
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--lr_scheduler_type", type=str, default="constant",
                   choices=["constant","constant_with_warmup","linear","cosine","cosine_with_restarts","polynomial","inverse_sqrt"])
    p.add_argument("--warmup_steps", type=int, default=0)
    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--logging_steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--bf16", action="store_true", help="Prefer on AMD/ROCm or Hopper-class GPUs")
    p.add_argument("--fp16", action="store_true", help="Use only if bf16 is unavailable/undesired")
    p.add_argument("--dataloader_num_workers", type=int, default=2)
    p.add_argument("--resume", action="store_true", help="Resume from last checkpoint in output_dir")
    return p.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)

    # --------- Load model & tokenizer (consistent checkpoint) ----------
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token  # LLaMA 3.x best practice

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None),
    )

    # Disable cache during training to avoid warnings & extra memory
    if getattr(model.config, "use_cache", True):
        model.config.use_cache = False

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # --------- Data ----------
    texts = load_texts(args.data_path)
    train_ds = RawTextDataset(texts)
    collator = CausalLMCollator(tokenizer, max_length=args.max_length, pad_to_multiple_of=8)

    # --------- Training arguments ----------
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        dataloader_num_workers=args.dataloader_num_workers,
        report_to="none",
        bf16=args.bf16,
        fp16=(args.fp16 and not args.bf16),
        gradient_checkpointing=args.gradient_checkpointing,
        ddp_find_unused_parameters=False,  # safer for decoder-only fine-tuning
    )

    # --------- Trainer ----------
    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        data_collator=collator,  # on-the-fly tokenization + masking
    )

    # Small diagnostics
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    eff_batch = args.per_device_train_batch_size * args.gradient_accumulation_steps * world_size
    print(f"[Info] World size: {world_size}  |  Effective batch size per optimizer step: {eff_batch}")

    # --------- Train ----------
    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_state()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
"""
