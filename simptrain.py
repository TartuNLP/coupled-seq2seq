#!/usr/bin/env python3

import json

from trainllm import _cmdline_args
from aux import log

from accelerate import Accelerator
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer


def load_training_data(path, tokenizer, cmd_args):
    # Load data
    with open(path, "r") as f:
        data = json.load(f)
    dataset = Dataset.from_dict({"text": data})

    # Tokenize
    def tokenize_fn(examples):
        log(f"I am being summoned! {len(examples['text'])} examples")
        tokens = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=cmd_args.max_length,
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens
    tokenized = dataset.map(tokenize_fn, batched=True)

    return tokenized


def get_training_args(cmdline_args):
    acc = Accelerator()
    world_size = acc.num_processes
    log(f"Nr of processes (GPUs): {world_size}")

    assert cmdline_args.batch_size % (cmdline_args.nr_sents_per_gpu * world_size) == 0, \
        "Batch size must be divisible by the number of GPUs and nr of sents per GPU"

    accum_steps = cmdline_args.batch_size // (cmdline_args.nr_sents_per_gpu * world_size)

    # Define training
    tr_args = TrainingArguments(
        output_dir=cmdline_args.save_location,
        per_device_train_batch_size=cmdline_args.nr_sents_per_gpu,
        gradient_accumulation_steps=accum_steps,
        num_train_epochs=cmdline_args.epochs,
        save_steps=cmdline_args.save_steps,
        save_total_limit=3,
        logging_steps=cmdline_args.log_steps,
        learning_rate=cmdline_args.lr,
        report_to="none"
    )

    return tr_args


def simple_train():
    cmd_args = _cmdline_args()

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cmd_args.mdl_id)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(cmd_args.mdl_id)

    tokenized_train_data = load_training_data(cmd_args.train_file, tokenizer, cmd_args)

    training_args = get_training_args(cmd_args)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_data,
        tokenizer=tokenizer
    )

    trainer.train()

if __name__ == "__main__":
    simple_train()