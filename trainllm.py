#!/usr/bin/env python3

import promptops
from aux import log, CmdlineArgs, load_model, load_tokenizer, env_stuff
from data import load_training_data


import json
import os

from datetime import datetime

from accelerate import Accelerator
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    logging,
    TrainerCallback
)

TESTING_LOCALLY = False

"""
1/3 This simply reads in command-line arguments 
"""

def _cmdline_args():
    description = """Train or tune decoder models"""

    result = CmdlineArgs(description,
                         pos_arg_list=["mdl_id", "save_location", "train_file"],
                         pos_arg_types=[str, str, str],
                         kw_arg_dict={ "continue_training": False, "save_steps": 100, "lr": 1.5e-5,
                            "batch_size": 1024, "nr_sents_per_gpu": 4, "log_steps": 1, "epochs": 4,
                            "max_length": 2000, "prompt_format": promptops.PF_SMUGRI_MT,
                            "sharing": "none",
                            "gradckpt": False,
                            "sft_output_field": "none",
                            "sft_delim": "none"})

    # if the directory args.save_location already exists, raise an exception:
    if not result.continue_training and os.path.exists(result.save_location):
        raise Exception(f"Path '{result.save_location}' exists, won't overwrite - did you mean to set continue_training=True?.")

    if result.nr_sents_per_gpu == 0:
        result.nr_sents_per_gpu = result.batch_size

    if result.sft_delim == "none":
        result.sft_delim = None

    if result.sft_output_field == "none":
        result.sft_output_field = None

    log(f"Launched as {result}")

    return result

"""
2/3 This here is used in training in order to report timing and predictions 
"""

class StepTimerCallback(TrainerCallback):
    def __init__(self):
        self._step_start = None
        self.lengths = []
        self.abs_start = datetime.now()

        self.actual_first_step = None

        self.zero = self.abs_start - self.abs_start

    # called right before each training step
    def on_step_begin(self, args, state, control, **kwargs):
        self._step_start = datetime.now()

    # called right after each training step
    def on_step_end(self, args, state, control, **kwargs):
        now = datetime.now()

        if self.actual_first_step is None:
            log(f"It took {now - self.abs_start} to start training")
            self.actual_first_step = state.global_step - 1

        elapsed = now - self._step_start
        tot_elapsed = now - self.abs_start
        self.lengths.append(elapsed)

        avg = sum(self.lengths, start=self.zero) / len(self.lengths)

        prediction = avg * (state.max_steps - state.global_step)

        # you can use logging.get_logger(...) instead of print
        print(f"[step {state.global_step}/{state.max_steps}] took {elapsed}, avg {avg}; approx {prediction} remaining")

"""
3/3 Finally, the filling of TrainingArguments and the launching of Trainer:
"""


def get_deepspeed_conf(cmdline_args, accum_steps):
    return {
        'train_batch_size': cmdline_args.batch_size,
        'train_micro_batch_size_per_gpu': cmdline_args.nr_sents_per_gpu,
        'gradient_accumulation_steps': accum_steps,
        "bf16": { "enabled": True },

        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": { "device": "none" },
            "allgather_partitions": True,
            "overlap_comm": False,
            "allgather_bucket_size": 500000000,
            "reduce_scatter": True,
            "reduce_bucket_size": 500000000,
            "contiguous_gradients": True
        },

        "gradient_clipping": 1.0,
        "steps_per_print": 20,
        "wall_clock_breakdown": False
    }


def get_fsdp_conf():
    fsdp = "full_shard auto_wrap"
    fsdp_config = {
        "use_orig_params": True,
        "sync_module_states": True,
        "forward_prefetch": True,
        "limit_all_gathers": True,
        "reshard_after_forward": True,
        "fsdp_min_num_params": 1e7,
        "transformer_layer_cls_to_wrap": ["LlamaDecoderLayer"],
        # DO NOT enable cpu_offload on LUMI unless desperate
    }
    return fsdp, fsdp_config


def get_training_args(cmdline_args, acc):
    world_size = acc.num_processes

    assert cmdline_args.batch_size % (cmdline_args.nr_sents_per_gpu * world_size) == 0, \
        "Batch size must be divisible by the number of GPUs and nr of sents per GPU"

    accum_steps = cmdline_args.batch_size // (cmdline_args.nr_sents_per_gpu * world_size)

    log(f"Nr of processes (GPUs): {world_size}, per-device batch: {cmdline_args.nr_sents_per_gpu}, accum. steps: {accum_steps}")

    dpspd = get_deepspeed_conf(cmdline_args, accum_steps) if cmdline_args.sharing == "deepspeed" else None
    fsdp, fsdp_config = get_fsdp_conf() if cmdline_args.sharing == "fsdp" else "", None

    log(f"DeepSpeed: {dpspd}, fsdp: {fsdp}, fsdp_config: {fsdp_config}, gradckpt: {cmdline_args.gradckpt}")

    tr_args = TrainingArguments(
        output_dir=cmdline_args.save_location,
        per_device_train_batch_size=cmdline_args.nr_sents_per_gpu,
        gradient_accumulation_steps=accum_steps,
        num_train_epochs=cmdline_args.epochs,
        save_steps=cmdline_args.save_steps,
        save_total_limit=10,
        logging_steps=cmdline_args.log_steps,
        deepspeed=dpspd,
        fsdp=fsdp,
        fsdp_config=fsdp_config,
        learning_rate=cmdline_args.lr,
        save_strategy="epoch",
        disable_tqdm=True,
        report_to="none",
        # Optional but often helpful on LUMI/ROCm if you enable it in your args:
        bf16=(not TESTING_LOCALLY),
        ddp_find_unused_parameters=False,
        #dataloader_num_workers=1,
        #group_by_length=True,
        log_level="debug",
        optim="adamw_torch",
        #gradient_checkpointing=True,
        #dataloader_persistent_workers=True
    )

    return tr_args


def simple_train():
    cmd_args = _cmdline_args()
    acc = Accelerator()
    device = acc.device if not cmd_args.gradckpt else None

    training_args = get_training_args(cmd_args, acc)

    tokenizer = load_tokenizer(cmd_args.mdl_id, acc)
    tokenizer.padding_side = "left"
    model = load_model(cmd_args.mdl_id, device, acc, attention=("eager" if TESTING_LOCALLY else "flash_attention_2"))
    if cmd_args.gradckpt:
        model.gradient_checkpointing_enable()

    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    log(f"Load data", accelerator=acc)
    tokenized_train_data = load_training_data(cmd_args.train_file, tokenizer, cmd_args)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=1,
        random_replace_prob=0,
        mask_replace_prob=0,
        pad_to_multiple_of=8,
    )

    log(f"Preparing to train", accelerator=acc)

    clbks = [StepTimerCallback] if acc.is_main_process else []
    TrCl = LoggingKillingTrainer if TESTING_LOCALLY else Trainer

    trainer = TrCl(
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

    #log(f"Done, saving model", accelerator=acc)
    #trainer.save_model()
    log(f"Done training", accelerator=acc)


"""
This replaces the trainer, in order to
print out the final batch when training,
and commit harakiri. So only for temporary
debugging-related usage
"""
class LoggingKillingTrainer(Trainer):
    def compute_loss(self, model, inputs, **kwargs):
        log(f"Here is the batch for training: {inputs}")
        raise NotImplementedError
        #return super().compute_loss(model, inputs, **kwargs)

if __name__ == "__main__":
    if not TESTING_LOCALLY:
        env_stuff()

    simple_train()
