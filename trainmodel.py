#!/usr/bin/env python3

import os
import torch

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_scheduler
from accelerate import Accelerator

from translate import hf_tok, encode
from data import MultilingualDatasetIterator
from aux import log, SameLineLogger, CmdlineArgs
from langconv import lang_set_maybe_smugri
from collections import namedtuple
from modelops import mdl_param_count, to_cpl_spec, load_loss_list, load_data_state, save_all_models

_CmdlineArgs = namedtuple("CmdlineArgs", "coupled_mdl_id train_data_file dev_data_file coupled_langs anchor_mdl_id anchor_langs save_location".split())

def freeze_model(model):
    for n, p in model.named_parameters():
        p.requires_grad = False


def load_hf_mdl_and_tok(mdl_id, tok_id=None, verbose=False):
    if tok_id is None:
        tok_id = mdl_id

    tokenizer = AutoTokenizer.from_pretrained(tok_id, token=hf_tok)
    model = AutoModelForSeq2SeqLM.from_pretrained(mdl_id, token=hf_tok)

    if verbose:
        mdl_size, _ = mdl_param_count(model)
        log(f"Loaded {mdl_id} with {mdl_size} params, voc size {model.config.vocab_size}")

    return model, tokenizer


def get_lps_from_specs(coupling_specs):
    lang_set = {lang for spec in coupling_specs for lang in spec.lang_set}

    for src_lang in lang_set:
        for tgt_lang in lang_set:
            if src_lang != tgt_lang:
                yield f"{src_lang}-{tgt_lang}"


def report_devices(accelerator = None, mdl = None):
    if torch.cuda.is_available():
        # Get the visible devices from CUDA
        visible_devices = torch.cuda.device_count()

        #log(f"Number of visible GPUs: {visible_devices}")
        msg = f"{visible_devices} GPUs:"

        # List the actual GPUs being used
        gpu_names = [torch.cuda.get_device_name(i) for i in range(visible_devices)]
        for i, name in enumerate(gpu_names):
            mem_alloc = torch.cuda.memory_allocated(i) / 1024**2
            mem_res = torch.cuda.memory_reserved(i) / 1024**2

            msg += f"  {i}: alloc {mem_alloc:.2f} Mb / res {mem_res:.2f} Mb;"

            #log(f"  GPU {i}: {name}, alloc: {mem_alloc:.2f} Mb (reserved: {mem_res:.2f} Mb)")

        log(msg)
    elif accelerator is not None and accelerator.device.type == "mps":
        mem_alloc = torch.mps.current_allocated_memory() / 1024**2
        log(f"Device being used: {accelerator.device}, mem alloc: {mem_alloc} Mb")
    else:
        log(f"No acceleration")

    if mdl is not None:
        log(f"Model device: {mdl.device}")


def chain_params(coupling_specs):
    for spec in coupling_specs:
        yield from spec.model.parameters()


class SwitchingAccelerator:
    def __init__(self, coupling_specs, train_set, train_kwargs):
        self.coupling_specs = coupling_specs

        self.train_set = train_set
        self.kwargs = train_kwargs

        self.train_loss_list = []

        self._init_acc_and_stuff()

    def _init_acc_and_stuff(self):
        self.accelerator = Accelerator(gradient_accumulation_steps=self.kwargs.accum_steps)

        epoch_len = len(self.train_set)
        train_len = epoch_len * self.kwargs.epochs

        optimizer = torch.optim.AdamW(chain_params(self.coupling_specs), lr=self.kwargs.lr)
        lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=int(train_len * 0.1),
                                     num_training_steps=train_len)
        models = [s.model for s in self.coupling_specs]

        self.train_set, self.optimizer, self.lr_scheduler, *self.models = self.accelerator.prepare(
            self.train_set, optimizer, lr_scheduler, *models)

        if self.kwargs.continue_training:
            self.accelerator.load_state(self.kwargs.mdl_id)
            self.data_state = load_data_state(self.kwargs.mdl_id)
            self.train_loss_list = load_loss_list(self.kwargs.mdl_id)
        else:
            self.data_state = (0, 0)
            self.train_loss_list = []

        self.accelerator.register_for_checkpointing(self.optimizer, self.lr_scheduler, *self.models)
        self.accelerator.save_state(self.kwargs.save_location)

        #self._save_all(*self.data_state)

    def train(self):
        #train_dl_acc, optimizer_acc, *models_acc = self.accelerator.prepare(train_dataloader, self.optimizer, *models)
        self._main_loop()

        self.accelerator.wait_for_everyone()

        unwr_coupled_model = self.accelerator.unwrap_model(self.models[0])

        return unwr_coupled_model, self.train_loss_list

    def _prepare_inputs(self, batch_with_idxs):
        weird_inputs, src_k, tgt_k, _ = batch_with_idxs

        batch_size = weird_inputs['input_ids'].size()[0]

        proc_batch_size = batch_size / self.accelerator.num_processes

        from_proc_idx = int(self.accelerator.process_index * proc_batch_size)
        to_proc_idx = int((self.accelerator.process_index + 1) * proc_batch_size)

        unweird_inputs = {k: weird_inputs[k][from_proc_idx:to_proc_idx].to(self.accelerator.device)
                          for k in weird_inputs}

        return unweird_inputs, src_k, tgt_k

    def _main_loop(self):
        if self.accelerator.is_main_process:
            logger = SameLineLogger(len(self.train_set), self.kwargs.epochs)
            logger.line_start()
        else:
            logger = None

        self.models[0].train()

        batch_idx = 0

        for epoch_idx in range(self.data_state[1], self.kwargs.epochs):

            for batch_with_bin_idxs in self.train_set:
                if batch_idx > self.data_state[0]:
                    inputs, src_k, tgt_k = self._prepare_inputs(batch_with_bin_idxs)

                    encoder_vecs = encode(self.models[src_k], inputs)
                    outputs = self.models[tgt_k](attention_mask=inputs['attention_mask'], labels=inputs['labels'], encoder_outputs=encoder_vecs)

                    loss = outputs.loss

                    self.train_loss_list.append((loss.item(), src_k, tgt_k))

                    self.accelerator.backward(loss)

                    self._step_and_perhaps_save(logger, batch_idx, epoch_idx, float(loss.item()))
                    
                batch_idx += 1

        if self.accelerator.is_main_process:
            logger.line_break()

    def _step_and_perhaps_save(self, logger, batch_i, epoch_i, loss):
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()

        if self.accelerator.is_main_process and ((batch_i + 1) % self.kwargs.log_steps == 0):
            logger.step(batch_i, loss)

        if ((batch_i + 1) % self.kwargs.save_steps == 0) or ((batch_i + 1) % len(self.train_set) == 0):
            self.accelerator.wait_for_everyone()

            if self.accelerator.is_main_process:
                logger.line_break()
                log(f"Saving at {batch_i + 1} steps, epoch {epoch_i + 1}")

                self._save_all(batch_i, epoch_i)

                logger.line_start()

    def _save_all(self, batch_i, epoch_i):
        epoch_len = len(self.train_set)

        ckpt_name = f"checkpoint-e{epoch_i + 1}-b{batch_i + 1:06}" if (
                    (batch_i + 1) % epoch_len) else f"checkpoint-e{epoch_i + 1}-full"

        this_location = os.path.join(self.kwargs.save_location, ckpt_name)
        if os.path.exists(this_location):
            raise FileExistsError("Cannot overwrite existing checkpoint")

        model_to_save = self.accelerator.unwrap_model(self.models[0])

        save_all_models(this_location, model_to_save, self.coupling_specs[0].tokenizer,
                        self.coupling_specs, self.train_loss_list, self.accelerator,
                        data_state=(batch_i, epoch_i))


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
    if os.path.exists(args.save_location):
        raise Exception(f"Save location '{args.save_location}' already exists, don't want to overwrite")

    return args


def yes_i_called_this_function_do_main():
    args = _cmdline_args()

    log("loading coupled model and tokenizer")
    main_model, main_tokenizer = load_hf_mdl_and_tok(args.mdl_id, verbose=True)

    coupling_specs = to_cpl_spec(args.langs, main_model, main_tokenizer, args.save_location)

    if args.anchor_mdl_id:
        log("loading anchor model and tokenizer")
        anchor_model, anchor_tokenizer = load_hf_mdl_and_tok(args.anchor_mdl_id, verbose=True)
        freeze_model(anchor_model)

        coupling_specs += to_cpl_spec(args.anchor_langs, anchor_model, anchor_tokenizer, args.anchor_mdl_id)

    train_set = MultilingualDatasetIterator(args.train_pretok_file)

    acc_trainer = SwitchingAccelerator(coupling_specs, train_set, args)

    upd_model, loss_list = acc_trainer.train()

    save_all_models(args.save_location, upd_model, main_tokenizer, coupling_specs, loss_list, acc_trainer.accelerator)

if __name__ == "__main__":
    yes_i_called_this_function_do_main()
