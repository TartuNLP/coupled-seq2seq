#!/usr/bin/env python3

import sys
import os
import torch

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_scheduler
from accelerate import Accelerator, DataLoaderConfiguration

from translate import hf_tok, encode
from data import MultilingualDatasetIterator
from aux import log, maybe_smugri_, SameLineLogger, CmdlineArgs
from collections import namedtuple
from coupling import to_cpl_spec, save_all_models
from modelops import mdl_param_count

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
    def __init__(self, coupling_specs, train_set, save_location, train_kwargs):
        self.coupling_specs = coupling_specs

        self.train_set = train_set
        self.save_location = save_location
        self.kwargs = train_kwargs

        self.train_loss_list = []

        #dl_conf = DataLoaderConfiguration(split_batches=True)
        self.accelerator = Accelerator(gradient_accumulation_steps=self.kwargs.accum_steps) #, dataloader_config=dl_conf)

        self.optimizer = torch.optim.AdamW(chain_params(coupling_specs), lr=self.kwargs.lr)
        self.lr_scheduler = get_scheduler("linear", optimizer=self.optimizer, num_warmup_steps=200,
                                          num_training_steps=len(train_set))

    def train(self):
        #train_dataloader = DataLoader(self.train_set)
        models = [s.model for s in self.coupling_specs]

        #train_dl_acc, optimizer_acc, *models_acc = self.accelerator.prepare(train_dataloader, self.optimizer, *models)
        train_dl_acc, optimizer_acc, *models_acc = self.accelerator.prepare(self.train_set, self.optimizer, *models)

        self.train_loss_list = []

        self._main_loop(models_acc, optimizer_acc, train_dl_acc)

        self.accelerator.wait_for_everyone()

        unwr_coupled_model = self.accelerator.unwrap_model(models_acc[0])

        return unwr_coupled_model, self.train_loss_list

    def _main_loop(self, models, optimizer, train_set):
        if self.accelerator.is_main_process:
            logger = SameLineLogger(len(self.train_set), self.kwargs.epochs)
            logger.line_start()
        else:
            logger = None

        models[0].train()

        batch_idx = 0

        for epoch_idx in range(self.kwargs.epochs):

            for batch_with_bin_idxs in train_set:
                weird_inputs, src_k, tgt_k, _ = batch_with_bin_idxs

                batch_size = weird_inputs['input_ids'].size()[0]

                proc_batch_size = batch_size / self.accelerator.num_processes

                from_proc_idx = int(self.accelerator.process_index * proc_batch_size)
                to_proc_idx = int((self.accelerator.process_index + 1) * proc_batch_size)

                unweird_inputs = {k: weird_inputs[k][from_proc_idx:to_proc_idx].to(self.accelerator.device) for k in weird_inputs}

                encoder_vecs = encode(models[src_k], unweird_inputs)
                outputs = models[tgt_k](attention_mask=unweird_inputs['attention_mask'], labels=unweird_inputs['labels'], encoder_outputs=encoder_vecs)

                loss = outputs.loss

                self.train_loss_list.append((loss.item(), src_k, tgt_k))

                self.accelerator.backward(loss)

                optimizer.step()
                self.lr_scheduler.step()
                optimizer.zero_grad()

                self._step_and_perhaps_save(logger, batch_idx, epoch_idx, float(loss.item()), models[0])
                batch_idx += 1

        if self.accelerator.is_main_process:
            logger.line_break()

    def _step_and_perhaps_save(self, logger, batch_i, epoch_i, loss, model):
        epoch_len = len(self.train_set)

        if self.accelerator.is_main_process and not (batch_i + 1) % self.kwargs.log_steps:
            logger.step(batch_i, loss)

        if not ((batch_i + 1) % self.kwargs.save_steps) or not ((batch_i + 1) % epoch_len):
            if self.accelerator.is_main_process:
                logger.line_break()

            log(f"Saving at {batch_i + 1} steps, epoch {epoch_i + 1}")

            if self.accelerator.is_main_process:
                ckpt_name = f"checkpoint-e{epoch_i + 1}-b{batch_i + 1}" if ((batch_i + 1) % epoch_len) else f"checkpoint-e{epoch_i + 1}-full"

                this_location = os.path.join(self.save_location, ckpt_name)
                if os.path.exists(this_location):
                    raise FileExistsError("Cannot overwrite existing checkpoint")

                model_to_save = self.accelerator.unwrap_model(model)
                save_all_models(this_location, model_to_save, self.coupling_specs[0].tokenizer, self.coupling_specs, loss_list=self.train_loss_list, trainer=self.accelerator)

            if self.accelerator.is_main_process:
                logger.line_start()


def _cmdline_args(input_values):
    description = """Train or tune models - TODO"""

    pos_args = ["mdl_id", "save_location", "train_file", "langs"]

    kw_args = { "anchor_mdl_id": None, "anchor_langs": None, "batch_size": 16,
                "save_steps": 1500, "lr": 1.5e-5, "accum_steps": 1, "log_steps": 100, "epochs": 4  }

    #post-process the arguments
    args = CmdlineArgs(description, pos_arg_list=pos_args, kw_arg_dict=kw_args, input_args=input_values)

    args.langs = maybe_smugri_(args.langs)

    if args.anchor_langs is not None:
        args.anchor_langs = maybe_smugri_(args.anchor_langs)

    # if the directory args.save_location already exists, raise an exception:
    if os.path.exists(args.save_location):
        raise Exception(f"Save location '{args.save_location}' already exists, don't want to overwrite")

    log(f"Launched as {args}")

    return args


def yes_i_called_this_function_do_main(iv):
    args = _cmdline_args(iv)

    log("loading coupled model and tokenizer")
    main_model, main_tokenizer = load_hf_mdl_and_tok(args.mdl_id, verbose=True)

    coupling_specs = to_cpl_spec(args.langs, main_model, main_tokenizer, args.save_location)

    if args.anchor_mdl_id is not None:
        log("loading anchor model and tokenizer")
        anchor_model, anchor_tokenizer = load_hf_mdl_and_tok(args.anchor_mdl_id, verbose=True)
        freeze_model(anchor_model)

        coupling_specs += to_cpl_spec(args.anchor_langs, anchor_model, anchor_tokenizer, args.anchor_mdl_id)

    train_set = MultilingualDatasetIterator(args.train_file, args.batch_size)

    acc_trainer = SwitchingAccelerator(coupling_specs, train_set, args.save_location, args)

    upd_model, loss_list = acc_trainer.train()

    save_all_models(args.save_location, upd_model, coupled_tokenizer, coupling_specs, loss_list=loss_list)

if __name__ == "__main__":
    input_values = sys.argv[1:] if len(sys.argv) > 1 \
        else ["models/smol", "models/smol_upd", "data/smugri4a-dev.json", "smugri"]

    yes_i_called_this_function_do_main(input_values)
