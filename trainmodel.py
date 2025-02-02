#!/usr/bin/env python3

import sys
import os
import torch
import time

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_scheduler
from accelerate import Accelerator, DataLoaderConfiguration
from torch.utils.data import DataLoader, DistributedSampler

from translate import hf_tok, encode
from data import MultilingualDatasetIterator, make_path_compatible
from aux import log, maybe_smugri, to_kwargs, SameLineLogger
from collections import namedtuple
from coupling import to_cpl_spec, save_all_models
from initmodel import mdl_param_count
from random import random

CmdlineArgs = namedtuple("CmdlineArgs", "coupled_mdl_id train_data_file dev_data_file coupled_langs anchor_mdl_id anchor_langs save_location".split())

host_remote = True


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


def cmdline_args():
    try:
        kwargs, filt_args = to_kwargs(sys.argv)

        coupled_mdl_id = filt_args[1]
        train_data_file = filt_args[2]
        raw_coupled_langs = maybe_smugri(filt_args[3])

        coupled_langs = set(raw_coupled_langs.split(","))

        if len(filt_args) > 4:
            anchor_mdl_id = filt_args[4]
            raw_anchor_langs = maybe_smugri(filt_args[5])

            anchor_langs = set(raw_anchor_langs.split(","))

            mdl_name_suff = "-mix" if (coupled_langs & anchor_langs) else "-cpl"

            mdl_name_suff += "-" + make_path_compatible(anchor_mdl_id)
        else:
            anchor_mdl_id = None
            anchor_langs = None

            mdl_name_suff = "-indtrained"

        if "bigmix" in train_data_file:
            mdl_name_suff += "-big"

        result = CmdlineArgs(coupled_mdl_id, train_data_file, None, coupled_langs, anchor_mdl_id, anchor_langs,
                             coupled_mdl_id + mdl_name_suff)

        return result, kwargs

    except IndexError:
        sys.stderr.write(f"Usage: {sys.argv[0]}  coupled_mdl_id  train_data_file  dev_data_file  coupled_langs  [anchor_mdl_id  anchor_langs]\n")
        sys.stderr.write("       coupled_mdl_id:            ID of HuggingFace model to train or fine-tune, coupled to the anchor model\n")
        sys.stderr.write("       train_data_file, dev_data_file: self-explanatory\n")
        sys.stderr.write("       coupled_langs:  comma-separated list of language codes used in train and dev data that are to be sent to the coupled model\n")
        sys.stderr.write("       anchor_mdl_id (optional):  ID of HuggingFace model to be used as anchor (pre-trained multilingual model)\n")
        sys.stderr.write("       anchor_langs (optional):   comma-separated list of language codes used in train and dev data that are to be sent to the anchor model\n")
        sys.exit(1)


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
    def read_kwargs(self, kwargs):
        type_list = [int, float, int, int, int]
        kw_names = ["save_steps", "lr", "accum_steps", "log_steps", "epochs"]
        default_values = [10000, 1.5e-5, 1, 100, 4]

        kw_with_dv = { kn: (dv if kn not in kwargs else typ(kwargs[kn])) for kn, dv, typ in zip(kw_names, default_values, type_list)}

        return namedtuple("kwargs", kw_names)(*[kw_with_dv[k] for k in kw_names])

    def __init__(self, coupling_specs, train_set, save_location, train_kwargs):
        self.coupling_specs = coupling_specs

        self.train_set = train_set
        self.save_location = save_location
        self.kwargs = self.read_kwargs(train_kwargs)

        self.train_loss_list = []

        dl_conf = DataLoaderConfiguration(split_batches=True)
        self.accelerator = Accelerator(gradient_accumulation_steps=self.kwargs.accum_steps, dataloader_config=dl_conf)

        self.optimizer = torch.optim.AdamW(chain_params(coupling_specs), lr=self.kwargs.lr)
        self.lr_scheduler = get_scheduler("linear", optimizer=self.optimizer, num_warmup_steps=200,
                                          num_training_steps=len(train_set))

    def train(self):
        logger = SameLineLogger(self.train_set)
        logger.line_start()

        train_dataloader = DataLoader(self.train_set)
        models = [s.model for s in self.coupling_specs]

        train_dl_acc, optimizer_acc, *models_acc = self.accelerator.prepare(train_dataloader, self.optimizer, *models)

        self.train_loss_list = []

        self._main_loop(logger, models_acc, optimizer_acc, train_dl_acc)

        logger.line_break()

        self.accelerator.wait_for_everyone()

        unwr_coupled_model = self.accelerator.unwrap_model(models_acc[0])

        return unwr_coupled_model, self.train_loss_list

    def _main_loop(self, logger, models, optimizer, train_set):
        models[0].train()
        batch_idx = 0

        for epoch_idx in range(self.kwargs.epochs):
            for batch_with_bin_idxs in train_set:
                weird_inputs, src_k, tgt_k, _ = batch_with_bin_idxs

                unweird_inputs = {k: weird_inputs[k][0] for k in weird_inputs}

                encoder_vecs = encode(models[src_k], unweird_inputs)
                outputs = models[tgt_k](attention_mask=unweird_inputs['attention_mask'], labels=unweird_inputs['labels'], encoder_outputs=encoder_vecs)

                loss = outputs.loss

                self.train_loss_list.append((loss.item(), src_k.item(), tgt_k.item()))

                self.accelerator.backward(loss)

                optimizer.step()
                self.lr_scheduler.step()
                optimizer.zero_grad()

                self._step_and_perhaps_save(logger, batch_idx, epoch_idx, float(loss.item()), models[0])
                batch_idx += 1

    def _step_and_perhaps_save(self, logger, batch_i, epoch_i, loss, model):
        logger.step(batch_i, epoch_i, loss)

        if not ((batch_i + 1) % self.kwargs.save_steps):
            logger.line_break()

            log(f"Saving at {batch_i + 1} steps, epoch {epoch_i + 1}")

            if self.accelerator.is_main_process:
                this_location = os.path.join(self.save_location, f"checkpoint-{epoch_i + 1}-{batch_i + 1}")
                if os.path.exists(this_location):
                    raise FileExistsError("Cannot overwrite existing checkpoint")

                model_to_save = self.accelerator.unwrap_model(model)
                save_all_models(this_location, model_to_save, self.coupling_specs[0].tokenizer, self.coupling_specs, loss_list=self.train_loss_list, trainer=self.accelerator)

            logger.line_start()

    def debug_accelerator(self):
        train_dataloader = DataLoader(self.train_set)

        train_dl_acc = self.accelerator.prepare(train_dataloader)

        for epoch_idx in range(self.kwargs.epochs):
            for batch, src_k, tgt_k, batch_idx in train_dl_acc:
                sys.stderr.write(f"Handling batch nr {batch_idx.item()}: {batch['input_ids'].size()}; epoch {epoch_idx}, on {self.accelerator.process_index} / {self.accelerator.local_process_index} / {self.accelerator.num_processes}\n")
                time.sleep(0.5 + random()/2)


def do_main():
    if not host_remote:
        #sys.argv = ["X", "models/smol", "data/smugri4a-dev.json", "smugri", "facebook/nllb-200-distilled-600m", "smugri-high"]
        sys.argv = ["X", "models/smol", "data/smugri4a-dev.json", "smugri", "debugging=yes"]

    args, train_kwargs = cmdline_args()

    log(f"Launched as {args}")

    # if the directory args.save_location already exists, raise an exception:
    if os.path.exists(args.save_location):
        raise Exception(f"Save location '{args.save_location}' already exists, don't want to overwrite")

    log("loading coupled model and tokenizer")
    coupled_model, coupled_tokenizer = load_hf_mdl_and_tok(args.coupled_mdl_id, verbose=True)

    coupling_specs = to_cpl_spec(args.coupled_langs, coupled_model, coupled_tokenizer, args.save_location)

    if args.anchor_mdl_id is not None:
        log("loading anchor model and tokenizer")
        anchor_model, anchor_tokenizer = load_hf_mdl_and_tok(args.anchor_mdl_id, verbose=True)
        freeze_model(anchor_model)

        coupling_specs += to_cpl_spec(args.anchor_langs, anchor_model, anchor_tokenizer, args.anchor_mdl_id)

    lp_set = set(get_lps_from_specs(coupling_specs))

    batch_size = int(train_kwargs['batch']) if 'batch' in train_kwargs else 16

    train_set = MultilingualDatasetIterator(args.train_data_file, batch_size)

    acc_trainer = SwitchingAccelerator(coupling_specs, train_set, args.save_location, train_kwargs)

    if 'debugging' in train_kwargs:
        acc_trainer.debug_accelerator()
    else:
        upd_model, loss_list = acc_trainer.train()

        save_all_models(args.save_location, upd_model, coupled_tokenizer, coupling_specs, loss_list=loss_list)

if __name__ == "__main__":
    host_remote = len(sys.argv) > 1

    do_main()
