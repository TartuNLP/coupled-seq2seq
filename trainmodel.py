#!/usr/bin/env python3

import sys
import os
import torch

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_scheduler
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from accelerate import Accelerator
from datetime import datetime

from translate import hf_tok
from data import MultilingualBatchingDataset, make_path_compatible
from aux import log, maybe_smugri, to_kwargs, get_changed_config, same_line_log
from collections import namedtuple
from vivisect import vivisect_save_chkpt, vivisect_train_step, vivisect_eval_step, \
    to_cpl_spec, save_all_models, switch_modules
from langconv import is_nllb, is_madlad
from initmodel import mdl_param_count

CmdlineArgs = namedtuple("CmdlineArgs", "coupled_mdl_id train_data_file dev_data_file coupled_langs anchor_mdl_id anchor_langs save_location".split())

host_remote = True


def freeze_model(model):
    for n, p in model.named_parameters():
        p.requires_grad = False


def train_args(name, batch_size, **kw):
    prelim_result = Seq2SeqTrainingArguments(
        name,
        eval_strategy="steps",
        eval_steps=25000,
        learning_rate=1.5e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=1,
        weight_decay=0.01,
        save_strategy="steps",
        save_steps=100000,
        logging_steps=100000,
        max_steps=1500000,
        # num_train_epochs=3,
        # predict_with_generate=True
    )

    result = get_changed_config(prelim_result, ["skip_training", "batch"], **kw)

    return result


def load_hf_mdl_and_tok(mdl_id, tok_id=None, verbose=False):
    if tok_id is None:
        tok_id = mdl_id

    tokenizer = AutoTokenizer.from_pretrained(tok_id, token=hf_tok)
    model = AutoModelForSeq2SeqLM.from_pretrained(mdl_id, token=hf_tok, device_map="auto")

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


class SameLineLogger:
    def __init__(self, train_set):
        self.total = len(train_set)
        self.log_after = []
        self.log_len = 0

        self.start_time = datetime.now()

    def line_start(self):
        same_line_log(str(datetime.now()) + ": training batches ")

    def step(self, i, loss):
        passed_time = datetime.now() - self.start_time

        time_per_batch = passed_time / (i + 1)

        prediction = time_per_batch * (self.total - i - 1)

        msg = f"{i + 1} / {self.total}, loss={loss}, {time_per_batch}/iter, {prediction} to finish        "

        new_len = same_line_log(msg, self.log_len)

        self.log_len = new_len

    def line_break(self):
        sys.stderr.write("\n")


def report_devices(accelerator):
    if torch.cuda.is_available():
        # Get the visible devices from CUDA
        visible_devices = torch.cuda.device_count()
        log(f"Number of visible GPUs: {visible_devices}")

        # List the actual GPUs being used
        gpu_names = [torch.cuda.get_device_name(i) for i in range(visible_devices)]
        log("GPUs being used:")
        for i, name in enumerate(gpu_names):
            log(f"  GPU {i}: {name}")
    else:
        log(f"Device being used: {accelerator.device}")


class SwitchingAccelerator:
    def read_kwargs(self, kwargs):
        type_list = [int, float]
        kw_names = ["save_steps", "lr"]
        default_values = [10000, 1.5e-5]

        kw_with_dv = { kn: (dv if kn not in kwargs else typ(kwargs[kn])) for kn, dv, typ in zip(kw_names, default_values, type_list)}

        return namedtuple("kwargs", kw_names)(*[kw_with_dv[k] for k in kw_names])

    def __init__(self, model, coupling_specs, train_set, save_location, train_kwargs):
        self.model_to_train = model
        self.coupling_specs = coupling_specs
        self.train_set = train_set
        self.save_location = save_location
        self.kwargs = self.read_kwargs(train_kwargs)

        self.train_loss_list = []

        self.accelerator = Accelerator()

        report_devices(self.accelerator)

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=self.kwargs.lr)
        self.lr_scheduler = get_scheduler("linear", optimizer=self.optimizer, num_warmup_steps=200,
                                          num_training_steps=len(train_set))

    def _encode(self, model, inputs):
        if is_nllb(model):
            enc = model.model.encoder
        elif is_madlad(model):
            enc = model.base_model.encoder
        else:
            raise NotImplementedError(f"Model {model} is not supported yet.")

        inputs_without_labels = { k: inputs[k] for k in inputs if k != "labels" }

        return enc(**inputs_without_labels)

    def _main_loop(self, logger, models, optimizer, train_set):
        for m in models:
            m.train()

        for i, batch_with_idxs in enumerate(train_set):
            batch, src_k, tgt_k = batch_with_idxs
            inputs = batch.to(self.accelerator.device)

            encoder_vecs = self._encode(models[src_k], inputs)

            outputs = models[tgt_k](**inputs, encoder_outputs=encoder_vecs)
            loss = outputs.loss

            self.train_loss_list.append((loss.item(), src_k, tgt_k))

            self.accelerator.backward(loss)

            optimizer.step()
            self.lr_scheduler.step()
            optimizer.zero_grad()

            avg_loss_vals = [i[0] for i in self.train_loss_list[-10:]]
            avg_loss = sum(avg_loss_vals)/len(avg_loss_vals)

            self._step_and_perhaps_save(logger, i, avg_loss, models[0])

    def _step_and_perhaps_save(self, logger, i, loss, model):
        logger.step(i, loss)

        if not ((i + 1) % self.kwargs.save_steps):
            logger.line_break()

            log(f"Saving at {i + 1} steps")

            if self.accelerator.is_main_process:
                this_location = os.path.join(self.save_location, f"checkpoint-{i + 1}")
                if os.path.exists(this_location):
                    raise FileExistsError("Cannot overwrite existing checkpoint")

                model_to_save = self.accelerator.unwrap_model(model)
                save_all_models(this_location, model_to_save, self.coupling_specs[0].tokenizer, self.coupling_specs)

            logger.line_start()

    def train(self):
        logger = SameLineLogger(self.train_set)
        logger.line_start()
        torch.distributed.init_process_group("nccl", rank=0, world_size=8)
        models_acc = self.accelerator.prepare(*[torch.nn.parallel.DistributedDataParallel(s.model) for s in self.coupling_specs])

        optimizer_acc = self.accelerator.prepare(self.optimizer)
        train_set_acc = self.accelerator.prepare(self.train_set)

        self.train_loss_list = []

        self._main_loop(logger, models_acc, optimizer_acc, train_set_acc)

        logger.line_break()
        self.accelerator.wait_for_everyone()
        self.accelerator.end_training()

def do_training(model, model_name, train_set, val_set, batch_size, cpl_specs, train_kwargs):
    args = train_args(model_name, batch_size=batch_size, **train_kwargs)

    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=train_set,
        eval_dataset=val_set,
        # data_collator=data_collator,
        # tokenizer=tokenizer,
        # compute_metrics=prep_metric_func(tokenizer),
    )

    vivisect_save_chkpt(trainer, cpl_specs, cpl_specs[0].tokenizer)

    # tok_cpl_specs = integrate_tokenizer_with_cpl_specs(cpl_specs)
    vivisect_train_step(trainer, cpl_specs)
    vivisect_eval_step(trainer, cpl_specs)

    log("Started training")

    trainer.train()

    log("Finished training, saving models")

    trainer.save_state()


def do_main():
    if not host_remote:
        sys.argv = ["X", "models/smol", "data/smugri4a-dev.json", "smugri"]

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

    train_set = MultilingualBatchingDataset(args.train_data_file, coupling_specs, batch_size,
                                            tracing_msg="TRAIN", verbose=True, leave_only=lp_set)

    if 'skip_training' not in train_kwargs:
        acc_trainer = SwitchingAccelerator(coupled_model, coupling_specs, train_set, args.save_location, train_kwargs)

        acc_trainer.train()

        save_all_models(args.save_location, coupled_model, coupled_tokenizer, coupling_specs)


if __name__ == "__main__":
    host_remote = len(sys.argv) > 1

    do_main()
