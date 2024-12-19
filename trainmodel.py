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
    to_cpl_spec, save_all_models


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
    if host_remote:
        model = AutoModelForSeq2SeqLM.from_pretrained(mdl_id, token=hf_tok, device_map="auto")
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(mdl_id, token=hf_tok)

    if verbose:
        log(f"Loaded {mdl_id}, tokenizer voc size {len(tokenizer)}, model voc size {model.config.vocab_size}")

    return model, tokenizer


def cmdline_args():
    try:
        kwargs, filt_args = to_kwargs(sys.argv)

        coupled_mdl_id = filt_args[1]
        train_data_file = filt_args[2]
        dev_data_file = filt_args[3]
        raw_coupled_langs = maybe_smugri(filt_args[4])

        coupled_langs = set(raw_coupled_langs.split(","))

        if len(filt_args) > 5:
            anchor_mdl_id = filt_args[5]
            raw_anchor_langs = maybe_smugri(filt_args[6])

            anchor_langs = set(raw_anchor_langs.split(","))

            mdl_name_suff = "-mix" if (coupled_langs & anchor_langs) else "-cpl"

            mdl_name_suff += "-" + make_path_compatible(anchor_mdl_id)
        else:
            anchor_mdl_id = None
            anchor_langs = None

            mdl_name_suff = "-indtrained"

        if "bigmix" in train_data_file:
            mdl_name_suff += "-big"


        result = CmdlineArgs(coupled_mdl_id, train_data_file, dev_data_file, coupled_langs, anchor_mdl_id, anchor_langs,
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
        same_line_log("")


def do_accelerated_training(model, save_location, train_set, cpl_specs, kwargs):
    save_steps = 10000 if "save_steps" not in kwargs else int(kwargs["save_steps"])
    l_rate = 1.5e-5 if "lr" not in kwargs else float(kwargs["lr"])

    accelerator = Accelerator()

    optimizer = torch.optim.AdamW(model.parameters(), lr=l_rate)
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=200, num_training_steps=len(train_set))

    # Step 5: Prepare with Accelerator
    model, optimizer, train_set = accelerator.prepare(model, optimizer, train_set)

    logger = SameLineLogger(train_set)
    logger.line_start()

    for i, batch in enumerate(train_set):
        inputs = batch.to(accelerator.device)
        outputs = model(**inputs)
        loss = outputs.loss
        accelerator.backward(loss)

        if accelerator.accumulate(model):
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        logger.step(i, loss)

        if not ((i+1) % save_steps):
            logger.line_break()

            log(f"Saving at {i+1} steps")

            if accelerator.is_main_process:
                this_location = os.path.join(save_location, f"checkpoint-{i+1}")
                if os.path.exists(this_location):
                    raise FileExistsError("Cannot overwrite existing checkpoint")

                save_all_models(this_location, model, cpl_specs[0].tokenizer, cpl_specs)

            logger.line_start()

    logger.line_break()
    accelerator.wait_for_everyone()


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


def dud():
    return (CmdlineArgs("models/smol",
                       "data/liv_train.json", "data/liv_train.json",
                       {"liv", "et", "lv", "en"}, None, None, "-indtmp"),
            {})


def do_main():
    args, train_kwargs = cmdline_args() if host_remote else dud()

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
        do_accelerated_training(coupled_model, args.save_location, train_set, coupling_specs, train_kwargs)

        save_all_models(args.save_location, coupled_model, coupled_tokenizer, coupling_specs)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        host_remote = True
    else:
        host_remote = False

    do_main()
