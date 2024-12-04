#!/usr/bin/env python3

import sys

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

from translate import hf_tok, maybe_smugri
from data import get_tr_pairs, MultilingualBatchingDataset
from aux import log
from collections import namedtuple
from vivisect import vivisect_save_chkpt, vivisect_train_step, vivisect_eval_step, \
    to_cpl_spec, save_all_models

CmdlineArgs = namedtuple("CmdlineArgs", "coupled_mdl_id train_data_file dev_data_file coupled_langs anchor_mdl_id anchor_langs save_location".split())

host_remote = True


def freeze_model(model):
    for n, p in model.named_parameters():
        p.requires_grad = False


def train_args(name, batch_size):
    return Seq2SeqTrainingArguments(
        name,
        eval_strategy="steps",
        eval_steps=100000,
        learning_rate=1.5e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=1,
        weight_decay=0.01,
        save_strategy="steps",
        save_steps=100000,
        logging_steps=100000,
        max_steps=500000,
        # num_train_epochs=3,
        # predict_with_generate=True
    )


def train_args_tmp(name, batch_size):
    return Seq2SeqTrainingArguments(
        name,
        eval_strategy="steps",
        eval_steps=10000,
        learning_rate=1.5e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=1,
        weight_decay=0.01,
        save_strategy="steps",
        save_steps=10000,
        logging_steps=10000,
        max_steps=10000,
    )


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
        coupled_mdl_id = sys.argv[1]
        train_data_file = sys.argv[2]
        dev_data_file = sys.argv[3]
        raw_coupled_langs = maybe_smugri(sys.argv[4])

        coupled_langs = set(raw_coupled_langs.split(","))

        if len(sys.argv) > 5:
            anchor_mdl_id = sys.argv[5]
            raw_anchor_langs = maybe_smugri(sys.argv[6])

            anchor_langs = set(raw_anchor_langs.split(","))

            mdl_name_suff = "-mix" if (coupled_langs & anchor_langs) else "-cpl"

            mdl_name_suff += "-" + anchor_mdl_id.replace("/", "_").replace(":", "-")
        else:
            anchor_mdl_id = None
            anchor_langs = None

            mdl_name_suff = "-indtrained"

        result = CmdlineArgs(coupled_mdl_id, train_data_file, dev_data_file, coupled_langs, anchor_mdl_id, anchor_langs,
                             coupled_mdl_id + mdl_name_suff)

        return result

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


#def integrate_tokenizer_with_cpl_specs(cpl_specs):
#    result = defaultdict(set)
#
#    for idx, spec_tuple in enumerate(cpl_specs):
#        for lang in spec_tuple.lang_set:
#            lang_tok_id = spec_tuple.tokenizer.get_lang_id(lang)
#            result[lang_tok_id].add(idx)
#
#    return result, cpl_specs


def do_training(model, model_name, train_set, val_set, batch_size, cpl_specs):
    #args = train_args(model_name, batch_size=batch_size)
    args = train_args_tmp(model_name, batch_size=batch_size)

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
    #return CmdlineArgs("models/smol", "data/train.json", "data/dev.json", {"fi", "en"}, "facebook/m2m100_418M", {"fi", "en"}, "-indtmp")
    return CmdlineArgs("models/smol", "data/liv_train.json", "data/liv_train.json", {"liv", "et", "lv", "en"}, None, None, "-indtmp")


def do_main():
    args = cmdline_args() if host_remote else dud()

    log(f"Launched as {args}")

    log("loading coupled model and tokenizer")
    coupled_model, coupled_tokenizer = load_hf_mdl_and_tok(args.coupled_mdl_id, verbose=True)

    coupling_specs = to_cpl_spec(args.coupled_langs, coupled_model, coupled_tokenizer, args.save_location)

    if args.anchor_mdl_id is not None:
        log("loading anchor model and tokenizer")
        anchor_model, anchor_tokenizer = load_hf_mdl_and_tok(args.anchor_mdl_id, verbose=True)
        freeze_model(anchor_model)

        coupling_specs += to_cpl_spec(args.anchor_langs, anchor_model, anchor_tokenizer, args.anchor_mdl_id)

    lp_set = set(get_lps_from_specs(coupling_specs))
    log(f"loading training data {str(lp_set)[:50]}")
    # lp_set = { "en-fi" }

    train_set_pairs = list(get_tr_pairs(filename=args.train_data_file, leave_only=lp_set))
    val_set_pairs = list(get_tr_pairs(filename=args.dev_data_file, leave_only=lp_set))

    batch_size = 16

    train_set = MultilingualBatchingDataset(train_set_pairs, coupling_specs, batch_size=batch_size, tracing_msg="TRAIN", verbose=True)
    val_set = MultilingualBatchingDataset(val_set_pairs, coupling_specs, batch_size=batch_size, tracing_msg="VAL", verbose=True)

    do_training(coupled_model, args.save_location, train_set, val_set, batch_size, coupling_specs)

    save_all_models(args.save_location, coupled_model, coupled_tokenizer, coupling_specs)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        host_remote = True
    else:
        host_remote = False

    do_main()
