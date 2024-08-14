import sys
import types
import pdb
import os
import json

from collections import namedtuple

from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from torch import reshape

# CouplingSpecTuple = namedtuple("CouplingSpecPair", ["lang_set", "model", "tokenizer", "model_id"])
CouplingSpecTuple = namedtuple("CouplingSpecPair",
                               ["lang_set", "voc_size", "encoder", "decoder", "lm_head", "tokenizer", "model_id"])

MODULE_CONFIG_FILE = "coupled_module_config.json"


def inject_enc_dec_tracing(model, msg, prefx):
    old_fwd = model.forward

    def new_func(self, *args, **kwargs):
        input_ids = kwargs['input_ids']
        max_val = max(reshape(input_ids, (-1,)).tolist())
        langs = set(input_ids[:, 0].tolist())

        if max_val >= self.config.vocab_size:
            sys.stderr.write("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n")
            sys.stderr.write("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n")
            sys.stderr.write("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n")
            sys.stderr.write(
                f"\nDEBUG: {msg}, max input val: {max_val}, langs: {langs} / #L: {len(self.layers)}, #V: {self.config.vocab_size}, D: {self.config.d_model}, FFN: {self.config.__dict__[prefx + '_ffn_dim']};\n")
            sys.stderr.write("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n")
            sys.stderr.write("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n")
            sys.stderr.write("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n")

        return old_fwd(*args, **kwargs)

    model.forward = types.MethodType(new_func, model)


def vivisect(tgt_mdl, coupling_spec, input_index, output_index):
    tgt_mdl.model.encoder = coupling_spec[input_index].encoder
    tgt_mdl.model.decoder = coupling_spec[output_index].decoder
    tgt_mdl.lm_head = coupling_spec[output_index].lm_head
    tgt_mdl.config.vocab_size = coupling_spec[output_index].voc_size


def get_index_from_tensor(input_tensor, tok_cpl_lang_dict):
    language_ids = set(input_tensor[:, 0].tolist())

    idxs = {tok_cpl_lang_dict[lang_id] for lang_id in language_ids}

    if len(idxs) != 1:
        raise ValueError("Batch includes languages from multiple bins/modules, not kosher!")
    else:
        return idxs.pop()


def perform_module_switching(model, tok_coupling_spec, inputs):
    tok_cpl_lang_dict, coupling_spec = tok_coupling_spec

    input_index = get_index_from_tensor(inputs['input_ids'], tok_cpl_lang_dict)
    output_index = get_index_from_tensor(inputs['labels'], tok_cpl_lang_dict)

    vivisect(model, coupling_spec, input_index, output_index)


def vivisect_eval_step(trainer_obj, tok_coupling_spec):
    old_func = trainer_obj.prediction_step

    def new_prediction_step(self, model, inputs, prediction_loss_only, ignore_keys):
        perform_module_switching(model, tok_coupling_spec, inputs)

        return old_func(model, inputs, prediction_loss_only, ignore_keys)

    trainer_obj.prediction_step = types.MethodType(new_prediction_step, trainer_obj)


def vivisect_train_step(trainer_obj, tok_coupling_spec):
    old_func = trainer_obj.training_step

    def new_training_step(self, model, inputs):
        perform_module_switching(model, tok_coupling_spec, inputs)

        return old_func(model, inputs)

    trainer_obj.training_step = types.MethodType(new_training_step, trainer_obj)


def restore_base_model(model, cpl_specs):
    vivisect(model, cpl_specs, 0, 0)


def vivisect_save_chkpt(trainer_obj, cpl_specs, tokenizer):
    old_func = trainer_obj._save_checkpoint

    def new_func(self, model, trial, metrics=None):
        restore_base_model(model, cpl_specs)

        # run the original function
        old_func(model, trial, metrics)

        # also save the tokenizer into the checkpoint folder
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)

        tokenizer.save_pretrained(output_dir)
        save_module_config(output_dir, cpl_specs)

    trainer_obj._save_checkpoint = types.MethodType(new_func, trainer_obj)


def save_module_config(model_dir, coupling_specs):
    config = [{'lang_set': list(spec.lang_set), 'model_id': spec.model_id} for spec in coupling_specs]

    with open(os.path.join(model_dir, MODULE_CONFIG_FILE), "w") as f:
        json.dump(config, f, indent=2, sort_keys=True)
        f.write("\n")


def load_module_config(model_dir):
    try:
        with open(os.path.join(model_dir, MODULE_CONFIG_FILE), "r") as f:
            config = json.load(f)
            return config
    except FileNotFoundError:
        return [{"model_id": model_dir, "lang_set": {}}]


def save_all_models(location, model, tokenizer, cpl_specs):
    restore_base_model(model, cpl_specs)

    model.save_pretrained(location)
    tokenizer.save_pretrained(location)
    save_module_config(location, cpl_specs)
