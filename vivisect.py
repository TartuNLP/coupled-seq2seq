import sys
import types
import pdb
import os
import json

from collections import namedtuple

from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from torch import reshape
from aux import debug
from langconv import is_nllb, is_madlad

CouplingSpecTuple = namedtuple("CouplingSpecPair",
                               ["lang_set", "voc_size", "encoder", "decoder", "lm_head", "tokenizer", "model_id"])

MODULE_CONFIG_FILE = "coupled_module_config.json"


def inject_enc_dec_tracing(model, msg, prefx):
    old_fwd = model.forward

    def new_func(self, *args, **kwargs):
        input_ids = kwargs['input_ids']

        langs = set(input_ids[:, 0].tolist())
        max_val = max(reshape(input_ids, (-1,)).tolist())

        #if max_val >= self.config.vocab_size:
        #    sys.stderr.write("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n")
        #    sys.stderr.write("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n")
        #    sys.stderr.write("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n")
        #    sys.stderr.write(
        #        f"\nDEBUG: {msg}, max input val: {max_val}, langs: {langs} / #L: {len(self.layers)}, #V: {self.config.vocab_size}, D: {self.config.d_model}, FFN: {self.config.__dict__[prefx + '_ffn_dim']};\n")
        #    sys.stderr.write("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n")
        #    sys.stderr.write("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n")
        #    sys.stderr.write("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n")

        return old_fwd(*args, **kwargs)

    model.forward = types.MethodType(new_func, model)


def switch_modules(tgt_mdl, coupling_spec, input_index, output_index):
    debug(f"Switching modules to {input_index} / {output_index}")

    if input_index == 1 and output_index == 1:
        raise ValueError("Dud!")

    if input_index is not None:
        if is_nllb(tgt_mdl):
            tgt_mdl.model.encoder = coupling_spec[input_index].encoder
        elif is_madlad(tgt_mdl):
            tgt_mdl.base_model.encoder = coupling_spec[input_index].encoder
        else:
            raise NotImplementedError(f"Model {tgt_mdl} is not supported yet.")

    if output_index is not None:
        if is_nllb(tgt_mdl):
            tgt_mdl.model.decoder = coupling_spec[output_index].decoder
        elif is_madlad(tgt_mdl):
            tgt_mdl.base_model.decoder = coupling_spec[output_index].decoder
        else:
            raise NotImplementedError(f"Model {tgt_mdl} is not supported yet.")

        tgt_mdl.lm_head = coupling_spec[output_index].lm_head
        tgt_mdl.config.vocab_size = coupling_spec[output_index].voc_size


#def get_idxs_from_tensor(input_tensor, tok_cpl_lang_dict):
#    language_ids = set(input_tensor[:, 0].tolist())
#
#    result = { 0, 1 }
#
#    for lang_id in language_ids:
#        result &= tok_cpl_lang_dict[lang_id]
#
#    if len(result) == 0:
#        raise ValueError("Batch includes languages from multiple bins/modules, not kosher!")
#    else:
#        return result


#def get_indeces_from_tensors(tok_cpl_lang_dict, input_tensor, output_tensor):
#    src_idxs = get_idxs_from_tensor(input_tensor, tok_cpl_lang_dict)
#    tgt_idxs = get_idxs_from_tensor(output_tensor, tok_cpl_lang_dict)
#
#    result = mix_and_sample_idxs_carefully(src_idxs, tgt_idxs)
#
#    if result[0] is None:
#        raise ValueError("Batch is bad, very bad!")
#    else:
#        return result


def extract_index_from_tensor(tensor):
    if tensor[0,0] >= 1e9:
        tensor[0,0] -= 1 << 30
        return 1, tensor
    else:
        return 0, tensor


def switch_modules_according_to_input(model, coupling_spec, inputs):
    input_index, new_input_ids = extract_index_from_tensor(inputs['input_ids'])
    output_index, new_labels = extract_index_from_tensor(inputs['labels'])

    debug(f"Switching according to input: {inputs}, result: {input_index} / {output_index}")

    switch_modules(model, coupling_spec, input_index, output_index)

    inputs['input_ids'] = new_input_ids
    inputs['labels'] = new_labels

    return inputs


def vivisect_eval_step(trainer_obj, coupling_spec):
    old_func = trainer_obj.prediction_step

    def new_prediction_step(self, model, inputs, prediction_loss_only, ignore_keys):
        upd_inputs = switch_modules_according_to_input(model, coupling_spec, inputs)

        return old_func(model, upd_inputs, prediction_loss_only, ignore_keys)

    trainer_obj.prediction_step = types.MethodType(new_prediction_step, trainer_obj)


def vivisect_train_step(trainer_obj, coupling_spec):
    old_func = trainer_obj.training_step

    def new_training_step(self, model, inputs):
        upd_inputs = switch_modules_according_to_input(model, coupling_spec, inputs)

        return old_func(model, upd_inputs)

    trainer_obj.training_step = types.MethodType(new_training_step, trainer_obj)


def restore_base_model(model, cpl_specs):
    switch_modules(model, cpl_specs, 0, 0)


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
    config = [{'lang_set': list(spec.lang_set), 'model_id': spec.model_id if i > 0 else model_dir} for i, spec in enumerate(coupling_specs)]

    with open(os.path.join(model_dir, MODULE_CONFIG_FILE), "w") as f:
        json.dump(config, f, indent=2, sort_keys=True)
        f.write("\n")


def to_cpl_spec(langs, model, tokenizer, location):
    if is_nllb(model):
        enc = model.model.encoder
        dec = model.model.decoder
    elif is_madlad(model):
        enc = model.base_model.encoder
        dec = model.base_model.decoder
    else:
        raise NotImplementedError(f"Model {model} is not supported yet.")

    return [CouplingSpecTuple(langs, model.config.vocab_size, enc, dec, model.lm_head, tokenizer, location)]


def load_module_config(model_dir):
    return [{"model_id": model_dir, "lang_set": {}}]
    #try:
    #    with open(os.path.join(model_dir, MODULE_CONFIG_FILE), "r") as f:
    #        config = json.load(f)
    #        return config
    #except FileNotFoundError:
    #    return [{"model_id": model_dir, "lang_set": {}}]


def save_all_models(location, model, tokenizer, cpl_specs):
    restore_base_model(model, cpl_specs)

    model.save_pretrained(location)
    tokenizer.save_pretrained(location)
    save_module_config(location, cpl_specs)
