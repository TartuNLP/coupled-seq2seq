import os
import json

from collections import namedtuple

from langconv import is_nllb, is_madlad, langs_to_mdl_type, get_mdl_type

CouplingSpecTuple = namedtuple("CouplingSpecPair", ["lang_set", "tokenizer", "model_id", "model"])

MODULE_CONFIG_FILE = "coupled_module_config.json"

"""
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
            raise NotImplementedError(f"Model {tgt_mdl.__class__.__name__} is not supported yet.")

        tgt_mdl.lm_head = coupling_spec[output_index].lm_head
        tgt_mdl.config.vocab_size = coupling_spec[output_index].voc_size
"""

"""def switch_modules_according_to_input(model, coupling_spec, inputs):
    input_index, new_input_ids = extract_index_from_tensor(inputs['input_ids'])
    output_index, new_labels = extract_index_from_tensor(inputs['labels'])

    debug(f"Switching according to input: {inputs}, result: {input_index} / {output_index}")

    switch_modules(model, coupling_spec, input_index, output_index)

    inputs['input_ids'] = new_input_ids
    inputs['labels'] = new_labels

    return inputs
"""

"""def vivisect_eval_step(trainer_obj, coupling_spec):
    old_func = trainer_obj.prediction_step

    def new_prediction_step(self, model, inputs, prediction_loss_only, ignore_keys):
        upd_inputs = switch_modules_according_to_input(model, coupling_spec, inputs)

        return old_func(model, upd_inputs, prediction_loss_only, ignore_keys)

    trainer_obj.prediction_step = types.MethodType(new_prediction_step, trainer_obj)


def vivisect_train_step(trainer_obj, coupling_spec):
    old_func = trainer_obj.training_step

    def new_training_step(self, model, inputs, num_items_in_batch):
        upd_inputs = switch_modules_according_to_input(model, coupling_spec, inputs)

        return old_func(model, upd_inputs, num_items_in_batch)

    trainer_obj.training_step = types.MethodType(new_training_step, trainer_obj)
"""

"""def restore_base_model(model, cpl_specs):
    switch_modules(model, cpl_specs, 0, 0)
"""

"""def vivisect_save_chkpt(trainer_obj, cpl_specs, tokenizer):
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
"""

def save_module_config(model_dir, coupling_specs):
    config = [{'lang_set': list(spec.lang_set), 'model_id': spec.model_id if i > 0 else model_dir} for i, spec in enumerate(coupling_specs)]

    with open(os.path.join(model_dir, MODULE_CONFIG_FILE), "w") as f:
        json.dump(config, f, indent=2, sort_keys=True)
        f.write("\n")


def to_cpl_spec(langs, model, tokenizer, location):
    mdl_type = get_mdl_type(model)
    cpl_langs = set(langs_to_mdl_type(mdl_type, langs))

    return [CouplingSpecTuple(cpl_langs, tokenizer, location, model)]


def load_module_config(model_dir):
    return [{"model_id": model_dir, "lang_set": {}}]
    #try:
    #    with open(os.path.join(model_dir, MODULE_CONFIG_FILE), "r") as f:
    #        config = json.load(f)
    #        return config
    #except FileNotFoundError:
    #    return [{"model_id": model_dir, "lang_set": {}}]


def save_loss_list(location, loss_list):
    if loss_list is not None:
        with open(os.path.join(location, "loss_list.json"), "w") as f:
            json.dump(loss_list, f, indent=2, sort_keys=True)
            f.write("\n")
    else:
        raise Exception("No loss list to save")


def save_training_state(location, trainer):
    if trainer is not None:
        trainer.state.save_to_json(os.path.join(location, "trainer_state.json"))


def save_all_models(location, model, tokenizer, cpl_specs, loss_list=None, trainer=None):
    model.save_pretrained(location)
    tokenizer.save_pretrained(location)
    save_module_config(location, cpl_specs)

    save_loss_list(location, loss_list)

    save_training_state(location, trainer)
