import os
import json

from collections import namedtuple

from langconv import langs_to_mdl_type, get_mdl_type

CouplingSpecTuple = namedtuple("CouplingSpecPair", ["lang_set", "tokenizer", "model_id", "model"])

MODULE_CONFIG_FILE = "coupled_module_config.json"
DATA_STATE_FILE = "data_state.json"


def save_module_config(model_dir, coupling_specs):
    config = [{'lang_set': list(spec.lang_set), 'model_id': spec.model_id if i > 0 else model_dir} for i, spec in enumerate(coupling_specs)]

    with open(os.path.join(model_dir, MODULE_CONFIG_FILE), "w") as f:
        json.dump(config, f, indent=2, sort_keys=True)
        f.write("\n")


def to_cpl_spec(langs, model, tokenizer, location):
    mdl_type = get_mdl_type(tokenizer)
    cpl_langs = set(langs_to_mdl_type(mdl_type, langs))

    return [CouplingSpecTuple(cpl_langs, tokenizer, location, model)]


def load_module_config(model_dir):
    try:
        with open(os.path.join(model_dir, MODULE_CONFIG_FILE), "r") as f:
            config = json.load(f)
            return config
    except FileNotFoundError:
        return [{"model_id": model_dir, "lang_set": {}}]


def save_loss_list(location, loss_list):
    if loss_list is not None:
        with open(os.path.join(location, "loss_list.json"), "w") as f:
            json.dump(loss_list, f, indent=2, sort_keys=True)
            f.write("\n")
    else:
        raise Exception("No loss list to save")


def save_training_state(location, trainer):
    if trainer is not None:
        trainer.save_state(location)


def save_data_state(location, data):
    if data is not None:
        with open(os.path.join(location, DATA_STATE_FILE), "w") as f:
            json.dump(data, f, indent=2, sort_keys=True)


def load_data_state(location):
    try:
        with open(os.path.join(location, DATA_STATE_FILE), "r") as f:
            data = json.load(f)
            return data
    except FileNotFoundError:
        return 0, 0


def save_all_models(location, model, tokenizer, cpl_specs, loss_list=None, trainer=None, data_state=None):
    model.save_pretrained(location)
    tokenizer.save_pretrained(location)
    save_module_config(location, cpl_specs)

    save_loss_list(location, loss_list)

    save_training_state(location, trainer)

    save_data_state(location, data_state)