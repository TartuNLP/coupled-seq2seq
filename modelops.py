import json
import os
from collections import namedtuple

from langconv import get_mdl_type, langs_to_mdl_type

CouplingSpecTuple = namedtuple("CouplingSpecPair", ["lang_set", "tokenizer", "model_id", "model"])
MODULE_CONFIG_FILE = "coupled_module_config.json"
DATA_STATE_FILE = "data_state.json"
LOSS_LIST_FILE = "loss_list.json"


def mdl_param_count(model):
    result = 0
    embedding_size = -1

    for n, p in model.named_parameters():
        this_count = 1

        for s in p.shape:
            this_count *= s

        result += this_count

        # if n == "model.shared.weight":

        if "shared.weight" in n:
            embedding_size = this_count

    return result, embedding_size


def to_cpl_spec(langs, model, tokenizer, location):
    mdl_type = get_mdl_type(tokenizer)
    cpl_langs = set(langs_to_mdl_type(mdl_type, langs))

    return [CouplingSpecTuple(cpl_langs, tokenizer, location, model)]


def _save_json_config(model_dir, filename, data):
    with open(os.path.join(model_dir, filename), "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def _load_json_config(model_dir, filename):
    try:
        with open(os.path.join(model_dir, filename), "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def save_module_config(model_dir, coupling_specs):
    config = [{'lang_set': list(spec.lang_set), 'model_id': spec.model_id if i > 0 else model_dir} for i, spec in enumerate(coupling_specs)]
    _save_json_config(model_dir, MODULE_CONFIG_FILE, config)


def load_module_config(model_dir):
    result = _load_json_config(model_dir, MODULE_CONFIG_FILE)

    return result if result is not None else [{"model_id": model_dir, "lang_set": {}}]


def load_loss_list(model_dir):
    result = _load_json_config(model_dir, LOSS_LIST_FILE)
    return result if result is not None else []


def save_loss_list(location, loss_list):
    if loss_list is not None:
        _save_json_config(location, LOSS_LIST_FILE, loss_list)
    else:
        raise Exception("No loss list to save")


def save_training_state(location, trainer):
    if trainer is not None:
        trainer.save_state(location)


def save_data_state(location, data):
    if data is not None:
        _save_json_config(location, DATA_STATE_FILE, data)


def load_data_state(location):
    result = _load_json_config(location, DATA_STATE_FILE)
    return result if result is not None else (0, 0)


def save_all_models(location, model, tokenizer, cpl_specs, loss_list=None, trainer=None, data_state=None):
    if not os.path.exists(location):
        os.makedirs(location)

    if trainer is not None:
        trainer.wait_for_everyone()

    model.save_pretrained(location)

    tokenizer.save_pretrained(location)

    save_training_state(location, trainer)

    save_module_config(location, cpl_specs)

    save_loss_list(location, loss_list)

    save_data_state(location, data_state)
