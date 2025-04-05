import json
import os
from collections import namedtuple

import torch

from aux import log
from langconv import get_mdl_type, langs_to_mdl_type

CouplingSpecTuple = namedtuple("CouplingSpecPair", ["lang_set", "tokenizer", "postokenizer", "model_id", "model"])

hf_tok = None
with open("hf_token", 'r') as fh:
    hf_tok = fh.read().strip()



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


def to_cpl_spec(langs, model, tokenizer, postokenizer, location):
    mdl_type = get_mdl_type(tokenizer)
    cpl_langs = set(langs_to_mdl_type(mdl_type, langs))

    return [CouplingSpecTuple(cpl_langs, tokenizer, postokenizer, location, model)]


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


def save_all_models(location, model, tokenizer, cpl_specs, trainer=None):
    if not os.path.exists(location):
        os.makedirs(location)

    if trainer is not None:
        trainer.save_state(location)

    model.config.save_pretrained(location)
    model.generation_config.save_pretrained(location)

    tokenizer.save_pretrained(location)

    save_module_config(location, cpl_specs)


def report_devices(msg = "", accelerator = None, mdl = None):
    if torch.cuda.is_available():
        # Get the visible devices from CUDA
        visible_devices = torch.cuda.device_count()

        #log(f"Number of visible GPUs: {visible_devices}")
        msg = f"{msg:30} {visible_devices} GPUs:"

        # List the actual GPUs being used
        gpu_names = [torch.cuda.get_device_name(i) for i in range(visible_devices)]
        for i, name in enumerate(gpu_names):
            mem_alloc = torch.cuda.memory_allocated(i) / 1024**2
            mem_res = torch.cuda.memory_reserved(i) / 1024**2

            if mem_alloc > 0.01 or mem_res > 0.01:
                msg += f"  {i}: alloc {mem_alloc:.2f} Mb / res {mem_res:.2f} Mb;"

        log(msg)
    elif accelerator is not None and accelerator.device.type == "mps":
        mem_alloc = torch.mps.current_allocated_memory() / 1024**2
        log(f"{msg:30} device being used: {accelerator.device}, mem alloc: {mem_alloc} Mb")
    else:
        log(f"No acceleration")

    if mdl is not None:
        log(f"Model device: {mdl.device}")


def is_gen_ai(mdl_id):
    lc = mdl_id.lower()
    return not ("madlad" in lc or "nllb" in lc or "m2m" in lc or "bart" in lc)


