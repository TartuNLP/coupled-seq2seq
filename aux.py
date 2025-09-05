#!/usr/bin/env python3
import os
import socket

import numpy as np
import pickle
import sys

from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def log(msg, accelerator=None, all_threads=False):
    if accelerator is not None and all_threads:
        report_proc = f" ({accelerator.process_index+1}/{accelerator.num_processes})"
    else:
        report_proc = ""

    if accelerator is None or accelerator.is_main_process or all_threads:
        sys.stderr.write(str(datetime.now()) + report_proc + ": " + msg + '\n')


def _same_line_log(msg, len_to_del=0):
    """if sys.stderr.isatty():
        if len_to_del > 0:
            sys.stderr.write("\b" * len_to_del)

        new_len = len(msg)

        sys.stderr.write(msg)
        sys.stderr.flush()

        return new_len
    else:"""
    log(msg)


def debug(msg):
    pass
    ### log("\n(DEBUG) " + msg)


def maybe_convert(value):
    try:
        return int(value)
    except (ValueError, TypeError):
        try:
            return float(value)
        except (ValueError, TypeError):
            return value


def get_changed_config(conf, args):
    arg_dict = args.to_dict()

    for kwarg in arg_dict:
        if hasattr(conf, kwarg) and arg_dict[kwarg] is not None:
            setattr(conf, kwarg, maybe_convert(arg_dict[kwarg]))

    return conf


class SameLineLogger:
    def __init__(self, epoch_len, epoch_num, data_state):
        self.epoch_len = epoch_len
        self.epoch_num = epoch_num
        self.start_global_step = epoch_len * data_state.epoch_idx + data_state.elem_idx

        self.totalx = epoch_len * epoch_num

        self.log_after = []
        self.log_len = 0

        self.start_time = datetime.now()

    def line_start(self):
        _same_line_log(str(datetime.now()) + ": training batches ")

    def step(self, global_batch_idx, epoch_batch_idx, epoch_idx, loss, lr, grad):
        passed_time = datetime.now() - self.start_time
        time_per_batch = passed_time / (global_batch_idx - self.start_global_step)
        prediction = time_per_batch * (self.totalx - global_batch_idx)

        msg = f"{epoch_batch_idx} / {self.epoch_len}, epoch {epoch_idx + 1} / {self.epoch_num}, loss={loss}, avg {time_per_batch}/iter, {prediction} to finish, LR={lr:.2e}, grad={grad:.2e}        "

        new_len = _same_line_log(msg, self.log_len)

        self.log_len = new_len

    def line_break(self):
        sys.stderr.write("\n")


class CmdlineArgs:
    def __init__(self,
                 description,
                 pos_arg_list=None,
                 pos_arg_types=None,
                 kw_arg_dict=None,
                 input_args=None):

        self.description = description

        self.raw_pos_arg_list = pos_arg_list if pos_arg_list is not None else []
        self.raw_pos_arg_types = pos_arg_types \
            if pos_arg_types is not None \
            else [None] * len(self.raw_pos_arg_list)

        self.kw_arg_dict_with_defaults = kw_arg_dict if kw_arg_dict is not None else {}

        kw_vals, cmdline_values = self._to_kwargs(sys.argv[1:] if input_args is None else input_args)

        self._maybe_help(cmdline_values)

        self._handle_positional_args(cmdline_values)

        self._handle_keyword_args(kw_vals)

    @staticmethod
    def _to_kwargs(arg_list):
        key_args = dict(raw_entry.lstrip("-").split("=") for raw_entry in arg_list if "=" in raw_entry)
        filtered_arg_list = [arg for arg in arg_list if "=" not in arg]

        return key_args, filtered_arg_list

    def _handle_keyword_args(self, kw_vals):
        for kw in self.kw_arg_dict_with_defaults:
            if kw in kw_vals:
                val = self._convert_kw(kw_vals, kw)
                del kw_vals[kw]
            else:
                val = self.kw_arg_dict_with_defaults[kw]

            setattr(self, kw, val)

        if kw_vals:
            extra_keys = ", ".join(kw_vals.keys())
            msg = f"command-line keyword arguments '{extra_keys}' are not recognized."

            self._help_message_and_die(extra=msg)

    def _convert_kw(self, kw_vals, kw):
        if self.kw_arg_dict_with_defaults[kw] is None:
            return kw_vals[kw]
        else:
            this_typ = type(self.kw_arg_dict_with_defaults[kw])

            try:
                return False if this_typ == bool and kw_vals[kw] == 'False' else this_typ(kw_vals[kw])
            except ValueError:
                self._help_message_and_die(extra=f"could not convert '{kw_vals[kw]}' to '{this_typ}'")

    def _sanity_check_pos_args(self, cmdline_values):
        cmdline_len = len(cmdline_values)

        if cmdline_len < len(self.raw_pos_arg_list):
            self._help_message_and_die(
                extra=f"positional arguments missing: {', '.join(self.raw_pos_arg_list[cmdline_len:])}")

        if cmdline_len > len(self.raw_pos_arg_list):
            self._help_message_and_die(
                extra=f"superfluous positional arguments: {', '.join(cmdline_values[len(self.raw_pos_arg_list):])}")

    def _handle_positional_args(self, cmdline_values):
        self._sanity_check_pos_args(cmdline_values)

        for arg, val, typ in zip(self.raw_pos_arg_list, cmdline_values, self.raw_pos_arg_types):
            try:
                val = val if typ is None else typ(val)
            except ValueError:
                self._help_message_and_die(extra=f"could not convert '{val}' to '{typ}'")

            setattr(self, arg, val)

    def _maybe_help(self, cmdline_values):
        if len(cmdline_values) == 1 and cmdline_values[0] in {"--help", "-h", "-?"}:
            self._help_message_and_die()

    def _help_message_and_die(self, extra=None):
        sys.stderr.write("Help message: " + self.description + "\n")

        if self.raw_pos_arg_list:
            args_descr = ", ".join([f"'{arg}' ({typ.__name__  if typ is not None else 'any'})"
                                    for arg, typ in zip(self.raw_pos_arg_list, self.raw_pos_arg_types)])

            sys.stderr.write(f"Positional arguments: {args_descr}\n")

        if self.kw_arg_dict_with_defaults:
            kw_descr = ", ".join([f"'{kw}' (default: {val})"
                                  for kw, val in self.kw_arg_dict_with_defaults.items()])

            sys.stderr.write(f"Keyword arguments: {kw_descr}\n")

        if extra is not None:
            sys.stderr.write("Error: " + extra + "\n")

        sys.stderr.write("\n")
        sys.exit(-1)

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()
                if k not in {'description', 'raw_pos_arg_list', 'raw_pos_arg_types', 'kw_arg_dict_with_defaults'}}

    def __str__(self):
        return str(self.to_dict())

    def __repr__(self):
        return self.__str__()

if __name__ == "__main__":
    for dname in sys.argv[1:]:
        d = np.load(dname + "/custom_checkpoint_1.pkl", allow_pickle=True)
        p = pickle.loads(d['custom_checkpoint_1/data.pkl'])
        print(dname, p)


def load_model(mdl_id, device, accelerator=None, attention="flash_attention_2"):
    log(f"Load model", accelerator=accelerator)
    model = AutoModelForCausalLM.from_pretrained(mdl_id,
                                                 low_cpu_mem_usage=False,
                                                 torch_dtype=torch.bfloat16,
                                                 attn_implementation=attention)

    model.config.use_cache = False
    model = model.to(device)
    log(f"Model loaded on device: {model.device}.", accelerator=accelerator)

    return model


def load_tokenizer(mdl_id, accelerator=None):
    log(f"Load tokenizer", accelerator=accelerator)
    tokenizer = AutoTokenizer.from_pretrained(mdl_id)

    tokenizer.pad_token = "<|reserved_special_token_100|>"
    tokenizer.mask_token = "<|reserved_special_token_130|>"

    return tokenizer


def env_stuff():
    os.environ.setdefault("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "---"))
    os.environ.setdefault("RANK", os.environ.get("SLURM_PROCID", "0"))
    os.environ.setdefault("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1"))
    os.environ.setdefault("MASTER_ADDR", os.environ.get("SLURM_LAUNCH_NODE_IPADDR", "127.0.0.1"))
    os.environ.setdefault("MASTER_PORT", "29500")  # pick an open port

    # Optional: make sure each process selects its own GPU
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    try:
        log(
            f"host={socket.gethostname()} "
            f"RANK={os.environ['RANK']}/{os.environ['WORLD_SIZE']} "
            f"LOCAL_RANK={os.environ['LOCAL_RANK']} "
            f"HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES')} "
            f"ROCR_VISIBLE_DEVICES={os.environ.get('ROCR_VISIBLE_DEVICES')} "
            f"cuda_count={torch.cuda.device_count()} curr_dev={torch.cuda.current_device()}"
        )
    except AssertionError:
        log(
            f"host={socket.gethostname()} "
            f"RANK={os.environ['RANK']}/{os.environ['WORLD_SIZE']} "
            f"LOCAL_RANK={os.environ['LOCAL_RANK']} "
            f"HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES')} "
            f"ROCR_VISIBLE_DEVICES={os.environ.get('ROCR_VISIBLE_DEVICES')} "
            f"no cuda"
        )
