import sys
from datetime import datetime

SMUGRI_LOW = "fkv,izh,kca,koi,kpv,krl,liv,lud,mdf,mhr,mns,mrj,myv,olo,sjd,sje,sju,sma,sme,smj,smn,sms,udm,vep,vot,vro"
SMUGRI_HIGH = "deu,eng,est,fin,hun,lvs,nor,rus,swe"
SMUGRI = "deu,eng,est,fin,fkv,hun,izh,kca,koi,kpv,krl,liv,lud,lvs,mdf,mhr,mns,mrj,myv,nor,olo,rus,sjd,sje,sju,sma,sme,smj,smn,sms,swe,udm,vep,vot,vro"


def log(msg):
    sys.stderr.write(str(datetime.now()) + ": " + msg + '\n')


def same_line_log(msg, len_to_del=0):
    """if sys.stderr.isatty():
        if len_to_del > 0:
            sys.stderr.write("\b" * len_to_del)

        new_len = len(msg)

        sys.stderr.write(msg)
        sys.stderr.flush()

        return new_len
    else:"""
    sys.stderr.write(msg + "\n")


def debug(msg):
    pass
    ### log("\n(DEBUG) " + msg)


def log_2dict(twod_dict, msg):
    for k1 in twod_dict:
        for k2 in twod_dict[k1]:
            log(f"DEBUG {msg}: {k1}, {k2} --> {twod_dict[k1][k2]}")


def maybe_smugri(lang_def):
    if lang_def == "smugri-low":
        return SMUGRI_LOW
    elif lang_def == "smugri-high":
        return SMUGRI_HIGH
    elif lang_def == "smugri":
        return SMUGRI
    else:
        return lang_def


def smugri_back(lang_list):
    sll = sorted(lang_list)

    sll_str = ",".join(sll)

    if sll_str == SMUGRI_LOW:
        return "smugri-low"
    elif sll_str == SMUGRI_HIGH:
        return "smugri-high"
    elif sll_str == SMUGRI:
        return "smugri-full"
    else:
        return sll_str


def maybe_convert(value):
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def get_changed_config(conf, extra_keys=[], **kw):
    for extra_key in extra_keys:
        if extra_key in kw:
            del kw[extra_key]

    for kwarg in kw:
        if kwarg in conf.__dict__:
            conf.__dict__[kwarg] = maybe_convert(kw[kwarg])
        else:
            raise KeyError(f'key "{kwarg}" is not in model config')

    return conf


class SameLineLogger:
    def __init__(self, epoch_len, epoch_num):
        self.epoch_len = epoch_len
        self.epoch_num = epoch_num

        self.totalx = epoch_len * epoch_num

        self.log_after = []
        self.log_len = 0

        self.start_time = datetime.now()

    def line_start(self):
        same_line_log(str(datetime.now()) + ": training batches ")

    def step(self, batch_i, loss):
        passed_time = datetime.now() - self.start_time

        time_per_batch = passed_time / (batch_i + 1)

        prediction = time_per_batch * (self.totalx - batch_i - 1)

        batch_i_in_epoch = batch_i % self.epoch_len
        curr_epoch_i = batch_i // self.epoch_len

        msg = f"{batch_i_in_epoch + 1} / {self.epoch_len}, epoch {curr_epoch_i+1} / {self.epoch_num}, loss={loss}, {time_per_batch}/iter, {prediction} to finish        "

        new_len = same_line_log(msg, self.log_len)

        self.log_len = new_len

    def line_break(self):
        sys.stderr.write("\n")

"""    def read_kwargs(self, kwargs):
        type_list = [int, float, int, int, int]
        kw_names = ["save_steps", "lr", "accum_steps", "log_steps", "epochs"]
        default_values = [1500, 1.5e-5, 1, 100, 2]

        kw_with_dv = { kn: (dv if kn not in kwargs else typ(kwargs[kn])) for kn, dv, typ in zip(kw_names, default_values, type_list)}

        return namedtuple("kwargs", kw_names)(*[kw_with_dv[k] for k in kw_names])"""


def _to_kwargs(arg_list):
    key_args = dict(raw_entry.split("=") for raw_entry in arg_list if "=" in raw_entry)
    filtered_arg_list = [arg for arg in arg_list if "=" not in arg]

    return key_args, filtered_arg_list


class CmdlineArgs:
    def __init__(self,
                 description,
                 pos_arg_list,
                 pos_arg_types=None,
                 kw_arg_dict={},
                 input_args=None):

        self.description = description

        self.raw_pos_arg_list = pos_arg_list
        self.raw_pos_arg_types = pos_arg_types \
            if pos_arg_types is not None \
            else [None] * len(self.raw_pos_arg_list)

        self.kw_arg_dict_with_defaults = kw_arg_dict

        kw_vals, cmdline_values = _to_kwargs(sys.argv[1:] if input_args is None else input_args)

        self._maybe_help(cmdline_values)

        self._handle_positional_args(cmdline_values)

        self._handle_keyword_args(kw_vals)

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
                return this_typ(kw_vals[kw])
            except ValueError:
                self._help_message_and_die(extra=f"could not convert '{kw_vals[kw]}' to '{this_typ}'")

    def _sanity_check_pos_args(self, cmdline_values):
        cmdline_len = len(cmdline_values)

        if cmdline_len < len(self.raw_pos_arg_list):
            self._help_message_and_die(
                extra=f"positional arguments missing: {', '.join(self.raw_pos_arg_list[cmdline_len:])}")

        if cmdline_len > len(self.raw_pos_arg_list):
            self._help_message_and_die(
                extra=f"superfluous positional arguments: {', '.join(cmdline_len[len(self.raw_pos_arg_list):])}")

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

        sys.exit(-1)