import sys
from datetime import datetime

SMUGRI_LOW = "fkv,izh,kca,koi,kpv,krl,liv,lud,mdf,mhr,mns,mrj,myv,olo,sjd,sje,sju,sma,sme,smj,smn,sms,udm,vep,vot,vro"
SMUGRI_HIGH = "deu,eng,est,fin,hun,lvs,nor,rus,swe"
SMUGRI = "deu,eng,est,fin,fkv,hun,izh,kca,koi,kpv,krl,liv,lud,lvs,mdf,mhr,mns,mrj,myv,nor,olo,rus,sjd,sje,sju,sma,sme,smj,smn,sms,swe,udm,vep,vot,vro"


def log(msg):
    sys.stderr.write(str(datetime.now()) + ": " + msg + '\n')


def same_line_log(msg, len_to_del=0):
    if sys.stderr.isatty():
        if len_to_del > 0:
            sys.stderr.write("\b" * len_to_del)

        new_len = len(msg)

        sys.stderr.write(msg)
        sys.stderr.flush()

        return new_len
    else:
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


def to_kwargs(arg_list):
    key_args = dict(raw_entry.split("=") for raw_entry in arg_list if "=" in raw_entry)
    filtered_arg_list = [arg for arg in arg_list if "=" not in arg]

    return key_args, filtered_arg_list


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
