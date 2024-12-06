import sys
from datetime import datetime


def log(msg):
    sys.stderr.write(str(datetime.now()) + ": " + msg + '\n')


def debug(msg):
    pass
    ### log("\n(DEBUG) " + msg)

def log_2dict(twod_dict, msg):

    for k1 in twod_dict:
        for k2 in twod_dict[k1]:
            log(f"DEBUG {msg}: {k1}, {k2} --> {twod_dict[k1][k2]}")


SMUGRI_LOW = "fkv,izh,kca,koi,kpv,krl,liv,lud,mdf,mhr,mns,mrj,myv,olo,sjd,sje,sju,sma,sme,smj,smn,sms,udm,vep,vot,vro"
SMUGRI_HIGH = "deu,eng,est,fin,hun,lvs,nor,rus,swe"
SMUGRI = "deu,eng,est,fin,fkv,hun,izh,kca,koi,kpv,krl,liv,lud,lvs,mdf,mhr,mns,mrj,myv,nor,olo,rus,sjd,sje,sju,sma,sme,smj,smn,sms,swe,udm,vep,vot,vro"


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
