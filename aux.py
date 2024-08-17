import sys
from datetime import datetime


def log(msg):
    sys.stderr.write(str(datetime.now()) + ": " + msg + '\n')


def debug(msg):
    pass
    # log("\n(DEBUG) " + msg)

def log_2dict(twod_dict, msg):
    for k1 in twod_dict:
        for k2 in twod_dict[k1]:
            log(f"DEBUG {msg}: {k1}, {k2} --> {twod_dict[k1][k2]}")
