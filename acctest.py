#!/usr/bin/env python3

import sys
import os
import torch

from datetime import datetime
from accelerate import Accelerator
from trainmodel import report_devices


def log(msg):
    now = datetime.now()
    sys.stderr.write(f"{now}: {msg}\n")


def main():
    log(f"Cuda available: {torch.cuda.is_available()}")

    report_devices()

    acc = Accelerator()

    log(acc.state)


if __name__ == '__main__':
    main()