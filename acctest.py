#!/usr/bin/env python3

import sys
import os
import torch

from datetime import datetime
from accelerate import Accelerator, DataLoaderConfiguration
from trainmodel import report_devices


def log(msg):
    now = datetime.now()
    sys.stderr.write(f"{now}: {msg}\n")


def main():
    log(f"Cuda available: {torch.cuda.is_available()}")

    report_devices()

    acc = Accelerator(mixed_precision="bf16", gradient_accumulation_steps=1, cpu=False,
                      dataloader_config=DataLoaderConfiguration(split_batches=True, dispatch_batches=True))

    log(acc.state)


if __name__ == '__main__':
    main()