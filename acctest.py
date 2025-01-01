#!/usr/bin/env python3

import sys
import torch

from datetime import datetime
from accelerate import Accelerator, DataLoaderConfiguration


def log(msg):
    now = datetime.now()
    sys.stderr.write(f"{now}: {msg}\n")


def main():
    log(f"Cuda available: {torch.cuda.is_available()}")

    log(f"Nr. of Cuda devices: {torch.cuda.device_count()}")

    acc = Accelerator()
    #acc = Accelerator(mixed_precision="bf16", gradient_accumulation_steps=1, cpu=False, dataloader_config=DataLoaderConfiguration(split_batches=True, dispatch_batches=True))

    log(f"Accelerator state: {str(acc.state)}")


if __name__ == '__main__':
    main()