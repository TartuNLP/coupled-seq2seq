#!/usr/bin/env python3

import sys
import torch

from datetime import datetime
from accelerate import Accelerator, DataLoaderConfiguration
from transformers import AutoModelForSeq2SeqLM


def log(msg):
    now = datetime.now()
    sys.stderr.write(f"{now}: {msg}\n")


def gpu_mem_report(accelerator = None):
    if torch.cuda.is_available():
        log("Allocated memory per GPU: " + " / ".join([f"{i}: {torch.cuda.memory_allocated(i)/(1024**2):.2f} Mb" for i in range(torch.cuda.device_count())]))
    else:
        if accelerator is not None:
            device_type = accelerator.device.type
            log(f"No CUDA, accelerator device: {device_type}")
            if device_type == "mps":
                log(f"MPS alloc memory: {torch.mps.current_allocated_memory()/(1024**2)} Mb")
        else:
            log("No CUDA, no accelerator")


def main():
    log(f"Cuda available: {torch.cuda.is_available()}")

    log(f"Nr. of Cuda devices: {torch.cuda.device_count()}")

    acc = Accelerator()

    # acc = Accelerator(mixed_precision="bf16", gradient_accumulation_steps=1, cpu=False, dataloader_config=DataLoaderConfiguration(split_batches=True, dispatch_batches=True))

    log(f"Accelerator state: {str(acc.state)}")

    gpu_mem_report(acc)

    log("Loading NLLB")

    nllb = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600m")

    log("Passing NLLB through the accelerator")

    nllb_acc = acc.prepare(nllb)

    gpu_mem_report(acc)


if __name__ == '__main__':
    main()