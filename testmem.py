#!/usr/bin/env python3

import torch.optim
import sys
import subprocess
import random

from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler, AutoModelForSeq2SeqLM
from datasets import load_dataset

from aux import CmdlineArgs, log
from langconv import is_dec_only_llm
from modelops import report_devices, hf_tok
from tokops import load_tokenizer, tokenizeit


def run_test(mdl_id, batch_sizes, ctxlen, acc):
    #state = AcceleratorState()
    log(f"Num proc: {acc.num_processes}, proc ID: {acc.process_index}")

    report_devices("Initial state:", accelerator=acc)

    t, pt = load_tokenizer(mdl_id) # AutoTokenizer.from_mpretrained(mdl_id, token=hf_tok)
    if is_dec_only_llm(t):
        m = AutoModelForCausalLM.from_pretrained(mdl_id, token=hf_tok, torch_dtype=torch.bfloat16)
        log("Decoder-only model")
    else:
        m = AutoModelForSeq2SeqLM.from_pretrained(mdl_id, token=hf_tok, torch_dtype=torch.bfloat16)
        log("Encoder-decoder model")

    opt = torch.optim.AdamW(m.parameters(), lr=1e-5)
    lrs = get_scheduler("linear", optimizer=opt, num_warmup_steps=100, num_training_steps=1000)
    opt, lrs, m = acc.prepare(opt, lrs, m)

    report_devices("Models in VRAM:", accelerator=acc)
    m.train()

    ds = load_dataset("Helsinki-NLP/europarl", "en-et")
    max_idx = len(ds['train'])

    for batch_size in batch_sizes:
        print("")

        for _ in range(10):
            inp_idx = random.randint(0, max_idx-batch_size)

            raw_inp = [ds['train'][i]['translation']['et'] for i in range(inp_idx, inp_idx+batch_size)]

            if is_dec_only_llm(t):
                inp = tokenizeit((t, pt), raw_inp, ctxlen, is_target=False, is_llm=True)
            else:
                inp = tokenizeit((t, pt), raw_inp, ctxlen, is_target=False, is_llm=False)

            inp['labels'] = inp['input_ids']
            inp.to(m.device)

            outputs = m(**inp)

            loss = outputs.loss
            report_devices(f"While training:", accelerator=acc)
            log(f"Batches    : {[inp[k].size() for k in 'input_ids labels attention_mask'.split(' ')]}")
            log(f"Batch total: {sum([inp[k].size()[0] * inp[k].size()[1] for k in 'input_ids labels attention_mask'.split(' ')])}")

            try:
                if acc.is_main_process:
                    result = subprocess.run(['rocm-smi'], capture_output=True, text=True)
                    print(result.stdout)
            except:
                pass

            acc.backward(loss)
            acc.wait_for_everyone()

        report_devices(f"Models gradients in VRAM, batch size {batch_size}:", accelerator=acc)

    print(f"Testing {mdl_id} with batch size {batch_size}: success!")



if __name__ == "__main__":
    if len(sys.argv) > 1:
        args = CmdlineArgs("Test the VRAM usage by a model with different batch sizes, comma-separated",
                           pos_arg_list=["mdl_id", "batch_sizes"],
                           kw_arg_dict={"ctxlen": 2048})

        clean_bs = [int(bs) for bs in args.batch_sizes.split(",")]
        mdl_id = args.mdl_id
        ctxlen = args.ctxlen
    else:
        mdl_id = "meta-llama/Llama-3.2-1B"
        clean_bs = [16, 32, 64]
        ctxlen = 2048

    acc = Accelerator()
    try:
        run_test(mdl_id, clean_bs, ctxlen, acc)
    except Exception as e:
        if acc.is_main_process:
            raise e
