#!/usr/bin/env python3

import torch.optim
import sys
import subprocess

from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler, AutoModelForSeq2SeqLM

from aux import CmdlineArgs, log
from langconv import is_dec_only_llm
from modelops import report_devices, hf_tok
from tokops import load_tokenizer


def run_test(mdl_id, batch_sizes, ctxlen, acc):
    #state = AcceleratorState()
    log(f"Num proc: {acc.num_processes}, proc ID: {acc.process_index}")

    report_devices("Initial state:", accelerator=acc)

    t = load_tokenizer(mdl_id) # AutoTokenizer.from_mpretrained(mdl_id, token=hf_tok)
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

    txt = "Kui ma laulan, kui me leelon, laulan lained laksuma, siis jääb küla kuulamaie"

    for batch_size in batch_sizes:
        print("")
        raw_inp = [txt] * batch_size
        if is_dec_only_llm(t):
            t.pad_token = '<|reserved_special_token_0|>'
            inp = t(raw_inp, return_tensors="pt", max_length=ctxlen, truncation=True, add_special_tokens=True, padding=True, padding_side='left')
        else:
            inp = t(raw_inp, return_tensors="pt", max_length=ctxlen, truncation=True, add_special_tokens=True,
                    padding=True)

        inp['labels'] = inp['input_ids']
        inp.to(m.device)

        for _ in range(3):
            outputs = m(**inp)
            loss = outputs.loss
            acc.backward(loss)
            acc.wait_for_everyone()

        report_devices(f"Models gradients in VRAM, batch size {batch_size}:", accelerator=acc)

    try:
        if acc.is_main_process:
            result = subprocess.run(['rocm-smi'], capture_output=True, text=True)
            print(result.stdout)
    except:
        pass


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
