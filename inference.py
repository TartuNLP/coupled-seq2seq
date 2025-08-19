#!/usr/bin/env python3

import promptops

from aux import CmdlineArgs, log
from data import get_data_loader
from trainllm import env_stuff, load_model, load_tokenizer


import sys
import torch
import json
import torch.distributed as dist

from accelerate import Accelerator

from datetime import datetime

"""
This currently assumes the batch size to be 1. With larger batches the padding tokens went
into the decoder. Right-padding as a solution?
"""
def llm_generate(model, tokenizer, tok_batch, debug=False, max_len=2000):
    tok_batch['input_ids'] = tok_batch['input_ids'].to(model.device)
    tok_batch['attention_mask'] = tok_batch['attention_mask'].to(model.device)
    start_time = datetime.now()

    if debug:
        log(f"Tokenized input: {tok_batch['input_ids']}")

    raw_output_toks = model.generate(**tok_batch, tokenizer=tokenizer,
                                 do_sample=False, num_beams=4, max_length=max_len, top_p=None, temperature=None,
                                 eos_token_id=[tokenizer.eos_token_id,
                                               tokenizer.convert_tokens_to_ids("<|reserved_special_token_14|>")])

    #clean_output_toks = remove_prompt_from_output(tok_batch['attention_mask'], raw_output_toks, filler_id)
    assert len(raw_output_toks) == 1, "Only batch size=1 supported %-("
    gen_idx = len(tok_batch['attention_mask'][0])

    if debug:
        log(f"Full tokenized output: {raw_output_toks[0]}")
        log(f"Full tokens: {tokenizer.convert_ids_to_tokens(raw_output_toks[0])}")
        full_out = tokenizer.batch_decode([raw_output_toks[0]], skip_special_tokens=True)
        log(f"Full text: {full_out[0]}")

    clean_output_toks = raw_output_toks[0][gen_idx:]
    clean_outputs = tokenizer.batch_decode([clean_output_toks], skip_special_tokens=True)

    if debug:
        log(f"Pruned tokenized output: {clean_output_toks}")
        log(f"Pruned tokens: {tokenizer.convert_ids_to_tokens(clean_output_toks)}")
        log(f"Cleaned output: {clean_outputs[0]}")

        end_time = datetime.now()
        log(f"This took: {end_time - start_time}")

    return clean_outputs


def reassemble_multi(list_of_lists):
    result = []

    for gen_idx in range(len(list_of_lists[0])):
        for i in range(len(list_of_lists)):
            if gen_idx < len(list_of_lists[i]):
                result.append(list_of_lists[i][gen_idx])

    return result


def predict(model, tokenizer, data_loader, accel, multi=False, debug=False, max_len=2000):
    outs_final = []

    with torch.no_grad():
        for idx, batch in enumerate(data_loader):
            if idx % accel.num_processes == accel.process_index:
                start_time = datetime.now()
                outputs = llm_generate(model, tokenizer, batch, debug=debug, max_len=max_len)
                end_time = datetime.now()
                log(f"Generated for {idx} in proc {accel.process_index} in {end_time - start_time}")
                outs_final += outputs

    if multi:
        accel.wait_for_everyone()

        rank0_buffer = [None] * accel.num_processes if accel.is_main_process else None
        dist.gather_object(outs_final, rank0_buffer, dst=0)
        if accel.is_main_process:
            outs_final = reassemble_multi(rank0_buffer)
        else:
            outs_final = None

    return outs_final


def _cmdline_args():
    inputs = sys.argv[1:]

    description = """Predict output for an input via prompting"""

    pos_args = ["mdl_id"]

    #post-process the arguments
    args = CmdlineArgs(description, pos_args, input_args=inputs,
                       kw_arg_dict={"debug": False,
                                    "input_file": "none",
                                    "output_file": "none",
                                    "multiproc": False,
                                    "max_len": 2000,
                                    "prompt_format": promptops.PF_ALPACA})

    if args.input_file == "none":
        args.input_file = None
    if args.output_file == "none":
        args.output_file = None

    log(f"Launched as {args}")

    return args


def save_all(outputs, args, acc):
    if acc.is_main_process:
        if args.output_file is None:
            log("Writing to STDOUT")
            out_fh = sys.stdout
        else:
            out_fh = open(args.output_file, "w")

        if args.prompt_format in {promptops.PF_RAW, promptops.PF_RAWLINES}:
            for line in outputs:
                out_fh.write(line + "\n")
        else:
            json.dump(outputs, out_fh)


def and_i_called_this_function_do_main_too():
    args = _cmdline_args()

    if args.multiproc:
        env_stuff()

    acc = Accelerator()
    device = acc.device

    log(f"Device: {device}.", accelerator=acc)

    if not args.multiproc and not acc.is_main_process:
        log("Not launched in multi-processing mode, exiting non-main process.")
        sys.exit(0)

    tokenizer = load_tokenizer(args.mdl_id, acc)

    data_loader = get_data_loader(args.input_file, args.prompt_format, tokenizer, debug=args.debug)

    model = load_model(args.mdl_id, device, acc, attention="eager")
    model.eval()

    log(f"Device: {model.device}.", accelerator=acc)

    log("Model loaded, starting to generate")
    outputs = predict(model, tokenizer, data_loader, acc, multi=args.multiproc, debug=args.debug, max_len=args.max_len)

    save_all(outputs, args, acc)

    log("Done")


if __name__ == "__main__":
    and_i_called_this_function_do_main_too()
