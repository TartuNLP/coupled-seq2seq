#!/usr/bin/env python3

import sys
import json

import promptops
import torch

from accelerate import Accelerator, data_loader
from transformers import AutoModelForCausalLM, DataCollatorForLanguageModeling, AutoTokenizer

from torch.utils.data import Dataset as TorchDataset, DataLoader
from aux import CmdlineArgs, log
from datetime import datetime

from simptrain import env_stuff


def remove_prompt_from_output(att_mask, tokens, filler_id):
    shape = list(att_mask.shape)
    shape[1] = tokens.size(1)
    a_padded = att_mask.new_zeros(*shape)

    # copy original data
    a_padded[:, :att_mask.size(1), ...] = att_mask

    return (1-a_padded) * tokens + filler_id * a_padded


def llm_generate(model, tokenizer, tok_batch, debug=False, max_len=3000, filler_id=128030):
    tok_batch['input_ids'] = tok_batch['input_ids'].to(model.device)
    tok_batch['attention_mask'] = tok_batch['attention_mask'].to(model.device)
    start_time = datetime.now()

    if debug:
        log(f"Tokenized input: {tok_batch['input_ids']}")
        #log(f"Att mask: {tok_batch['attention_mask']}")

    raw_output_toks = model.generate(**tok_batch, tokenizer=tokenizer,
                                 do_sample=False, num_beams=4, max_length=max_len, top_p=None,
                                 temperature=None)

    #clean_output_toks = remove_prompt_from_output(tok_batch['attention_mask'], raw_output_toks, filler_id)
    assert len(raw_output_toks) == 1, "Only batch size=1 supported %-("
    gen_idx = len(tok_batch['attention_mask'][0])

    clean_output_toks = raw_output_toks[0][gen_idx:]

    if debug:
        log(f"Raw tokenized full output: {raw_output_toks}")
        log(f"Raw tokenized output: {clean_output_toks}")
        log(f"Raw tokens: {tokenizer.convert_ids_to_tokens(clean_output_toks)}")

    clean_outputs = tokenizer.batch_decode([clean_output_toks], skip_special_tokens=True)

    if debug:
        end_time = datetime.now()
        log(f"Cleaned output: {clean_outputs}")
        log(f"This took: {end_time - start_time}")

    return clean_outputs


def predict(model, tokenizer, data_loader, accel, debug=False):
    outs_local = []

    with torch.no_grad():
        for idx, batch in enumerate(data_loader):
            if idx % accel.num_processes == accel.process_index:
                start_time = datetime.now()
                outputs = llm_generate(model, tokenizer, batch, debug=debug, max_len=4000)
                end_time = datetime.now()
                log(f"Generated for {idx} in proc {accel.process_index} in {end_time - start_time}")
                outs_local += outputs

    return outs_local


def read_input(path, format):
    with open(path, 'r') as fh:
        if format == promptops.PF_RAW:
            result = fh.read()
        elif format == promptops.PF_RAWLINES:
            result = fh.readlines()
        else:
            result = json.load(fh)

        return result


class LazyTokenizingInferenceDataset(TorchDataset):
    def __init__(self, texts, tokenizer, prompt_format, max_length=512, debug=False):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_format = prompt_format
        self.debug = debug

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        entry = self.texts[idx]

        prompt = promptops.prep_prompt(entry, self.prompt_format, inference=True)
        result = promptops.tokenize_str(self.tokenizer, prompt, add_eos=False)

        if self.debug:
            log(f"Input: {prompt}")
            log(f"Tokenized: {result}")

        return result


def get_data_loader(path, prompt_format, tokenizer, debug=False):
    inputs = read_input(path, prompt_format)

    dataset = LazyTokenizingInferenceDataset(inputs, tokenizer, prompt_format, debug=debug)

    data_coll = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=None,  # helps performance; set None if you prefer exact lengths
    )

    data_loader = DataLoader(dataset, collate_fn=data_coll, batch_size=1)

    return data_loader


def _cmdline_args(inputs):
    description = """Predict output for an input via prompting"""

    pos_args = ["mdl_id", "input_file", "output_file"]

    #post-process the arguments
    args = CmdlineArgs(description, pos_args, input_args=inputs,
                       kw_arg_dict={"debug": False,
                                    "prompt_format": promptops.PF_ALPACA})

    log(f"Launched as {args}")

    return args


def combine_outputs(acc, output_file):
    #if idx % accel.num_processes == accel.process_index:

    indiv_contents = []

    for i in range(acc.num_processes):
        with open(f"{output_file}.{i}", "r") as fh:
            content = fh.readlines()
            indiv_contents.append(content)

    with open(output_file, "w") as fh:
        for gen_idx in range(len(indiv_contents[0])):
            for i in range(len(indiv_contents)):
                if gen_idx < len(indiv_contents[i]):
                    fh.write(indiv_contents[i][gen_idx])


def save_all(outputs, args, acc):
    if args.prompt_format not in {promptops.PF_RAW, promptops.PF_RAWLINES}:
        outputs = [o.replace("\n", "<<BR>>") for o in outputs]

    ind_out_file = f"{args.output_file}.{acc.process_index}"

    with open(ind_out_file, "w", encoding="utf-8") as f_out:
        f_out.write("\n".join(outputs) + "\n")

    log(f"Saved {len(outputs)} rows to {ind_out_file}")

    acc.wait_for_everyone()

    if acc.is_main_process:
        combine_outputs(acc, args.output_file)


def and_i_called_this_function_do_main_too(iv):
    args = _cmdline_args(iv)

    acc = Accelerator()

    model = AutoModelForCausalLM.from_pretrained(args.mdl_id,
                                                 low_cpu_mem_usage=False,
                                                 torch_dtype=torch.bfloat16,
                                                 attn_implementation="eager")
    model.config.use_cache = False
    model = model.to(acc.device)
    model.eval()

    log(f"Device: {model.device}.", accelerator=acc)

    tokenizer = AutoTokenizer.from_pretrained(args.mdl_id)
    tokenizer.pad_token = tokenizer.eos_token

    data_loader = get_data_loader(args.input_file, args.prompt_format, tokenizer, debug=args.debug)

    log("Model loaded, starting to translate")
    outputs = predict(model, tokenizer, data_loader, acc, debug=args.debug)

    save_all(outputs, args, acc)

    log("Done")


if __name__ == "__main__":
    input_values = sys.argv[1:]

    env_stuff()

    and_i_called_this_function_do_main_too(input_values)
