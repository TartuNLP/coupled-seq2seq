#!/usr/bin/env python3

import sys
import re

from accelerate import Accelerator
from transformers import AutoModelForCausalLM, DataCollatorForLanguageModeling, AutoTokenizer

from simptrain import tokenize_for_inference
from torch.utils.data import Dataset as TorchDataset, DataLoader
import torch
from aux import CmdlineArgs, log


def remove_prompt_from_output(att_mask, tokens, filler_id):
    shape = list(att_mask.shape)
    shape[1] = tokens.size(1)
    a_padded = att_mask.new_zeros(*shape)

    # copy original data
    a_padded[:, :att_mask.size(1), ...] = att_mask

    return (1-a_padded) * tokens + filler_id * a_padded

def get_lang_pred(raw_txt):
    m = re.search(r'<\|reserved_special_token_13\|>(.*?)<\|reserved_special_token_14\|>', raw_txt)

    pre_result = m.group(1) if m else None

    if pre_result is None:
        return "-"
    else:
        return re.sub(r'<\|[^|]+\|>', '', pre_result).strip()

def llm_generate(model, tokenizer, tok_batch, mode, debug=False, max_len=1000):
    if debug:
        log(f"Tokenized input: {tok_batch['input_ids']}")
        log(f"Att mask: {tok_batch['attention_mask']}")

    tok_batch['input_ids'] = tok_batch['input_ids'].to(model.device)
    tok_batch['attention_mask'] = tok_batch['attention_mask'].to(model.device)

    if mode == 'lid':
        stop_strings = ["<|reserved_special_token_14|>", "<|reserved_special_token_16|>", "<|end_of_text|>"]
    else:
        stop_strings = ["<|end_of_text|>"]

    raw_outputs = model.generate(**tok_batch, tokenizer=tokenizer,
                                 do_sample=False, num_beams=5, max_length=max_len, top_p=None,
                                 temperature=None, stop_strings=stop_strings)

    if debug:
        log(f"Raw tokenized output: {raw_outputs}")

    clean_outputs = remove_prompt_from_output(tok_batch['attention_mask'], raw_outputs, 128030)

    # 3. output token IDs --> output text

    if mode == 'lid':
        pre_result = tokenizer.batch_decode(raw_outputs, skip_special_tokens=False)
        if debug:
            log(f"Raw pre tokenized output: {pre_result}")
        result = [get_lang_pred(e) for e in pre_result]
        if debug:
            log(f"Result: {result}")
    else:
        if debug:
            debresult = tokenizer.batch_decode(clean_outputs, skip_special_tokens=False)
            log(f"Raw result: {debresult}")

        result = tokenizer.batch_decode(clean_outputs, skip_special_tokens=True)

    return result


def generative_translate(model, tokenizer, input_texts, input_language, output_language, mode, debug=False):
    all_outputs = list()

    if mode in {'lid', 'raw'}:
        input_language = None
        output_language = None

    dataset = LazyTokenizingInferenceDataset(input_texts, tokenizer, mode,
                                             src_lang=input_language, tgt_lang=output_language, debug=debug)

    data_coll = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,  # helps performance; set None if you prefer exact lengths
    )

    data_loader = DataLoader(dataset, collate_fn=data_coll, batch_size=8)

    for inp_batch in data_loader:
        these_outputs = llm_generate(model, tokenizer, inp_batch, mode, debug=debug, max_len=200)

        all_outputs += these_outputs

    return all_outputs


def _cmdline_args(inputs):
    description = """Translate STDIN text with a translation model"""

    pos_args = ["mdl_id", "from_lang", "to_lang"]

    #post-process the arguments
    args = CmdlineArgs(description, pos_args, input_args=inputs, kw_arg_dict={"debug": False, "mode": "translate"})

    log(f"Launched as {args}")

    return args


class LazyTokenizingInferenceDataset(TorchDataset):
    def __init__(self, texts, tokenizer, mode, src_lang=None, tgt_lang=None, max_length=512, debug=False):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode # translate / lid / raw
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.debug = debug

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        entry = self.texts[idx]

        result = tokenize_for_inference(self.tokenizer, entry, self.src_lang, self.tgt_lang, self.mode, debug=self.debug)

        return result


def and_i_called_this_function_do_main_too(iv):
    args = _cmdline_args(iv)

    raw_input = sys.stdin.read().rstrip()

    if args.mode == 'raw':
        inputs = [raw_input]
    else:
        inputs = raw_input.split("\n")

    acc = Accelerator()

    model = AutoModelForCausalLM.from_pretrained(args.mdl_id,
                                                 low_cpu_mem_usage=True,
                                                 torch_dtype=torch.bfloat16,
                                                 device_map=acc.device,
                                                 attn_implementation="eager")

    log(f"Device: {model.device}.", accelerator=acc)

    tokenizer = AutoTokenizer.from_pretrained(args.mdl_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<|reserved_special_token_100|>"

    log("Model loaded, starting to translate")
    outputs = generative_translate(model, tokenizer, inputs, args.from_lang, args.to_lang, args.mode, debug=args.debug)

    print("\n".join(outputs))

    log("Done...")


if __name__ == "__main__":
    input_values = sys.argv[1:]

    and_i_called_this_function_do_main_too(input_values)
