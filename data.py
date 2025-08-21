#!/usr/bin/env python3
import promptops

import json
import sys

from random import shuffle

from torch.utils.data import Dataset as TorchDataset, DataLoader

from aux import log


def tokenize_str(tokenizer, entry, add_eos=True, max_len=3000, for_inf=False):
    if for_inf:
        tokens = tokenizer(
            entry,
            truncation=True,
            max_length=max_len,
            return_attention_mask=True,
            return_tensors="pt"
        )
    else:
        tokens = tokenizer(
            entry,
            truncation=True,
            max_length=max_len,
            return_attention_mask=True
        )

    if add_eos:
        tokens['attention_mask'].append(1)
        tokens['input_ids'].append(tokenizer.eos_token_id)

    return tokens

"""
Load texts into memory and allow to loop through it,
returning tokenized tensors.

Currently no support for text data that does not fit into memory,
need to add it. Or do HF datasets have something out of the box? 
"""
class LazyTokenizingDataset(TorchDataset):
    def __init__(self, texts, tokenizer, max_length=512, prompt_format="raw"):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_format = prompt_format

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # Return plain Python lists; let the collator pad & build labels.
        entry = self.texts[idx]

        prompt = promptops.prep_prompt(entry, self.prompt_format)

        return tokenize_str(self.tokenizer, prompt)


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
        result = tokenize_str(self.tokenizer, prompt, add_eos=False, for_inf=True)

        if self.debug:
            log(f"Input: {prompt}")
            log(f"Tokenized: {result}")

        return result


def read_input(path, formt):
    if path is None:
        log("Reading from STDIN")
        fh = sys.stdin
    else:
        log(f"Reading from {path}")
        fh = open(path, 'r')

    if formt == promptops.PF_RAW:
        result = [fh.read()]
    elif formt == promptops.PF_RAWLINES:
        result = fh.readlines()
    else:
        result = json.load(fh)

    return result


def get_data_loader(path, prompt_format, tokenizer, debug=False):
    inputs = read_input(path, prompt_format)

    dataset = LazyTokenizingInferenceDataset(inputs, tokenizer, prompt_format, debug=debug)

    """
    data_coll = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=None,  # helps performance; set None if you prefer exact lengths
    )

    data_loader = DataLoader(dataset, collate_fn=data_coll, batch_size=1)
    """

    return dataset



def load_training_data(path, tokenizer, cmd_args):
    with open(path, "r") as f:
        data = json.load(f)

    train_set_iter = LazyTokenizingDataset(data, tokenizer, cmd_args.max_length, cmd_args.prompt_format)

    return train_set_iter


if __name__ == '__main__':
    all_data = []

    for input_file in sys.argv[1:]:
        with open(input_file, "r") as f:
            this_data = json.load(f)
            all_data += this_data

    shuffle(all_data)

    json.dump(all_data, sys.stdout)
