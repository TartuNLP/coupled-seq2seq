#!/usr/bin/env python3

import json
import sys

from torch.utils.data import IterableDataset
from random import shuffle, randint
from aux import log
from tokops import tokenize_batch


def prep_llm_input(ljmftpl):
    #{'task': 'translate' / 'approx-translate' / 'generate',
    # 'src_segm': src_segm,
    # 'tgt_segm': tgt_segm,
    # 'src_lang': src_lang,
    # 'tgt_lang': tgt_lang}

    result = f"{ljmftpl['src_segm']}\n=====\nis in {ljmftpl['src_lang']}"

    if ljmftpl['task'] in {'translate', 'approx-translate'}:
        result += f"; {ljmftpl['task']} to {ljmftpl['tgt_lang']}:\n{ljmftpl['tgt_segm']}"

    return result

def make_path_compatible(filename):
    return filename.replace("/", "_").replace(":", "-")

def do_list_in_batches(data, batch_size):
    i = 0

    while i < len(data):
        yield data[i:i + batch_size]
        i += batch_size


class DataState:
    def __init__(self, elem_idx = 0, shard_idx = 0, epoch_idx = None):
        self.elem_idx = elem_idx
        self.shard_idx = shard_idx
        self.epoch_idx = epoch_idx

    def state_dict(self):
        return {'elem_idx': self.elem_idx, 'shard_idx': self.shard_idx, 'epoch_idx': self.epoch_idx}

    def load_state_dict(self, state_dict):
        self.elem_idx = state_dict['elem_idx']
        self.shard_idx = state_dict['shard_idx']
        self.epoch_idx = state_dict['epoch_idx']

    def copy_from(self, src_ds, epoch_idx = None):
        self.shard_idx = src_ds.shard_idx
        self.elem_idx = src_ds.elem_idx

        if epoch_idx is not None:
            self.epoch_idx = epoch_idx

    def __str__(self):
        return 'DataState(elem_idx={}, shard_idx={}, epoch_idx={})'.format(self.elem_idx, self.shard_idx, self.epoch_idx)

    def __repr__(self):
        return self.__str__()


class BatchingIterator(IterableDataset):
    def __init__(self, batched_data, batch_size, tokenizer, max_len=8000):
        assert len(batched_data[0]) == batch_size, "loaded data batch size and specified batch size differ"

        self.batched_data = batched_data

        self.tokenizer = tokenizer
        self.max_len = max_len

        self.curr_elem_idx = 0

        self.data_len = len(self.batched_data)

    def __len__(self):
        return self.data_len

    def __iter__(self):
        #self.curr_elem_idx = 0
        return self

    def where_are_we(self):
        return DataState(shard_idx=0, elem_idx=self.curr_elem_idx)

    def thats_where(self, data_state):
        self.curr_elem_idx = data_state.elem_idx

    def _tokenize(self, prepped_segm_list):
        #self.tokenizer.pad_token = '<|reserved_special_token_0|>'
        #tokenized_batch = self.tokenizer(prepped_segm_list, return_tensors="pt", max_length=self.max_len,
        #                           truncation=True, add_special_tokens=True,
        #                           padding=True)
        tokenized_batch = tokenize_batch(self.tokenizer, prepped_segm_list)
        return tokenized_batch, self.curr_elem_idx + 1

    def __next__(self):
        if self.curr_elem_idx >= self.data_len:
            self.curr_elem_idx = 0
            raise StopIteration
        else:
            batch = self._tokenize(self.batched_data[self.curr_elem_idx])
            self.curr_elem_idx += 1
            return batch

if __name__ == '__main__':
    # open a list of tuples, save a list of batches of strings made of these tuples
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    try:
        batch_size = int(sys.argv[3])
    except IndexError:
        batch_size = None

    log("Reading data")
    # read the tuples
    with open(input_file, "r") as f:
        raw_data = json.load(f)

    log("Making strings")
    # make strings out of tuples
    unsorted_data_in_elems = [prep_llm_input(s) for s in raw_data]

    if batch_size is None:
        final_data = unsorted_data_in_elems
    else:
        # if last batch is undersized, get some random elements to compensate
        while len(unsorted_data_in_elems) % batch_size != 0:
            new_elem_idx = randint(0, len(unsorted_data_in_elems) - 1)
            unsorted_data_in_elems.append(unsorted_data_in_elems[new_elem_idx])

        log("Sorting and grouping")
        # sort by length
        sorted_data_in_elems = sorted(unsorted_data_in_elems, key=lambda x: len(x), reverse=True)

        # group into batches
        final_data = list(do_list_in_batches(sorted_data_in_elems, batch_size))

    log("Shuffling")
    # shuffle the batches / sentences
    shuffle(final_data)

    log("Saving")
    # save the result
    with open(output_file, "w") as f:
        json.dump(final_data, f, indent=2)
