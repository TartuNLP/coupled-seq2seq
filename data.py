#!/usr/bin/env python3

from torch.utils.data import IterableDataset
from random import shuffle, randint


def prep_llm_input(ljmftpl):
    #{'task': 'translate' / 'approx-translate' / 'generate',
    # 'src_segm': src_segm,
    # 'tgt_segm': tgt_segm,
    # 'src_lang': src_lang,
    # 'tgt_lang': tgt_lang}

    # it's a tuple
    if ljmftpl['task'] in {'translate', 'approx-translate'}:
        return (f"{ljmftpl['src_segm']}\n=====\n{ljmftpl['task']} from {ljmftpl['src_lang']}; " +
                f"to {ljmftpl['tgt_lang']}:\n{ljmftpl['tgt_segm']}")

    elif ljmftpl['task'] == 'generate':
        return f"{ljmftpl['src_segm']}\n=====\nis in {ljmftpl['src_lang']};"

    else:
        raise NotImplementedError


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
    def __init__(self, segment_list, batch_size, tokenizer, max_len=8000):
        self.batch_size = batch_size
        self.batched_data = []
        self._prep_batched_data(segment_list)

        self.tokenizer = tokenizer
        self.max_len = max_len

        self.curr_elem_idx = 0

        self.data_len = len(self.batched_data)

    def _prep_batched_data(self, segment_list):
        unsorted_data_in_elems = [prep_llm_input(s) for s in segment_list]
        sorted_data_in_elems = sorted(unsorted_data_in_elems, key=lambda x: len(x), reverse=True)

        self.batched_data = list(do_list_in_batches(sorted_data_in_elems, self.batch_size))

        while len(self.batched_data[-1]) < self.batch_size:
            self.batched_data[-1].append(self.batched_data[-1][-1])

        # shuffle(self.batched_data)

    def __len__(self):
        return self.data_len

    def __iter__(self):
        self.curr_elem_idx = 0
        return self

    def where_are_we(self):
        return DataState(shard_idx=0, elem_idx=self.curr_elem_idx)

    def thats_where(self, data_state):
        self.curr_elem_idx = data_state.elem_idx

    def _tokenize(self, prepped_segm_list):
        #{'task': 'translate',
        # 'src_segm': src_segm,
        # 'tgt_segm': tgt_segm,
        # 'src_lang': src_lang,
        # 'tgt_lang': tgt_lang}

        self.tokenizer.pad_token = '<|reserved_special_token_0|>'
        tokenized_batch = self.tokenizer(prepped_segm_list, return_tensors="pt", max_length=self.max_len,
                                   truncation=True, add_special_tokens=True,
                                   padding=True)
        return tokenized_batch, self.curr_elem_idx + 1

    def __next__(self):
        if self.curr_elem_idx >= self.data_len:
            raise StopIteration
        else:
            batch = self._tokenize(self.batched_data[self.curr_elem_idx])
            self.curr_elem_idx += 1
            return batch

