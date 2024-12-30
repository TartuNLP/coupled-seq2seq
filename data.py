#!/usr/bin/env python3

import json
import os
import sys
import torch

from torch.utils.data import IterableDataset
from collections import namedtuple, defaultdict
from random import randrange, shuffle

from aux import log, smugri_back, maybe_smugri
from langconv import any_to_madlad, any_to_nllb, is_nllb, is_madlad

TrPair = namedtuple('TrPair', ["src_lang", "tgt_lang", "input", "output"])
#DataEntry = namedtuple('DataEntry', ["tr_pair", "prepared", "src_bin_idx", "tgt_bin_idx"])


def make_path_compatible(filename):
    return filename.replace("/", "_").replace(":", "-")

def do_list_in_batches(data, batch_size):
    i = 0

    while i < len(data):
        yield data[i:i + batch_size]
        i += batch_size


def do_bins_in_shuffled_batches(bins, batch_size):
    result_list = []

    for src_k in bins:
        for tgt_k in bins[src_k]:
            if src_k == 0 or tgt_k == 0:
                result_list += [(e, src_k, tgt_k) for e in do_list_in_batches(bins[src_k][tgt_k], batch_size)]

    shuffle(result_list)

    return result_list


def _post_proc(text, lang):
    if lang == 'liv' and "’" in text and "O’R" not in text:
        return text.replace("’", "")
    else:
        return text


def clean_entry(entry, leave_out):
    result = {k: _post_proc(entry[k], k) for k in entry if entry[k].strip() and k not in leave_out}
    return result


def load_json_datax(path, leave_out={"fr"}, skip_cats=True, load_mono=True):
    with open(path, 'r') as f:
        data = json.load(f)

        if skip_cats:
            # skip categories
            resx = [clean_entry(entry, leave_out)
                    for cat in data for entry in cat['sentences']]
            res = [e for e in resx if e]
        else:
            raise NotImplementedError

            # resx = {cat['source']: [clean_entry(entry, leave_out) for entry in cat['sentences']] for cat in data}
            # res = {k: resx[k] for k in resx if resx[k]}

        return res


def get_tr_pairs(raw_data=None, filename=None, leave_out=None, leave_only=None):
    if filename is not None:
        raw_data = load_json_datax(filename)

    if raw_data is None:
        raise ValueError("Neither file nor data are provided")

    i = 0
    log("Loading data")
    for tup in raw_data:
        for l1 in tup:
            for l2 in tup:
                if l1 != l2 and not "dia" in l1 and not "dia" in l2:
                    if leave_out is None or f"{l1}-{l2}" not in leave_out:
                        if leave_only is None or f"{l1}-{l2}" in leave_only:
                            i += 1
                            if not i % 1000000:
                                log(f"Loaded {i/1000000}M pairs")
                            dia_key = f"{l2}-dia"

                            input = tup[l1]
                            if dia_key in tup:
                                input = f"<{tup[dia_key]}> {input}"

                            yield TrPair(l1, l2, input, tup[l2])


def split_by_lang(tr_pairs=None, filename=None):
    result = defaultdict(list)

    if filename is not None:
        tr_pairs = load_json_datax(filename)

    for tup in tr_pairs:
        for l1 in tup:
            for l2 in tup:
                if l1 != l2 and not "dia" in l1 and not "dia" in l2:
                    lp = f"{l1}-{l2}"
                    result[lp].append((tup[l1], tup[l2]))

    return result


def data_iter_for_tok_train(raw_data, langs_to_include):
    for tup in raw_data:
        for lang in tup:
            if lang in langs_to_include:
                yield tup[lang]


def lang_bin_mapping(coupling_specs):
    lang_to_idx = defaultdict(set)

    for i, spec_pair in enumerate(coupling_specs):
        for lang in spec_pair.lang_set:
            lang_to_idx[lang].add(i)

    return lang_to_idx


def mix_and_sample_idxs_carefully(src_idxs, tgt_idxs):
    idx_pairs = [(s, t) for s in src_idxs for t in tgt_idxs if not (s == 1 and t == 1)]

    if len(idx_pairs) == 0:
        result = (None, None)
    else:
        pair_idx = randrange(len(idx_pairs))
        result = idx_pairs[pair_idx]

    # debug(f"src lang: {tr_pair.src_lang}, tgt_lang: {tr_pair.tgt_lang}, idx list: {idx_pairs}, result: {result}")

    return result


def inject_bin_indices(batch, src_k, tgt_k):
    batch['input_ids'][0,0] += src_k << 30

    batch['labels'][0,0] += tgt_k << 30


class MultilingualBatchingDataset(IterableDataset):
    def _post_proc_bins(self, bins):
        for src_k in bins:
            for tgt_k in bins[src_k]:
                while len(bins[src_k][tgt_k]) % self.batch_size != 0:
                    rnd_elem_idx = randrange(len(bins[src_k][tgt_k]))
                    rnd_elem = bins[src_k][tgt_k][rnd_elem_idx]
                    bins[src_k][tgt_k].append(rnd_elem)

                shuffle(bins[src_k][tgt_k])
        return bins

    def _get_idxs(self, tr_pair):
        src_idxs = self._lang_to_idx[tr_pair.src_lang]
        tgt_idxs = self._lang_to_idx[tr_pair.tgt_lang]

        return mix_and_sample_idxs_carefully(src_idxs, tgt_idxs)

    def _fill_bins(self, filename):
        bins = defaultdict(lambda: defaultdict(list))

        for tr_pair in get_tr_pairs(filename=filename):
            src_bin_idx, tgt_bin_idx = self._get_idxs(tr_pair)

            if src_bin_idx is not None and tgt_bin_idx is not None:
                bins[src_bin_idx][tgt_bin_idx].append(tr_pair)

        return self._post_proc_bins(bins)

    def report_update_stats(self, bins):
        total = 0
        totalx = 0
        updates = 0
        duds = 0

        enc_count = 0
        dec_count = 0

        for src_k in bins:
            for tgt_k in bins[src_k]:
                l = len(bins[src_k][tgt_k])

                total += l
                if src_k == 0 or tgt_k == 0:
                    totalx += l
                updates += l * (1 - (src_k + tgt_k) / 2)

                enc_count += l * (1 - src_k)
                dec_count += l * (1 - tgt_k)

                if src_k == 1 and tgt_k == 1:
                    duds += 1

        log(f"### Ratio of coupled model updates: {100 * updates / total:.2f}% ({100 * updates / totalx:.2f}%); " + \
            f"frozen meaningless updates: {100 * duds / total:.2f}%; " + \
            f"enc samples: {enc_count}, dec samples: {dec_count}")

    def tokenize_input(self, cplspec, input_list, rawbatch):
        src_tokenizer = cplspec.tokenizer
        src_tokenizer.src_lang = rawbatch[0].src_lang
        prep_batch_grouped = src_tokenizer(text=input_list, return_tensors="pt",
                                           padding="longest", truncation=True, max_length=512)
        if is_nllb(src_tokenizer):
            src_lang_list = [any_to_nllb(e.src_lang) for e in rawbatch]
            src_lang_vec = src_tokenizer.convert_tokens_to_ids(src_lang_list)
            prep_batch_grouped['input_ids'][:,0] = torch.tensor(src_lang_vec)

        return prep_batch_grouped

    def tokenize_output(self, tgttokenizer, rawbatch):
        outputs = [e.output for e in rawbatch]
        tgttokenizer.tgt_lang = rawbatch[0].tgt_lang
        labels = tgttokenizer(text_target=outputs, return_tensors="pt", padding="longest", truncation=True,
                               max_length=256)
        if is_nllb(tgttokenizer):
            tgt_lang_list = [any_to_nllb(e.tgt_lang) for e in rawbatch]
            tgt_lang_vec = tgttokenizer.convert_tokens_to_ids(tgt_lang_list)
            labels['input_ids'][:, 0] = torch.tensor(tgt_lang_vec)

        return labels

    def tokenize_and_pad(self, raw_batch, src_k, tgt_k):
        tgt_tokenizer = self.coupling_specs[tgt_k].tokenizer

        if is_madlad(tgt_tokenizer):
            inputs = [f"{any_to_madlad(e.tgt_lang)} {e.input}" for e in raw_batch]
        else:
            inputs = [e.input for e in raw_batch]

        prep_batch_grouped = self.tokenize_input(self.coupling_specs[src_k], inputs, raw_batch)
        labels = self.tokenize_output(tgt_tokenizer, raw_batch)
        prep_batch_grouped['labels'] = labels['input_ids']

        # inject_bin_indices(prep_batch_grouped, src_k, tgt_k)

        #split_prep_batch = [{k: prep_batch_grouped[k][i] for k in prep_batch_grouped}
        #                    for i, trp in enumerate(raw_batch)]

        return prep_batch_grouped

    def _bins_to_tokenized_batched_data(self, bins):
        i = 0
        log("Tokenizing data")

        self.data = []

        for raw_batch, src_k, tgt_k in do_bins_in_shuffled_batches(bins, self.batch_size):
            i += 1
            if not i % 10000:
                log(f"Tokenized {i} batches")

            prepared_batch = self.tokenize_and_pad(raw_batch, src_k, tgt_k)
            self.data.append((prepared_batch, src_k, tgt_k))

    def _prepare_new_data(self, filename):
        bins = self._fill_bins(filename)

        self.report_update_stats(bins)

        self._bins_to_tokenized_batched_data(bins)

    def _get_data_cache_location(self, filename):
        dirname = filename + "-tokcache"
        if not os.path.isdir(dirname):
            os.mkdir(dirname)

        name = dirname + "/batch-" + str(self.batch_size)

        for spec_tuple in self.coupling_specs:
            lang_set_str = smugri_back(spec_tuple.lang_set)

            name += f"-{lang_set_str}-{make_path_compatible(spec_tuple.model_id)}"

        return name + ".pt"

    def _load_data_from_cache(self, filename):
        cache_location = self._get_data_cache_location(filename)

        there_is_a_cache = os.path.exists(cache_location)

        if there_is_a_cache:
            log(f"Loading data from cache ({cache_location})")
            self.data = torch.load(cache_location)
        else:
            log(f"Cache not found ({cache_location}), need to tokenize anew")

        return there_is_a_cache

    def _save_cache(self, filename):
        cache_location = self._get_data_cache_location(filename)

        if os.path.exists(cache_location):
            raise Exception("Cache already exists")

        torch.save(self.data, cache_location)
        log("Saved data into cache")

    def load_group_and_tokenize_data(self, filename):
        did_it_work = self._load_data_from_cache(filename)

        if not did_it_work:
            self._prepare_new_data(filename)
            self._save_cache(filename)

    def __init__(self, tr_file, coupling_specs, batch_size, tracing_msg="just a set", max_src_len=256,
                 max_tgt_len=256, verbose=False, leave_only=None):
        self.msg = tracing_msg
        self.batch_size = batch_size

        self.coupling_specs = coupling_specs

        # init lang to idx
        self._lang_to_idx = lang_bin_mapping(coupling_specs)

        # collect data into bins and fill self.data:
        self.load_group_and_tokenize_data(tr_file)

    def __iter__(self):
        self.i = 0
        self.prev = None
        return self

    def __next__(self):
        if self.i < len(self.data):
            res = self.data[self.i]
            self.i += 1

            return res
        else:
            raise StopIteration

    def __len__(self):
        return len(self.data)


def dump_to_stdout(filename=None, lang_or_lp=None):
    if not filename:
        filename = sys.argv[1]
    if not lang_or_lp:
        lang_or_lp = sys.argv[2]

    if "-" in lang_or_lp:
        lp = lang_or_lp
        i = 0
        for tr_pair in get_tr_pairs(filename=filename, leave_only={lp}):
            i += 1
            print(tr_pair.input + "\t" + tr_pair.output)
    else:
        langs = maybe_smugri(lang_or_lp)
        lang_set = set(langs.split(","))
        raw_data = load_json_datax(filename)
        data_iter = data_iter_for_tok_train(raw_data, lang_set)
        i = 0
        for snt in data_iter:
            print(snt)
            i += 1


def do_stats(filename):
    stats = defaultdict(int)
    raw_data = load_json_datax(filename)

    for data in raw_data:
        langs = sorted([k for k in data.keys() if data[k].strip() != ""])
        stats["-".join(langs)] += 1
    for k in stats:
        print(k, stats[k])


def lang_from_name(filename):
    return filename.split(".")[-1]


def moses_to_json(file1, file2):
    result = list()

    l1 = lang_from_name(file1)
    l2 = lang_from_name(file2)

    with open(file1, "r") as h1, open(file2, "r") as h2:
        for line1 in h1:
            line2 = h2.readline()

            result.append({l1: line1.strip(), l2: line2.strip()})

    return result


def multi_moses_to_json(output_file, init_json, input_file_tuples):
    try:
        with open(init_json, "r") as h:
            result = json.load(h)
    except:
        result = list()

    for input_file_tuple in input_file_tuples:
        this_result = moses_to_json(*input_file_tuple)
        result.append({"source": f"{input_file_tuple[0]}-{input_file_tuple[1]}", "sentences": this_result})

    with open(output_file, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)


def group_tuples(input_tuples):
    return [(input_tuples[2 * i], input_tuples[2 * i + 1]) for i in range(int(len(input_tuples) / 2))]


def combine_two_jsons(json_target, json_addition):
    for k in json_addition:
        if k in json_target:
            json_target[k] += json_addition[k]
        else:
            json_target[k] = json_addition[k]


def combine_jsons(filelist):
    result = dict()

    for filename in filelist:
        data = json.load(open(filename))

        combine_two_jsons(result, data)

    json.dumps(result)


if __name__ == "__main__":
    # dump_to_stdout(filename="data/train.json", lang_or_lp="fi")
    dump_to_stdout()
    # multi_moses_to_json(sys.argv[1], sys.argv[2], group_tuples(sys.argv[3:]))

    # combine_jsons(sys.argv[1:])

    # do_stats("data/train.json")

"""
en-et-liv-lv 382
en-liv-lv 128
liv 41596
liv-lv 56
en-et-liv 1
et-liv 2778
en-liv 89
en 7
et 16
et-liv-lv 11431
et-lv 1
"""
