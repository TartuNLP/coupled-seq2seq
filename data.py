#!/usr/bin/env python3

import json
import os
import sys
import torch

from torch.utils.data import IterableDataset
from collections import namedtuple, defaultdict
from random import randrange, shuffle

from aux import log, smugri_back, maybe_smugri
from langconv import any_to_madlad, any_to_nllb, is_nllb, is_madlad, get_mdl_type, any_to_mdl_type

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


def get_tr_pairs(raw_data=None, filename=None, leave_out=None, leave_only=None, model_type=None):
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

                            conv_l1 = any_to_mdl_type(model_type, l1)
                            conv_l2 = any_to_mdl_type(model_type, l2)

                            yield TrPair(conv_l1, conv_l2, input, tup[l2])


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
    lang_to_idx = dict()

    for i, spec_pair in enumerate(coupling_specs):
        for lang in spec_pair.lang_set:
            if lang not in lang_to_idx:
                lang_to_idx[lang] = {i}
            else:
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

def get_data_cache_location(filename, batch_size, idx = None):
    dirname = filename + "-tokcache"
    if not os.path.isdir(dirname):
        os.mkdir(dirname)

    name = (dirname + "/batch-" + str(batch_size))

    name += r"-meta.json" if idx is None else f"-{idx:04}.pt"

    return name



class MultilingualBatchingCachingDataset:
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

    def _fill_bins(self):
        bins = defaultdict(lambda: defaultdict(list))

        for tr_pair in get_tr_pairs(filename=self.filename, model_type=self.model_type):
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
        # log(str(self._lang_to_idx))

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

    def _bins_to_tokenized_batched_cached_data(self, bins):
        shard_i = 0
        batch_i = 0

        metainfo = []
        data = []

        log("Tokenizing data")

        for raw_batch, src_k, tgt_k in do_bins_in_shuffled_batches(bins, self.batch_size):
            batch_i += 1
            if not batch_i % 10000:
                log(f"Tokenized {batch_i + shard_i * self.shard_size} batches (shard {shard_i})")

            prepared_batch = self.tokenize_and_pad(raw_batch, src_k, tgt_k)
            data.append((prepared_batch, src_k, tgt_k))

            if batch_i >= self.shard_size:
                shard_i += 1
                batch_i = 0
                fn = self._save_cache_file(data, self.filename, shard_i)
                metainfo.append({'shard_filename': fn, 'shard_size': len(data)})

                del data

                data = []

        if len(data) > 0:
            log(f"Tokenized {batch_i} batches (shard {shard_i})")
            shard_i += 1
            fn = self._save_cache_file(data, self.filename, shard_i)
            metainfo.append({'shard_filename': fn, 'shard_size': len(data)})

        meta_fn = get_data_cache_location(self.filename, self.batch_size)
        with open(meta_fn, 'w') as f:
            json.dump(metainfo, f)

        del data

    """def _load_data_from_cache(self, filename):
        cache_location = self._get_data_cache_location(filename)

        there_is_a_cache = os.path.exists(cache_location)

        if there_is_a_cache:
            log(f"Loading data from cache ({cache_location})")
            self.data = torch.load(cache_location)
        else:
            log(f"Cache not found ({cache_location}), need to tokenize anew")

        return there_is_a_cache"""

    def _save_cache_file(self, data, filename, idx):
        cache_location = get_data_cache_location(filename, self.batch_size, idx)

        if os.path.exists(cache_location):
            raise Exception("Cache already exists")

        torch.save(data, cache_location)
        log("Saved data into cache")
        return cache_location

    def set_model_type(self):
        result = None

        for spec_tuple in self.coupling_specs:
            this_type = get_mdl_type(spec_tuple.tokenizer)
            if result is None:
                result = this_type
            else:
                assert result == this_type, "in this implementation model types (NLLB/MADLAD) must be the same for all included models"

        return result


    def __init__(self, tr_file, coupling_specs, batch_size, tracing_msg="just a set", max_src_len=256,
                 max_tgt_len=256, verbose=False, leave_only=None, shard_size=1000000):
        self.msg = tracing_msg
        self.batch_size = batch_size
        self.shard_size = shard_size

        self.filename = tr_file

        self.coupling_specs = coupling_specs
        self.model_type = self.set_model_type()

        # init lang to idx
        self._lang_to_idx = lang_bin_mapping(coupling_specs)

    def cache_data(self):
        # collect data into bins and cache it
        bins = self._fill_bins()

        self.report_update_stats(bins)

        self._bins_to_tokenized_batched_cached_data(bins)

class MultilingualDatasetIterator(IterableDataset):
    def _load_metafile(self, filename, batch_size):
        cache_metafile = get_data_cache_location(filename, batch_size)

        with open(cache_metafile, 'r') as f:
            self.metainfo = json.load(f)

    def _init_curr_shard(self):
        cache_location = self.metainfo[self.curr_shard_idx]['shard_filename']
        self.curr_shard_data = torch.load(cache_location)

    def __init__(self, filename, batch_size):
        self._load_metafile(filename, batch_size)

        self.data_len = sum([e['shard_size'] for e in self.metainfo])

        self.curr_shard_idx = 0
        self._init_curr_shard()

        self.curr_elem_idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        try:
            result = self.curr_shard_data[self.curr_elem_idx]
            self.curr_elem_idx += 1
        except IndexError:
            self.curr_shard_idx += 1

            if self.curr_shard_idx >= len(self.metainfo):
                raise StopIteration
            else:
                self._init_curr_shard()
                self.curr_elem_idx = 0
                result = self.curr_shard_data[self.curr_elem_idx]
                self.curr_elem_idx += 1

        return result

    def __len__(self):
        return self.data_len


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
