#!/usr/bin/env python3

import json
import os
import sys
import torch
import re

from torch.utils.data import IterableDataset
from collections import namedtuple, defaultdict
from random import randrange, shuffle
from pathlib import Path

from aux import log
from langconv import any_to_madlad, any_to_nllb, is_nllb, is_madlad, get_mdl_type, any_to_mdl_type, is_dec_only_llm, \
    base_to_nllb
from tokops import tokenizeit

TrPair = namedtuple('TrPair', ["src_lang", "tgt_lang", "input", "output"])


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


def load_json_data(path, leave_out={}, skip_cats=True, load_mono=True):
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


def get_tr_pairs(raw_data=None, filename=None, leave_out=None, leave_only=None, model_type=None, exclude_set=None):
    if filename is not None:
        raw_data = load_json_data(filename)

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

                            if exclude_set is None or (tup[l1] not in exclude_set[l1] and tup[l2] not in exclude_set[l2]):
                                input = tup[l1]
                                if dia_key in tup:
                                    input = f"<{tup[dia_key]}> {input}"

                                conv_l1 = any_to_mdl_type(model_type, l1)
                                conv_l2 = any_to_mdl_type(model_type, l2)

                                if not snt_is_fishy(input, conv_l1) and not snt_is_fishy(tup[l2], conv_l2):
                                    yield TrPair(conv_l1, conv_l2, input, tup[l2])


def split_by_lang(filename, model_type):
    result = defaultdict(list)

    # if filename is not None:
        # tr_pairs = load_json_datax(filename)

    tr_pairs = get_tr_pairs(filename=filename, model_type=model_type)

    for tup in tr_pairs:
        #for l1 in tup:
        #    for l2 in tup:
        #        if l1 != l2 and not "dia" in l1 and not "dia" in l2:
        l1 = tup.src_lang
        l2 = tup.tgt_lang
        lp = f"{l1}-{l2}"
        result[lp].append((tup.input, tup.output))

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

def get_data_cache_location(cache_meta_path, idx):
    cache_folder, cache_file = os.path.split(cache_meta_path)

    if cache_folder:
        Path(cache_folder).mkdir(parents=True, exist_ok=True)

    if cache_meta_path.endswith(".json"):
        return cache_meta_path[:-5] + f"_{idx:04}.pt"
    else:
        raise ValueError(f"Expected a json file for the cache meta-location ({cache_meta_path})")


def make_gen_text(elem, tok, for_inference=False):
    return (f"== From: {elem.src_lang}\n== To: {elem.tgt_lang}\n== Input: {elem.input}\n== Output: " +
            ("" if for_inference else "{elem.output}{tok.eos_token}"))


class MultilingualBatchingCachingDataset:
    def _post_proc_bins(self, bins):
        for src_k in bins:
            for tgt_k in bins[src_k]:
                while len(bins[src_k][tgt_k]) % self.args.batch_size != 0:
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

        for tr_pair in get_tr_pairs(filename=self.filename, model_type=self.model_type, exclude_set=self.exclude_set):
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
        #prep_batch_grouped = src_tokenizer(text=input_list, return_tensors="pt",
        #                                   padding="longest", truncation=True, max_length=self.args.max_snt_len)
        prep_batch_grouped = tokenizeit((src_tokenizer, cplspec.postokenizer), input_list, self.args.max_snt_len, False)

        if is_nllb(src_tokenizer):
            src_lang_list = [any_to_nllb(e.src_lang) for e in rawbatch]
            src_lang_vec = src_tokenizer.convert_tokens_to_ids(src_lang_list)
            prep_batch_grouped['input_ids'][:,0] = torch.tensor(src_lang_vec)

        return prep_batch_grouped

    def tokenize_output(self, tgttokenizer, tgtposttok, rawbatch):
        outputs = [e.output for e in rawbatch]
        tgttokenizer.tgt_lang = rawbatch[0].tgt_lang
        #labels = tgttokenizer(text_target=outputs, return_tensors="pt",
        #                      padding="longest", truncation=True, max_length=self.args.max_snt_len)
        labels = tokenizeit((tgttokenizer, tgtposttok), outputs, self.args.max_snt_len, True)

        if is_nllb(tgttokenizer):
            tgt_lang_list = [any_to_nllb(e.tgt_lang) for e in rawbatch]
            tgt_lang_vec = tgttokenizer.convert_tokens_to_ids(tgt_lang_list)
            labels['input_ids'][:, 0] = torch.tensor(tgt_lang_vec)

        return labels

    def tokenize_gen_batch(self, raw_batch):
        tokenizer = self.coupling_specs[0].tokenizer
        tokenizer.pad_token = '<|reserved_special_token_0|>'
        tokenizer.padding_side = 'left'

        texts = [make_gen_text(e, tokenizer) for e in raw_batch]

        batch = tokenizer(texts, return_tensors="pt", max_length=512, truncation=True, add_special_tokens=True, padding=True)

        return batch

    def tokenize_and_pad(self, raw_batch, src_k, tgt_k):
        tgt_tokenizer = self.coupling_specs[tgt_k].tokenizer
        tgt_postok = self.coupling_specs[tgt_k].postokenizer

        if is_madlad(tgt_tokenizer):
            inputs = [f"{any_to_madlad(e.tgt_lang)} {e.input}" for e in raw_batch]
        else:
            inputs = [e.input for e in raw_batch]

        prep_batch_grouped = self.tokenize_input(self.coupling_specs[src_k], inputs, raw_batch)
        labels = self.tokenize_output(tgt_tokenizer, tgt_postok, raw_batch)
        prep_batch_grouped['labels'] = labels['input_ids']

        # inject_bin_indices(prep_batch_grouped, src_k, tgt_k)

        #split_prep_batch = [{k: prep_batch_grouped[k][i] for k in prep_batch_grouped}
        #                    for i, trp in enumerate(raw_batch)]

        return prep_batch_grouped

    def _bins_to_tokenized_batched_cached_data(self, bins, cache_path):
        shard_i = 0
        batch_i = 0
        total_i = 0

        metainfo = []
        data = []

        log("Tokenizing data")

        for raw_batch, src_k, tgt_k in do_bins_in_shuffled_batches(bins, self.args.batch_size):
            batch_i += 1
            if not batch_i % 10000:
                log(f"Tokenized {batch_i + shard_i * self.args.shard_size} batches (shard {shard_i})")

            if is_dec_only_llm(self.coupling_specs[tgt_k].tokenizer):
                prepared_batch = self.tokenize_gen_batch(raw_batch)
                data.append((prepared_batch, total_i))
            else:
                prepared_batch = self.tokenize_and_pad(raw_batch, src_k, tgt_k)
                data.append((prepared_batch, src_k, tgt_k, total_i))

            if batch_i >= self.args.shard_size:
                shard_i += 1
                batch_i = 0
                fn = self._save_cache_file(data, cache_path, shard_i)
                metainfo.append({'shard_filename': fn, 'shard_size': len(data)})

                del data

                data = []

            total_i += 1

        if len(data) > 0:
            fn = self._save_cache_file(data, cache_path, shard_i + 1)
            metainfo.append({'shard_filename': fn, 'shard_size': len(data)})

        with open(cache_path, 'w') as f:
            json.dump(metainfo, f)

        del data

    @staticmethod
    def _save_cache_file(data, cache_location, idx):
        cache_location = get_data_cache_location(cache_location, idx)

        if os.path.exists(cache_location):
            raise Exception("Cache already exists")

        torch.save(data, cache_location)
        log(f"Saved data into cache (shard {idx})")

        return cache_location

    def set_model_type(self):
        result = None

        for spec_tuple in self.coupling_specs:
            this_type = get_mdl_type(spec_tuple.tokenizer)
            if result is None:
                result = this_type
            else:
                assert result == this_type, "in this implementation model types (NLLB/MADLAD/...) must be the same for all included models"

        return result


    def __init__(self, tr_file, coupling_specs, args):
        self.args = args
        self.filename = tr_file
        self.coupling_specs = coupling_specs

        self.exclude_set = _dev_to_dict(args.exclude_set) if args.exclude_set is not None else None

        self.model_type = self.set_model_type()

        # init lang to idx
        self._lang_to_idx = lang_bin_mapping(coupling_specs)

    def load_and_cache_data(self, cache_path):
        # collect data into bins and cache it
        bins = self._fill_bins()

        self.report_update_stats(bins)

        self._bins_to_tokenized_batched_cached_data(bins, cache_path)


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


class MultilingualDatasetIterator(IterableDataset):
    def _load_metafile(self, cache_metafile):
        with open(cache_metafile, 'r') as f:
            self.metainfo = json.load(f)
            self.data_len = sum([e['shard_size'] for e in self.metainfo])

    def _init_curr_shard(self):
        cache_location = self.metainfo[self.curr_shard_idx]['shard_filename']

        self.curr_shard_data = torch.load(cache_location, weights_only=False)

        assert len(self.curr_shard_data) == self.metainfo[self.curr_shard_idx]['shard_size']

    def __init__(self, filename):
        self.curr_shard_idx = 0
        self.curr_elem_idx = 0
        self.prev_shard_sum_len = 0

        if filename is not None:
            self._load_metafile(filename)

    def __iter__(self):
        self._init_curr_shard()
        return self

    def where_are_we(self):
        return DataState(shard_idx=self.curr_shard_idx, elem_idx=self.curr_elem_idx)

    def thats_where(self, data_state):
        self.curr_shard_idx = data_state.shard_idx
        self.curr_elem_idx = data_state.elem_idx
        self.prev_shard_sum_len = sum([e['shard_size'] for i, e in enumerate(self.metainfo) if i < self.curr_shard_idx])

    def __next__(self):
        try:
            result_data = self.curr_shard_data[self.curr_elem_idx]

            self.curr_elem_idx += 1
        except IndexError:
            self.prev_shard_sum_len += self.metainfo[self.curr_shard_idx]['shard_size']
            self.curr_shard_idx += 1

            if self.curr_shard_idx >= len(self.metainfo):
                self.__init__(None)
                raise StopIteration
            else:
                self._init_curr_shard()
                self.curr_elem_idx = 0

                result_data = self.curr_shard_data[self.curr_elem_idx]

                self.curr_elem_idx += 1

        index_in_epoch = self.prev_shard_sum_len + self.curr_elem_idx
        return result_data, index_in_epoch

    def __len__(self):
        return self.data_len


def upd_lc(dct, lang, snt):
    l = len(snt.split(" "))

    if l <= 10:
        k = '1:  1..10'
    elif l <= 15:
        k = '2: 11..15'
    elif l <= 20:
        k = '3: 16..20'
    else:
        k = '4:    >20'

    dct[k] += 1

    return l


def dump_to_stdout():
    filename = sys.argv[1]

    lc_src = defaultdict(int)

    tot_len = 0
    tot_count = 0

    for tr_pair in get_tr_pairs(filename=filename):
        print(tr_pair.src_lang + "\t" + tr_pair.input + "\t" + tr_pair.tgt_lang + "\t" + tr_pair.output)

        tot_len += upd_lc(lc_src, tr_pair.src_lang, tr_pair.input)
        tot_len += upd_lc(lc_src, tr_pair.tgt_lang, tr_pair.output)

        tot_count += 2

    totes = sum(lc_src.values())
    for k in sorted(lc_src):
        sys.stderr.write(f"{k}: {100*lc_src[k]/totes:.1f}%\n")
    sys.stderr.write(f"Avg length: {tot_len/float(tot_count):.1f}\n")


def do_stats(filename):
    stats = defaultdict(int)
    raw_data = load_json_data(filename)

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


def _dev_to_dict(filename):
    result = defaultdict(lambda: defaultdict(int))

    for dev_sample in load_json_data(filename):
        for lang in dev_sample:
            if not "dia" in lang:
                result[lang][dev_sample[lang]] = 1

    return result


def check_cross_pollination(small_path, large_path):
    print("preparing dev set")
    dct = _dev_to_dict(small_path)

    print("reading train set")
    for train_sample in load_json_data(large_path):
        for lang in train_sample:
            if not "dia" in lang and lang in dct:
                snt = train_sample[lang]

                if snt in dct[lang]:
                    dct[lang][snt] += 1

    print("---------------------")
    print("contamination report:")
    print("---------------------")
    for lang in dct:
        total = 0
        counts = 0
        freqs = 0

        for snt in dct[lang]:
            total += 1
            if dct[lang][snt] > 1:
                counts += 1
                freqs += (dct[lang][snt] - 1)

        print(f"{lang}: contaminated: {counts} ({100*counts/float(total):.1f}%), total occurrence: {freqs}")


def char_class(c):
    lc = c.lower()
    if re.match("[a-z]", lc):
        return "latn"
    elif re.match("[а-я]", lc):
        return "cyrl"
    else:
        return "other"


def snt_is_fishy(snt_raw, lang, detailed=False):
    snt = re.sub(r'^<[^>]+> ', '', snt_raw)

    snt_db = defaultdict(int)
    for c in snt:
        c_c = char_class(c)
        snt_db[c_c] += 1

    tot = snt_db['latn'] + snt_db['cyrl']

    if tot > 0:
        if snt_db['latn'] / tot > 0.7:
            this_is = 'latn'
        elif snt_db['cyrl'] / tot > 0.7:
            this_is = 'cyrl'
        else:
            this_is = 'mix'

        should_be = any_to_nllb(lang).split("_")[1].lower()

        if should_be != this_is:
            return (True, this_is, should_be) if detailed else True

    return (False, None, None) if detailed else False


def script_stats():
    db = defaultdict(lambda: defaultdict(int))

    # corp = []

    for raw_line in sys.stdin:
        lang, snt_raw = raw_line.strip().split("\t")

        is_fishy, this_is, should_be = snt_is_fishy(snt_raw, lang, detailed=True)
        if is_fishy:
            print(f"{lang}: should be {should_be}, is actually {this_is}:\n{snt_raw}")


if __name__ == "__main__":
    # check_cross_pollination(sys.argv[1], sys.argv[2])
    # multi_moses_to_json(sys.argv[1], sys.argv[2], group_tuples(sys.argv[3:]))
    # combine_jsons(sys.argv[1:])
    # do_stats("data/train.json")

    # dump_to_stdout()
    script_stats()
