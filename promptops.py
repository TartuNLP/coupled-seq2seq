from aux import log

PF_RAW = "raw"
PF_SMUGRI = "smugri_lid_md"
PF_ALPACA = "alpaca"

INF_PROMPT_LID = "<|reserved_special_token_12|>{src_segm}<|reserved_special_token_13|>"
INF_PROMPT_MT = INF_PROMPT_LID + "{src_lang}<|reserved_special_token_14|>{task} to {tgt_lang}<|reserved_special_token_15|>"

_TRAIN_PROMPT_PREF = "<|reserved_special_token_12|>{src_segm}<|reserved_special_token_13|>{src_lang}"
_TRAIN_PROMPT_MID = "<|reserved_special_token_14|>{task} to {tgt_lang}<|reserved_special_token_15|>{tgt_segm}"
_TRAIN_PROMPT_SUF = "<|reserved_special_token_16|><|end_of_text|>"

TRAIN_PROMPT_PARA = _TRAIN_PROMPT_PREF + _TRAIN_PROMPT_MID + _TRAIN_PROMPT_SUF
TRAIN_PROMPT_MONO = _TRAIN_PROMPT_PREF + _TRAIN_PROMPT_SUF

ALPACA_PROMPT = ("Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}")


def prep_prompt(data, prompt_format):
    if prompt_format == PF_RAW:
        # data is a string, return it
        return data

    elif prompt_format == PF_SMUGRI:
        # data has src_segm, src_lang, tgt_lang, etc
        return _prep_ljmf_entry(data)

    elif prompt_format == PF_ALPACA:
        # data has instruction and input in it
        return _prep_alpaca_entry(data)


def _prep_alpaca_entry(entry):
    prompt = ALPACA_PROMPT.format(**entry)

    return prompt


def tokenize_str(tokenizer, entry, add_eos=True, max_len=2000):
    tokens = tokenizer(
        entry,
        truncation=True,
        max_length=max_len,
        return_attention_mask=True,
    )

    if add_eos:
        tokens['attention_mask'].append(1)
        tokens['input_ids'].append(tokenizer.eos_token_id)

    return tokens


def tokenize_for_inference(tokenizer, src_segm, src_lang=None, tgt_lang=None, task="translate", debug=False):
    if task == "lid":
        prompt = INF_PROMPT_LID.format(src_segm=src_segm)
    elif task in {"translate", "approx-translate"}:
        prompt = INF_PROMPT_MT.format(src_segm=src_segm, src_lang=src_lang, tgt_lang=tgt_lang, task=task)
    else:
        prompt = src_segm

    if debug:
        log(prompt)

    return tokenize_str(tokenizer, prompt, add_eos=False)


def _prep_ljmf_entry(entry):
    if entry['task'] in {'translate', 'approx-translate'} and entry['tgt_segm'] and entry['tgt_lang']:
        prompt = TRAIN_PROMPT_PARA.format(**entry)
    else:
        prompt = TRAIN_PROMPT_MONO.format(**entry)

    return prompt
