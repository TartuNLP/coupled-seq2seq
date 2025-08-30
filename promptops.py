
# first, keyword identifiers for selecting prompt templates in scripts:

PF_RAW = "raw"
PF_RAWLINES = "rawlines"
PF_SMUGRI_MT = "smugri_mt"
PF_SMUGRI_LID = "smugri_lid"
PF_ALPACA = "alpaca"
PF_PIVOT = "eurollm_pivot"

# now the prompt templates themselves, SMUGRI LID / MT template:

SMUGRI_INF_PROMPT_LID = "<|reserved_special_token_12|>{src_segm}<|reserved_special_token_13|>"

_SMUGRI_INF_PROMPT_TMPMID = "<|reserved_special_token_14|>{task} to {tgt_lang}<|reserved_special_token_15|>"
SMUGRI_INF_PROMPT_MT = SMUGRI_INF_PROMPT_LID + "{src_lang}" + _SMUGRI_INF_PROMPT_TMPMID

SMUGRI_PROMPT_TRAIN_MONO = SMUGRI_INF_PROMPT_LID + "{src_lang}"
_SMUGRI_TRAIN_PROMPT_MID = _SMUGRI_INF_PROMPT_TMPMID + "{tgt_segm}"

SMUGRI_PROMPT_TRAIN_PARA = SMUGRI_PROMPT_TRAIN_MONO + _SMUGRI_TRAIN_PROMPT_MID

# Alpaca instructions prompt template:

ALPACA_PROMPT_INF = ("Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n")

ALPACA_PROMPT_TRAIN = (ALPACA_PROMPT_INF + "{output}")

# EuroLLM format:

EUROLLM_TEMPLATE = [{'role': 'system',
             'content': "You are a powerful AI translator, the best model to "
                        "produce translations in all European languages and more. "
                        "When you are asked to translate, you always write 'I understand, here is a translation:'. "
                        "After that you respond with the translation output in the requested language, "
                        "which perfectly preserves the meaning and stylistics and is overall a perfect and "
                        "usable translation and text segment in the requested language."},
            {'role': 'user',
             'content': "Translate the following text segment "
                        "from {hi_lang} to {new_hi_res_lang}:\n{hi_segm}"}]

def prep_prompt(data, prompt_format, inference=False, tok=None):
    if prompt_format in {PF_RAW, PF_RAWLINES}:
        # data is a string, return it
        return data

    elif prompt_format == PF_PIVOT:
        assert inference, "Pivoting template with EuroLLM 9B is meant for inference only"
        return _prep_eurollm_entry(data, tok)

    elif prompt_format in {PF_SMUGRI_MT, PF_SMUGRI_LID}:
        # data has src_segm, src_lang, tgt_lang, etc
        return _prep_ljmf_entry(data, prompt_format, inference)

    elif prompt_format == PF_ALPACA:
        # data has instruction and input in it
        return _prep_alpaca_entry(data, inference)

    else:
        raise NotImplementedError(f"Prompt format {prompt_format} is not implemented.")


def _prep_eurollm_entry(entry, tok):
    input_msgs = EUROLLM_TEMPLATE
    input_msgs[1]['content'] = input_msgs[1]['content'].format(**entry)

    return tok.apply_chat_template(input_msgs, tokenize=False, add_generation_prompt=True)


def _prep_alpaca_entry(entry, inference=False):
    fmt = ALPACA_PROMPT_INF if inference else ALPACA_PROMPT_TRAIN
    prompt = fmt.format(**entry)
    return prompt


def _prep_ljmf_entry(entry, fmt, inference=False):
    if inference:
        if fmt == PF_SMUGRI_MT:
            prompt = SMUGRI_INF_PROMPT_MT.format(**entry)
        elif fmt == PF_SMUGRI_LID:
            prompt = SMUGRI_INF_PROMPT_LID.format(**entry)
        else:
            raise NotImplementedError(f"Prompt format {fmt} is not implemented.")
    else:
        if entry['task'] in {'translate', 'approx-translate'} and entry['tgt_segm'] and entry['tgt_lang']:
            prompt = SMUGRI_PROMPT_TRAIN_PARA.format(**entry)
        else:
            prompt = SMUGRI_PROMPT_TRAIN_MONO.format(**entry)

    return prompt
