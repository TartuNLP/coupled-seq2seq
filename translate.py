#!/usr/bin/env python3

import sys

from aux import CmdlineArgs, log
from tokops import tokenize_batch
from trainllm import load_hf_tokenizer, load_hf_model
from data import do_list_in_batches, prep_llm_input


def llm_generate(model, tokenizer, input_texts, debug=False, max_len=8000):
    tok_batch = tokenize_batch(tokenizer, input_texts)

    tok_batch['input_ids'] = tok_batch['input_ids'].to(model.device)
    tok_batch['attention_mask'] = tok_batch['attention_mask'].to(model.device)

    raw_outputs = model.generate(**tok_batch, num_beams=5, do_sample=False)

    # 3. output token IDs --> output text
    pre_result = tokenizer.batch_decode(raw_outputs, skip_special_tokens=True)

    result = [o[len(i):].strip().replace("\n", "<<BR>>") for i, o in zip(input_texts, pre_result)]

    return result


def generative_translate(model, tokenizer, input_texts, input_language, output_language, debug=False):
    all_outputs = list()

    for inp_batch in do_list_in_batches(input_texts, 8):
        these_outputs = llm_generate(model, tokenizer, input_texts, debug=debug, max_len=8000)

        all_outputs += these_outputs

    return all_outputs


def _cmdline_args(inputs):
    description = """Translate STDIN text with a translation model"""

    pos_args = ["mdl_id", "from_lang", "to_lang"]

    #post-process the arguments
    args = CmdlineArgs(description, pos_args, input_args=inputs, kw_arg_dict={"debug": False})

    log(f"Launched as {args}")

    return args


def and_i_called_this_function_do_main_too(iv):
    args = _cmdline_args(iv)

    inputs = [prep_llm_input({
        'src_segm': line.strip(),
        'src_lang': args.from_lang,
        'tgt_lang': args.to_lang,
        'task': 'translate',
        'tgt_segm': '' }) for line in sys.stdin]
    # inputs = ["See on ikka tore uudis.", "Ma ikka katsetaks ka täpitähtedega tõlkimist.", "Mis tähed on täpitähed?"]

    log(f"Inputs: {inputs}")

    model = load_hf_model(args.mdl_id)
    tokenizer = load_hf_tokenizer(args.mdl_id)

    log("Model loaded, starting to translate")
    outputs = generative_translate(model, tokenizer, inputs, args.from_lang, args.to_lang, debug=args.debug)

    print("\n".join(outputs))

    log("Done...")


if __name__ == "__main__":
    input_values = sys.argv[1:] if len(sys.argv) > 1 \
        else ["models/nllb", "et", "en"]

    and_i_called_this_function_do_main_too(input_values)
