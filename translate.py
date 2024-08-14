#!/usr/bin/env python3

import sys

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from data import do_list_in_batches, lang_bin_mapping
from vivisect import CouplingSpecTuple, load_module_config
from collections import defaultdict

hf_tok = "TODO"

host_remote = True


def prepare_for_translation(input_text, tokenizer, input_language, output_language = None, for_output = False):
   if for_output:
      tokenizer.tgt_lang = output_language

      labels = tokenizer(text_target=input_text, return_tensors="pt")
      return labels['input_ids']
   else:
      tokenizer.src_lang = input_language

      inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
      if host_remote:
         inputs.to("cuda:0")

      frc_bos = tokenizer.get_lang_id(output_language) if output_language is not None else None

      return inputs, frc_bos


def finalize_translation(outputs, tokenizer, output_language):
   result = tokenizer.batch_decode(outputs, skip_special_tokens=True)

   return result


def loadtokenizer(mdlname = "facebook/m2m100_418M"):
   tokenizer = AutoTokenizer.from_pretrained(mdlname, token=hf_tok)
   return tokenizer


def loadmodel(mdlname = "facebook/m2m100_418M"):
   if host_remote:
      model = AutoModelForSeq2SeqLM.from_pretrained(mdlname, token=hf_tok, device_map="auto")
   else:
      model = AutoModelForSeq2SeqLM.from_pretrained(mdlname, token=hf_tok)
   return model


def coupled_encode(coupling_specs, lang_to_bin, input_language, input_texts):
   this = coupling_specs[lang_to_bin[input_language]]

   # 0. input text --> input token IDs
   inputs, _ = prepare_for_translation(input_texts, this.tokenizer, input_language)

   # 1. input token IDs --> encoder vectors
   encoder = this.model.get_encoder()

   embeddings = encoder(**inputs)

   return embeddings

def coupled_generate(coupling_specs, lang_to_bin, output_language, encoder_embeddings):
   dec_idx = lang_to_bin[output_language]
   main_model = coupling_specs[0].model

   if dec_idx != 0:
      tmp_decoder = coupling_specs[dec_idx].model.get_decoder()

      cached_decoder = main_model.get_decoder()

      main_model.decoder = tmp_decoder
   else:
      cached_decoder = None

   tokenizer = coupling_specs[dec_idx].tokenizer

   # 2. encoder vectors --> output token IDs
   frc_bos = tokenizer.get_lang_id(output_language)
   raw_outputs = main_model.generate(forced_bos_token_id=frc_bos, encoder_outputs=encoder_embeddings)

   # 3. output token IDs --> output text
   result = finalize_translation(raw_outputs, tokenizer, output_language)

   if dec_idx != 0:
      main_model.decoder = cached_decoder

   return result


def make_uniq(lang_to_bin):
   result = defaultdict(lambda: 0)

   for lang in lang_to_bin:
      bin_set = lang_to_bin[lang]
      result[lang] = 0 if 0 in bin_set else list(bin_set)[0]

   return result

def coupled_translate(coupling_specs, input_texts, input_language, output_language):
   lang_to_bin = make_uniq(lang_bin_mapping(coupling_specs))

   outputs = list()

   for inp_batch in do_list_in_batches(input_texts, 16):
      encoder_embeddings = coupled_encode(coupling_specs, lang_to_bin, input_language, inp_batch)

      these_outputs = coupled_generate(coupling_specs, lang_to_bin, output_language, encoder_embeddings)

      outputs += these_outputs

   return outputs


def load_and_init_module_config(model_id):
   config = load_module_config(model_id)

   coupling_specs = list()

   for entry in config:
      lang_set = entry["lang_set"]
      model_id = entry["model_id"]

      model = loadmodel(model_id)
      tokenizer = loadtokenizer(model_id)

      coupling_specs.append(CouplingSpecTuple(lang_set=lang_set, model=model, tokenizer=tokenizer, model_id=model_id))

   return coupling_specs

if __name__ == "__main__":
   if len(sys.argv) > 1:
      host_remote = True
   else:
      host_remote = False

   if not host_remote:
      sys.argv = ("X", "models/m2m_orig", "et", "de")

   mdl_id = sys.argv[1]

   from_lang = sys.argv[2]
   to_lang = sys.argv[3]

   module_config = load_and_init_module_config(mdl_id)

   if host_remote:
      inputs = [line.strip() for line in sys.stdin]
   else:
      inputs = ["See on ikka tore uudis."]

   outputs = coupled_translate(module_config, inputs, from_lang, to_lang)

   print("\n".join(outputs))
