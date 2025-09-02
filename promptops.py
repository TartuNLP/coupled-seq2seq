
# first, keyword identifiers for selecting prompt templates in scripts:

PF_RAW = "raw"
PF_RAWLINES = "rawlines"
PF_SMUGRI_MT = "smugri_mt"
PF_SMUGRI_LID = "smugri_lid"
PF_ALPACA = "alpaca"
PF_PIVOT = "eurollm_pivot"
PF_TR_FLT = "eurollm_tr_flt"

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

EUROLLM_TEMPLATE_BASE = """<|im_start|>system
{system_instruction}<|im_end|>
<|im_start|>user
{user_instruction}<|im_end|>
<|im_start|>assistant
"""

EUROLLM_TEMPLATE_FILTER = EUROLLM_TEMPLATE_BASE.format(
    system_instruction="You are a large language model, whose sole task is to respond to user queries. Think "
                       "carefully, and after careful deliberation respond to the user, following their instructions.",
    user_instruction="Your task is to detect hallucinations of a translation model. Here is the original input "
                     "text:\n\n{hi_segm}\n\nand this next here should be its translation into {new_hi_res_lang}:\n\n"
                     "{hyp-translation}\n\nThink carefully: is this second text actually a translation of the original "
                     "input text into {new_hi_res_lang}? Respond with yes or no, no additional comments or discussions.",
    user_instructionx="Your task is to evaluate the quality of a translation pair: an original text and its "
                     "translation. The goal is to check if both are in their specified languages and if the "
                     "translation is appropriate. First the texts and then the definition of how you should respond. "
                     "This is extremely important to get right, so take a deep breath and respond correctly.\n\n"
                     "The original text is:\n\n"
                     "{hi_segm}\n\n"
                     "You need to check if this text is in {hi_lang}. The translation is:\n\n"
                     "{hyp-translation}\n\n"
                     "You need to check if it is in {new_hi_res_lang}. "
                     "Mainly, check if the 2nd text is actually an appropriate translation of the 1st text.\n"
                     "Respond with a single phrase only, with no additional comments or explanations:\n"
                     "- if the texts are appropriate enough translations of each other and are in the expected "
                     "languages, respond with the phrase 'appropriate translation',\n"
                     "- if the translation is not fully appropriate and precise and has some issues, but is still a "
                     "translation of the original text, and it is definitely in {new_hi_res_lang},"
                     "respond with the phrase 'some issues',\n"
                     "- if the 2nd text actually is a very bad translation of the 1st text, or is not its translation "
                     "at all, or has extra text in it, respond with the phrase 'wrong translation',\n"
                     "- if the original text is not actually in {hi_lang}, then respond with the phrase "
                     "'wrong input language',\n"
                     "- finally, if the 2nd text is not in {new_hi_res_lang}, respond with the phrase "
                     "'wrong output language'.\n"
                     "Now, take a deep breath, and output your response, depending on the specified languages and texts",
)

MULTILING_MSG = {
    'English': { 'system_instruction': "You are a powerful AI translator, the best model to produce translations "
                                       "from any European language into English. When you are asked to translate, you "
                                       "respond with the translation in the requested language, which perfectly "
                                       "preserves the meaning and stylistics and is overall a perfect and usable "
                                       "translation and text segment into English.",
                 'text_is_in': "Your task is to translate the following text; the language of this text is: ",
                 'postinstruction': "Now translate that text into English" },
    'Russian': { 'system_instruction': "Ты — мощный ИИ-переводчик, лучшая модель для перевода с любого европейского "
                                       "языка на русский. Когда тебя просят перевести, ты отвечаешь переводом на "
                                       "требуемом языке, который идеально сохраняет смысл и стилистику и в целом "
                                       "является совершенным и пригодным переводом и текстовым фрагментом на русском.",
                 'text_is_in': "Твоя задача — перевести текст; язык этого текста",
                 'postinstruction': "Теперь переведи этот текст на русский" },
    'Estonian': {'system_instruction': "Sa oled võimas tehisintellektil põhinev tõlkija, parim mudel, mis suudab "
                                       "tõlkida kõigist Euroopa keeltest eesti keelde. Kui sinult palutakse tõlkida, "
                                       "vastad sa tõlkega soovitud keeles, mis säilitab täiuslikult tähenduse ja stiili"
                                       " ning on igati ideaalne ja kasutuskõlblik tõlge ja tekstilõik eesti keeles.",
                 'text_is_in': "Sinu ülesanne on tõlkida tekst; selle teksti keel on",
                 'postinstruction': "Nüüd tõlgi see tekst eesti keelde"},
    'Latvian': {'system_instruction': "Tu esi spēcīgs mākslīgā intelekta tulkotājs, labākais modelis, lai veiktu "
                                      "tulkojumus no jebkuras Eiropas valodas latviešu valodā. Kad no tevis tiek lūgts "
                                      "tulkot, tu atbildi ar tulkojumu pieprasītajā valodā, kas nevainojami saglabā "
                                      "nozīmi un stilistiku un kopumā ir perfekts un lietojams tulkojums un "
                                      "teksta fragments latviešu valodā.",
                'text_is_in': "Tavs uzdevums ir iztulkot tekstu; šī teksta valoda ir",
                'postinstruction': "Tagad iztulko šo tekstu latviešu valodā"},
    'Finnish': {'system_instruction': "Olet tehokas tekoälykääntäjä, paras malli tuottamaan käännöksiä mistä tahansa "
                                      "eurooppalaisesta kielestä suomeen. Kun sinulta pyydetään käännöstä, vastaat "
                                      "pyydetyllä kielellä annetulla käännöksellä, joka säilyttää täydellisesti "
                                      "merkityksen ja tyylin ja on kokonaisuudessaan täydellinen ja käyttökelpoinen "
                                      "käännös ja tekstijakso suomeksi.",
                'text_is_in': "Tehtäväsi on kääntää teksti; tämän tekstin kieli on",
                'postinstruction': "Nyt käännä tuo teksti suomeksi"},
    'Hungarian': {'system_instruction': "Te egy nagy teljesítményű mesterséges intelligencia fordító vagy, a legjobb "
                                        "modell bármely európai nyelvről magyarra történő fordításra. Amikor "
                                        "fordításra kérnek, a kért nyelven adod meg a fordítást, amely tökéletesen "
                                        "megőrzi a jelentést és a stílust, és összességében hibátlan, használható "
                                        "magyar nyelvű fordítás és szövegrész lesz.",
                  'text_is_in': "A feladatod egy szöveg lefordítása; ennek a szövegnek a nyelve",
                  'postinstruction': "Most fordítsd le ezt a szöveget magyarra"},
    'Swedish': {'system_instruction': "Du är en kraftfull AI-översättare, den bästa modellen för att översätta från "
                                      "vilket europeiskt språk som helst till svenska. När du blir ombedd att "
                                      "översätta svarar du med översättningen på det begärda språket, som fullständigt "
                                      "bevarar betydelsen och stilen och som i sin helhet är en perfekt och användbar "
                                      "översättning och text på svenska.",
                'text_is_in': "Din uppgift är att översätta en text; språket i denna text är",
                'postinstruction': "Nu översätt den texten till svenska"},
    'Norwegian': {'system_instruction': "Du er en kraftig AI-oversetter, den beste modellen for å oversette fra "
                                        "ethvert europeisk språk til norsk. Når du blir bedt om å oversette, svarer "
                                        "du med oversettelsen på det ønskede språket, som perfekt bevarer meningen og "
                                        "stilen og som totalt sett er en fullkommen og brukbar oversettelse og "
                                        "tekstbit på norsk.",
                  'text_is_in': "Din oppgave er å oversette en tekst; språket i denne teksten er",
                  'postinstruction': "Oversett nå den teksten til norsk"}
}

EUROLLM_USER_MSG_TEMPLATE = """{text_is_in}: {hi_lang}.

{hi_segm}

{postinstruction}"""

#EUROLLM_USER_MSG_TEMPLATE = """{hi_segm}
#{postinstruction}"""

def prep_prompt(data, prompt_format, inference=False, tok=None):
    if prompt_format in {PF_RAW, PF_RAWLINES}:
        # data is a string, return it
        return data

    elif prompt_format == PF_PIVOT:
        assert inference, "Pivoting template with EuroLLM 9B is meant for inference only"
        return _prep_eurollm_entry(data)

    elif prompt_format == PF_TR_FLT:
        return _prep_eurollm_flt_entry(data)

    elif prompt_format in {PF_SMUGRI_MT, PF_SMUGRI_LID}:
        # data has src_segm, src_lang, tgt_lang, etc
        return _prep_ljmf_entry(data, prompt_format, inference)

    elif prompt_format == PF_ALPACA:
        # data has instruction and input in it
        return _prep_alpaca_entry(data, inference)

    else:
        raise NotImplementedError(f"Prompt format {prompt_format} is not implemented.")


def _prep_eurollm_entry(entry):
    output_lang = entry['new_hi_res_lang']
    user_msg = EUROLLM_USER_MSG_TEMPLATE.format(**entry, **MULTILING_MSG[output_lang])
    result = EUROLLM_TEMPLATE_BASE.format(**MULTILING_MSG[output_lang], user_instruction=user_msg)
    return result


def _prep_eurollm_flt_entry(entry):
    result = EUROLLM_TEMPLATE_FILTER.format(**entry)
    return result


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
