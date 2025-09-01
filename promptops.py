
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
    system_instruction="You are an AI assistant, helping users with queries related to language data. "
                       "You respond helpfully, concisely and laconically to what the user requests. "
                       "It is of utmost importance to address the requests as correctly as possible. "
                       "So, take a deep breath, and carefully respond to the following request.",
    user_instruction="Your task is to determine if a pair of texts is in the specified languages and if the second "
                     "text is a correct translation of the first text. Your response should consist of a single word. "
                     "If the pair of texts is "
                     "indeed in the specified languages and they are translations of each other, then respond with "
                     "the word 'perfect'. If the texts are in correct languages and are not perfect translations of "
                     "each other, but still approximately correspond to each other by meaning, or if there are any "
                     "other minor issues, then respond with the word 'approximate'. If either text is NOT in the "
                     "specified language, or if the texts completely differ in meaning, then respond with the word "
                     "'wrong'. Do not provide any explanations or comments, just a single word as response.\n\n"
                     "Here are the texts, the first text should be in {hi_lang}, and the text is:\n\n"
                     "{hi_segm}\n\n"
                     "And here is the second text in the pair, it should be in {new_hi_res_lang}:\n\n"
                     "{hyp-translation}\n\n"
                     "Now, think carefully: is the latest text in {new_hi_res_lang} or not, and is it a translation of the first text? "
                     "Respond as instructed with a single word, 'perfect'/'approximate'/'wrong'.",
)

MULTILING_MSG = {
    'English': { 'system_instruction': "You are a powerful AI translator, the best model to produce translations "
                                       "from any European language into English. When you are asked to translate, you "
                                       "respond with the translation in the requested language, which perfectly "
                                       "preserves the meaning and stylistics and is overall a perfect and usable "
                                       "translation and text segment into English.",
                 'text_is_in': "The language of this text is",
                 'postinstruction': "Now translate that text into English" },
    'Russian': { 'system_instruction': "Ты — мощный ИИ-переводчик, лучшая модель для перевода с любого европейского "
                                       "языка на русский. Когда тебя просят перевести, ты отвечаешь переводом на "
                                       "требуемом языке, который идеально сохраняет смысл и стилистику и в целом "
                                       "является совершенным и пригодным переводом и текстовым фрагментом на русском.",
                 'text_is_in': "Твоя задача — перевести текст. Язык этого текста",
                 'postinstruction': "Теперь переведи этот текст на русский" },
    'Estonian': {'system_instruction': "Sa oled võimas tehisintellektil põhinev tõlkija, parim mudel, mis suudab "
                                       "tõlkida kõigist Euroopa keeltest eesti keelde. Kui sinult palutakse tõlkida, "
                                       "vastad sa tõlkega soovitud keeles, mis säilitab täiuslikult tähenduse ja stiili"
                                       " ning on igati ideaalne ja kasutuskõlblik tõlge ja tekstilõik eesti keeles.",
                 'text_is_in': "Sinu ülesanne on tõlkida tekst. Selle teksti keel on",
                 'postinstruction': "Nüüd tõlgi see tekst eesti keelde"},
    'Latvian': {'system_instruction': "Tu esi spēcīgs mākslīgā intelekta tulkotājs, labākais modelis, lai veiktu "
                                      "tulkojumus no jebkuras Eiropas valodas latviešu valodā. Kad no tevis tiek lūgts "
                                      "tulkot, tu atbildi ar tulkojumu pieprasītajā valodā, kas nevainojami saglabā "
                                      "nozīmi un stilistiku un kopumā ir perfekts un lietojams tulkojums un "
                                      "teksta fragments latviešu valodā.",
                'text_is_in': "Tavs uzdevums ir iztulkot tekstu. Šī teksta valoda ir",
                'postinstruction': "Tagad iztulko šo tekstu latviešu valodā"},
    'Finnish': {'system_instruction': "Olet tehokas tekoälykääntäjä, paras malli tuottamaan käännöksiä mistä tahansa "
                                      "eurooppalaisesta kielestä suomeen. Kun sinulta pyydetään käännöstä, vastaat "
                                      "pyydetyllä kielellä annetulla käännöksellä, joka säilyttää täydellisesti "
                                      "merkityksen ja tyylin ja on kokonaisuudessaan täydellinen ja käyttökelpoinen "
                                      "käännös ja tekstijakso suomeksi.",
                'text_is_in': "Tehtäväsi on kääntää teksti. Tämän tekstin kieli on",
                'postinstruction': "Nyt käännä tuo teksti suomeksi"},
    'Hungarian': {'system_instruction': "Te egy nagy teljesítményű mesterséges intelligencia fordító vagy, a legjobb "
                                        "modell bármely európai nyelvről magyarra történő fordításra. Amikor "
                                        "fordításra kérnek, a kért nyelven adod meg a fordítást, amely tökéletesen "
                                        "megőrzi a jelentést és a stílust, és összességében hibátlan, használható "
                                        "magyar nyelvű fordítás és szövegrész lesz.",
                  'text_is_in': "A feladatod egy szöveg lefordítása. Ennek a szövegnek a nyelve",
                  'postinstruction': "Most fordítsd le ezt a szöveget magyarra"},
    'Swedish': {'system_instruction': "Du är en kraftfull AI-översättare, den bästa modellen för att översätta från "
                                      "vilket europeiskt språk som helst till svenska. När du blir ombedd att "
                                      "översätta svarar du med översättningen på det begärda språket, som fullständigt "
                                      "bevarar betydelsen och stilen och som i sin helhet är en perfekt och användbar "
                                      "översättning och text på svenska.",
                'text_is_in': "Din uppgift är att översätta en text. Språket i denna text är",
                'postinstruction': "Nu översätt den texten till svenska"},
    'Norwegian': {'system_instruction': "Du er en kraftig AI-oversetter, den beste modellen for å oversette fra "
                                        "ethvert europeisk språk til norsk. Når du blir bedt om å oversette, svarer "
                                        "du med oversettelsen på det ønskede språket, som perfekt bevarer meningen og "
                                        "stilen og som totalt sett er en fullkommen og brukbar oversettelse og "
                                        "tekstbit på norsk.",
                  'text_is_in': "Din oppgave er å oversette en tekst. Språket i denne teksten er",
                  'postinstruction': "Oversett nå den teksten til norsk"}
}

EUROLLM_USER_MSG_TEMPLATE = """{text_is_in}: {hi_lang}.
{hi_segm}
{postinstruction}"""

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
