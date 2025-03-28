"""
Convert lang. codes between different schemas

NLLB uses codes like "eng_Latn": ISO-639-3 and script

MADLAD uses codes like "<2en>": ISO-639-1 for where it's available, ISO-639-3 elsewhere
(and some codes include the script, but we'll ignore them here)

Functions at the end of the file (any_to_nllb, any_to_madlad) should
cope with a lang code in any style ('en', 'eng', 'eng_Latn', '<2en>', '<2eng>', etc)
and convert them to corresponding representations (NLLB/MADLAD).
"""
from collections import defaultdict

SMUGRI_LOW = "fkv,izh,kca,koi,kpv,krl,liv,lud,mdf,mhr,mns,mrj,myv,olo,sjd,sje,sju,sma,sme,smj,smn,sms,udm,vep,vot,vro"
SMUGRI_HIGH = "deu,eng,est,fin,hun,lvs,nor,rus,swe"
SMUGRI = SMUGRI_HIGH + "," + SMUGRI_LOW

import pycountry

# madlad all codes
MADLAD_CODES = ['<2meo>', '<2lo>', '<2Grek>', '<2ada>', '<2ps>', '<2arn>', '<2Armn>', '<2to>', '<2raj>', '<2bas>', '<2ny>', '<2>', '<2zza>', '<2Thai>', '<2kaa_Latn>', '<2yap>', '<2en_xx_simple>', '<2ta>', '<2bg_Latn>', '<2mkn>', '<2lhu>', '<2gu_Latn>', '<2nzi>', '<2uz>', '<2pis>', '<2cfm>', '<2min>', '<2fon>', '<2tn>', '<2msi>', '<2sw>', '<2Tfng>', '<2teo>', '<2taj>', '<2pap>', '<2sd>', '<2Jpan>', '<2tca>', '<2sr>', '<2an>', '<2fr>', '<2gor>', '<2az>', '<2qvi>', '<2pck>', '<2cak>', '<2ltg>', '<2sah>', '<2tly_IR>', '<2ts>', '<2yo>', '<2hne>', '<2bzj>', '<2tuc>', '<2sh>', '<2da>', '<2gui>', '<2translate>', '<2et>', '<2sja>', '<2nhe>', '<2scn>', '<2dje>', '<2pt>', '<2nog>', '<2fil>', '<2mai>', '<2lb>', '<2bm>', '<2Guru>', '<2gom>', '<2hr>', '<2kg>', '<2uk>', '<2rw>', '<2izz>', '<2Telu>', '<2wuu>', '<2Deva>', '<2or>', '<2is>', '<2om>', '<2iso>', '<2sn>', '<2kjh>', '<2tbz>', '<2suz>', '<2bjn>', '<2lv>', '<2mfe>', '<2tcy>', '<2tyz>', '<2ksw>', '<2nds_NL>', '<2ms>', '<2mam>', '<2ubu>', '<2hil>', '<2mh>', '<2gl>', '<2bew>', '<2ilo>', '<2kbd>', '<2toj>', '<2quf>', '<2jam>', '<2Beng>', '<2tyv>', '<2lmo>', '<2ace>', '<2cab>', '<2sq>', '<2ug>', '<2kac>', '<2ay>', '<2mag>', '<2Arab>', '<2mrj>', '<2cs>', '<2bci>', '<2doi>', '<2zu>', '<2ndc_ZW>', '<2smt>', '<2ho>', '<2ss>', '<2he>', '<2twu>', '<2kjg>', '<2pag>', '<2Latn>', '<2gym>', '<2sus>', '<2zh_Latn>', '<2mps>', '<2lg>', '<2ko>', '<2se>', '<2guc>', '<2mr>', '<2mwl>', '<2dwr>', '<2din>', '<2ffm>', '<2maz>', '<2nia>', '<2nl>', '<2Knda>', '<2jv>', '<2noa>', '<2udm>', '<2kr>', '<2de>', '<2ar>', '<2ZW>', '<2dln>', '<2mn>', '<2ml>', '<2crh>', '<2ha>', '<2ks>', '<2qvc>', '<2fur>', '<2myv>', '<2nv>', '<2ak>', '<2Gujr>', '<2cce>', '<2nso>', '<2sg>', '<2rmc>', '<2mas>', '<2mni>', '<2frp>', '<2my>', '<2xal>', '<2th>', '<2bik>', '<2bho>', '<2inb>', '<2Mlym>', '<2oj>', '<2back_translated>', '<2tet>', '<2gsw>', '<2ff>', '<2hy>', '<2otq>', '<2el>', '<2agr>', '<2br>', '<2alt>', '<2tzo>', '<2chm>', '<2transliterate>', '<2hu>', '<2btx>', '<2vi>', '<2iba>', '<2bg>', '<2gub>', '<2li>', '<2ace_Arab>', '<2qub>', '<2ktu>', '<2bru>', '<2bbc>', '<2ca>', '<2hvn>', '<2sat_Latn>', '<2ku>', '<2shn>', '<2djk>', '<2krc>', '<2io>', '<2ig>', '<2chk>', '<2sm>', '<2Mymr>', '<2Kore>', '<2ary>', '<2lu>', '<2fa>', '<2spp>', '<2af>', '<2ti>', '<2Tibt>', '<2emp>', '<2enq>', '<2kl>', '<2be>', '<2srn>', '<2ms_Arab_BN>', '<2kri>', '<2gd>', '<2mk>', '<2syr>', '<2kmz_Latn>', '<2CA>', '<2ium>', '<2abt>', '<2ngu>', '<2tab>', '<2it>', '<2ru>', '<2ann>', '<2msm>', '<2fo>', '<2ne>', '<2akb>', '<2kv>', '<2jac>', '<2ceb>', '<2ang>', '<2tdx>', '<2tr>', '<2kbp>', '<2mgh>', '<2az_RU>', '<2acf>', '<2tg>', '<2dov>', '<2pau>', '<2mg>', '<2fuv>', '<2nn>', '<2Hant>', '<2hui>', '<2ml_Latn>', '<2ja>', '<2lus>', '<2te>', '<2qu>', '<2rom>', '<2tsg>', '<2el_Latn>', '<2cr_Latn>', '<2ur>', '<2fi>', '<2shp>', '<2brx>', '<2laj>', '<2sda>', '<2lij>', '<2st>', '<2bn>', '<2zxx_xx_dtynoise>', '<2yua>', '<2no>', '<2fr_CA>', '<2miq>', '<2trp>', '<2es>', '<2ch>', '<2mass>', '<2os>', '<2bts>', '<2ady>', '<2lrc>', '<2seh>', '<2adh>', '<2new>', '<2mak>', '<2grc>', '<2nus>', '<2tzj>', '<2nut>', '<2gu>', '<2oc>', '<2ppk>', '<2Hans>', '<2tzh>', '<2si>', '<2wo>', '<2nyu>', '<2Hebr>', '<2mad>', '<2tll>', '<2kr_Arab>', '<2pon>', '<2mbt>', '<2kw>', '<2bjn_Arab>', '<2gn>', '<2eu>', '<2dz>', '<2kaa>', '<2crh_Latn>', '<2te_Latn>', '<2ky>', '<2kn_Latn>', '<2kum>', '<2fip>', '<2ksd>', '<2sk>', '<2NL>', '<2ctd_Latn>', '<2Khmr>', '<2gbm>', '<2Cans>', '<2haw>', '<2gag>', '<2Taml>', '<2cnh>', '<2bim>', '<2ms_Arab>', '<2Thaa>', '<2kha>', '<2tvl>', '<2Cyrl>', '<2chr>', '<2dtp>', '<2ba>', '<2nan_Latn_TW>', '<2ro>', '<2ctu>', '<2Ethi>', '<2zh>', '<2ln>', '<2ve>', '<2xh>', '<2skr>', '<2ber>', '<2niq>', '<2ibb>', '<2jvn>', '<2tks>', '<2av>', '<2ahk>', '<2tk>', '<2tt>', '<2ka>', '<2tsc>', '<2km>', '<2co>', '<2id>', '<2prs>', '<2rki>', '<2kmb>', '<2ks_Deva>', '<2ify>', '<2wal>', '<2arz>', '<2amu>', '<2rm>', '<2pa>', '<2RU>', '<2ce>', '<2hi>', '<2eo>', '<2taq>', '<2ga>', '<2qxr>', '<2la>', '<2bi>', '<2rwo>', '<2dyu>', '<2zh_Hant>', '<2mt>', '<2bqc>', '<2bn_Latn>', '<2zne>', '<2szl>', '<2lt>', '<2sl>', '<2hif>', '<2alz>', '<2ber_Latn>', '<2ckb>', '<2wa>', '<2Cher>', '<2msb>', '<2gom_Latn>', '<2ru_Latn>', '<2crs>', '<2kk>', '<2gvl>', '<2qvz>', '<2bar>', '<2qup>', '<2bgp>', '<2bo>', '<2su>', '<2tzm>', '<2IR>', '<2sv>', '<2srm>', '<2rn>', '<2bus>', '<2jiv>', '<2awa>', '<2gv>', '<2knj>', '<2as>', '<2quc>', '<2en>', '<2sa>', '<2bug>', '<2quy>', '<2hi_Latn>', '<2nds>', '<2kek>', '<2mrw>', '<2kos>', '<2cy>', '<2ta_Latn>', '<2kn>', '<2nr>', '<2ape>', '<2bs>', '<2iu>', '<2nnb>', '<2Geor>', '<2rcf>', '<2meu>', '<2cac>', '<2cuk>', '<2bua>', '<2vec>', '<2so>', '<2fj>', '<2gof>', '<2koi>', '<2cv>', '<2guh>', '<2war>', '<2pl>', '<2cbk>', '<2kj>', '<2dv>', '<2mdf>', '<2fy>', '<2am>', '<2sc>', '<2taq_Tfng>', '<2mi>', '<2zap>', '<2mqy>', '<2yi>', '<2kwi>', '<2hmn>', '<2tiv>', '<2sxn>', '<2hus>', '<2ban>', '<2nij>', '<2tlh>', '<2Orya>', '<2quh>', '<2ee>', '<2ht>', '<2bum>', '<2stq>']

# NLLB all codes
NLLB_CODES = ['ace_Arab', 'ace_Latn', 'acm_Arab', 'acq_Arab', 'aeb_Arab', 'afr_Latn', 'ajp_Arab', 'aka_Latn', 'amh_Ethi', 'apc_Arab', 'arb_Arab', 'ars_Arab', 'ary_Arab', 'arz_Arab', 'asm_Beng', 'ast_Latn', 'awa_Deva', 'ayr_Latn', 'azb_Arab', 'azj_Latn', 'bak_Cyrl', 'bam_Latn', 'ban_Latn', 'bel_Cyrl', 'bem_Latn', 'ben_Beng', 'bho_Deva', 'bjn_Arab', 'bjn_Latn', 'bod_Tibt', 'bos_Latn', 'bug_Latn', 'bul_Cyrl', 'cat_Latn', 'ceb_Latn', 'ces_Latn', 'cjk_Latn', 'ckb_Arab', 'crh_Latn', 'cym_Latn', 'dan_Latn', 'deu_Latn', 'dik_Latn', 'dyu_Latn', 'dzo_Tibt', 'ell_Grek', 'eng_Latn', 'epo_Latn', 'est_Latn', 'eus_Latn', 'ewe_Latn', 'fao_Latn', 'pes_Arab', 'fij_Latn', 'fin_Latn', 'fon_Latn', 'fra_Latn', 'fur_Latn', 'fuv_Latn', 'gla_Latn', 'gle_Latn', 'glg_Latn', 'grn_Latn', 'guj_Gujr', 'hat_Latn', 'hau_Latn', 'heb_Hebr', 'hin_Deva', 'hne_Deva', 'hrv_Latn', 'hun_Latn', 'hye_Armn', 'ibo_Latn', 'ilo_Latn', 'ind_Latn', 'isl_Latn', 'ita_Latn', 'jav_Latn', 'jpn_Jpan', 'kab_Latn', 'kac_Latn', 'kam_Latn', 'kan_Knda', 'kas_Arab', 'kas_Deva', 'kat_Geor', 'knc_Arab', 'knc_Latn', 'kaz_Cyrl', 'kbp_Latn', 'kea_Latn', 'khm_Khmr', 'kik_Latn', 'kin_Latn', 'kir_Cyrl', 'kmb_Latn', 'kon_Latn', 'kor_Hang', 'kmr_Latn', 'lao_Laoo', 'lvs_Latn', 'lij_Latn', 'lim_Latn', 'lin_Latn', 'lit_Latn', 'lmo_Latn', 'ltg_Latn', 'ltz_Latn', 'lua_Latn', 'lug_Latn', 'luo_Latn', 'lus_Latn', 'mag_Deva', 'mai_Deva', 'mal_Mlym', 'mar_Deva', 'min_Latn', 'mkd_Cyrl', 'plt_Latn', 'mlt_Latn', 'mni_Beng', 'khk_Cyrl', 'mos_Latn', 'mri_Latn', 'zsm_Latn', 'mya_Mymr', 'nld_Latn', 'nno_Latn', 'nob_Latn', 'npi_Deva', 'nso_Latn', 'nus_Latn', 'nya_Latn', 'oci_Latn', 'gaz_Latn', 'ory_Orya', 'pag_Latn', 'pan_Guru', 'pap_Latn', 'pol_Latn', 'por_Latn', 'prs_Arab', 'pbt_Arab', 'quy_Latn', 'ron_Latn', 'run_Latn', 'rus_Cyrl', 'sag_Latn', 'san_Deva', 'sat_Beng', 'scn_Latn', 'shn_Mymr', 'sin_Sinh', 'slk_Latn', 'slv_Latn', 'smo_Latn', 'sna_Latn', 'snd_Arab', 'som_Latn', 'sot_Latn', 'spa_Latn', 'als_Latn', 'srd_Latn', 'srp_Cyrl', 'ssw_Latn', 'sun_Latn', 'swe_Latn', 'swh_Latn', 'szl_Latn', 'tam_Taml', 'tat_Cyrl', 'tel_Telu', 'tgk_Cyrl', 'tgl_Latn', 'tha_Thai', 'tir_Ethi', 'taq_Latn', 'taq_Tfng', 'tpi_Latn', 'tsn_Latn', 'tso_Latn', 'tuk_Latn', 'tum_Latn', 'tur_Latn', 'twi_Latn', 'tzm_Tfng', 'uig_Arab', 'ukr_Cyrl', 'umb_Latn', 'urd_Arab', 'uzn_Latn', 'vec_Latn', 'vie_Latn', 'war_Latn', 'wol_Latn', 'xho_Latn', 'ydd_Hebr', 'yor_Latn', 'yue_Hant', 'zho_Hans', 'zho_Hant', 'zul_Latn']

MDL_NLLB = "MDL_NLLB"
MDL_MADLAD = "MDL_MADLAD"
MDL_NEUROTOLGE = "MDL_NEUROTÕLGE"

_iso3_to_script = dict([nllb_code.split("_") for nllb_code in NLLB_CODES])

iso3_to_nllb = { code: f"{code}_{_iso3_to_script[code]}" for code in _iso3_to_script }

iso3_to_nllb['lav'] = "lvs_Latn"
iso3_to_nllb['yid'] = "ydd_Hebr"

for lang in "fkv izh krl liv lud olo sje sju sma sme smj smn sms vep vot vro".split():
    iso3_to_nllb[lang] = f"{lang}_Latn"

for lang in "kca koi kpv mdf mhr mns mrj myv sjd udm".split():
    iso3_to_nllb[lang] = f"{lang}_Cyrl"


_rev_joshi = defaultdict(lambda: "?")

for k in "krl,sma,vep,smj,smn,lud,liv,izh,vot,kca,sms,sje,mns,fkv,sju,sjd".split(","):
    _rev_joshi[k] = "0"
for k in "kpv,sme,mhr,udm,olo,myv,mdf,vro,mrj,koi".split(","):
    _rev_joshi[k] = "1"
for k in SMUGRI_HIGH.split(","):
    _rev_joshi[k] = "2+"


def guess_script(lang):
    return "Unk"


def get_high_set():
    return set(SMUGRI_HIGH.split(",")) - {"deu", "swe"}


def clean_lang(raw_lang):
    if "<2" in raw_lang:
        raw_lang = raw_lang[2:-1]

    if "_" in raw_lang:
        return raw_lang.split("_")[0]
    else:
        return raw_lang


def any_to_base(lang):
    clang = clean_lang(lang)

    res = pycountry.languages.get(alpha_2=clang)

    if res is None:
        return pycountry.languages.get(alpha_3=clang)
    else:
        return res


def base_to_nllb(lang_entry=None, lang_code=None):
    if lang_code is None:
        lang_code = lang_entry.alpha_3

    try:
        #script = iso3_to_script[lang_code]
        return iso3_to_nllb[lang_code]
    except KeyError:
        script = guess_script(lang_code)
        return f"{lang_code}_{script}"


def base_to_madlad(lang_entry=None, lang_code=None):
    if lang_code is None:
        if hasattr(lang_entry, 'alpha_2'):
            lang_code = lang_entry.alpha_2
        else:
            lang_code = lang_entry.alpha_3

    return f"<2{lang_code}>"


def any_to_something(lang, conv_func):
    base = any_to_base(lang)

    if base is None:
        clang = clean_lang(lang)
        return conv_func(None, clang)
    else:
        return conv_func(base)


def run_test(src_list, tgt_list, conv_func, msg_prefix, verbose=False):
    ok_count = 0
    err_count = 0
    fail_count = 0

    for raw_c in src_list:
        try:
            test = conv_func(raw_c)
            if test in tgt_list:
                ok_count += 1
            else:
                fail_count += 1
                if verbose:
                    print("FAIL:", test)
        except KeyError:
            err_count += 1
            if verbose:
                print("ERR:", raw_c)

    print(f"{msg_prefix}: {ok_count} good, {fail_count} fail, {err_count} err")


def any_to_madlad(lang):
    return any_to_something(lang, base_to_madlad)


def any_to_nllb(lang):
    return any_to_something(lang, base_to_nllb)


def any_to_neurotolge(lang):
    l = any_to_base(lang).alpha_3

    return l if l != 'lvs' else 'lv'


def any_to_mdl_type(mdl_type, lang):
    if mdl_type == MDL_NLLB:
        return any_to_nllb(lang)
    elif mdl_type == MDL_MADLAD:
        return any_to_madlad(lang)
    elif mdl_type is None:
        return lang
    else:
        raise ValueError(f"Unknown mdl_type {mdl_type}")

def langs_to_madlad(lang_set):
    return [any_to_madlad(l) for l in lang_set] if lang_set is not None else []


def langs_to_nllb(lang_set):
    return [any_to_nllb(l) for l in lang_set] if lang_set is not None else []


if __name__ == "__main__":
    run_test(NLLB_CODES, MADLAD_CODES, any_to_madlad, "NLLB to MADLAD")
    run_test(NLLB_CODES, NLLB_CODES, any_to_nllb, "NLLB to NLLB")
    run_test(MADLAD_CODES, NLLB_CODES, any_to_nllb, "MADLAD TO NLLB")
    run_test(MADLAD_CODES, MADLAD_CODES, any_to_madlad, "MADLAD TO MADLAD")


def is_nllb(object):
    """
    Check if the object is an NLLB model or tokenizer
    """
    name = object.__class__.__name__.lower()
    return "m2m100" in name or "nllb" in name


def is_madlad(object):
    """
    Check if the object is a MADLAD model or tokenizer
    """
    return "t5" in object.__class__.__name__.lower()


def get_mdl_type(obj):
    obj = obj.module if hasattr(obj, "module") else obj

    if is_nllb(obj):
        return MDL_NLLB
    elif is_madlad(obj):
        return MDL_MADLAD
    else:
        raise ValueError(f"Object {obj} is not supported")


def langs_to_mdl_type(mdl_type, lang_set):
    if mdl_type == MDL_NLLB:
        return langs_to_nllb(lang_set)
    elif mdl_type == MDL_MADLAD:
        return langs_to_madlad(lang_set)
    else:
        raise ValueError(f"Model type {mdl_type} is not supported")


def get_joshi_class(lang_code):
    norm_code = any_to_base(lang_code)

    if norm_code is None:
        return "?"
    else:
        norm_code = norm_code.alpha_3

    return _rev_joshi[norm_code]

def lang_set_maybe_smugri(lang_def):
    if lang_def == "smugri-low":
        preresult = SMUGRI_LOW
    elif lang_def == "smugri-high":
        preresult = SMUGRI_HIGH
    elif lang_def == "smugri":
        preresult = SMUGRI
    else:
        preresult = lang_def

    return set(preresult.split(","))


def smugri_back(lang_list):
    sll = sorted(lang_list)

    sll_str = ",".join(sll)

    if sll_str == SMUGRI_LOW:
        return "smugri-low"
    elif sll_str == SMUGRI_HIGH:
        return "smugri-high"
    elif sll_str == SMUGRI:
        return "smugri-full"
    else:
        return sll_str
