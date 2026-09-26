import pandas as pd

from business_entity_resolution.src.normalize import (
    OUT_COLS, basic_clean, normalize_address, normalize_frame, normalize_name, part_admin)


def name(raw):
    n = normalize_name(raw)
    return {"norm": n[0], "core": n[1], "nospace": n[2], "url": n[3], "nonlatin": n[4]}


def addr(raw):
    a = normalize_address(raw)
    return {"norm": a[0], "core": a[1], "nums": a[2], "first": a[3], "admin": a[4], "empty": a[5]}


def test_basic_clean():
    assert basic_clean("  Àmicale & Pàrenthese--Co.  ") == "amicale and parenthese co"


# ---------------------------------------------------------------- names (real examples, facts.md section 6)

def test_accent_and_legal_suffix_same_core():
    assert name("Kochar Góld Private")["core"] == name("Kochar Gold Private Limited")["core"] == "kochar gold"


def test_website_name_same_nospace():
    assert name("lifeinvestments.com")["nospace"] == name("Life Investments")["nospace"] == "lifeinvestments"
    assert name("lifeinvestments.com")["url"] == 1 and name("Life Investments")["url"] == 0


def test_hashtag_name_same_nospace():
    assert name("#empiremolecular")["nospace"] == name("Empire Molecular Inc.")["nospace"] == "empiremolecular"


def test_dotted_llc_same_core():
    assert name("Summit Society L.L.C.")["core"] == name("Summit Society LLC")["core"] == "summit society"


def test_pipe_suffix_and_junk_prefix_removed():
    n = name("-- Holloway Peak Inc Seafood | www.hollowaypeak.com")
    assert n["core"] == "holloway peak seafood" and n["url"] == 1


def test_duplicate_tokens_and_honorific():
    assert name("Shri Kochar Kochar Traders")["core"] == "kochar traders"


def test_core_falls_back_to_norm_when_empty():
    n = name("The Company")
    assert n["norm"] == "the company" and n["core"] == "the company"


def test_transliterated_hindi_legal_forms_removed():
    # real train names; anyascii gives "praivet limited" and "pra li"
    assert name("जय एंटरप्राइजेज प्राइवेट लिमिटेड")["core"] == "jy emtrpraijej"
    assert name("गुरु बिजनेस प्रा. लि.")["core"] == "guru bijnes"


def test_nonlatin_flag():
    assert name("राम मार्केटिंग प्राइवेट लिमिटेड")["nonlatin"] == 1
    assert name("Àmicale Góld")["nonlatin"] == 0  # accented Latin is still Latin


# ---------------------------------------------------------------- addresses

def test_leading_zeros_stripped():
    a = addr("00272 LAGO GRANDE DRIVE")
    assert a["first"] == "272" and a["norm"] == "272 lago grande drive"


def test_us_abbreviation_same_norm():
    assert addr("41 BIRCH HILL DR")["norm"] == addr("41 Birch Hill Drive")["norm"] == "41 birch hill drive"


def test_french_r_is_rue():
    assert "rue" in addr("63 R. DE DIEPPE, LILLE, Hauts-de-France")["norm"].split()


def test_null_token_removed():
    a = addr("NULL, 12 Main Road, null")
    assert "null" not in a["norm"].split() and a["norm"] == "12 main road"
    assert addr("NULL")["empty"] == 1 and addr("")["empty"] == 1


def test_st_ambiguity_documented():
    # One mapping for both countries: "st" -> street, "ste" -> suite (never saint / sainte).
    # Both sides of a pair go through the same rule, so a French "St Michel" still matches itself.
    assert addr("105 Elm St, Ste 200")["norm"] == "105 elm street suite 200"
    assert addr("15 St Michel")["norm"] == "15 street michel"


def test_numbers_and_ranges():
    a = addr("KH NO. -570/13, 1000 34-1002")
    assert a["nums"] == "570 13 1000 34 1002" and a["first"] == "570"
    assert addr("Mack Rd, Haltom City, Texas")["first"] == ""
    # digit runs inside tokens count: "155C" gives 155, not the unit number
    assert addr("155C Sullivant Dr, Unit 106, Marysville, Ohio")["nums"] == "155 106"
    assert addr("6B/79- Mukundapur Road")["first"] == "6"


def test_admin_split():
    admin = {"texas": "tx", "tx": "tx", "west bengal": "wb"}
    assert part_admin(["texas"], admin) == ("tx", [True])
    assert part_admin(["west", "bengal"], admin) == ("wb", [True, True])
    assert part_admin(["tx", "75001"], admin) == ("tx", [True, False])  # zip stays in addr_core
    assert part_admin(["haltom", "city", "texas"], admin) == (None, [False, False, False])  # whole part only
    assert part_admin(["a", "b"], {}) == (None, [False, False])


def test_normalize_frame_schema():
    df = pd.DataFrame({"entity_id": ["S2-1"], "business_name": ["Acme LLC"],
                       "business_address": ["NULL"], "country": ["France"], "source": [2]})
    out = normalize_frame(df)
    assert list(out.columns) == OUT_COLS
    assert out.loc[0, "addr_empty"] == 1 and str(out["source"].dtype) == "int8"
