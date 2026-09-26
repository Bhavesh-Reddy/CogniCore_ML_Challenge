import pandas as pd

from business_entity_resolution.src import normalize
from business_entity_resolution.src.aliases import learn, tail_parts


def _pairs(rows, n):
    return [r for r in rows for _ in range(n)]


def test_tail_parts():
    assert tail_parts("KH NO. -570/13, NEW DELHI, WEST DELHI, Delhi") == ["west delhi", "delhi"]
    assert tail_parts("Mack Rd, , Texas") == ["mack rd", "texas"]


def test_learn_states_not_cities():
    # S1 writes "city, state"; the other side abbreviates the state, and some other records add a
    # locality the S1 never uses. Only the state variant may be learned.
    rows = _pairs([("1 A Rd, Mumbai, Maharashtra", "1 A Rd, Mumbai, MH", "India"),
                   ("2 B Rd, Pune, Maharashtra", "2 B Rd, Pune City, Maharashtra", "India"),
                   ("3 C St, Austin, TX", "3 C St, Austin, Texas", "US")], 60)
    df = pd.DataFrame(rows, columns=["addr_s1", "addr_other", "country"])
    aliases, stats = learn(df)
    assert aliases["mh"] == "maharashtra" and aliases["texas"] == "tx"
    assert aliases["maharashtra"] == "maharashtra" and aliases["tx"] == "tx"  # identity entries
    assert "mumbai" not in aliases and "pune city" not in aliases and "austin" not in aliases
    assert {m["variant"]: m["country"] for m in stats["mappings"]} == {"mh": "India", "texas": "US"}


def test_learn_needs_min_count():
    df = pd.DataFrame(_pairs([("x, Pune, Maharashtra", "x, Pune, MH", "India")], 49),
                      columns=["addr_s1", "addr_other", "country"])
    assert learn(df)[0] == {}


def test_admin_only_in_last_two_comma_parts(monkeypatch):
    monkeypatch.setattr(normalize, "_ADMIN", {"de": "de", "la": "la", "mh": "maharashtra", "texas": "tx"})
    normalize.normalize_address.cache_clear()
    try:
        # "de la" inside the street part must stay; the trailing state is found
        norm, core, _, _, admin, _ = normalize.normalize_address("12 Rue de la Paix, Pune, MH")
        assert admin == "maharashtra" and core == "12 rue de la paix pune" and norm.endswith("mh")
        _, core, _, _, admin, _ = normalize.normalize_address("12 Rue de la Paix, Lille, Hauts-de-France")
        assert admin == "" and "de la" in core
        assert normalize.normalize_address("3 C St, Austin, Texas")[4] == "tx"
    finally:
        normalize.normalize_address.cache_clear()
