"""Text normalisation of names and addresses (contracts.md section 3, noise patterns in facts.md section 6).

Usage: python -m business_entity_resolution.src.normalize --split {train,test} [--limit N] [--force]

Country is never used here: the same rules run on every record, so France is handled like any other country.
A --limit run writes to work/norm/limit{N}/ so it can never be mistaken for a full output.
"""
import argparse
import json
import re
import time
import unicodedata
from functools import lru_cache
from multiprocessing import Pool
from typing import Dict, Optional, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from anyascii import anyascii

from . import config
from .io_utils import peak_mem_gb

CHUNK_ROWS = 500_000
NORM = config.WORK / "norm"
ALIASES = config.WORK / "aliases.json"
OUT_COLS = ["entity_id", "source", "country", "name_norm", "name_core", "name_nospace", "name_has_url",
            "name_nonlatin", "addr_norm", "addr_core", "addr_nums", "addr_first_num", "addr_admin", "addr_empty"]

_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_DIGITS = re.compile(r"\d+")


def basic_clean(s: str) -> str:
    """anyascii -> lowercase -> '&' to ' and ' -> non [a-z0-9] to space -> collapse spaces -> strip."""
    s = anyascii(s).lower().replace("&", " and ")
    return _NON_ALNUM.sub(" ", s).strip()


# ----------------------------------------------------------------------------- names

_DOMAIN_END = r"\.(?:com|in|org|net|co|fr)\b"
_URL_PARTS = re.compile(r"https?://|www\.|" + _DOMAIN_END, re.IGNORECASE)
_HAS_URL = re.compile(r"https?://|www\.|[a-z0-9]" + _DOMAIN_END, re.IGNORECASE)

LEGAL_FORMS = {"llc", "inc", "incorporated", "corp", "corporation", "co", "company", "ltd", "limited", "pvt",
               "private", "llp", "lp", "pc", "plc", "pllc", "sarl", "sas", "sasu", "eurl", "sa", "sci", "snc",
               "cie", "gmbh",
               "praivet"}  # anyascii of Devanagari "प्राइवेट" (facts.md section 7)
FILLER_WORDS = {"center", "centre", "services", "service", "group", "and", "the", "of", "fka", "aka", "dba"}
HONORIFICS = {"shri", "sri", "smt", "mr", "mrs", "dr", "md", "dmd"}
NAME_STOP = LEGAL_FORMS | FILLER_WORDS | HONORIFICS
# spaced-out forms that basic_clean produces from "L.L.C.", "P.C.", "F/K/A", "M.D.",
# and "pra li" from Devanagari "प्रा. लि." (Pvt. Ltd.)
NAME_STOP_PHRASES = re.compile(r"\b(?:l l c|p c|f k a|a k a|m d|pra li)\b")


def _has_nonlatin(raw: str) -> int:
    if raw.isascii():
        return 0
    for ch in raw:
        if ch.isalpha() and not unicodedata.name(ch, "").startswith("LATIN"):
            return 1
    return 0


def _dedupe_adjacent(tokens):
    out = []
    for t in tokens:
        if not out or out[-1] != t:
            out.append(t)
    return out


@lru_cache(maxsize=2_000_000)
def normalize_name(raw: str) -> Tuple[str, str, str, int, int]:
    """(name_norm, name_core, name_nospace, name_has_url, name_nonlatin)."""
    s = raw.split("|", 1)[0]  # "Acme | www.acme.com" -> "Acme"
    s = _URL_PARTS.sub(" ", s)
    name_norm = basic_clean(s)
    core = NAME_STOP_PHRASES.sub(" ", name_norm)
    tokens = _dedupe_adjacent([t for t in core.split() if t not in NAME_STOP])
    name_core = " ".join(tokens) or name_norm
    return (name_norm, name_core, name_core.replace(" ", ""),
            int(bool(_HAS_URL.search(raw))), _has_nonlatin(raw))


# ----------------------------------------------------------------------------- addresses

# Whole-token expansions. Where US and France disagree we keep ONE mapping so both sides of a pair
# normalise the same way: "st" -> street (not saint), "ste" -> suite (not sainte). US + India are
# 100% of train and 85% of test S1 (facts.md section 4).
ADDR_ABBR: Dict[str, str] = {
    # US
    "st": "street", "rd": "road", "ave": "avenue", "av": "avenue", "blvd": "boulevard", "dr": "drive",
    "ln": "lane", "ct": "court", "cir": "circle", "pl": "place", "pkwy": "parkway", "hwy": "highway",
    "trl": "trail", "ter": "terrace", "sq": "square", "n": "north", "s": "south", "e": "east", "w": "west",
    "ste": "suite", "apt": "apartment", "so": "south",
    # India
    "no": "number", "h": "house", "hno": "house", "nr": "near", "opp": "opposite", "flr": "floor",
    "blk": "block", "sec": "sector", "mg": "marg",
    # France (av, pl, st, ste already mapped above)
    "r": "rue", "bd": "boulevard", "all": "allee", "imp": "impasse", "che": "chemin", "rte": "route",
    "fbg": "faubourg",
}

_ADMIN: Optional[Dict[str, str]] = None
_ADMIN_MAX_TOKENS = 1


def _admin_table() -> Dict[str, str]:
    global _ADMIN, _ADMIN_MAX_TOKENS
    if _ADMIN is None:
        _ADMIN = json.loads(ALIASES.read_text(encoding="utf-8"))["admin"] if ALIASES.exists() else {}
        _ADMIN_MAX_TOKENS = max((len(k.split()) for k in _ADMIN), default=1)
    return _ADMIN


def split_admin(tokens, admin: Dict[str, str], max_len: int) -> Tuple[str, str]:
    """Greedy longest match of alias phrases -> (addr_admin, addr_core)."""
    if not admin:
        return "", " ".join(tokens)
    found, core, i = [], [], 0
    while i < len(tokens):
        for n in range(min(max_len, len(tokens) - i), 0, -1):
            canon = admin.get(" ".join(tokens[i:i + n]))
            if canon is not None:
                if canon not in found:
                    found.append(canon)
                i += n
                break
        else:
            core.append(tokens[i])
            i += 1
    return " ".join(found), " ".join(core)


@lru_cache(maxsize=2_000_000)
def normalize_address(raw: str) -> Tuple[str, str, str, str, str, int]:
    """(addr_norm, addr_core, addr_nums, addr_first_num, addr_admin, addr_empty)."""
    tokens = []
    for t in basic_clean(raw).split():
        if t == "null":
            continue
        if t.isdigit():
            t = t.lstrip("0") or "0"
        tokens.append(ADDR_ABBR.get(t, t))
    addr_norm = " ".join(tokens)
    # every digit run, so house numbers like "155c" or "6b/79" still give 155 / 6
    nums = [m.lstrip("0") or "0" for m in _DIGITS.findall(addr_norm)]
    addr_admin, addr_core = split_admin(tokens, _admin_table(), _ADMIN_MAX_TOKENS)
    return addr_norm, addr_core, " ".join(nums), (nums[0] if nums else ""), addr_admin, int(not addr_norm)


# ----------------------------------------------------------------------------- frames and CLI

def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    names = [normalize_name(x) for x in df["business_name"].tolist()]
    addrs = [normalize_address(x) for x in df["business_address"].tolist()]
    out = pd.DataFrame(names, columns=["name_norm", "name_core", "name_nospace", "name_has_url", "name_nonlatin"])
    a = pd.DataFrame(addrs, columns=["addr_norm", "addr_core", "addr_nums", "addr_first_num", "addr_admin",
                                     "addr_empty"])
    out = pd.concat([out, a], axis=1)
    out.insert(0, "entity_id", df["entity_id"].to_numpy())
    out.insert(1, "source", df["source"].to_numpy())
    out.insert(2, "country", df["country"].to_numpy())
    for c in ("name_has_url", "name_nonlatin", "addr_empty"):
        out[c] = out[c].astype("int8")
    out["source"] = out["source"].astype("int8")
    for c in OUT_COLS:
        if c not in ("source", "name_has_url", "name_nonlatin", "addr_empty"):
            out[c] = out[c].astype(str)
    return out[OUT_COLS]


def _split_frame(df: pd.DataFrame, n: int):
    step = -(-len(df) // n)
    return [df.iloc[i:i + step].reset_index(drop=True) for i in range(0, len(df), step)]


def normalize_file(src, dst, limit: Optional[int], pool: Pool) -> int:
    pf = pq.ParquetFile(src)
    cols = ["entity_id", "business_name", "business_address", "country", "source"]
    writer, n_done = None, 0
    tmp = dst.with_suffix(".tmp")
    try:
        for batch in pf.iter_batches(batch_size=CHUNK_ROWS, columns=cols):
            df = batch.to_pandas()
            if limit is not None:
                df = df.iloc[:limit - n_done]
            parts = pool.map(normalize_frame, _split_frame(df, config.N_JOBS))
            table = pa.Table.from_pandas(pd.concat(parts, ignore_index=True), preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(tmp, table.schema)
            writer.write_table(table)
            n_done += len(df)
            if limit is not None and n_done >= limit:
                break
    finally:
        if writer is not None:
            writer.close()
    tmp.replace(dst)
    return n_done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--limit", type=int, default=None, help="rows per source file (dev runs)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out_dir = NORM / f"limit{args.limit}" if args.limit else NORM
    out_dir.mkdir(parents=True, exist_ok=True)
    _admin_table()
    print(f"normalize --split {args.split}: aliases {'loaded' if _ADMIN else 'not found, addr_admin empty'}, "
          f"{config.N_JOBS} workers -> {out_dir}", flush=True)
    t_all, rows_all = time.time(), 0
    with Pool(config.N_JOBS) as pool:
        for k in (1, 2, 3):
            dst = out_dir / f"{args.split}_s{k}.parquet"
            if dst.exists() and not args.force:
                print(f"  {dst.name}: exists, skipping (use --force)")
                continue
            t = time.time()
            n = normalize_file(config.WORK / "raw" / f"{args.split}_s{k}.parquet", dst, args.limit, pool)
            dt = time.time() - t
            rows_all += n
            print(f"  {dst.name}: {n:,} rows in {dt:.1f}s ({n / dt:,.0f} rows/s)", flush=True)
    dt = time.time() - t_all
    if rows_all:
        print(f"normalize: {rows_all:,} rows in {dt:.1f}s ({rows_all / dt:,.0f} rows/s), "
              f"peak memory (main process) {peak_mem_gb()} GB")


if __name__ == "__main__":
    main()
