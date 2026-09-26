"""Learn admin-region aliases (state names, codes, native-script forms) from matched TRAIN pairs only.

Usage: python -m business_entity_resolution.src.aliases [--force]

Rules (thresholds in config.ALIAS_*):
- Sample ALIAS_SAMPLE gt pairs (seed SEED), take the last 2 comma-parts of both addresses after basic_clean.
- Canonical values are S1 forms that are the S1's LAST comma-part at least ALIAS_MIN_COUNT times and in at
  least ALIAS_LAST_SHARE of the S1 records containing them (states; cities sit before the state).
- Variant v -> canonical c when v is not itself canonical, is not used by S1 as a city, is the last part of
  the other address in at least ALIAS_LAST_SHARE of its occurrences, and count(v, c) >= ALIAS_MIN_COUNT and
  >= ALIAS_MIN_SHARE of v's occurrences. count(v, c) only counts records where v does not already appear on
  the S1 side. The canonical side is always the S1 form.
- Every canonical also maps to itself.
Writes work/aliases.json ({"admin": {variant: canonical}}) and reports/aliases.md.
"""
import argparse
import json
import time
from collections import Counter, defaultdict

import pandas as pd

from . import config
from .io_utils import peak_mem_gb
from .normalize import basic_clean

OUT = config.WORK / "aliases.json"
REPORT = config.REPORTS / "aliases.md"
SANITY = [("mh", "maharashtra"), ("mharastr", "maharashtra"), ("tx", "texas"), ("texas", "tx"),
          ("krnatk", "karnataka"), ("dl", "delhi")]


def tail_parts(addr: str, n: int = 2) -> list:
    parts = [p for p in (basic_clean(x) for x in addr.split(",")) if p]
    return parts[-n:]


def load_sample() -> pd.DataFrame:
    gt = pd.read_parquet(config.WORK / "gt_pairs.parquet").sample(config.ALIAS_SAMPLE, random_state=config.SEED)
    s1 = pd.read_parquet(config.WORK / "raw" / "train_s1.parquet", columns=["entity_id", "business_address",
                                                                            "country"])
    s1 = s1[s1["entity_id"].isin(set(gt["s1_id"]))].rename(columns={"entity_id": "s1_id",
                                                                      "business_address": "addr_s1"})
    ids = set(gt["other_id"])
    others = []
    for k in (2, 3):
        o = pd.read_parquet(config.WORK / "raw" / f"train_s{k}.parquet", columns=["entity_id", "business_address"])
        others.append(o[o["entity_id"].isin(ids)])
    other = pd.concat(others).rename(columns={"entity_id": "other_id", "business_address": "addr_other"})
    df = gt.merge(s1, on="s1_id").merge(other, on="other_id")
    assert len(df) == len(gt), f"join lost rows: {len(df):,} of {len(gt):,}"
    return df


def learn(df: pd.DataFrame):
    """Return (aliases dict, stats) from a frame with addr_s1, addr_other, country."""
    occ, co, co_country = Counter(), Counter(), defaultdict(Counter)
    s1_last, s1_any, other_last = Counter(), Counter(), Counter()
    for a1, a2, country in zip(df["addr_s1"], df["addr_other"], df["country"]):
        t1, t2 = tail_parts(a1), tail_parts(a2)
        s1_side = set(t1)
        if t1:
            s1_last[t1[-1]] += 1
        if t2:
            other_last[t2[-1]] += 1
        for x in s1_side:
            s1_any[x] += 1
        for v in set(t2):
            occ[v] += 1
            if v in s1_side:
                continue
            for c in s1_side:
                co[v, c] += 1
                co_country[v, c][country] += 1

    def is_last(counter_last, counter_all, x):
        return counter_last[x] / counter_all[x] >= config.ALIAS_LAST_SHARE

    canon = {c for c, n in s1_last.items() if n >= config.ALIAS_MIN_COUNT and is_last(s1_last, s1_any, c)}
    s1_city = {x for x, n in s1_any.items() if n >= config.ALIAS_MIN_COUNT and not is_last(s1_last, s1_any, x)}
    best = {}
    for (v, c), n in co.items():
        if (c in canon and v not in canon and v not in s1_city and is_last(other_last, occ, v)
                and n >= config.ALIAS_MIN_COUNT and n >= config.ALIAS_MIN_SHARE * occ[v]):
            if v not in best or n > best[v][1]:
                best[v] = (c, n)

    aliases = {c: c for c in sorted(canon)}
    aliases.update({v: c for v, (c, _) in sorted(best.items())})

    def country_of(v, c):
        by = co_country[v, c]
        return by.most_common(1)[0][0] if by else "?"

    mappings = [{"variant": v, "canonical": c, "count": n, "occurrences": occ[v],
                 "country": country_of(v, c)} for v, (c, n) in best.items()]
    mappings.sort(key=lambda m: -m["count"])
    identities = sorted(canon, key=lambda c: -s1_last[c])
    return aliases, {"mappings": mappings, "identities": identities, "s1_last": s1_last}


def sanity(aliases: dict) -> list:
    lines = []
    for v, c in SANITY:
        got = aliases.get(v)
        if got == c and v != c:
            lines.append(f"{v} -> {c}: learned")
        elif got is not None:
            lines.append(f"{v} -> {c}: not learned ({v} maps to {got!r})")
        else:
            lines.append(f"{v} -> {c}: not learned")
    return lines


def write_report(aliases: dict, stats: dict, n_pairs: int, sanity_lines: list, elapsed: float) -> None:
    top = stats["mappings"][:60]
    lines = [
        "# Admin-region aliases (Step 2.1)",
        "",
        f"Learned from {n_pairs:,} sampled matched TRAIN pairs (seed {config.SEED}), no external data. "
        f"{len(stats['identities'])} canonical values (identity entries) + {len(stats['mappings'])} variants "
        f"= {len(aliases)} entries in `work/aliases.json`.",
        "",
        f"Rules: variant count >= {config.ALIAS_MIN_COUNT} and >= {config.ALIAS_MIN_SHARE:.0%} of its "
        f"occurrences; both canonical and variant must be the LAST comma-part in >= "
        f"{config.ALIAS_LAST_SHARE:.0%} of their records (this keeps states and drops cities and localities); "
        "canonical = the S1 form. See the aliases.py docstring.",
        "",
        "**France:** there is no France data in train, so no France aliases were learned. French regions and "
        "departments (e.g. Hauts-de-France / Nord) stay as plain address tokens. Hand-written French street-type "
        "expansions (r -> rue, av -> avenue, bd -> boulevard, ...) are in normalize.py.",
        "",
        "## Sanity check",
        "",
        *[f"- {s}" for s in sanity_lines],
        "",
        f"## Top {len(top)} mappings by count, grouped by country",
        "",
        "Country = the S1 country in most of the pairs behind the mapping.",
    ]
    for country in sorted({m["country"] for m in top}):
        lines += ["", f"### {country}", "", "| variant | canonical | count | variant occurrences |", "|---|---|---|---|"]
        lines += [f"| {m['variant']} | {m['canonical']} | {m['count']:,} | {m['occurrences']:,} |"
                  for m in top if m["country"] == country]
    lines += ["", "## Canonical values (identity entries), most frequent first", "",
              ", ".join(f"{c} ({stats['s1_last'][c]:,})" for c in stats["identities"]), "",
              f"Elapsed {elapsed:.1f}s.", ""]
    REPORT.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if OUT.exists() and not args.force:
        print(f"aliases: {OUT} exists, skipping (use --force to rebuild)")
        print("\n".join(sanity(json.loads(OUT.read_text(encoding="utf-8"))["admin"])))
        return
    t0 = time.time()
    df = load_sample()
    print(f"aliases: {len(df):,} sampled pairs joined ({time.time() - t0:.1f}s)", flush=True)
    aliases, stats = learn(df)
    OUT.write_text(json.dumps({"admin": aliases}, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    sanity_lines = sanity(aliases)
    elapsed = time.time() - t0
    write_report(aliases, stats, len(df), sanity_lines, elapsed)
    print(f"aliases: {len(stats['identities'])} canonical + {len(stats['mappings'])} variants -> {OUT}")
    print("sanity:")
    print("\n".join(f"  {s}" for s in sanity_lines))
    print(f"aliases: done in {elapsed:.1f}s, peak memory {peak_mem_gb()} GB -> {REPORT}")


if __name__ == "__main__":
    main()
