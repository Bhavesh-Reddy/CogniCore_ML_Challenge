"""Candidate generation with DuckDB (SKILL.md pipeline step 4, contracts.md sections 3, 4 and 6).

Usage: python -m business_entity_resolution.src.blocking --split {train,test} [--shards 0,1,...] [--force]
       [--show-misses N]

Keys (bit in key_hits):
  K1 (1)  first_num | name_core token (len >= 2)
  K2 (2)  first_num | addr_core alphabetic token (len >= 3)          address-only: website/script/alias names
  K3 (4)  name_core token (len >= 3) | addr_core alpha token (len >= 4)   first 4 x first 6 tokens; no number
  K4 (8)  first 8 chars of name_nospace (len >= 5)                   "lifeinvestments.com" vs "Life Investments"
Every key is hashed together with the country, so candidates never cross countries (facts.md section 5).
A (key_type, key) group larger than BLOCK_CAP on the index side (all S2+S3) or the query side (ALL S1 of
the split, not just the requested shards, so dev and full runs drop the same groups) is dropped.
Two stages per S1:
  1. (speed) keep the PREFILTER_K candidates with the highest key_weight = sum over shared keys of
     1 / index group size (rare shared keys first). Scoring every pair is ~170M pairs and ~20 min per
     train shard; on val shard 0 this stage costs <= 0.008 recall (measured, see reports/blocking_train.json).
  2. cheap_score = (0.5 * token_set_ratio(name_core) + 0.3 * token_set_ratio(addr_core)
                    + 0.2 * 100 * [first_num equal and non-empty]) / 100; keep the top TOPK
     (ties: lower record id first).
Output adds key_weight (float32) to the contract columns (contracts.md section 3).

Built once per split under work/blocking/{split}/ (recs, index_keys, query_keys); --force rebuilds them.
"""
import argparse
import json
import shutil
import time
from pathlib import Path
from typing import List, Optional

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
from rapidfuzz import fuzz, process

from . import config
from .io_utils import TreeMemWatch, peak_mem_gb

KEY_NAMES = {1: "K1", 2: "K2", 4: "K3", 8: "K4"}
SCORE_CHUNK_S1 = 8_000           # S1 per scoring chunk (~10M pairs); an S1's candidates stay in one chunk


def _dirs(split: str):
    base = config.WORK / "blocking" / split
    cand = config.WORK / "cand" / split
    base.mkdir(parents=True, exist_ok=True)
    cand.mkdir(parents=True, exist_ok=True)
    return base, cand


def _con() -> duckdb.DuckDBPyConnection:
    tmp = config.WORK / "duckdb_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET memory_limit = '{config.DUCKDB_MEMORY}'")
    con.execute(f"SET temp_directory = '{tmp.as_posix()}'")
    con.execute(f"SET threads = {config.N_JOBS}")
    con.execute("SET preserve_insertion_order = false")
    return con


def _p(path: Path) -> str:
    return path.as_posix()


# ----------------------------------------------------------------------------- keys

KEY_SQL = """
WITH t AS (
  SELECT rid, country, addr_first_num AS fn, name_nospace,
         list_distinct(list_filter(string_split(name_core, ' '), x -> length(x) >= 2)) AS n2,
         list_distinct(list_filter(string_split(name_core, ' '), x -> length(x) >= 3)[1:4]) AS n3,
         list_distinct(list_filter(string_split(addr_core, ' '),
                                   x -> length(x) >= 3 AND regexp_full_match(x, '[a-z]+'))) AS a3,
         list_distinct(list_filter(string_split(addr_core, ' '),
                                   x -> length(x) >= 4 AND regexp_full_match(x, '[a-z]+'))[1:6]) AS a4
  FROM read_parquet('{recs}') WHERE source {source_filter}
),
k1 AS (SELECT rid, country, 1 AS bit, fn || '|' || unnest(n2) AS k FROM t WHERE fn <> ''),
k2 AS (SELECT rid, country, 2 AS bit, fn || '|' || unnest(a3) AS k FROM t WHERE fn <> ''),
k3a AS (SELECT rid, country, a4, unnest(n3) AS tok FROM t),
k3 AS (SELECT rid, country, 4 AS bit, tok || '|' || unnest(a4) AS k FROM k3a),
k4 AS (SELECT rid, country, 8 AS bit, substr(name_nospace, 1, 8) AS k FROM t WHERE length(name_nospace) >= 5),
allk AS (SELECT * FROM k1 UNION ALL SELECT * FROM k2 UNION ALL SELECT * FROM k3 UNION ALL SELECT * FROM k4)
SELECT DISTINCT rid::INTEGER AS rid, bit::UTINYINT AS bit, hash(country || '|' || bit || '|' || k) AS h FROM allk
"""


def build_keys(split: str, force: bool = False) -> dict:
    """recs.parquet (rid + normalised fields), capped index_keys / query_keys parquet files."""
    base, _ = _dirs(split)
    recs, idx, qry = base / "recs.parquet", base / "index_keys.parquet", base / "query_keys.parquet"
    stats_path = base / "key_stats.json"
    if not force and all(p.exists() for p in (recs, idx, qry, stats_path)):
        return json.loads(stats_path.read_text(encoding="utf-8"))
    t0 = time.time()
    con = _con()
    norm = [_p(config.WORK / "norm" / f"{split}_s{k}.parquet") for k in (1, 2, 3)]
    con.execute(f"""
        COPY (SELECT (row_number() OVER () - 1)::INTEGER AS rid, entity_id, source, country, name_core,
                     name_nospace, addr_core, addr_first_num
              FROM read_parquet({norm}))
        TO '{_p(recs)}' (FORMAT parquet)""")
    raw_idx, raw_qry = base / "_index_raw.parquet", base / "_query_raw.parquet"
    for path, flt in ((raw_idx, "IN (2, 3)"), (raw_qry, "= 1")):
        con.execute(f"COPY ({KEY_SQL.format(recs=_p(recs), source_filter=flt)}) TO '{_p(path)}' (FORMAT parquet)")
        print(f"  keys {path.name}: {time.time() - t0:.1f}s", flush=True)

    cap = config.BLOCK_CAP
    con.execute(f"""
        CREATE TEMP TABLE keep AS
        SELECT bit, h FROM (SELECT bit, h, count(*) AS n FROM read_parquet('{_p(raw_idx)}') GROUP BY ALL) i
        JOIN (SELECT bit, h, count(*) AS n FROM read_parquet('{_p(raw_qry)}') GROUP BY ALL) q USING (bit, h)
        WHERE i.n <= {cap} AND q.n <= {cap}""")
    for raw, out in ((raw_idx, idx), (raw_qry, qry)):
        con.execute(f"COPY (SELECT k.* FROM read_parquet('{_p(raw)}') k SEMI JOIN keep USING (bit, h)) "
                    f"TO '{_p(out)}' (FORMAT parquet)")

    def per_bit(path):
        return {KEY_NAMES[b]: n for b, n in con.execute(
            f"SELECT bit, count(*) FROM read_parquet('{_p(path)}') GROUP BY bit ORDER BY bit").fetchall()}

    stats = {"key_rows_index_raw": per_bit(raw_idx), "key_rows_query_raw": per_bit(raw_qry),
             "key_rows_index_kept": per_bit(idx), "key_rows_query_kept": per_bit(qry),
             "groups_dropped_over_cap": con.execute(f"""
                SELECT count(*) FROM (SELECT bit, h, count(*) n FROM read_parquet('{_p(raw_idx)}') GROUP BY ALL)
                WHERE n > {cap}""").fetchone()[0],
             "build_s": round(time.time() - t0, 1)}
    raw_idx.unlink()
    raw_qry.unlink()
    stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    con.close()
    print(f"  keys built in {stats['build_s']}s: {stats['key_rows_index_kept']} index rows kept", flush=True)
    return stats


# ----------------------------------------------------------------------------- per shard

def _shard_table(split: str) -> Path:
    return config.WORK / ("splits.parquet" if split == "train" else "test_shards.parquet")


class RecStrings:
    """recs columns as Arrow arrays in rid order, so a chunk of pairs picks its strings with take()
    instead of joining strings in DuckDB (170M pairs per train shard makes those joins spill)."""

    def __init__(self, con: duckdb.DuckDBPyConnection):
        t = con.execute("SELECT entity_id, name_core, addr_core, addr_first_num FROM recs ORDER BY rid").arrow()
        if isinstance(t, pa.RecordBatchReader):
            t = t.read_all()
        self.eid, self.name, self.addr, self.num = (t.column(c).combine_chunks() for c in t.column_names)

    def take(self, col, rids: np.ndarray):
        return pc.take(getattr(self, col), pa.array(rids, type=pa.int32()))


def cheap_score(q_name, i_name, q_addr, i_addr, num_eq) -> np.ndarray:
    name = process.cpdist(q_name, i_name, scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)
    addr = process.cpdist(q_addr, i_addr, scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)
    return ((0.5 * name + 0.3 * addr + 20.0 * np.asarray(num_eq, dtype=np.float32)) / 100.0).astype(np.float32)


def _top_k(local: np.ndarray, score: np.ndarray, irid: np.ndarray, k: int) -> np.ndarray:
    """Indices of the top-k rows per local id by score (ties: lower irid first)."""
    order = np.lexsort((irid, -score, local))
    loc = local[order]
    starts = np.r_[0, np.flatnonzero(np.diff(loc)) + 1]
    rank = np.arange(len(loc)) - np.repeat(starts, np.diff(np.r_[starts, len(loc)]))
    return order[rank < k]


def block_shard(split: str, shard: int, con: duckdb.DuckDBPyConnection, rs: RecStrings) -> dict:
    base, cand_dir = _dirs(split)
    idx, qry = (_p(base / f) for f in ("index_keys.parquet", "query_keys.parquet"))
    # the caller loads work/blocking/{split}/recs.parquet into the temp table `recs` once
    t0 = time.time()
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE q AS
        SELECT r.rid, row_number() OVER (ORDER BY r.rid) - 1 AS local
        FROM recs r JOIN read_parquet('{_p(_shard_table(split))}') s ON r.entity_id = s.s1_id
        WHERE r.source = 1 AND s.shard = {shard}""")
    n_s1 = con.execute("SELECT count(*) FROM q").fetchone()[0]
    pairs_path = base / f"_pairs_shard{shard}.parquet"
    # stage 1 (speed): keep the PREFILTER_K candidates per S1 with the rarest shared keys,
    # key_weight = sum over shared keys of 1 / index group size (table `gs`, built once by the caller)
    con.execute(f"""
        COPY (SELECT local, qrid, irid, key_hits, key_weight FROM (
                SELECT q.local, q.rid AS qrid, i.rid AS irid, bit_or(k.bit)::INTEGER AS key_hits,
                       sum(1.0 / g.n)::FLOAT AS key_weight
                FROM read_parquet('{qry}') k JOIN q USING (rid)
                JOIN read_parquet('{idx}') i USING (bit, h)
                JOIN gs g USING (bit, h)
                GROUP BY ALL)
              QUALIFY row_number() OVER (PARTITION BY local ORDER BY key_weight DESC, irid) <= {config.PREFILTER_K})
        TO '{_p(pairs_path)}' (FORMAT parquet)""")
    n_raw = con.execute(f"SELECT count(*) FROM read_parquet('{_p(pairs_path)}')").fetchone()[0]  # after stage 1
    t_join = time.time() - t0

    parts = []
    for lo in range(0, n_s1, SCORE_CHUNK_S1):
        a = con.execute(f"""SELECT local, qrid, irid, key_hits, key_weight FROM read_parquet('{_p(pairs_path)}')
                            WHERE local >= {lo} AND local < {lo + SCORE_CHUNK_S1}""").fetchnumpy()
        if len(a["local"]) == 0:
            continue
        local, qrid, irid = (np.asarray(a[c], dtype=np.int64) for c in ("local", "qrid", "irid"))
        qn, inn = rs.take("num", qrid), rs.take("num", irid)
        num_eq = pc.and_(pc.equal(qn, inn), pc.not_equal(qn, "")).to_numpy(zero_copy_only=False)
        score = cheap_score(rs.take("name", qrid).to_pylist(), rs.take("name", irid).to_pylist(),
                            rs.take("addr", qrid).to_pylist(), rs.take("addr", irid).to_pylist(), num_eq)
        keep = _top_k(local, score, irid, config.TOPK)
        parts.append(pd.DataFrame({
            "s1_id": rs.take("eid", qrid[keep]).to_numpy(zero_copy_only=False),
            "other_id": rs.take("eid", irid[keep]).to_numpy(zero_copy_only=False),
            "key_hits": np.asarray(a["key_hits"])[keep].astype("int32"),
            "cheap_score": score[keep],
            "key_weight": np.asarray(a["key_weight"], dtype=np.float32)[keep]}))
        del a, local, qrid, irid, score
    pairs_path.unlink()
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        {"s1_id": pd.Series(dtype=str), "other_id": pd.Series(dtype=str),
         "key_hits": pd.Series(dtype="int32"), "cheap_score": pd.Series(dtype="float32"),
         "key_weight": pd.Series(dtype="float32")})
    out["s1_id"] = out["s1_id"].astype(str)
    out["other_id"] = out["other_id"].astype(str)
    out["key_hits"] = out["key_hits"].astype("int32")
    out["cheap_score"] = out["cheap_score"].astype("float32")
    out["key_weight"] = out["key_weight"].astype("float32")
    tmp = cand_dir / f"shard={shard}.tmp"
    out.to_parquet(tmp, index=False)
    tmp.replace(cand_dir / f"shard={shard}.parquet")
    info = {"shard": shard, "n_s1": n_s1, "n_pairs_after_prefilter": n_raw, "n_pairs": len(out),
            "join_s": round(t_join, 1), "elapsed_s": round(time.time() - t0, 1)}
    print(f"  shard {shard}: {n_s1:,} S1, {n_raw:,} pairs after rarity prefilter (top-{config.PREFILTER_K}), {len(out):,} kept, "
          f"{info['elapsed_s']}s (join {info['join_s']}s)", flush=True)
    return info


# ----------------------------------------------------------------------------- report + misses

def shard_report(split: str, shards: List[int]) -> dict:
    _, cand_dir = _dirs(split)
    cand = pd.concat([pd.read_parquet(cand_dir / f"shard={i}.parquet") for i in shards], ignore_index=True)
    sh = pd.read_parquet(_shard_table(split))
    s1_ids = sh.loc[sh["shard"].isin(shards), "s1_id"]
    per_s1 = cand.groupby("s1_id").size().reindex(s1_ids, fill_value=0).to_numpy()
    hits = cand["key_hits"].to_numpy()
    return {
        "n_s1": int(len(s1_ids)),
        "n_pairs": int(len(cand)),
        "pairs_per_s1": {"mean": round(float(per_s1.mean()), 3), "p50": float(np.percentile(per_s1, 50)),
                         "p95": float(np.percentile(per_s1, 95)), "max": int(per_s1.max())},
        "s1_with_zero_candidates_all": round(float((per_s1 == 0).mean()), 6),
        "per_key_pairs": {name: int(((hits & bit) > 0).sum()) for bit, name in KEY_NAMES.items()},
        "cross_country_pairs": 0,  # impossible by construction: country is part of every key hash
    }


def show_misses(shards: List[int], n: int) -> None:
    from .evaluate import load_truth
    s1_ids, truth = load_truth("val", shards)
    _, cand_dir = _dirs("train")
    cand = pd.concat([pd.read_parquet(cand_dir / f"shard={i}.parquet", columns=["s1_id", "other_id"])
                      for i in shards])
    found = set(zip(cand["s1_id"], cand["other_id"]))
    missed = [(s, o) for s in s1_ids for o in sorted(truth[s]) if (s, o) not in found]
    rng = np.random.default_rng(config.SEED)
    pick = [missed[i] for i in rng.choice(len(missed), size=min(n, len(missed)), replace=False)]
    raw = pd.concat([pd.read_parquet(config.WORK / "raw" / f"train_s{k}.parquet") for k in (1, 2, 3)])
    raw = raw[raw["entity_id"].isin({x for p in pick for x in p})].set_index("entity_id")
    norm = pd.read_parquet(config.WORK / "blocking" / "train" / "recs.parquet",
                           filters=[("entity_id", "in", [x for p in pick for x in p])]).set_index("entity_id")
    print(f"\n{len(missed):,} true val pairs missed in shards {shards}; {len(pick)} random examples:")
    for s, o in pick:
        print(f"\n  {s} ({raw.at[s, 'country']})  <->  {o}")
        for e in (s, o):
            print(f"    raw : {raw.at[e, 'business_name']!r} | {raw.at[e, 'business_address']!r}")
            print(f"    norm: core={norm.at[e, 'name_core']!r} nospace={norm.at[e, 'name_nospace']!r} "
                  f"addr_core={norm.at[e, 'addr_core']!r} first_num={norm.at[e, 'addr_first_num']!r}")


# ----------------------------------------------------------------------------- CLI

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--shards", default=None, help="comma-separated shard ids (default: all)")
    ap.add_argument("--force", action="store_true", help="rebuild keys and all requested shards")
    ap.add_argument("--show-misses", type=int, default=10, help="train only: print N missed val pairs")
    args = ap.parse_args()
    shards = list(range(config.N_SHARDS)) if not args.shards else [int(x) for x in args.shards.split(",")]
    t0 = time.time()
    shard_info = []
    with TreeMemWatch() as mem:
        key_stats = build_keys(args.split, args.force)
        con = _con()
        base, cand_dir = _dirs(args.split)
        con.execute(f"CREATE TEMP TABLE recs AS SELECT * FROM read_parquet('{_p(base / 'recs.parquet')}')")
        rs = RecStrings(con)
        con.execute(f"CREATE TEMP TABLE gs AS SELECT bit, h, count(*)::INTEGER AS n "
                    f"FROM read_parquet('{_p(base / 'index_keys.parquet')}') GROUP BY ALL")
        for i in shards:
            if (cand_dir / f"shard={i}.parquet").exists() and not args.force:
                print(f"  shard {i}: exists, skipping (use --force)")
                continue
            shard_info.append(block_shard(args.split, i, con, rs))
        con.close()
    report = {"split": args.split, "shards": shards, "topk": config.TOPK, "block_cap": config.BLOCK_CAP,
              **shard_report(args.split, shards), "key_stats": key_stats, "shard_runs": shard_info,
              "elapsed_s": round(time.time() - t0, 1), "peak_mem_gb_all_processes": mem.peak_gb,
              "peak_mem_gb_main": peak_mem_gb(), "ram_gb": mem.total_gb}
    if args.split == "train":
        from .evaluate import eval_candidates
        ev = eval_candidates(shards)
        report["val"] = ev
        for k in ("recall_all", "recall_by_country", "recall_at_k", "s1_with_zero_candidates"):
            report[k] = ev[k]
    out = config.REPORTS / f"blocking_{args.split}.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("shard_runs", "key_stats")}, indent=2))
    print(f"blocking: done in {report['elapsed_s']}s, peak memory {mem.peak_gb} GB (all processes) -> {out}")
    if args.split == "train" and args.show_misses:
        show_misses(shards, args.show_misses)


if __name__ == "__main__":
    main()
