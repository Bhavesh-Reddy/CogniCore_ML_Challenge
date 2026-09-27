"""Pairwise features for candidate pairs (SKILL.md pipeline step 5, contracts.md sections 3 and 4).

Usage: python -m business_entity_resolution.src.features --split {train,test} [--shards 0,1,...] [--force]

Country is never a feature (it is open-set; France is unseen in train). FEATURES below is the single, ordered
source of feature names for train.py and predict.py.

Conventions: string similarities are 0-100 (JaroWinkler 0-1). Address similarities are -1 when either
address is empty (rapidfuzz scores two empty strings as 100). *_eq features are 1 equal / 0 different /
-1 one side missing.
Competition features (n_s1_claiming, claim_rank) are computed over ALL candidate pairs of the shards in the
same run, so dev runs (4 shards) see fewer competing S1 than full runs; run all shards together for real use.
Each shard runs in its own subprocess so memory is returned between shards (in one process it grew until
the machine swapped: 234s -> 630s per shard on 16 GB).
"""
import argparse
import json
import subprocess
import sys
import time
from typing import List

import duckdb
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

from . import config
from .io_utils import TreeMemWatch, peak_mem_gb

FEATURES = [
    # name similarity
    "name_ratio", "name_tset", "name_tsort", "name_partial", "name_jw",
    "nospace_ratio", "nospace_partial", "namenorm_tset", "name_jacc",
    # address similarity
    "addr_tset", "addr_partial", "addrnorm_tsort", "addr_jacc",
    # numbers and admin region
    "num_jacc", "first_num_eq", "s1_num_share", "admin_eq",
    # other-record flags and lengths
    "other_addr_empty", "other_name_has_url", "other_name_nonlatin", "other_source",
    "name_len_ratio", "addr_len_ratio",
    # blocking
    "key_k1", "key_k2", "key_k3", "key_k4", "cheap_score", "key_weight",
    # group (per s1_id)
    "n_candidates", "cs_rank", "cs_gap", "name_tset_rank", "addr_tset_rank",
    # competition (per other_id)
    "n_s1_claiming", "claim_rank",
]
NORM_COLS = ["name_norm", "name_core", "name_nospace", "name_has_url", "name_nonlatin", "addr_norm",
             "addr_core", "addr_nums", "addr_first_num", "addr_admin", "source"]
CHUNK_S1 = 40_000  # S1 per chunk (~2M pairs); all candidates of an S1 stay in one chunk


def _p(path) -> str:
    return path.as_posix()


def _con() -> duckdb.DuckDBPyConnection:
    tmp = config.WORK / "duckdb_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET memory_limit = '{config.DUCKDB_MEMORY}'")
    con.execute(f"SET temp_directory = '{tmp.as_posix()}'")
    con.execute(f"SET threads = {config.N_JOBS}")
    return con


# ----------------------------------------------------------------------------- pure feature helpers

def _cp(a, b, scorer) -> np.ndarray:
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32)


def _jaccard(a: List[str], b: List[str]) -> np.ndarray:
    out = np.empty(len(a), dtype=np.float32)
    for i, (x, y) in enumerate(zip(a, b)):
        sx, sy = set(x.split()), set(y.split())
        u = len(sx | sy)
        out[i] = len(sx & sy) / u if u else -1.0
    return out


def _numbers(a: List[str], b: List[str]):
    """(num_jacc, s1_num_share): -1 when a side has no numbers."""
    jac = np.empty(len(a), dtype=np.float32)
    share = np.empty(len(a), dtype=np.float32)
    for i, (x, y) in enumerate(zip(a, b)):
        sx, sy = set(x.split()), set(y.split())
        if not sx or not sy:
            jac[i] = -1.0
        else:
            jac[i] = len(sx & sy) / len(sx | sy)
        share[i] = len(sx & sy) / len(sx) if sx else -1.0
    return jac, share


def _eq3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """1 equal, 0 different, -1 either side empty."""
    missing = (a == "") | (b == "")
    return np.where(missing, -1.0, (a == b).astype(np.float32)).astype(np.float32)


def _admin_eq(a: List[str], b: List[str]) -> np.ndarray:
    out = np.empty(len(a), dtype=np.float32)
    for i, (x, y) in enumerate(zip(a, b)):
        if not x or not y:
            out[i] = -1.0
        else:
            out[i] = 1.0 if set(x.split()) & set(y.split()) else 0.0
    return out


def _len_ratio(a: pd.Series, b: pd.Series) -> np.ndarray:
    la, lb = a.str.len().to_numpy(np.float32), b.str.len().to_numpy(np.float32)
    hi = np.maximum(la, lb)
    return np.where(hi > 0, np.minimum(la, lb) / np.where(hi > 0, hi, 1), 0.0).astype(np.float32)


def pair_features(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise features from a frame with q_* (S1) and o_* (other) normalised columns."""
    f = pd.DataFrame(index=df.index)
    L = {c: df[c].tolist() for c in df.columns if c.startswith(("q_", "o_")) and df[c].dtype != "int8"}
    f["name_ratio"] = _cp(L["q_name_core"], L["o_name_core"], fuzz.ratio)
    f["name_tset"] = _cp(L["q_name_core"], L["o_name_core"], fuzz.token_set_ratio)
    f["name_tsort"] = _cp(L["q_name_core"], L["o_name_core"], fuzz.token_sort_ratio)
    f["name_partial"] = _cp(L["q_name_core"], L["o_name_core"], fuzz.partial_ratio)
    f["name_jw"] = _cp(L["q_name_core"], L["o_name_core"], JaroWinkler.normalized_similarity)
    f["nospace_ratio"] = _cp(L["q_name_nospace"], L["o_name_nospace"], fuzz.ratio)
    f["nospace_partial"] = _cp(L["q_name_nospace"], L["o_name_nospace"], fuzz.partial_ratio)
    f["namenorm_tset"] = _cp(L["q_name_norm"], L["o_name_norm"], fuzz.token_set_ratio)
    f["name_jacc"] = _jaccard(L["q_name_core"], L["o_name_core"])

    addr_missing = ((df["q_addr_norm"] == "") | (df["o_addr_norm"] == "")).to_numpy()
    for name, a, b, scorer in (("addr_tset", "q_addr_core", "o_addr_core", fuzz.token_set_ratio),
                               ("addr_partial", "q_addr_core", "o_addr_core", fuzz.partial_ratio),
                               ("addrnorm_tsort", "q_addr_norm", "o_addr_norm", fuzz.token_sort_ratio)):
        f[name] = np.where(addr_missing, -1.0, _cp(L[a], L[b], scorer)).astype(np.float32)
    f["addr_jacc"] = np.where(addr_missing, -1.0, _jaccard(L["q_addr_core"], L["o_addr_core"])).astype(np.float32)

    f["num_jacc"], f["s1_num_share"] = _numbers(L["q_addr_nums"], L["o_addr_nums"])
    f["first_num_eq"] = _eq3(df["q_addr_first_num"].to_numpy(), df["o_addr_first_num"].to_numpy())
    f["admin_eq"] = _admin_eq(L["q_addr_admin"], L["o_addr_admin"])

    f["other_addr_empty"] = (df["o_addr_norm"] == "").astype(np.float32)
    f["other_name_has_url"] = df["o_name_has_url"].astype(np.float32)
    f["other_name_nonlatin"] = df["o_name_nonlatin"].astype(np.float32)
    f["other_source"] = df["o_source"].astype(np.float32)
    f["name_len_ratio"] = _len_ratio(df["q_name_core"], df["o_name_core"])
    f["addr_len_ratio"] = _len_ratio(df["q_addr_norm"], df["o_addr_norm"])

    hits = df["key_hits"].to_numpy()
    for j, bit in enumerate((1, 2, 4, 8), start=1):
        f[f"key_k{j}"] = ((hits & bit) > 0).astype(np.float32)
    f["cheap_score"] = df["cheap_score"].astype(np.float32)
    f["key_weight"] = df["key_weight"].astype(np.float32)
    return f


def group_features(s1: pd.Series, f: pd.DataFrame) -> pd.DataFrame:
    g = f.groupby(s1.to_numpy(), sort=False)
    out = pd.DataFrame(index=f.index)
    out["n_candidates"] = g["cheap_score"].transform("size")
    out["cs_rank"] = g["cheap_score"].rank(ascending=False, method="min")
    out["cs_gap"] = f["cheap_score"] - g["cheap_score"].transform("max")
    out["name_tset_rank"] = g["name_tset"].rank(ascending=False, method="min")
    out["addr_tset_rank"] = g["addr_tset"].rank(ascending=False, method="min")
    return out.astype(np.float32)


# ----------------------------------------------------------------------------- per shard

def build_shard(split: str, shard: int, con: duckdb.DuckDBPyConnection) -> dict:
    t0 = time.time()
    cand = _p(config.WORK / "cand" / split / f"shard={shard}.parquet")
    label_sql = ("CASE WHEN g.s1_id IS NULL THEN 0 ELSE 1 END" if split == "train" else "0")
    label_join = (f"LEFT JOIN read_parquet('{_p(config.WORK / 'gt_pairs.parquet')}') g "
                  "ON g.s1_id = c.s1_id AND g.other_id = c.other_id" if split == "train" else "")
    qcols = ", ".join(f"a.{c} AS q_{c}" for c in NORM_COLS)
    ocols = ", ".join(f"o.{c} AS o_{c}" for c in NORM_COLS)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE s1list AS
        SELECT s1_id, (row_number() OVER (ORDER BY s1_id) - 1) // {CHUNK_S1} AS chunk
        FROM (SELECT DISTINCT s1_id FROM read_parquet('{cand}'))""")
    n_chunks = con.execute("SELECT coalesce(max(chunk) + 1, 0) FROM s1list").fetchone()[0]
    parts = []
    for ch in range(n_chunks):
        df = con.execute(f"""
            SELECT c.s1_id, c.other_id, c.key_hits, c.cheap_score, c.key_weight, {qcols}, {ocols},
                   comp.n_s1_claiming, comp.claim_rank, {label_sql} AS label
            FROM read_parquet('{cand}') c
            JOIN s1list l ON l.s1_id = c.s1_id AND l.chunk = {ch}
            JOIN norm a ON a.entity_id = c.s1_id
            JOIN norm o ON o.entity_id = c.other_id
            JOIN comp ON comp.s1_id = c.s1_id AND comp.other_id = c.other_id
            {label_join}""").df()
        f = pair_features(df)
        f = pd.concat([f, group_features(df["s1_id"], f)], axis=1)
        f["n_s1_claiming"] = df["n_s1_claiming"].astype(np.float32)
        f["claim_rank"] = df["claim_rank"].astype(np.float32)
        out = pd.concat([df[["s1_id", "other_id"]].astype(str), f[FEATURES].astype(np.float32)], axis=1)
        if split == "train":
            out["label"] = df["label"].astype("int8")
        parts.append(out)
        del df, f
    res = pd.concat(parts, ignore_index=True)
    out_dir = config.WORK / "feat" / split
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / f"shard={shard}.tmp"
    res.to_parquet(tmp, index=False)
    tmp.replace(out_dir / f"shard={shard}.parquet")
    dt = time.time() - t0
    info = {"shard": shard, "n_pairs": len(res), "elapsed_s": round(dt, 1), "pairs_per_s": round(len(res) / dt)}
    if split == "train":
        info["label_rate"] = round(float(res["label"].mean()), 6)
    print(f"  shard {shard}: {len(res):,} pairs in {dt:.1f}s ({len(res) / dt:,.0f} pairs/s)"
          + (f", label rate {info['label_rate']:.4f}" if split == "train" else ""), flush=True)
    return info


def label_comparison(split: str, shards: List[int]) -> pd.DataFrame:
    df = pd.concat([pd.read_parquet(config.WORK / "feat" / split / f"shard={i}.parquet",
                                    columns=FEATURES + ["label"]) for i in shards], ignore_index=True)
    rows = []
    for c in FEATURES:
        x = df[c]
        rows.append({"feature": c, "mean_label1": float(x[df["label"] == 1].mean()),
                     "mean_label0": float(x[df["label"] == 0].mean()), "n_nan": int(x.isna().sum()),
                     "constant": bool(x.nunique(dropna=False) <= 1)})
    return pd.DataFrame(rows)


def _worker(split: str, shard: int) -> None:
    """One shard in a fresh process (memory is released when it exits)."""
    con = _con()
    cand = _p(config.WORK / "cand" / split / f"shard={shard}.parquet")
    norm = [_p(config.WORK / "norm" / f"{split}_s{k}.parquet") for k in (1, 2, 3)]
    con.execute(f"""
        CREATE TEMP TABLE norm AS
        SELECT entity_id, {', '.join(NORM_COLS)} FROM read_parquet({norm})
        WHERE entity_id IN (SELECT s1_id FROM read_parquet('{cand}') UNION SELECT other_id FROM read_parquet('{cand}'))""")
    con.execute(f"CREATE TEMP TABLE comp AS SELECT * FROM read_parquet('{_p(_comp_path(split))}') "
                f"WHERE s1_id IN (SELECT DISTINCT s1_id FROM read_parquet('{cand}'))")
    info = build_shard(split, shard, con)
    con.close()
    info_path = config.WORK / "feat" / split / f"_info_shard{shard}.json"
    info_path.write_text(json.dumps(info) + "\n", encoding="utf-8")


def _comp_path(split: str):
    return config.WORK / "feat" / split / "_comp.parquet"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--shards", default=None, help="comma-separated shard ids (default: all)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--worker-shard", type=int, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.worker_shard is not None:
        _worker(args.split, args.worker_shard)
        return
    shards = list(range(config.N_SHARDS)) if not args.shards else [int(x) for x in args.shards.split(",")]
    feat_dir = config.WORK / "feat" / args.split
    feat_dir.mkdir(parents=True, exist_ok=True)
    todo = [i for i in shards if args.force or not (feat_dir / f"shard={i}.parquet").exists()]
    for i in sorted(set(shards) - set(todo)):
        print(f"  shard {i}: exists, skipping (use --force)")
    t0, infos = time.time(), []
    with TreeMemWatch() as mem:
        if todo:
            con = _con()
            cands = [_p(config.WORK / "cand" / args.split / f"shard={i}.parquet") for i in shards]
            con.execute(f"""
                COPY (SELECT s1_id, other_id,
                             count(*) OVER (PARTITION BY other_id) AS n_s1_claiming,
                             rank() OVER (PARTITION BY other_id ORDER BY cheap_score DESC) AS claim_rank
                      FROM read_parquet({cands}))
                TO '{_p(_comp_path(args.split))}' (FORMAT parquet)""")
            con.close()
            print(f"features --split {args.split}: competition table over shards {shards} "
                  f"({time.time() - t0:.1f}s)", flush=True)
            for i in todo:
                subprocess.run([sys.executable, "-m", "business_entity_resolution.src.features", "--split",
                                args.split, "--worker-shard", str(i)], check=True, cwd=config.ROOT)
                info_path = feat_dir / f"_info_shard{i}.json"
                infos.append(json.loads(info_path.read_text(encoding="utf-8")))
                info_path.unlink()
            _comp_path(args.split).unlink()
    n = sum(x["n_pairs"] for x in infos)
    dt = time.time() - t0
    report = {"split": args.split, "shards": shards, "features": FEATURES, "shard_runs": infos,
              "n_pairs": n, "elapsed_s": round(dt, 1), "pairs_per_s": round(n / dt) if n else None,
              "peak_mem_gb_all_processes": mem.peak_gb, "peak_mem_gb_main": peak_mem_gb()}
    if args.split == "train":
        cmp = label_comparison(args.split, shards)
        report["label_comparison"] = cmp.to_dict(orient="records")
        with pd.option_context("display.width", 140, "display.max_rows", 100):
            print(cmp.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        bad = cmp[(cmp["n_nan"] > 0) | cmp["constant"]]
        print("features with NaN or constant values:", "none" if bad.empty else bad["feature"].tolist())
    (config.REPORTS / f"features_{args.split}.json").write_text(json.dumps(report, indent=2) + "\n",
                                                               encoding="utf-8")
    print(f"features: {n:,} pairs in {dt:.1f}s ({report['pairs_per_s']} pairs/s), "
          f"peak memory {mem.peak_gb} GB (all processes)")


if __name__ == "__main__":
    main()
