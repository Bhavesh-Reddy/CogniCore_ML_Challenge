"""Score candidates and predictions on the val fold (contracts.md sections 4 and 6).

Usage:
  python -m business_entity_resolution.src.evaluate --what candidates [--shards 0,1,2,3] [--out reports/x.json]
  python -m business_entity_resolution.src.evaluate --what predictions --pred-path P [--threshold T] [--shards ...]

The official per-entity score is metric.f05_entity. best_threshold needs it for every S1 at every
threshold, so it evaluates the same formula on count arrays (_f05_counts); tests check that both agree.
"""
import argparse
import json
import time
from itertools import chain
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from . import config
from .metric import f05_entity, macro_f05

DEFAULT_GRID = np.arange(0.05, 0.96, 0.01)
RECALL_KS = (10, 20, 30, 50)


# ----------------------------------------------------------------------------- loading

def load_truth(fold: Optional[str] = "val", shards: Optional[Sequence[int]] = None
               ) -> Tuple[List[str], Dict[str, Set[str]]]:
    """S1 ids of the fold (optionally restricted to shards) and their true match sets.

    Singletons are included in the id list and map to an empty set.
    """
    splits = pd.read_parquet(config.WORK / "splits.parquet")
    if fold is not None:
        splits = splits[splits["fold"] == fold]
    if shards is not None:
        splits = splits[splits["shard"].isin(list(shards))]
    s1_ids = splits["s1_id"].tolist()
    pairs = pd.read_parquet(config.WORK / "gt_pairs.parquet")
    pairs = pairs[pairs["s1_id"].isin(set(s1_ids))]
    truth = {s: set() for s in s1_ids}
    for s1, other in zip(pairs["s1_id"].to_numpy(), pairs["other_id"].to_numpy()):
        truth[s1].add(other)
    return s1_ids, truth


def load_country(s1_ids: Iterable[str]) -> Dict[str, str]:
    raw = pd.read_parquet(config.WORK / "raw" / "train_s1.parquet", columns=["entity_id", "country"])
    raw = raw[raw["entity_id"].isin(set(s1_ids))]
    return dict(zip(raw["entity_id"], raw["country"]))


def load_candidates(shards: Sequence[int], split: str = "train") -> pd.DataFrame:
    parts = [pd.read_parquet(config.WORK / "cand" / split / f"shard={i}.parquet") for i in shards]
    return pd.concat(parts, ignore_index=True)


def _truth_pairs(truth: Dict[str, Set[str]]) -> pd.DataFrame:
    keys = list(truth)
    s1 = np.repeat(np.array(keys, dtype=object), [len(truth[k]) for k in keys])
    others = list(chain.from_iterable(truth[k] for k in keys))
    return pd.DataFrame({"s1_id": pd.Series(s1, dtype=str), "other_id": pd.Series(others, dtype=str)})


def _label(df: pd.DataFrame, truth: Dict[str, Set[str]]) -> np.ndarray:
    """Boolean array: is (s1_id, other_id) of each row a true pair. Uses integer keys, not a string merge."""
    tp = _truth_pairs(truth)
    n = len(df)
    s1 = pd.factorize(pd.concat([df["s1_id"].astype(str), tp["s1_id"]], ignore_index=True))[0]
    other, uniques = pd.factorize(pd.concat([df["other_id"].astype(str), tp["other_id"]], ignore_index=True))
    key = s1.astype(np.int64) * max(len(uniques), 1) + other
    return pd.Series(key[:n]).isin(key[n:]).to_numpy()


# ----------------------------------------------------------------------------- scoring helpers

def _f05_counts(tp, n_pred, n_true) -> np.ndarray:
    """metric.f05_entity evaluated element-wise on count arrays."""
    tp, n_pred, n_true = (np.asarray(a, dtype=np.float64) for a in (tp, n_pred, n_true))
    with np.errstate(divide="ignore", invalid="ignore"):
        p = tp / n_pred
        r = tp / n_true
        f = 1.25 * p * r / (0.25 * p + r)
    f = np.where(tp > 0, f, 0.0)
    return np.where(n_true == 0, (n_pred == 0).astype(np.float64), f)


def _pct(x: np.ndarray, q: float) -> float:
    return float(np.percentile(x, q)) if len(x) else 0.0


# ----------------------------------------------------------------------------- candidates

def candidate_stats(cand: pd.DataFrame, s1_ids: List[str], truth: Dict[str, Set[str]],
                    country: Dict[str, str]) -> dict:
    """Blocking quality over the given S1 ids (candidates for other S1 are ignored)."""
    ids = set(s1_ids)
    cand = cand[cand["s1_id"].isin(ids)].reset_index(drop=True)
    cand = cand.assign(_true=_label(cand, truth))
    n_true_pairs = sum(len(v) for v in truth.values())
    found = cand[cand["_true"]]

    per_s1 = cand.groupby("s1_id").size().reindex(s1_ids, fill_value=0).to_numpy()
    s1_country = pd.Series(country).reindex(s1_ids)
    true_by_country = pd.Series({s: len(truth[s]) for s in s1_ids}).groupby(s1_country).sum()
    found_by_country = found.groupby(found["s1_id"].map(country)).size()

    rank = cand.sort_values(["s1_id", "cheap_score", "other_id"], ascending=[True, False, True]) \
               .groupby("s1_id").cumcount() + 1
    found_rank = rank.reindex(found.index).to_numpy()

    key_hits = cand["key_hits"].to_numpy().astype(np.int64)
    found_hits = found["key_hits"].to_numpy().astype(np.int64)
    n_bits = int(key_hits.max()).bit_length() if len(key_hits) else 0
    per_key = {}
    for b in range(n_bits):
        per_key[f"K{b + 1}"] = {
            "pairs": int(((key_hits >> b) & 1).sum()),
            "recall": round(float(((found_hits >> b) & 1).sum()) / n_true_pairs, 6) if n_true_pairs else None,
        }

    found_sets: Dict[str, Set[str]] = {}
    for s1, o in zip(found["s1_id"].to_numpy(), found["other_id"].to_numpy()):
        found_sets.setdefault(s1, set()).add(o)

    return {
        "n_s1": len(s1_ids),
        "n_true_pairs": n_true_pairs,
        "n_cand_pairs": len(cand),
        "recall_all": round(len(found) / n_true_pairs, 6) if n_true_pairs else None,
        "recall_by_country": {c: round(float(found_by_country.get(c, 0)) / float(n), 6)
                              for c, n in true_by_country.items() if n > 0},
        "recall_at_k": {k: round(float((found_rank <= k).sum()) / n_true_pairs, 6) if n_true_pairs else None
                        for k in RECALL_KS},
        "s1_with_zero_candidates": round(float((per_s1 == 0).mean()), 6) if len(per_s1) else None,
        "pairs_per_s1": {"mean": round(float(per_s1.mean()), 3) if len(per_s1) else 0.0,
                         "p50": _pct(per_s1, 50), "p95": _pct(per_s1, 95),
                         "max": int(per_s1.max()) if len(per_s1) else 0},
        "per_key": per_key,
        "oracle_macro_f05": round(macro_f05(found_sets, truth, s1_ids), 6),
    }


def eval_candidates(shards: Sequence[int]) -> dict:
    s1_ids, truth = load_truth("val", shards)
    out = candidate_stats(load_candidates(shards), s1_ids, truth, load_country(s1_ids))
    return {"shards": list(shards), **out}


# ----------------------------------------------------------------------------- predictions

def _universe(pred_df, s1_ids, truth, country, shards):
    if s1_ids is None or truth is None:
        s1_ids, truth = load_truth("val", shards)
    if country is None:
        country = load_country(s1_ids)
    pred = pred_df[pred_df["s1_id"].isin(set(s1_ids))]
    return pred, s1_ids, truth, country


def eval_predictions(pred_df: pd.DataFrame, threshold: float, *, s1_ids: Optional[List[str]] = None,
                     truth: Optional[Dict[str, Set[str]]] = None, country: Optional[Dict[str, str]] = None,
                     shards: Optional[Sequence[int]] = None) -> dict:
    """Score post-processed predictions kept at prob >= threshold, over every S1 in the universe.

    The universe defaults to all val S1 (in `shards` if given); S1 absent from pred_df predict empty.
    """
    pred, s1_ids, truth, country = _universe(pred_df, s1_ids, truth, country, shards)
    kept = pred[pred["prob"] >= threshold]
    pred_sets: Dict[str, Set[str]] = {}
    for s1, o in zip(kept["s1_id"].to_numpy(), kept["other_id"].to_numpy()):
        pred_sets.setdefault(s1, set()).add(o)

    scores = pd.Series({s: f05_entity(pred_sets.get(s, set()), truth[s]) for s in s1_ids}, dtype=float)
    n_pred = sum(len(v) for v in pred_sets.values())
    n_true = sum(len(truth[s]) for s in s1_ids)
    tp = sum(len(pred_sets.get(s, set()) & truth[s]) for s in s1_ids)
    is_single = pd.Series({s: not truth[s] for s in s1_ids})
    by_country = scores.groupby(pd.Series(country).reindex(s1_ids).to_numpy()).mean()
    return {
        "threshold": round(float(threshold), 6),
        "n_s1": len(s1_ids),
        "macro_f05": round(float(scores.mean()), 6),
        "precision": round(tp / n_pred, 6) if n_pred else None,
        "recall": round(tp / n_true, 6) if n_true else None,
        "per_country": {c: round(float(v), 6) for c, v in by_country.items()},
        "macro_f05_singletons": round(float(scores[is_single].mean()), 6) if is_single.any() else None,
        "macro_f05_non_singletons": round(float(scores[~is_single].mean()), 6) if (~is_single).any() else None,
        "n_pred_pairs": n_pred,
    }


def threshold_curve(pred_df: pd.DataFrame, grid=DEFAULT_GRID, *, s1_ids=None, truth=None,
                    shards=None) -> pd.Series:
    """Macro F0.5 at every threshold of `grid` (keep prob >= t), vectorised over S1 and thresholds."""
    if s1_ids is None or truth is None:
        s1_ids, truth = load_truth("val", shards)
    grid = np.round(np.asarray(grid, dtype=np.float64), 6)
    n_s1, g = len(s1_ids), len(grid)

    # S1 in the universe get codes 0..n_s1-1; predictions for other S1 are dropped.
    codes = pd.factorize(pd.concat([pd.Series(s1_ids, dtype=str), pred_df["s1_id"].astype(str)],
                                   ignore_index=True))[0][n_s1:].astype(np.int64)
    inside = codes < n_s1
    pred = pred_df[inside]
    pred_s1 = codes[inside]
    is_true = _label(pred, truth)

    # Bucket b = number of grid values <= prob, so a row is kept at grid[j] iff j < b.
    b = np.searchsorted(grid, pred["prob"].to_numpy(np.float64), side="right")
    cell = pred_s1.astype(np.int64) * (g + 1) + b
    n_mat = np.bincount(cell, minlength=n_s1 * (g + 1)).reshape(n_s1, g + 1).astype(np.int32)
    tp_mat = np.bincount(cell[is_true], minlength=n_s1 * (g + 1)).reshape(n_s1, g + 1).astype(np.int32)
    n_true = np.array([len(truth[s]) for s in s1_ids], dtype=np.int32)[:, None]

    total = np.zeros(g, dtype=np.float64)
    for lo in range(0, n_s1, 100_000):  # chunks keep the float temporaries small
        hi = min(lo + 100_000, n_s1)
        # kept count at grid[j] = sum over buckets b > j
        n_kept = np.cumsum(n_mat[lo:hi, ::-1], axis=1, dtype=np.int32)[:, ::-1][:, 1:]
        tp_kept = np.cumsum(tp_mat[lo:hi, ::-1], axis=1, dtype=np.int32)[:, ::-1][:, 1:]
        total += _f05_counts(tp_kept, n_kept, n_true[lo:hi]).sum(axis=0)
    return pd.Series(total / n_s1, index=grid, name="macro_f05")


def best_threshold(pred_df: pd.DataFrame, grid=DEFAULT_GRID, **kw) -> Tuple[float, float]:
    """(threshold, macro F0.5) maximising macro F0.5; ties go to the lowest threshold."""
    curve = threshold_curve(pred_df, grid, **kw)
    j = int(np.argmax(curve.to_numpy()))
    return float(curve.index[j]), float(curve.iloc[j])


# ----------------------------------------------------------------------------- CLI

def _parse_shards(s: Optional[str]) -> List[int]:
    return list(range(config.N_SHARDS)) if not s else [int(x) for x in s.split(",")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", choices=["candidates", "predictions"], required=True)
    ap.add_argument("--shards", default=None, help="comma-separated shard ids (default: all)")
    ap.add_argument("--pred-path", default=None, help="parquet with s1_id, other_id, prob (post-processed)")
    ap.add_argument("--threshold", type=float, default=None, help="default: tuned with best_threshold")
    ap.add_argument("--out", default=None, help="JSON report path, e.g. reports/<name>.json")
    args = ap.parse_args()
    shards = _parse_shards(args.shards)
    t0 = time.time()

    if args.what == "candidates":
        report = eval_candidates(shards)
    else:
        if not args.pred_path:
            ap.error("--pred-path is required with --what predictions")
        s1_ids, truth = load_truth("val", shards)
        pred = pd.read_parquet(args.pred_path, columns=["s1_id", "other_id", "prob"])
        kw = {"s1_ids": s1_ids, "truth": truth}
        thr = args.threshold
        if thr is None:
            thr, _ = best_threshold(pred, **kw)
        report = {"shards": shards, "tuned": args.threshold is None,
                  **eval_predictions(pred, thr, country=load_country(s1_ids), **kw)}

    report["elapsed_s"] = round(time.time() - t0, 1)
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8", newline="\n") as f:
            f.write(text + "\n")
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
