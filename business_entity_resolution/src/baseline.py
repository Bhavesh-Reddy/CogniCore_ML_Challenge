"""Rule-based baseline: cheap_score + one-to-one + one tuned threshold (SKILL.md pipeline step 7).

Usage: python -m business_entity_resolution.src.baseline --split {train,test} [--shards 0,1,...]

train: one-to-one over ALL S1 in the loaded shards (both folds, as on test), tune the threshold on the val
       S1 of those shards with evaluate.best_threshold, write reports/baseline_val.json.
test:  same one-to-one, the saved threshold, write work/pred/test_baseline.parquet
       (s1_id, other_id, prob, kept) for write_outputs --baseline.
"""
import argparse
import json
import time

import numpy as np
import pandas as pd

from . import config
from .evaluate import best_threshold, eval_predictions, load_country, load_truth

REPORT = config.REPORTS / "baseline_val.json"
OUT = config.WORK / "pred" / "test_baseline.parquet"


def one_to_one(df: pd.DataFrame, score: str = "prob") -> pd.DataFrame:
    """Keep, for each other_id, only the S1 with the highest score (ties: smallest s1_id). facts.md section 5."""
    df = df.sort_values(["other_id", score, "s1_id"], ascending=[True, False, True], kind="mergesort")
    return df.drop_duplicates("other_id", keep="first").reset_index(drop=True)


def load_scores(split: str, shards) -> pd.DataFrame:
    parts = [pd.read_parquet(config.WORK / "cand" / split / f"shard={i}.parquet",
                             columns=["s1_id", "other_id", "cheap_score"]) for i in shards]
    df = pd.concat(parts, ignore_index=True).rename(columns={"cheap_score": "prob"})
    df["prob"] = df["prob"].astype(np.float32)
    return df


def run_train(shards) -> dict:
    t0 = time.time()
    df = load_scores("train", shards)
    n_before = len(df)
    df = one_to_one(df)
    s1_ids, truth = load_truth("val", shards)
    kw = {"s1_ids": s1_ids, "truth": truth}
    thr, _ = best_threshold(df, **kw)
    res = eval_predictions(df, thr, country=load_country(s1_ids), **kw)
    no_o2o = eval_predictions(load_scores("train", shards), thr, country={}, **kw)["macro_f05"]
    report = {"model": "baseline cheap_score", "shards": list(shards), "n_pairs_before_one_to_one": n_before,
              "n_pairs_after_one_to_one": len(df), "macro_f05_without_one_to_one_same_threshold": no_o2o,
              **res, "elapsed_s": round(time.time() - t0, 1)}
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def run_test(shards) -> dict:
    thr = json.loads(REPORT.read_text(encoding="utf-8"))["threshold"]
    df = one_to_one(load_scores("test", shards))
    df["kept"] = df["prob"] >= thr
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    return {"threshold": thr, "n_pairs_after_one_to_one": len(df), "n_kept": int(df["kept"].sum()),
            "n_s1_with_match": int(df.loc[df["kept"], "s1_id"].nunique())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--shards", default=None, help="comma-separated shard ids (default: all)")
    args = ap.parse_args()
    shards = list(range(config.N_SHARDS)) if not args.shards else [int(x) for x in args.shards.split(",")]
    out = run_train(shards) if args.split == "train" else run_test(shards)
    print(json.dumps(out, indent=2))
    if args.split == "test":
        print(f"-> {OUT}")


if __name__ == "__main__":
    main()
