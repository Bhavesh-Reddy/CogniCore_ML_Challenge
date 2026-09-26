"""Fold and shard assignment for train S1, shard assignment for test S1 (contracts.md section 3).

Usage: python -m business_entity_resolution.src.splits [--force]
"""
import argparse
import json
import time

import numpy as np
import pandas as pd

from . import config
from .io_utils import peak_mem_gb, stable_hash

RAW = config.WORK / "raw"
SPLITS = config.WORK / "splits.parquet"
TEST_SHARDS = config.WORK / "test_shards.parquet"
REPORT = config.REPORTS / "splits.json"


def shard_of(ids) -> np.ndarray:
    return np.fromiter((stable_hash(s) % config.N_SHARDS for s in ids), dtype="int16", count=len(ids))


def fold_of(ids) -> np.ndarray:
    is_val = np.fromiter((stable_hash(s + "#fold") % config.VAL_MOD == 0 for s in ids),
                         dtype=bool, count=len(ids))
    return np.where(is_val, "val", "fit")


def _ids(split: str) -> list:
    return pd.read_parquet(RAW / f"{split}_s1.parquet", columns=["entity_id"])["entity_id"].tolist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if SPLITS.exists() and TEST_SHARDS.exists() and REPORT.exists() and not args.force:
        print("splits: outputs exist, skipping (use --force to rebuild)")
        return
    t0 = time.time()

    train_ids = _ids("train")
    splits = pd.DataFrame({"s1_id": train_ids, "fold": fold_of(train_ids), "shard": shard_of(train_ids)})
    splits["s1_id"] = splits["s1_id"].astype(str)
    splits["fold"] = splits["fold"].astype(str)
    splits.to_parquet(SPLITS, index=False)
    print(f"  train: {len(splits):,} S1 -> {SPLITS.name}", flush=True)

    test_ids = _ids("test")
    test = pd.DataFrame({"s1_id": test_ids, "shard": shard_of(test_ids)})
    test["s1_id"] = test["s1_id"].astype(str)
    test.to_parquet(TEST_SHARDS, index=False)
    print(f"  test: {len(test):,} S1 -> {TEST_SHARDS.name}", flush=True)

    gt_s1 = pd.read_parquet(config.WORK / "gt_s1.parquet")
    merged = splits.merge(gt_s1, on="s1_id", how="left", validate="one_to_one")
    assert merged["n_matches"].notna().all(), "train S1 missing from gt_s1"
    singleton_rate = (merged["n_matches"] == 0).groupby(merged["fold"]).mean()

    elapsed = round(time.time() - t0, 1)
    report = {
        "n_fit": int((splits["fold"] == "fit").sum()),
        "n_val": int((splits["fold"] == "val").sum()),
        "n_test": len(test),
        "singleton_rate": {f: round(float(v), 6) for f, v in singleton_rate.items()},
        "shard_sizes_train": {int(k): int(v) for k, v in splits["shard"].value_counts().sort_index().items()},
        "shard_sizes_val": {int(k): int(v) for k, v in
                            splits.loc[splits["fold"] == "val", "shard"].value_counts().sort_index().items()},
        "shard_sizes_test": {int(k): int(v) for k, v in test["shard"].value_counts().sort_index().items()},
        "n_shards": config.N_SHARDS,
        "val_mod": config.VAL_MOD,
        "elapsed_s": elapsed,
        "peak_mem_gb": peak_mem_gb(),
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"splits: n_fit={report['n_fit']:,} n_val={report['n_val']:,} "
          f"singleton_rate={report['singleton_rate']} done in {elapsed}s, "
          f"peak memory {report['peak_mem_gb']} GB -> {REPORT}")


if __name__ == "__main__":
    main()
