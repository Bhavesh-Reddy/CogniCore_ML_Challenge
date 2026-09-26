"""TSV -> parquet for all source files, plus ground-truth pairs (contracts.md section 3).

Usage: python -m business_entity_resolution.src.ingest [--force]
"""
import argparse
import gc
import json
import time

import pandas as pd

from . import config
from .io_utils import peak_mem_gb, read_tsv

# facts.md section 3 (rows exclude header)
EXPECTED_ROWS = {
    "train_source1.tsv": 2_206_821,
    "train_source2.tsv": 5_034_616,
    "train_source3.tsv": 5_285_603,
    "train_ground_truth.tsv": 2_206_821,
    "test_source1.tsv": 1_732_544,
    "test_source2.tsv": 4_887_273,
    "test_source3.tsv": 5_082_316,
}
# facts.md section 5
EXPECTED_SINGLETONS = 123_247
EXPECTED_GT_PAIRS = 7_638_365
SOURCE_COLS = ["entity_id", "business_name", "business_address", "country"]

RAW = config.WORK / "raw"
GT_PAIRS = config.WORK / "gt_pairs.parquet"
GT_S1 = config.WORK / "gt_s1.parquet"
REPORT = config.REPORTS / "ingest.json"


def _check_rows(name: str, n: int) -> None:
    assert n == EXPECTED_ROWS[name], f"{name}: {n:,} rows, facts.md says {EXPECTED_ROWS[name]:,}"


def ingest_sources(rows: dict) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        for k in (1, 2, 3):
            name = f"{split}_source{k}.tsv"
            t = time.time()
            df = read_tsv(config.DATA_DIR / split / name)
            assert list(df.columns) == SOURCE_COLS, f"{name}: columns {list(df.columns)}"
            _check_rows(name, len(df))
            assert df["entity_id"].is_unique, f"{name}: duplicate entity_id"
            df["source"] = pd.Series(k, index=df.index, dtype="int8")
            df.to_parquet(RAW / f"{split}_s{k}.parquet", index=False)
            rows[name] = len(df)
            print(f"  {name}: {len(df):,} rows -> raw/{split}_s{k}.parquet "
                  f"({time.time() - t:.1f}s)", flush=True)
            del df
            gc.collect()


def ingest_ground_truth(rows: dict) -> dict:
    name = "train_ground_truth.tsv"
    t = time.time()
    gt = read_tsv(config.DATA_DIR / "train" / name)
    assert list(gt.columns) == ["source1_entity_id", "matched_entity_ids"], list(gt.columns)
    _check_rows(name, len(gt))
    assert gt["source1_entity_id"].is_unique, f"{name}: duplicate source1_entity_id"
    rows[name] = len(gt)

    train_s1 = pd.read_parquet(RAW / "train_s1.parquet", columns=["entity_id"])["entity_id"]
    assert set(gt["source1_entity_id"]) == set(train_s1), "ground truth S1 ids != train_source1 ids"
    del train_s1

    lists = gt["matched_entity_ids"].str.split(",")
    pairs = pd.DataFrame({"s1_id": gt["source1_entity_id"].repeat(lists.str.len()).to_numpy(),
                          "other_id": lists.explode().to_numpy()})
    pairs = pairs[pairs["other_id"] != ""].reset_index(drop=True)
    pairs["s1_id"] = pairs["s1_id"].astype(str)
    pairs["other_id"] = pairs["other_id"].astype(str)
    assert pairs["other_id"].is_unique, "an S2/S3 id is matched to more than one S1 (facts section 5)"
    assert pairs["other_id"].str.match(r"^S[23]-").all(), "ground truth contains a non S2-/S3- id"

    counts = pairs.groupby("s1_id").size()
    gt_s1 = pd.DataFrame({"s1_id": gt["source1_entity_id"].astype(str)})
    gt_s1["n_matches"] = gt_s1["s1_id"].map(counts).fillna(0).astype("int16")
    n_singletons = int((gt_s1["n_matches"] == 0).sum())
    assert n_singletons == EXPECTED_SINGLETONS, f"{n_singletons:,} singletons, facts.md says {EXPECTED_SINGLETONS:,}"
    assert len(pairs) == EXPECTED_GT_PAIRS, f"{len(pairs):,} gt pairs, facts.md says {EXPECTED_GT_PAIRS:,}"

    pairs.to_parquet(GT_PAIRS, index=False)
    gt_s1.to_parquet(GT_S1, index=False)
    hist = gt_s1["n_matches"].value_counts().sort_index()
    print(f"  {name}: {len(gt):,} S1, {len(pairs):,} pairs, {n_singletons:,} singletons "
          f"({time.time() - t:.1f}s)", flush=True)
    return {
        "n_gt_pairs": len(pairs),
        "n_gt_pairs_s2": int(pairs["other_id"].str.startswith("S2-").sum()),
        "n_gt_pairs_s3": int(pairs["other_id"].str.startswith("S3-").sum()),
        "n_singletons": n_singletons,
        "singleton_rate": round(n_singletons / len(gt_s1), 6),
        "mean_matches_per_s1": round(float(gt_s1["n_matches"].mean()), 4),
        "matches_per_s1_hist": {int(k): int(v) for k, v in hist.items()},
    }


def outputs_exist() -> bool:
    raws = [RAW / f"{s}_s{k}.parquet" for s in ("train", "test") for k in (1, 2, 3)]
    return all(p.exists() for p in raws + [GT_PAIRS, GT_S1, REPORT])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if outputs_exist() and not args.force:
        print("ingest: outputs exist, skipping (use --force to rebuild)")
        return
    t0 = time.time()
    rows = {}
    ingest_sources(rows)
    gt_stats = ingest_ground_truth(rows)
    elapsed = round(time.time() - t0, 1)
    report = {"rows": rows, "rows_match_facts": True, **gt_stats,
              "elapsed_s": elapsed, "peak_mem_gb": peak_mem_gb()}
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"ingest: done in {elapsed}s, peak memory {report['peak_mem_gb']} GB -> {REPORT}")


if __name__ == "__main__":
    main()
