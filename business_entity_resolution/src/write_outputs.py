"""Write output/candidate_pairs.tsv and output/matching_results.tsv, then run the official validator
(contracts.md section 5, facts.md section 1).

Usage:
  python -m business_entity_resolution.src.write_outputs --version vNN   model: work/pred/test/ + models/lgb_vNN.json
  python -m business_entity_resolution.src.write_outputs --baseline      work/pred/test_baseline.parquet (Step 2.5)
  python -m business_entity_resolution.src.write_outputs --dry-run       empty lists for every test S1
Add --check-ids to also run the validator's ID-existence check (uses a few GB).
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

import pandas as pd

from . import config
from .io_utils import load_test_s1_ids, write_id_list_tsv

MATCH_HEADER = "matched_entity_ids"
CAND_HEADER = "candidate_entity_ids"
VALIDATOR = config.ROOT / "utils" / "validate_submission.py"


class JoinedLists(Mapping):
    """s1_id -> list of ids, stored as one comma-joined string per S1 to keep memory low."""

    def __init__(self, joined: Dict[str, str]):
        self._joined = joined

    @classmethod
    def from_pairs(cls, df: pd.DataFrame) -> "JoinedLists":
        if df.empty:
            return cls({})
        agg = df.groupby("s1_id", sort=False)["other_id"].agg(",".join)
        return cls(dict(zip(agg.index, agg.to_numpy())))

    def __getitem__(self, s1: str) -> List[str]:
        return self._joined[s1].split(",")

    def __iter__(self):
        return iter(self._joined)

    def __len__(self) -> int:
        return len(self._joined)


def _examples(items, n: int = 5) -> str:
    items = list(items)
    return ", ".join(map(str, items[:n])) + (f", ... ({len(items)} total)" if len(items) > n else "")


def check_matches_in_candidates(cand_map: Mapping[str, Iterable[str]],
                                match_map: Mapping[str, Iterable[str]]) -> None:
    bad = []
    for s1, ids in match_map.items():
        extra = set(ids) - set(cand_map.get(s1, ()))
        if extra:
            bad.append(f"{s1}: {sorted(extra)[:3]}")
    if bad:
        raise ValueError(f"{len(bad)} S1 have matched ids that are not candidates, e.g. {_examples(bad)}")


def run_validator(out_dir: Path, test_dir: Path, check_ids: bool = False) -> int:
    cmd = [sys.executable, str(VALIDATOR), "--matching", str(out_dir / "matching_results.tsv"),
           "--candidate", str(out_dir / "candidate_pairs.tsv"), "--test-dir", str(test_dir)]
    if check_ids:
        cmd.append("--check-ids")
    return subprocess.run(cmd, cwd=config.ROOT).returncode


def write_submission(cand_map: Mapping[str, Iterable[str]], match_map: Mapping[str, Iterable[str]], *,
                     out_dir: Optional[Path] = None, s1_ids: Optional[List[str]] = None,
                     test_dir: Optional[Path] = None, check_ids: bool = False, validate: bool = True) -> int:
    """Write both TSVs (one row per test S1, in test_source1.tsv order) and return the validator exit code."""
    out_dir = Path(out_dir or config.OUTPUT)
    test_dir = Path(test_dir or config.DATA_DIR / "test")
    s1_ids = load_test_s1_ids() if s1_ids is None else s1_ids

    known = set(s1_ids)
    unknown = [s for m in (cand_map, match_map) for s in m if s not in known]
    if unknown:
        raise ValueError(f"{len(unknown)} S1 ids are not in the test S1 list, e.g. {_examples(unknown)}")
    check_matches_in_candidates(cand_map, match_map)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_id_list_tsv(out_dir / "candidate_pairs.tsv", CAND_HEADER, s1_ids, cand_map)
    write_id_list_tsv(out_dir / "matching_results.tsv", MATCH_HEADER, s1_ids, match_map)
    n_cand = sum(1 for s in s1_ids if cand_map.get(s))
    n_match = sum(1 for s in s1_ids if match_map.get(s))
    print(f"wrote {len(s1_ids):,} rows to {out_dir} ({n_cand:,} S1 with candidates, {n_match:,} with matches)",
          flush=True)
    return run_validator(out_dir, test_dir, check_ids) if validate else 0


# ----------------------------------------------------------------------------- CLI inputs

def _read_shards(folder: Path, columns=None) -> pd.DataFrame:
    files = sorted(folder.glob("shard=*.parquet"))
    if not files:
        raise FileNotFoundError(f"no shard=*.parquet files in {folder}")
    return pd.concat([pd.read_parquet(f, columns=columns) for f in files], ignore_index=True)


def _candidates() -> JoinedLists:
    return JoinedLists.from_pairs(_read_shards(config.WORK / "cand" / "test", ["s1_id", "other_id"]))


def _model_matches(version: str) -> JoinedLists:
    meta = json.loads((config.MODELS / f"lgb_{version}.json").read_text(encoding="utf-8"))
    pred = _read_shards(config.WORK / "pred" / "test")
    # predict.py (Step 3.2) stores post-processed probs and a kept flag; fall back to the stored threshold
    kept = pred["kept"].astype(bool) if "kept" in pred.columns else pred["prob"] >= meta["threshold"]
    print(f"model {version}: threshold {meta['threshold']}, {int(kept.sum()):,} kept of {len(pred):,} scored pairs")
    return JoinedLists.from_pairs(pred.loc[kept, ["s1_id", "other_id"]])


def _baseline_matches() -> JoinedLists:
    pred = pd.read_parquet(config.WORK / "pred" / "test_baseline.parquet")
    if "kept" in pred.columns:
        pred = pred[pred["kept"].astype(bool)]
    return JoinedLists.from_pairs(pred[["s1_id", "other_id"]])


def main() -> None:
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--version", help="model version, e.g. v02")
    mode.add_argument("--baseline", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    ap.add_argument("--check-ids", action="store_true", help="validator ID-existence check (a few GB of RAM)")
    args = ap.parse_args()
    t0 = time.time()

    if args.dry_run:
        cand_map, match_map = {}, {}
    else:
        cand_map = _candidates()
        match_map = _baseline_matches() if args.baseline else _model_matches(args.version)
    code = write_submission(cand_map, match_map, check_ids=args.check_ids)
    print(f"write_outputs: validator exit code {code}, {time.time() - t0:.1f}s")
    sys.exit(code)


if __name__ == "__main__":
    main()
