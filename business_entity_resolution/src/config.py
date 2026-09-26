"""Paths, seeds and constants. The only place for tunables (contracts.md section 2)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # student_resource/
DATA_DIR = Path(os.environ.get("ER_DATA_DIR", ROOT / "dataset"))
WORK, MODELS, REPORTS, OUTPUT = ROOT/"work", ROOT/"models", ROOT/"reports", ROOT/"output"
SEED = 42
N_SHARDS = 16          # S1 entities are split into shards by stable_hash(s1_id) % N_SHARDS
VAL_MOD = 5            # fold = "val" if stable_hash(s1_id + "#fold") % VAL_MOD == 0 else "fit"
TOPK = 50              # max candidates kept per S1 after blocking (tune with recall curve)
BLOCK_CAP = 1000       # a blocking key group larger than this on either side is dropped
DUCKDB_MEMORY = "6GB"  # lower on small machines
N_JOBS = max(1, os.cpu_count() - 2)

for _d in (WORK, MODELS, REPORTS, OUTPUT):
    _d.mkdir(parents=True, exist_ok=True)
