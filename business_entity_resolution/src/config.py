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
PREFILTER_K = 200      # blocking stage 1: candidates per S1 kept by key rarity before cheap_score
DUCKDB_MEMORY = "6GB"  # lower on small machines
N_JOBS = max(1, os.cpu_count() - 2)

# aliases.py (Step 2.1): admin-region aliases learned from matched train pairs
ALIAS_SAMPLE = 300_000     # gt pairs sampled (seed SEED)
ALIAS_MIN_COUNT = 50       # min co-occurrences for variant -> canonical, and min frequency of a canonical
ALIAS_MIN_SHARE = 0.8      # count(v, c) must be at least this share of v's occurrences
ALIAS_LAST_SHARE = 0.8     # an admin value is the LAST comma-part in at least this share of its records (states >= 0.90, cities <= 0.62 in train)

for _d in (WORK, MODELS, REPORTS, OUTPUT):
    _d.mkdir(parents=True, exist_ok=True)
