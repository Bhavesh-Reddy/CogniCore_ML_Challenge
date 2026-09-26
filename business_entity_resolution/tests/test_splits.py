import json
import os
import subprocess
import sys
from pathlib import Path

from business_entity_resolution.src import config
from business_entity_resolution.src.splits import fold_of, shard_of

ROOT = Path(__file__).resolve().parents[2]
IDS = ["S1-925783039", "S1-773889195", "S1-714132312", "S1-106407869", "S1-x"]
SNIPPET = (
    "import json, sys\n"
    "from business_entity_resolution.src.io_utils import stable_hash\n"
    "from business_entity_resolution.src.splits import fold_of, shard_of\n"
    "ids = json.loads(sys.argv[1])\n"
    "print(json.dumps({'hash': [stable_hash(s) for s in ids], 'builtin': [hash(s) for s in ids],\n"
    "                  'shard': shard_of(ids).tolist(), 'fold': fold_of(ids).tolist()}))\n"
)


def _run(hash_seed: str) -> dict:
    env = {**os.environ, "PYTHONHASHSEED": hash_seed}
    out = subprocess.run([sys.executable, "-c", SNIPPET, json.dumps(IDS)], cwd=ROOT, env=env,
                         capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def test_stable_hash_deterministic_across_processes():
    a, b = _run("1"), _run("2")
    assert a["builtin"] != b["builtin"]  # built-in hash() is salted per process
    assert a["hash"] == b["hash"]
    assert a["shard"] == b["shard"]
    assert a["fold"] == b["fold"]


def test_shard_and_fold_ranges():
    shards = shard_of(IDS)
    assert shards.min() >= 0 and shards.max() < config.N_SHARDS
    assert set(fold_of(IDS)) <= {"fit", "val"}
