# Contracts: layout, interfaces, file schemas

These contracts let four people build modules in parallel. Do not rename files, columns or CLI flags defined here.
If a contract must change, update this file in the same commit and tell the team.

## 1. Repository layout (repo root = `student_resource/`)

```
student_resource/                      <- git repo root; Claude Code is opened here
├── .claude/skills/amazon-er-challenge/ <- this skill
├── CLAUDE.md                          <- points to the skill
├── TEAM_EXECUTION_GUIDE.md            <- copy-paste prompts
├── dataset/                           <- GIVEN, never modified, git-ignored
├── utils/validate_submission.py       <- GIVEN, never modified
├── Documentation_template.md          <- GIVEN, filled in during packaging
├── business_entity_resolution/        <- becomes code/business_entity_resolution/ in the zip
│   ├── src/
│   │   ├── __init__.py
│   │   ├── config.py        paths, seeds, constants (only place for tunables)
│   │   ├── io_utils.py      read_tsv(), write_id_list_tsv(), stable_hash()
│   │   ├── ingest.py        TSV -> parquet, ground-truth pairs
│   │   ├── splits.py        fold + shard assignment
│   │   ├── metric.py        copy of reference/f05_metric.py
│   │   ├── evaluate.py      score predictions / candidates on the val fold
│   │   ├── normalize.py     text normalisation (country-agnostic + France rules)
│   │   ├── aliases.py       mines state/region alias tables from train pairs
│   │   ├── blocking.py      candidate generation (DuckDB)
│   │   ├── features.py      pairwise features (rapidfuzz)
│   │   ├── train.py         LightGBM training + threshold tuning
│   │   ├── predict.py       scoring + one-to-one post-processing
│   │   ├── write_outputs.py writes both output TSVs
│   │   ├── baseline.py      rule-based scorer for the first submission
│   │   └── run_pipeline.py  end-to-end entry point
│   ├── tests/               pytest unit tests
│   ├── README.md
│   └── requirements.txt
├── work/        (git-ignored) parquet intermediates
├── models/      LightGBM model files + metadata JSON (small, committed)
├── reports/     JSON/MD reports, committed; the ONLY source for numbers in the documentation
├── output/      (git-ignored) matching_results.tsv, candidate_pairs.tsv
└── submissions/ SUBMISSION_LOG.md committed; vNN/ copies git-ignored
```

Every script runs from the repo root as a module: `python -m business_entity_resolution.src.<module> [flags]`.
To make this work, `business_entity_resolution/__init__.py` and `business_entity_resolution/src/__init__.py` both exist (empty).

## 2. config.py (single source of tunables)

```python
ROOT = Path(__file__).resolve().parents[2]          # student_resource/
DATA_DIR = Path(os.environ.get("ER_DATA_DIR", ROOT / "dataset"))
WORK, MODELS, REPORTS, OUTPUT = ROOT/"work", ROOT/"models", ROOT/"reports", ROOT/"output"
SEED = 42
N_SHARDS = 16          # S1 entities are split into shards by stable_hash(s1_id) % N_SHARDS
VAL_MOD = 5            # fold = "val" if stable_hash(s1_id + "#fold") % VAL_MOD == 0 else "fit"
TOPK = 50              # max candidates kept per S1 after blocking (tune with recall curve)
BLOCK_CAP = 1000       # a blocking key group larger than this on either side is dropped
DUCKDB_MEMORY = os.environ.get("ER_DUCKDB_MEMORY", "4GB")  # override per machine via ER_DUCKDB_MEMORY
N_JOBS = max(1, os.cpu_count() - 2)
```
`stable_hash(s: str) -> int` = `zlib.crc32(s.encode("utf-8"))`. Never use Python's built-in `hash()`: it is salted per process.

## 3. Intermediate files (all parquet, all ID columns are strings)

| Path | Columns | Produced by |
|---|---|---|
| `work/raw/{split}_s{1,2,3}.parquet` | entity_id, business_name, business_address, country, source (int8: 1/2/3) | ingest |
| `work/gt_pairs.parquet` | s1_id, other_id | ingest (train only) |
| `work/gt_s1.parquet` | s1_id, n_matches (int16) | ingest |
| `work/splits.parquet` | s1_id, fold ("fit"/"val"), shard (int16) | splits (train S1) |
| `work/test_shards.parquet` | s1_id, shard | splits (test S1) |
| `work/norm/{split}_s{k}.parquet` | entity_id, source, country, name_norm, name_core, name_nospace, name_has_url, name_nonlatin, addr_norm, addr_core, addr_nums, addr_first_num, addr_admin, addr_empty | normalize |
| `work/aliases.json` | {"admin": {variant: canonical}} learned from train pairs | aliases |
| `work/cand/{split}/shard={i}.parquet` | s1_id, other_id, key_hits (int32 bitmask), cheap_score (float32) | blocking |
| `work/feat/{split}/shard={i}.parquet` | s1_id, other_id, feature columns (float32), label (int8, train only) | features |
| `work/pred/{split}/shard={i}.parquet` | s1_id, other_id, prob (float32) | predict |

`{split}` is `train` or `test`. Train blocking and features cover ALL train S1 entities (both folds), because the one-to-one step needs every competing S1 to behave as it will on test.

Column meanings from normalize:
- `name_norm`: anyascii → lowercase → `&`→`and` → punctuation to space → collapse spaces.
- `name_core`: name_norm minus legal suffixes, filler words, honorifics, URL parts and duplicated adjacent tokens. Lists live in `normalize.py` constants.
- `name_nospace`: name_core with spaces removed (catches `lifeinvestments.com` vs `Life Investments`).
- `addr_norm`: anyascii → lowercase → `null` removed → abbreviations expanded (street types, French `r`/`av`/`bd`) → leading zeros stripped from numbers.
- `addr_nums`: space-joined numeric tokens in order. `addr_first_num`: first of them or "".
- `addr_admin`: canonical state/region tokens found in the address via `aliases.json`. `addr_core`: addr_norm minus admin tokens.
- Features must NOT use `country` as an input. It is only a blocking partition key. France is unseen in training.

## 4. CLIs (exact flags)

```
python -m business_entity_resolution.src.ingest
python -m business_entity_resolution.src.splits
python -m business_entity_resolution.src.normalize --split {train,test} [--limit N]
python -m business_entity_resolution.src.aliases
python -m business_entity_resolution.src.blocking --split {train,test} [--shards 0,1,...]
python -m business_entity_resolution.src.evaluate --what candidates|predictions [--shards ...]
python -m business_entity_resolution.src.features --split {train,test} [--shards ...]
python -m business_entity_resolution.src.train --version vNN [--max-rows N]
python -m business_entity_resolution.src.predict --split {train,test} --version vNN
python -m business_entity_resolution.src.baseline --split {train,test}
python -m business_entity_resolution.src.write_outputs --version vNN   (or --baseline)
python -m business_entity_resolution.src.run_pipeline --version vNN    (test end-to-end)
```
Every stage skips shards whose output already exists unless `--force` is passed. Every stage logs row counts and elapsed time.

## 5. Output writing (only via `io_utils.write_id_list_tsv`)

- Start from the full list of test S1 IDs in `test_source1.tsv` order, so every S1 gets exactly one row.
- Deduplicate each list while keeping order. Keep only IDs starting with `S2-` or `S3-`.
- Write with plain Python `open(path, "w", encoding="utf-8", newline="\n")`: `f"{s1}\t{','.join(ids)}\n"`. Do not use pandas `to_csv` for these, because it may quote fields.
- Headers are exactly `source1_entity_id\tmatched_entity_ids` and `source1_entity_id\tcandidate_entity_ids`.
- `candidate_pairs.tsv` is written from `work/cand/test/` after the same filtering the model sees. Every matched ID must also be a candidate.

## 6. Report files (numbers for the documentation come only from here)

| File | Keys (minimum) |
|---|---|
| `reports/env_<member>.json` | member, os, python, ram_gb, cpus, packages{name: version} |
| `reports/ingest.json` | row counts per file, n_gt_pairs, n_singletons |
| `reports/splits.json` | n_fit, n_val, shard sizes |
| `reports/blocking_{split}.json` | n_pairs, pairs_per_s1 (mean, p50, p95, max), per_key_pairs, elapsed_s; train also: recall_all, recall_by_country, recall_at_k {10,20,30,50}, s1_with_zero_candidates |
| `reports/baseline_val.json` | threshold, macro_f05, precision, recall, per_country |
| `reports/train_vNN.json` | features, n_rows_fit, params, best_iter, threshold, val macro_f05 before/after one-to-one, per_country, feature_importance(top 20) |
| `reports/errors_vNN.md` | 30 false positives + 30 false negatives with both records printed |
| `reports/test_run_vNN.json` | n_s1, n_candidates, n_matched_pairs, share_s1_with_matches per country, mean matches per S1 per country, validator exit code |
| `submissions/SUBMISSION_LOG.md` | version, date-time IST, git commit, model, val macro F0.5, public LB F0.5, notes |
