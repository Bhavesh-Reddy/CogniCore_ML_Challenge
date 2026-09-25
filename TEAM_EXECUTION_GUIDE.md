# Team Execution Guide: Amazon ML Challenge 2026 (Business Entity Resolution)

**How to use this guide**
- Every step has an owner, its dependencies, and a prompt inside a code block. Copy the whole block and paste it into Claude Code, opened in your `student_resource/` folder.
- Do not paste a step until every step it depends on shows ✅ in the team chat.
- When Claude finishes, check its summary against the step's **Done when** line. Then post `✅ <step id> + key numbers` in the team chat.
- If Claude reports a failed check, do not work around it. Paste the output in the team chat and let the owner of the failing module fix it.
- The rules, facts and interfaces live in the skill at `.claude/skills/amazon-er-challenge/`. Every prompt tells Claude to use it.

## Roles

| Member | Role | Owns |
|---|---|---|
| **A** | Lead, integrator, **the only person who uploads to the portal**. Uses the machine with the most RAM for full-scale runs. | repo, ingest, splits, full runs, run_pipeline, submissions, zip |
| **B** | Blocking | metric, evaluate, blocking, predict + one-to-one post-processing |
| **C** | Matching model | normalize, features, LightGBM training |
| **D** | Outputs, quality and docs | output writer, aliases, baseline, France checks, README, documentation |

**Machines.** Everyone develops on their own machine using **dev mode = shards 0-3** of train. Member A runs **full mode = all 16 shards** and all test runs. Intermediate files under `work/` are never shared through git. They are rebuilt locally with the Sync prompt.

**Submissions.** At most 5 per day. Plan: v01 baseline, v02 first model, then only submit when the validation score improves. Keep at least 1 upload spare on the last day.

## Timeline overview

| Phase | Steps | Parallel? | Target |
|---|---|---|---|
| 0 Setup | 0.1 → 0.2 | 0.2 by all | first hour |
| 1 Foundations | 1A, 1B, 1C, 1D | all 4 in parallel | +3 h |
| 2 Candidates + baseline | 2.1 → 2.2 → 2.3 → 2.5 (A submits v01); 2.4 in parallel | partly | +8 h |
| 3 Model | 3.1, 3.2 → 3.3 (A submits v02) | partly | +14 h |
| 4 Improve | 4.x loops | all 4 | until 6 h before deadline |
| 5 Package | 5.1, 5.2, 5.3 | D + A | last 6 h |

---

## Phase 0: Setup

### Step 0.1: Create the repo and scaffold (Owner: A. Depends on: nothing)

The team repo is `https://github.com/Bhavesh-Reddy/CogniCore_ML_Challenge.git`. It already holds the skill, this guide and CLAUDE.md. Before pasting, add B, C and D as collaborators on GitHub, and make sure the repo is private.

```text
Use the amazon-er-challenge skill. Read SKILL.md and reference/contracts.md sections 1, 2 and 5.

Task: scaffold the repository in the current folder (student_resource/).
0. Connect this folder to the team repo first: if this folder is not its own git repo (check that `git rev-parse --show-toplevel` prints this folder), run git init, git remote add origin https://github.com/Bhavesh-Reddy/CogniCore_ML_Challenge.git, git fetch origin, git checkout -f -t origin/main. Otherwise git pull. The dataset/ folder must remain untouched and must never be committed.
1. Create the directory tree from contracts.md §1: business_entity_resolution/src/, business_entity_resolution/tests/, models/, reports/, submissions/. Create empty business_entity_resolution/__init__.py and business_entity_resolution/src/__init__.py.
2. Write business_entity_resolution/src/config.py exactly per contracts.md §2. It must create WORK, MODELS, REPORTS and OUTPUT directories on import if missing.
3. Write business_entity_resolution/src/io_utils.py with:
   - read_tsv(path, nrows=None) -> pd.DataFrame using sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE.
   - stable_hash(s) -> int using zlib.crc32(s.encode("utf-8")).
   - write_id_list_tsv(path, header_col, s1_ids_in_order, mapping) per contracts.md §5: one row per S1 in the given order, dedupe while keeping order, drop anything not starting with "S2-" or "S3-", plain open(..., "w", encoding="utf-8", newline="\n").
   - load_test_s1_ids() -> list[str] in file order from DATA_DIR/test/test_source1.tsv.
4. Copy .claude/skills/amazon-er-challenge/reference/f05_metric.py to business_entity_resolution/src/metric.py unchanged.
5. Write business_entity_resolution/tests/test_io_utils.py covering: read_tsv keeps the literal string "null", write_id_list_tsv output passes a round-trip read, dedupe works, S1- ids are dropped, empty lists produce "S1-x\t" lines.
6. Write business_entity_resolution/requirements.txt with the exact versions currently installed of: pandas, numpy, pyarrow, polars, duckdb, rapidfuzz, anyascii, lightgbm, tqdm, pytest. Install missing ones first with pip. Read versions from importlib.metadata; do not guess them.
7. Create submissions/SUBMISSION_LOG.md with a table header: | version | date-time IST | git commit | model | val macro F0.5 | public LB F0.5 | notes |.
8. Run: python -m business_entity_resolution.src.metric and python -m pytest business_entity_resolution/tests -q. Both must pass.
9. Confirm .gitignore excludes dataset/, work/, output/ and *.zip, and that `git status` shows no file larger than 50 MB. Commit everything with message "scaffold" and push to origin main.

Done when: metric prints PASS with 0.7143, pytest passes, push succeeded. Print `git log --oneline -1` and the list of committed files.
```
**Done when:** push succeeded and tests pass. Post the repo URL.

### Step 0.2: Join the repo and record your machine (Owner: A, B, C, D. Depends on: 0.1)

Members B, C, D: extract the dataset zip first, so that `student_resource/dataset/train` and `dataset/test` exist, then open Claude Code in `student_resource/`. Replace `<MEMBER>` with your letter.

```text
Use the amazon-er-challenge skill.

Task: connect this folder to the team repo and record this machine.
1. If this folder is not yet a git repo: git init, git remote add origin https://github.com/Bhavesh-Reddy/CogniCore_ML_Challenge.git, git fetch origin, git checkout -f -t origin/main. Otherwise: git pull. The dataset/ folder must remain untouched.
2. pip install -r business_entity_resolution/requirements.txt
3. Verify the dataset: count lines of every file in dataset/train and dataset/test with a streaming line count (do not load them into pandas). Compare with reference/facts.md §3 (lines = rows + 1 header). Report any mismatch.
4. Write reports/env_<MEMBER>.json with: member, os, python version, total RAM in GB, logical CPUs, and the installed versions of every package in requirements.txt.
5. Run python -m pytest business_entity_resolution/tests -q.
6. Commit reports/env_<MEMBER>.json with message "env <MEMBER>" and push.

Done when: line counts match facts.md, tests pass, env report pushed. Print RAM and CPU count.
```
**Done when:** everyone posts RAM and CPUs. Member A confirms they have the most RAM; if not, the member with the most RAM becomes the full-run machine and runs every step marked "A (full run)".

### Sync prompt (use any time you pull new code and need local intermediates)

```text
Use the amazon-er-challenge skill.
git pull. Then make sure the local work/ intermediates exist for dev mode by running, in order, only the stages whose code exists and whose outputs are missing (each stage skips existing outputs): ingest, splits, aliases, normalize --split train, normalize --split test, blocking --split train --shards 0,1,2,3, features --split train --shards 0,1,2,3. Stop at the first stage that fails and show its error. Print which stages ran and their elapsed time.
```

---

## Phase 1: Foundations (all four in parallel)

### Step 1A: Ingest and splits (Owner: A. Depends on: 0.1)

```text
Use the amazon-er-challenge skill. Follow contracts.md §3 exactly.

Task: implement and run ingest.py and splits.py.
1. business_entity_resolution/src/ingest.py:
   - For split in (train, test) and k in (1,2,3): read DATA_DIR/{split}/{split}_source{k}.tsv with io_utils.read_tsv, add column source (int8 = k), write work/raw/{split}_s{k}.parquet.
   - Ground truth: read train_ground_truth.tsv, explode matched_entity_ids on "," (empty string means no matches), write work/gt_pairs.parquet (s1_id, other_id) and work/gt_s1.parquet (s1_id, n_matches int16) including singletons with 0.
   - Assert every row count equals reference/facts.md §3. Assert entity_id uniqueness per file. Assert each other_id appears at most once in gt_pairs (facts §5).
   - Write reports/ingest.json per contracts §6.
2. business_entity_resolution/src/splits.py:
   - Train S1: shard = stable_hash(s1_id) % N_SHARDS; fold = "val" if stable_hash(s1_id + "#fold") % VAL_MOD == 0 else "fit". Write work/splits.parquet.
   - Test S1: shard the same way. Write work/test_shards.parquet.
   - Write reports/splits.json with n_fit, n_val, shard sizes, and the singleton rate in each fold.
3. Add tests in business_entity_resolution/tests/test_splits.py showing stable_hash is deterministic across two subprocesses.
4. Run both modules. Print elapsed time and peak memory (psutil if installed, otherwise skip).
5. Commit code + reports and push.

Done when: all asserts pass and reports/ingest.json matches facts.md §3.
```
**Done when:** A posts n_fit, n_val and the singleton rate per fold.

### Step 1B: Evaluator (Owner: B. Depends on: 0.1)

```text
Use the amazon-er-challenge skill. Use business_entity_resolution/src/metric.py; do not re-implement the formula.

Task: write business_entity_resolution/src/evaluate.py with these functions and a CLI (contracts §4):
1. load_truth(fold="val", shards=None) -> (s1_ids list, truth dict s1_id -> set(other_ids)). Uses work/gt_s1.parquet, work/gt_pairs.parquet, work/splits.parquet. Includes singletons.
2. eval_candidates(shards) -> dict: over val S1 in those shards: pair recall (true pairs found in candidates / all true pairs), recall by country, recall_at_k for k in (10,20,30,50) using cheap_score order, share of val S1 with zero candidates, mean/p50/p95/max candidates per S1, per-key recall using the key_hits bitmask, and oracle_macro_f05 = macro F0.5 when predicting exactly (candidates ∩ truth). Reads work/cand/train/shard={i}.parquet.
3. eval_predictions(pred_df, threshold) -> dict with macro_f05, micro precision, micro recall, macro_f05 per country, macro_f05 on singletons only and on non-singletons only. pred_df has s1_id, other_id, prob and has already been post-processed.
4. best_threshold(pred_df, grid=np.arange(0.05, 0.96, 0.01)) -> (threshold, macro_f05). Must be vectorised with pandas groupby, not a Python loop over S1 per threshold, so it runs in seconds on 500k S1.
5. CLI: --what candidates|predictions, --shards 0,1,2,3, --pred-path, --out reports/<name>.json.
6. Tests in business_entity_resolution/tests/test_evaluate.py with a synthetic 5-entity example whose macro F0.5 you compute by hand in the test comments, including one singleton predicted empty (1.0) and one singleton with a false match (0.0).
7. Run pytest. Commit and push.

Done when: tests pass. Real data is not needed yet.
```

### Step 1C: Normalization (Owner: C. Depends on: 0.1)

```text
Use the amazon-er-challenge skill. Read reference/facts.md §6 and §7 carefully; they list the real noise patterns. Follow the normalize columns in contracts.md §3.

Task: write business_entity_resolution/src/normalize.py.
1. basic_clean(s): anyascii → lowercase → "&" to " and " → any character that is not [a-z0-9] to space → collapse spaces → strip. Export it; aliases.py will import it.
2. Name functions:
   - name_norm = basic_clean(name) after removing URL parts: "www.", "http(s)://", and domain endings like ".com", ".in", ".org", ".net", ".co", ".fr", plus a leading "#" (hashtags) and anything after " | ".
   - name_core = name_norm minus: legal forms (llc, l l c, inc, incorporated, corp, corporation, co, company, ltd, limited, pvt, private, llp, lp, pc, p c, plc, pllc, sarl, sas, sasu, eurl, sa, sci, snc, cie, gmbh), filler words (center, centre, services, service, group, and, the, of, fka, aka, dba, f k a, a k a), honorifics (shri, sri, smt, mr, mrs, dr, m d, md, dmd), and adjacent duplicate tokens. If removal would leave it empty, fall back to name_norm.
   - name_nospace = name_core without spaces. name_has_url (0/1) from the raw name. name_nonlatin (0/1) if the raw name has any character outside Latin script.
3. Address functions:
   - Remove literal null tokens. basic_clean. Strip leading zeros from numbers ("00272" → "272").
   - Expand abbreviations with whole-token dictionaries: US (st street, rd road, ave/av avenue, blvd boulevard, dr drive, ln lane, ct court, cir circle, pl place, pkwy parkway, hwy highway, trl trail, ter terrace, sq square, n/s/e/w directions, ste suite, apt apartment, so south), India (no number, h hno house, nr near, opp opposite, flr floor, blk block, sec sector, mg marg), France (r rue, av avenue, bd boulevard, all allee, imp impasse, che chemin, pl place, rte route, fbg faubourg, st saint, ste sainte). Note that "st" is ambiguous (street vs saint); use one mapping and write a test that documents the choice.
   - addr_nums, addr_first_num per contracts §3.
   - addr_admin and addr_core use work/aliases.json if it exists; if not, addr_admin = "" and addr_core = addr_norm. Step 2.2 re-runs this after aliases exist.
   - addr_empty (0/1).
4. CLI --split {train,test} [--limit N]: process work/raw/{split}_s{1,2,3}.parquet in chunks of 500k rows with multiprocessing (config.N_JOBS), write work/norm/{split}_s{k}.parquet. Print rows/sec.
5. Tests in business_entity_resolution/tests/test_normalize.py using these REAL examples from facts.md: "Kochar Góld Private" vs "Kochar Gold Private Limited" (same name_core), "lifeinvestments.com" vs "Life Investments" (same name_nospace), "#empiremolecular" vs "Empire Molecular Inc." (same name_nospace), "00272 LAGO GRANDE DRIVE" (first num 272), "41 BIRCH HILL DR" vs "41 Birch Hill Drive" (same addr_norm), "63 R. DE DIEPPE" → contains "rue", "Summit Society L.L.C." vs "Summit Society LLC" (same name_core), an address with "NULL" (no "null" token remains).
6. Run with --limit 200000 on train only if work/raw exists (Step 1A); otherwise build a tiny parquet from the first 2000 lines of each train TSV for testing. Print 15 before/after examples for names and 15 for addresses.
7. Commit and push.

Done when: tests pass and the before/after examples look right. Do not run the full dataset yet.
```

### Step 1D: Output writer and format dry-run (Owner: D. Depends on: 0.1)

```text
Use the amazon-er-challenge skill. Read contracts.md §5 and facts.md §1.

Task: write business_entity_resolution/src/write_outputs.py and prove the output format with the official validator.
1. write_outputs.py:
   - Function write_submission(cand_map, match_map) → uses io_utils.load_test_s1_ids() and io_utils.write_id_list_tsv to write output/candidate_pairs.tsv and output/matching_results.tsv.
   - Before writing, assert that every matched id is in that S1's candidate list; if not, raise with 5 examples.
   - After writing, run: python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test, and return its exit code.
   - CLI: --version vNN reads work/pred/test/ and work/cand/test/ plus the threshold from models/lgb_vNN.json. --baseline reads baseline predictions (Step 2.5). --dry-run writes empty lists for every test S1.
2. Run --dry-run. The validator must print PASS. Also run the validator with --check-ids on the dry-run output.
3. Write a negative test script business_entity_resolution/tests/test_write_outputs.py that builds small fake outputs in a temp dir: a duplicate id in a list, a missing S1 row, an S1- id, a comma-separated header. Show that write_id_list_tsv prevents each of them.
4. Commit and push. Do not commit output/.

Done when: dry-run validator prints PASS. Paste its last 5 lines.
```

---

## Phase 2: Candidates and first submission

### Step 2.1: Mine alias tables from training data (Owner: D. Depends on: 1A, 1C)

```text
Use the amazon-er-challenge skill. Only training data may be used; no external lists of states or cities.

Task: write business_entity_resolution/src/aliases.py that learns admin-region aliases (state names, codes and native-script forms) from matched train pairs.
1. Take a 300k sample of work/gt_pairs.parquet (seed 42), join raw S1 and S2/S3 addresses.
2. Split each address on commas, apply normalize.basic_clean to each part, and take the last 1 and last 2 comma-parts from both sides.
3. Count co-occurrences of (S1 part, other part) where they differ. A variant v maps to canonical c when: count(v, c) >= 50 and it is at least 80% of v's co-occurrences. The canonical side is always the S1 form.
4. Also add identity entries for every canonical value with frequency >= 50.
5. Write work/aliases.json as {"admin": {variant: canonical}} and reports/aliases.md listing the top 60 mappings with counts, grouped by country.
6. Sanity-check that these appear if the data supports them, and print the mapping or "not learned" for each: mh→maharashtra, mharastr→maharashtra, tx→texas or texas→tx (whichever direction the S1 side uses), krnatk→karnataka, dl→delhi. Do not add any mapping by hand.
7. France has no training data, so no France aliases will be learned. Say so in reports/aliases.md. Hand-written French street-type expansions are already in normalize.py.
8. Commit code and reports/aliases.md, then push. work/aliases.json is rebuilt locally by the Sync prompt.

Done when: the sanity list is printed and the mapping file exists.
```

### Step 2.2: Full normalization run (Owner: C for own machine, A for the full-run machine. Depends on: 2.1)

```text
Use the amazon-er-challenge skill.
git pull. Rebuild work/aliases.json if missing (python -m business_entity_resolution.src.aliases).
Run normalize --split train and normalize --split test on the full data with --force. Report rows/sec, elapsed time and peak memory.
Then print, for each split and source: share of empty name_core, share of empty addr_norm, share of rows with addr_first_num, and share with non-empty addr_admin, broken down by country. Write this to reports/normalize_stats.json and commit it.
If memory exceeds 70% of RAM, reduce the chunk size and retry once; report what you changed.
```
**Done when:** stats are committed. Check that France rows have non-empty `addr_norm` and numbers, the same as US and India.

### Step 2.3: Blocking (Owner: B develops in dev mode, then A runs full. Depends on: 1B, 2.2)

```text
Use the amazon-er-challenge skill. Read SKILL.md pipeline step 4 and contracts.md §3-4. Facts to respect: matches never cross country (facts §5), 14.4% of true pairs share no name token (facts §6), no PIN codes exist (facts §6).

Task: write business_entity_resolution/src/blocking.py using DuckDB (memory_limit = config.DUCKDB_MEMORY, temp_directory = work/duckdb_tmp).
1. Key extraction from work/norm/{split}_s*.parquet. For each record, emit rows (entity_id, country, key_type, key):
   - K1 bit 1: first_num + "|" + each name_core token of length >= 2
   - K2 bit 2: first_num + "|" + each addr_core alphabetic token of length >= 3 (address-only key)
   - K3 bit 4: each name_core token of length >= 3 + "|" + each addr_core alphabetic token of length >= 4 (max 4 × 6 tokens per record)
   - K4 bit 8: first 8 characters of name_nospace (only if length >= 5)
   Records with an empty first_num simply produce no K1/K2 rows.
2. Index side = S2 + S3 of the split (all rows). Query side = S1 of the requested shards. Drop any (country, key_type, key) group whose size on either side exceeds config.BLOCK_CAP.
3. Join query keys to index keys on (country, key_type, key). Aggregate per (s1_id, other_id): key_hits = bit_or(bit).
4. cheap_score = 0.5 * token_set_ratio(name_core) + 0.3 * token_set_ratio(addr_core) + 0.2 * (first_num equal) × 100, computed with rapidfuzz.process.cpdist(..., workers=-1) and scaled to 0-1. Keep the top config.TOPK other_ids per s1_id by cheap_score.
5. Write work/cand/{split}/shard={i}.parquet with s1_id, other_id, key_hits, cheap_score. Process one shard at a time. Skip existing shards unless --force.
6. Write reports/blocking_{split}.json (contracts §6). For train, call evaluate.eval_candidates on the same shards and include its output.
7. Run in dev mode: --split train --shards 0,1,2,3. Print the evaluation, per-key recall, elapsed time per shard and peak memory.
8. Print 10 missed true pairs (val fold) with both raw records, so we can see why blocking missed them.
9. Commit code and the report. Push.

Gate: overall pair recall on val. Report it as is. If it is below 0.95, do not change keys yet. Post the missed examples and the per-key recall to the team first.
```
**Done when:** B posts recall, recall@k, pairs per S1 and seconds per shard. The team decides together whether to change `TOPK` or `BLOCK_CAP` in `config.py`.

Then **A (full run)** pastes:
```text
Use the amazon-er-challenge skill. git pull. Run the Sync prompt stages up to normalize if needed. Then run blocking --split train (all 16 shards) and blocking --split test. Report elapsed time per shard, peak memory, total candidate pairs for test, and the train val recall across all shards. Commit reports/blocking_train.json and reports/blocking_test.json. Confirm that every test S1 appears in work/cand/test with at least 0 rows (list how many test S1 have zero candidates, per country).
```

### Step 2.4: Features (Owner: C. Depends on: 2.3 dev-mode candidates. Can start coding during 2.3)

```text
Use the amazon-er-challenge skill. Read SKILL.md pipeline step 5. Never use country as a feature.

Task: write business_entity_resolution/src/features.py.
1. For each shard: read work/cand/{split}/shard={i}.parquet, join normalized fields of the S1 record and the other record from work/norm.
2. Compute with rapidfuzz.process.cpdist(workers=-1) (verify the signature with help() first):
   - name_core: fuzz.ratio, fuzz.token_set_ratio, fuzz.token_sort_ratio, fuzz.partial_ratio, JaroWinkler.normalized_similarity
   - name_nospace: fuzz.ratio, fuzz.partial_ratio
   - name_norm: fuzz.token_set_ratio
   - addr_core: fuzz.token_set_ratio, fuzz.partial_ratio; addr_norm: fuzz.token_sort_ratio
   - token Jaccard for name_core and addr_core (vectorised or with a fast Python set loop over chunks)
   - numbers: Jaccard of addr_nums sets, first_num_equal (1 equal, 0 different, -1 one side missing), share of S1 numbers present in the other address
   - admin_equal (1/0/-1 missing)
   - flags: other_addr_empty, other_name_has_url, other_name_nonlatin, other_source (2 or 3), name length ratio, address length ratio
   - key_hits split into 4 binary columns, cheap_score
   - group features per s1_id: n_candidates, rank of cheap_score (1 = best), cheap_score minus the group max, rank of name token_set and addr token_set within the group
   - competition features per other_id over all S1 in the processed shards: n_s1_claiming, rank of this S1's cheap_score among them
3. For train, add label = 1 if (s1_id, other_id) is in work/gt_pairs.parquet else 0.
4. Store all feature columns as float32. Write work/feat/{split}/shard={i}.parquet. Put the ordered feature list in business_entity_resolution/src/features.py as FEATURES, the single source for training and prediction.
5. Run on --split train --shards 0,1,2,3. Print pairs/sec, label rate, and for each feature the mean for label=1 vs label=0. Flag any feature with NaN or constant values.
6. Commit and push.

Done when: the per-feature label comparison is printed and no feature is NaN.
```

### Step 2.5: Baseline and first submission v01 (Owner: D builds, A runs full and submits. Depends on: 2.3 full run)

```text
Use the amazon-er-challenge skill. Read SKILL.md "F0.5 reasoning" and post-processing (pipeline step 7).

Task: write business_entity_resolution/src/baseline.py, a rule-based matcher used for the first leaderboard submission and as the bar the model must beat.
1. Score = cheap_score from work/cand/{split}.
2. One-to-one: for each other_id keep only the S1 with the highest score (ties broken by s1_id).
3. Tune a single threshold on val S1 with evaluate.best_threshold. Save it with the val result to reports/baseline_val.json (threshold, macro_f05, precision, recall, per_country, singletons vs non-singletons).
4. For --split test: apply the same one-to-one step and the saved threshold, then write work/pred/test_baseline.parquet.
5. Run on train dev shards first and report the val macro F0.5. Commit.
```
Then **A (full run)** pastes:
```text
Use the amazon-er-challenge skill. git pull. Run baseline --split train on all shards, then baseline --split test, then write_outputs --baseline. The validator must print PASS. Then write reports/test_run_v01.json (contracts §6) with per-country share of S1 with matches and mean matches per S1. Copy output/matching_results.tsv to submissions/v01/. Append a row to submissions/SUBMISSION_LOG.md with version v01, current IST date-time, git commit hash, model "baseline cheap_score", the val macro F0.5 from reports/baseline_val.json, public LB "pending". Commit, create git tag sub-v01, push with tags. Print the path of the file to upload.
```
**Done when:** A uploads `submissions/v01/matching_results.tsv` to the portal, gets a `SCORED` status, and fills the public score into the log.

---

## Phase 3: The model

### Step 3.1: Train LightGBM (Owner: C in dev mode, then A in full mode. Depends on: 2.4, 2.5)

```text
Use the amazon-er-challenge skill. Model must be LightGBM (MIT). Use FEATURES from features.py.

Task: write business_entity_resolution/src/train.py --version vNN [--shards ...] [--max-rows N].
1. Load work/feat/train for the given shards. Fit rows = fold "fit". Val rows = fold "val". If fit rows exceed --max-rows (default 20,000,000), sample whole S1 groups with seed 42.
2. LightGBM binary: learning_rate 0.05, num_leaves 127, min_data_in_leaf 100, feature_fraction 0.8, bagging_fraction 0.8, bagging_freq 1, lambda_l2 1.0, num_threads = config.N_JOBS, seed 42, up to 3000 rounds, early stopping 100 on val binary logloss.
3. Predict val. Apply the one-to-one step (keep the highest-probability S1 per other_id; include fit-fold S1 predictions in the competition so val sees realistic competition). Then evaluate.best_threshold on val S1 only.
4. Save models/lgb_vNN.txt and models/lgb_vNN.json with features, params, best_iter, threshold, and val metrics from evaluate.eval_predictions. Write reports/train_vNN.json (contracts §6) including macro F0.5 before and after one-to-one, per country, and the top 20 feature importances (gain).
5. Write reports/errors_vNN.md: 30 false positives and 30 false negatives from val at the chosen threshold, with both raw records, prob and key features for each.
6. Compare with reports/baseline_val.json. State clearly whether the model beats the baseline on the same shards.
7. Commit code, model files, reports. Push.

Gate: model val macro F0.5 > baseline val macro F0.5. If not, stop and report.
```

### Step 3.2: Prediction and post-processing (Owner: B. Depends on: 3.1 dev model)

```text
Use the amazon-er-challenge skill.

Task: write business_entity_resolution/src/predict.py --split {train,test} --version vNN.
1. Load models/lgb_vNN.txt and its JSON. Predict every shard of work/feat/{split} using FEATURES in the stored order. Assert the column order matches the JSON.
2. Concatenate predictions (s1_id, other_id, prob) for all shards, apply the one-to-one step across ALL shards, then apply the stored threshold. Write work/pred/{split}/shard={i}.parquet for both the raw probs and a kept flag.
3. For train, run evaluate.eval_predictions on val and assert it matches reports/train_vNN.json within 0.001 (same shards). This proves train and predict use the same logic.
4. Put the one-to-one function in one place (predict.py) and make train.py and baseline.py import it. Refactor them if needed and re-run their tests.
5. Commit and push.
```

### Step 3.3: End-to-end run and submission v02 (Owner: A, full mode. Depends on: 3.1, 3.2)

```text
Use the amazon-er-challenge skill.

Task: write business_entity_resolution/src/run_pipeline.py --version vNN that runs, for the test split and skipping finished stages: ingest, splits, aliases, normalize, blocking, features, predict, write_outputs. It must also be able to run training first with --train (train split: normalize, blocking, features, train). Log each stage's elapsed time to reports/test_run_vNN.json.
Then:
1. git pull. Run the Sync prompt stages for all 16 train shards. Run features --split train (all shards) and train --version v02 on all shards. Report the val macro F0.5 from reports/train_v02.json and compare with baseline.
2. Run run_pipeline --version v02 on test. The validator must print PASS. Fill reports/test_run_v02.json with per-country share of S1 with matches and mean matches per S1, and compare those per-country numbers with the val fold's true rates (train: share with matches = 1 - singleton rate, mean matches from facts §5). Flag France if its numbers differ from US/India by more than 2x.
3. Copy output/matching_results.tsv to submissions/v02/, append the log row, commit, tag sub-v02, push. Print the file to upload.
```
**Done when:** A uploads v02 and records the public score.

---

## Phase 4: Improvement loops (all members, until 6 hours before deadline)

Rules for every loop:
- Work in dev mode (shards 0-3). Compare against the dev-mode val score of the current best version, on the same shards.
- Keep a change only if val macro F0.5 improves. Put the before/after numbers in the commit message.
- Only A runs full mode and submits. Submit only when full-mode val improves over the last submitted version.

### Loop 4B: Blocking recall (Owner: B)
```text
Use the amazon-er-challenge skill. Read reports/blocking_train.json and the latest reports/errors_vNN.md.
Task: improve blocking recall without letting pairs-per-S1 grow more than 30%. Analyse 200 val true pairs missed by blocking (dev shards): group them by cause (no shared number, number format, street token spelling, empty address, script, other) and print counts per cause with 3 examples each. Propose at most 2 key changes that target the biggest causes, implement them behind new key bits, and re-run blocking + evaluate on dev shards. Report recall, oracle_macro_f05 and pairs per S1 before vs after. Keep a change only if recall improves and the pair budget holds. Commit with the numbers in the message.
```

### Loop 4C: Features and model (Owner: C)
```text
Use the amazon-er-challenge skill. Read the latest reports/errors_vNN.md and reports/train_vNN.json.
Task: group the 30 false positives and 30 false negatives by cause and print counts. Implement at most 3 new features that target the largest false-positive causes first (precision counts double in F0.5). Candidates to consider only if the errors support them: the number of name tokens that appear in the other address, match of the rarest name token (IDF computed from training S2/S3 only), same-name-different-number indicators, S2/S3 sibling agreement (how many other candidates of this S1 share this record's address numbers). Retrain on dev shards as a new version and compare val macro F0.5 with the previous version on the same shards. Keep only improving features. Commit with the numbers.
```

### Loop 4D: France and normalization robustness (Owner: D)
```text
Use the amazon-er-challenge skill. France has no labels, so this loop can only use checks, never tuning on France results.
Task:
1. Sample 300 France test S1 records with their top 5 candidates and model probabilities from work/pred/test. Print 20 cases around the threshold. Check whether French address patterns (rue/r., avenue/av, bis/ter, allée, impasse, chemin, région vs département names, accents) are normalized consistently. Check whether French legal forms are stripped from name_core.
2. Fix only normalization bugs that you can show with concrete examples, and add each example as a unit test.
3. Confirm the change does not lower US/India dev val macro F0.5. Report the per-country predicted matches-per-S1 on test before and after.
Commit with the numbers.
```

### Loop 4A: Threshold and post-processing (Owner: A)
```text
Use the amazon-er-challenge skill. Using full-mode val predictions of the current best version, test these post-processing variants and report val macro F0.5 for each, without retraining:
(a) current: one-to-one then one global threshold;
(b) separate thresholds for S2 and S3 records, tuned jointly on a grid;
(c) keep an S1's matches only if its best candidate prob exceeds a second threshold t_top (grid 0.3-0.9), otherwise predict empty;
(d) cap matches per S1 at the 99.9th percentile of true match counts (from facts §5 the maximum is 11).
Adopt a variant only if it beats (a) by at least 0.001. Guard against overfitting: tune on half of the val S1 (by stable hash) and confirm the gain on the other half. Write the variant and the numbers into models/lgb_vNN.json and reports/postprocess_vNN.json.
```

---

## Phase 5: Final package (start at least 6 hours before the deadline)

### Step 5.1: Code README and requirements (Owner: D. Depends on: final version chosen)
```text
Use the amazon-er-challenge skill.
Task: write business_entity_resolution/README.md for someone reproducing results on a fresh machine: prerequisites (Python version, RAM needed from the reports/env_*.json of the full-run machine), pip install -r requirements.txt, where to put the dataset (or ER_DATA_DIR), the exact commands to reproduce both output files end-to-end with run_pipeline (training included), expected runtime per stage taken from reports/test_run_vNN.json, and how to run the validator. Every module in src/ gets a one-line description and every public function gets a docstring (the guidelines require commented code). Regenerate requirements.txt from importlib.metadata for the packages actually imported in src/ (scan the imports). Confirm no import of any network library (requests, urllib, httpx, etc.) exists in src/. Commit and push.
```

### Step 5.2: Methodology document (Owner: D. Depends on: final version chosen)
```text
Use the amazon-er-challenge skill. Anti-hallucination rule: every number in the document must come from a file in reports/ or models/. Cite the file in a comment next to each number when drafting, then remove the comments. Write "not measured" for anything without a source.
Task: fill in Documentation_template.md (keep its headings) for the final version vNN. Team name <TEAM_NAME>, members <MEMBERS>, date. Cover: EDA insights (facts.md §4-6), approach type (blocking + LightGBM classifier + one-to-one post-processing), blocking keys and why (address-only keys for 14.4% zero-name-overlap pairs), total candidate pairs for test, blocking recall and recall@k on val, feature list grouped by name/address/other, model and parameters, threshold method (macro F0.5 on val, singletons included), results (val macro F0.5 overall, per country, singletons vs non-singletons, public LB scores from SUBMISSION_LOG.md), common false positives and false negatives from reports/errors_vNN.md, France handling, compliance (no external data; LightGBM MIT; anyascii ISC used only for preprocessing), and the code structure and entry point. Keep the core to about 2 pages; put tables and extra results in the appendix. Commit and push.
```

### Step 5.3: Build and check the zip (Owner: A. Depends on: 5.1, 5.2, final outputs)
```text
Use the amazon-er-challenge skill. Read facts.md §1 (final zip structure).
Task:
1. Confirm output/matching_results.tsv is byte-identical to the last uploaded version in submissions/ (or is the version we have decided to finalise), and that the validator prints PASS with --check-ids.
2. Build <TEAM_NAME>_submission.zip in the parent folder with exactly: output/matching_results.tsv, output/candidate_pairs.tsv, code/business_entity_resolution/ (src/, tests/, README.md, requirements.txt; no __pycache__, no work/ data, no dataset), and Documentation_template.md (filled).
3. List the zip contents with sizes and check the tree matches facts.md §1.
4. Smoke-test reproducibility: extract the zip to a temp folder, set ER_DATA_DIR to the dataset, and run the README's first command with a small limit, or at least import every module. Report the result.
5. Record the final version and the zip file name in SUBMISSION_LOG.md, commit, tag final, push.
```

---

## Quick reference

| Command | Purpose |
|---|---|
| `python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test` | Official format check. Must print PASS. |
| `python -m business_entity_resolution.src.metric` | Metric self-test (0.7143). |
| `python -m business_entity_resolution.src.evaluate --what candidates --shards 0,1,2,3` | Blocking recall on val. |
| `python -m business_entity_resolution.src.run_pipeline --version vNN` | Full test run. |

**If something goes wrong**
- Out of memory: lower `DUCKDB_MEMORY`, process fewer shards at once, or reduce `TOPK` after checking recall@k.
- Validator FAIL: never hand-edit the TSV. Fix the writer or the upstream stage and regenerate.
- Val score and leaderboard disagree a lot: check `reports/test_run_vNN.json` per-country rates first, France especially.
- Merge conflicts: only one person edits a module (see Roles). Shared files (`config.py`, `contracts.md`) change only with a team message.
