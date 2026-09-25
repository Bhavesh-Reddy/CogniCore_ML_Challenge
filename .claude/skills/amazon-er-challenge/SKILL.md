---
name: amazon-er-challenge
description: Guardrails, verified facts and the agreed pipeline for the Amazon ML Challenge 2026 Business Entity Resolution task (match Source 1 business records to Source 2/3 records, scored by macro F0.5). Use for ANY work in this repo - data loading, normalization, blocking, features, LightGBM training, threshold tuning, writing matching_results.tsv / candidate_pairs.tsv, validation, submission logging, packaging the final zip, or filling Documentation_template.md.
---

# Amazon ML Challenge 2026: Business Entity Resolution

You are helping a team of 4 execute a pre-planned pipeline. Each prompt you receive is one step of
`TEAM_EXECUTION_GUIDE.md`. Do exactly that step, meet its acceptance checks, and report.

## Read before acting

1. `reference/facts.md` holds every verified fact about the rules and the data. Treat it as ground truth.
2. `reference/contracts.md` holds the repo layout, module names, CLI flags, file schemas and report keys. Follow it exactly.
3. `reference/f05_metric.py` is the verified metric. Copy it. Never rewrite the formula.

Read only the sections you need. Do not re-profile the dataset unless the step asks you to.

## Non-negotiable rules

**Anti-hallucination**
- Never state a number (score, recall, row count, timing) that is not printed by code you ran in this session or stored in `reports/`. If it is not measured, say "not measured".
- Never invent column names, file paths, flags or library APIs. Check contracts.md. For a library API you are unsure of, run `python -c "help(...)"` or a 3-line test first.
- Never paste or `cat` a full dataset file. Files are 120-490 MB. Use `head`, `--limit`, or pandas `nrows`.
- If an acceptance check fails, stop and report the failure with its output. Do not silently loosen the check, skip the step or redesign the pipeline.
- When you finish, print a short summary: files changed, commands run, measured results, anything not done.

**Competition rules (disqualification risk)**
- No internet data of any kind in the pipeline: no geocoding, no business registries, no ER APIs, no downloaded gazetteers, no pretrained lookup tables. Hand-written abbreviation lists (Rd→Road, R.→Rue, SARL) and aliases mined from the TRAINING data are allowed.
- Final model: LightGBM (MIT). Any other model you add must be MIT or Apache-2.0 and at most 8B parameters. Check its license from package metadata before use. Never use `Unidecode` (GPL); use `anyascii`.
- `country` is an open set. Never filter, hard-code or one-hot `{US, India}`. Never use `country` as a model feature. Use it only as a blocking partition. Every test S1 entity, France included, must get a row.

**Data handling**
- Read TSVs only through `io_utils.read_tsv`, which uses `sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE`.
- Write output TSVs only through `io_utils.write_id_list_tsv` (see contracts §5).
- Process big data in shards (`config.N_SHARDS`). Stay under about 70% of the machine's RAM. Free DataFrames with `del` and `gc.collect()` between shards.
- Use fixed seeds (`config.SEED`) and `stable_hash`, never Python `hash()`.
- Never modify anything under `dataset/` or `utils/`.

## The pipeline

1. **Ingest**: TSV → parquet; ground truth → `(s1_id, other_id)` pairs.
2. **Splits**: train S1 → folds `fit` (80%) / `val` (20%) and 16 shards, by stable hash.
3. **Normalize**: anyascii transliteration, lowercase, abbreviation expansion (US, India, France), legal-suffix and filler removal for `name_core`, `name_nospace`, number extraction, admin (state/region) canonicalization using aliases mined from train pairs.
4. **Blocking** (DuckDB, within country): union of keys, each group capped at `BLOCK_CAP`, then top-`TOPK` per S1 by a cheap score. Initial keys:
   - K1 `first_num + name_core token`
   - K2 `first_num + address street token` (address-only; catches website, script and alias names)
   - K3 `name_core token + address street token` (catches missing house numbers)
   - K4 `name_nospace` 8-char prefix (catches `lifeinvestments.com`)
   - Measure recall per key and overall on the val fold. Blocking recall is the ceiling on recall.
5. **Features** (rapidfuzz, country-agnostic): name ratios (ratio, token_set, token_sort, partial, Jaro-Winkler) on name_core and name_nospace, token Jaccard, address token_set, number Jaccard, first-number equality, admin overlap, empty/URL/non-Latin flags, source (2 or 3), key-hit bits, cheap_score, rank and gap within the S1 group, and competition features (how many S1 entities claim this record, and this S1's rank among them).
6. **Model**: LightGBM binary classifier trained on `fit` shards. Early stopping on `val`.
7. **Post-processing**: each S2/S3 record goes to at most one S1, the one with the highest probability (facts §5). Then threshold. Tune the threshold to maximize macro F0.5 on `val`, singletons included.
8. **Outputs**: `candidate_pairs.tsv` = exactly what the model scored. `matching_results.tsv` = the kept pairs. Run the validator. It must print PASS.
9. **Log** every leaderboard upload in `submissions/SUBMISSION_LOG.md`, and create git tag `sub-vNN`.
10. **Package** the zip per contracts and facts §1. Fill the documentation only from `reports/`.

## F0.5 reasoning to keep in mind

- Precision counts double. When unsure, predict fewer matches. An S1 with a wrong extra match loses much more than one with a missed match.
- A singleton scores 1.0 only with an EMPTY prediction. Pushing low-confidence S1 entities to empty is often right.
- Optimize the threshold on the real macro metric, not on AUC or pairwise F1.
- France has no labels. Use the same global threshold. Compare the predicted matches-per-S1 for France with US/India in `reports/test_run_vNN.json` as a sanity check, and report it rather than tuning blind.

## Quality gates (stop and report if one fails)

| Gate | Check |
|---|---|
| Ingest | Row counts equal facts §3 exactly |
| Metric | `python -m business_entity_resolution.src.metric` prints PASS with 0.714 on the PDF example |
| Blocking | Report recall on val and pairs-per-S1. Recall should be at least 0.95; if lower, report the per-key recall and the misses before changing anything |
| Candidates | Every val S1 appears; no S1-, duplicate or cross-country candidates |
| Model | Val macro F0.5 of the model is higher than `reports/baseline_val.json` |
| Output | Validator prints PASS; row count = 1,732,544; matched ⊆ candidates |
