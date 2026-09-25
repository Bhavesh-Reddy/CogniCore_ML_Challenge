# Verified facts (source of truth)

Every number below was measured on the actual files on 2026-09-26 or quoted from the two official PDFs.
If a fact you need is not here, measure it with a script and save the result under `reports/`. Do not guess.

## 1. Official rules (quoted from the problem statement PDF and guidelines PDF)

| Rule | Source |
|---|---|
| Task: for every Source 1 (S1) entity, list all matching Source 2 (S2) / Source 3 (S3) records. 0, 1 or many matches. | Problem statement p.1 |
| All inputs and outputs are TAB-separated `.tsv`. | p.1 |
| Metric: macro F_0.5 per S1 entity, averaged over ALL S1 entities. Empty-truth + empty-pred = 1.0; empty-truth + any pred = 0.0. | p.6 |
| `matching_results.tsv` header: `source1_entity_id<TAB>matched_entity_ids`. IDs comma-joined, no quotes, no spaces. | p.3 |
| `candidate_pairs.tsv` header: `source1_entity_id<TAB>candidate_entity_ids`. It must be the exact set the final model scored. Matches must be a subset of candidates. | p.3-4 |
| One row per test S1 entity (exactly once). Empty list allowed. No duplicate IDs in a list. Only S2-/S3- IDs that exist in the test set. | p.3, p.5 |
| Final model must be MIT or Apache-2.0 licensed and at most 8B parameters. | p.5 |
| NO external data: no entity-resolution APIs, no business registries, no geocoding APIs, no internet augmentation. Disqualification. | p.7 |
| `country` is an open set. Test contains `France`, which is absent from training. Do not hard-code or one-hot `{US, India}`. | p.1-2 |
| Leaderboard: public = subset of test, private = rest. Final ranking = private. Submit predictions for the full test set. | p.6 |
| Max 5 leaderboard submissions per day. Challenge window 25 Sep 2026 00:00 IST to 27 Sep 2026 23:59 IST. | Guidelines p.1 |
| Keep version history of all submissions. | Guidelines p.1 |
| One login per participant, one device. No simultaneous logins. | Guidelines p.2 |
| Final zip: `output/` (both TSVs), `code/business_entity_resolution/` (`src/`, `README.md`, `requirements.txt`), filled `Documentation_template.md`. | p.4-5 |
| Guidelines ask for a 1-2 page approach document; the problem statement says the template has no page limit. Fill the template and keep it tight. | Both PDFs |
| Validator: `python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test` run from `student_resource/`. Exit 0 = PASS. | p.4 |

## 2. Reading the files correctly (verified)

```python
import csv, pandas as pd
pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)
```
- `keep_default_na=False` is required: addresses contain the literal strings `null` / `NULL`, which pandas would otherwise turn into NaN.
- `quoting=csv.QUOTE_NONE` gives row counts that equal `wc -l` minus the header for every file.
- Load time for all train files with pandas: about 60 s on the reference machine.

## 3. File sizes and row counts (rows exclude header)

| File | Rows | Size |
|---|---|---|
| train_source1.tsv | 2,206,821 | 201 MB |
| train_source2.tsv | 5,034,616 | 467 MB |
| train_source3.tsv | 5,285,603 | 481 MB |
| train_ground_truth.tsv | 2,206,821 | 122 MB |
| test_source1.tsv | 1,732,544 | 167 MB |
| test_source2.tsv | 4,887,273 | 486 MB |
| test_source3.tsv | 5,082,316 | 483 MB |

All `entity_id` values are unique within each file. Ground truth has exactly one row per train S1 entity.

## 4. Country distribution

| Split | Source | US | India | France |
|---|---|---|---|---|
| train | S1 | 1,323,633 | 883,188 | 0 |
| train | S2 | 3,016,817 | 2,017,799 | 0 |
| train | S3 | 3,170,056 | 2,115,547 | 0 |
| test | S1 | 663,106 | 809,986 | 259,452 |
| test | S2 | 1,871,330 | 2,312,565 | 703,378 |
| test | S3 | 1,945,701 | 2,405,000 | 731,615 |

The test mix differs from train. India is the largest test country, and France is 15% of test S1.

## 5. Ground-truth structure (train)

- Matches per S1: 0 → 123,247 (5.58% singletons); 1 → 119,157; 2 → 375,212; 3 → 530,841; 4 → 484,115; 5 → 321,957; 6 → 164,868; 7 → 63,968; 8 → 18,680; 9 → 4,205; 10 → 534; 11 → 37. Mean 3.46.
- S2 matches per S1 range 0-5. S3 matches per S1 range 0-6. So S2 and S3 are NOT deduplicated: one business can appear several times in the same source.
- **Every S2/S3 record is matched to at most one S1 entity** (7,638,365 matched IDs, all unique). This justifies a one-to-one post-processing step: assign each S2/S3 record to at most one S1.
- 73.4% of S2 records and 74.6% of S3 records match some S1. The remaining ~26% are unmatched distractors.
- **Matched pairs always share the same `country`** (100% of 691,347 sampled pairs). Blocking within country loses nothing.

## 6. Noise observed in matched pairs (sampled)

- Name token Jaccard between a S1 name and its matched record: 5th pct 0.0, 10th pct 0.0, median 0.67. **14.4% of matched pairs share no name token at all.** Causes seen:
  - website or hashtag names: `lifeinvestments.com`, `#empiremolecular`, `Shri vippub1icschool.com`
  - non-Latin scripts: Devanagari (~4.0% of matched S2/S3 names), other Indic scripts such as Kannada (~3.2%)
  - unrelated aliases: `Jaxdrex`, `Gilddrex`, `Halotavo F/K/A Roach Beverage Corp`
  - For these, the address is the only signal. Blocking must include address-only keys.
- Name noise: added/removed legal suffixes (Pvt, Private, Ltd, Limited, LLC, L.L.C., Corp, Inc, P.C., SARL, EURL, SAS, SASU, SCI), added filler words (`Center`, `Services`, `Group`), repeated words (`Kochar Kochar`), typos (`Soihely`, `Wheooitne`), accents added (`Góld`, `Ópton`), honorific prefixes (`Shri`, `Smt`), junk prefixes (`<<`, `--`, `#`, `[ ]`), `| www.site.com` suffixes (~3.2% of S2/S3 names contain a URL or pipe).
- Address noise: UPPERCASE in S2, leading zeros (`00272`), component reordering (`OH, Columbus, 5559 Orville Avenue`), abbreviations (Dr/Drive, Blvd, Cir, Ave, SO WINDSOR), state as code/name/native script (`MH`, `Maharashtra`, `महाराष्ट्र`; `TX`/`Texas`), literal `null`/`NULL` tokens (~2.5% of S2/S3), empty address (~3.3% of train S2/S3, ~2.7% of test), prefixes like `H.NO`, `No #`, `##`, `KH NO.`, ranges `1000 34-1002`.
- The first number in the address is equal in 79% of matched pairs where both sides have a number. 12.3% of matched S2/S3 addresses have no digit at all.
- **No 6-digit Indian PIN codes appear in the S1 or matched addresses of the sample.** Do not build a PIN-code blocking key.
- **38.3% of train S1 names are shared by more than one S1 entity** (for example `Primary Care Group` appears 253 times). Name alone cannot resolve entities.
- France examples (test only): `R.` = Rue, `AV` = Avenue, `ALLÉE`, `Impasse`, `Chemin`, `bis`, region vs department (`Hauts-de-France` vs `Nord`, `Nouvelle-Aquitaine` vs `Gironde`), legal forms SARL/SAS/SASU/EURL/SCI, accents (`Àmicale`, `Pàrenthese`).

## 7. Environment facts (reference machine)

- Windows 11, Python 3.12.8, 16 GB RAM, 20 logical CPUs, no CUDA GPU. Teammates' machines may differ; Step 0 records each one.
- Verified installed versions and licenses (from package metadata): rapidfuzz 3.14.6 MIT; lightgbm 4.7.0 MIT; duckdb 1.5.5 MIT; anyascii 0.3.3 ISC; polars 1.34.0 MIT; pyarrow 21.0.0 Apache-2.0; pandas 2.3.2 BSD-3; numpy 2.5.1 BSD-3; scikit-learn 1.7.2 BSD-3.
- The trained **final model is LightGBM (MIT)**. The other packages are preprocessing or tooling libraries.
- Do not use `Unidecode`: it is GPL-licensed. Use `anyascii` for transliteration.
- `anyascii` output on real values: `ಲೋಟಸ್ ಮಾರ್ಕೆಟಿಂಗ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್` → `lots marketimg praivet limited`; `महाराष्ट्र` → `mharastr`; `ಕರ್ನಾಟಕ` → `krnatk`; `दिल्ली` → `dilli`. Transliteration is lossy, so use fuzzy/char-level similarity on it, and learn native-script state aliases from training pairs.
