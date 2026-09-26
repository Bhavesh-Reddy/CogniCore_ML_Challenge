import numpy as np
import pandas as pd
import pytest

from business_entity_resolution.src.evaluate import (
    _f05_counts, best_threshold, candidate_stats, eval_predictions, threshold_curve)
from business_entity_resolution.src.metric import f05_entity, macro_f05

# Synthetic 5-entity val fold.
#   A (US)    truth {S2-1, S3-1}   preds S2-1 .90, S3-1 .80, S2-9 .70 (wrong)
#   B (US)    truth {S2-2}         preds S2-2 .60
#   C (US)    singleton            preds S2-3 .30 (wrong)
#   D (India) singleton            preds S3-4 .90 (wrong)
#   E (India) truth {S2-5, S3-5}   preds S2-5 .95, S3-6 .55 (wrong)
S1_IDS = ["A", "B", "C", "D", "E"]
TRUTH = {"A": {"S2-1", "S3-1"}, "B": {"S2-2"}, "C": set(), "D": set(), "E": {"S2-5", "S3-5"}}
COUNTRY = {"A": "US", "B": "US", "C": "US", "D": "India", "E": "India"}
PRED = pd.DataFrame(
    [("A", "S2-1", .90), ("A", "S3-1", .80), ("A", "S2-9", .70), ("B", "S2-2", .60),
     ("C", "S2-3", .30), ("D", "S3-4", .90), ("E", "S2-5", .95), ("E", "S3-6", .55)],
    columns=["s1_id", "other_id", "prob"])
KW = {"s1_ids": S1_IDS, "truth": TRUTH, "country": COUNTRY}

# At threshold 0.5, F0.5 = 1.25*P*R / (0.25*P + R):
#   A: pred 3, tp 2 -> P=2/3, R=1   -> 1.25*(2/3) / (1/6 + 1) = 5/7 = 0.714286
#   B: pred 1, tp 1                 -> 1.0
#   C: singleton, .30 dropped -> empty prediction -> 1.0
#   D: singleton, .90 kept    -> false match      -> 0.0
#   E: pred 2, tp 1 -> P=1/2, R=1/2 -> 0.5
#   macro = (5/7 + 1 + 1 + 0 + 0.5) / 5 = 0.642857
#   US = (5/7 + 1 + 1) / 3 = 0.904762, India = (0 + 0.5) / 2 = 0.25
#   singletons (C, D) = 0.5, non-singletons (A, B, E) = (5/7 + 1 + 0.5) / 3 = 0.738095
#   micro precision = tp 4 / pred 7, micro recall = tp 4 / true 5
MACRO_AT_05 = (5 / 7 + 1 + 1 + 0 + 0.5) / 5


def test_eval_predictions_hand_computed():
    r = eval_predictions(PRED, 0.5, **KW)
    assert r["macro_f05"] == pytest.approx(MACRO_AT_05, abs=1e-6)
    assert r["precision"] == pytest.approx(4 / 7, abs=1e-6)
    assert r["recall"] == pytest.approx(4 / 5, abs=1e-6)
    assert r["per_country"]["US"] == pytest.approx((5 / 7 + 2) / 3, abs=1e-6)
    assert r["per_country"]["India"] == pytest.approx(0.25, abs=1e-6)
    assert r["macro_f05_singletons"] == pytest.approx(0.5, abs=1e-6)
    assert r["macro_f05_non_singletons"] == pytest.approx((5 / 7 + 1.5) / 3, abs=1e-6)


def test_eval_predictions_matches_metric_module():
    pred_sets = {"A": {"S2-1", "S3-1", "S2-9"}, "B": {"S2-2"}, "D": {"S3-4"}, "E": {"S2-5", "S3-6"}}
    assert macro_f05(pred_sets, TRUTH, S1_IDS) == pytest.approx(MACRO_AT_05, abs=1e-12)


def test_s1_missing_from_predictions_counts_as_empty():
    # Only A predicted; B, E lose (0), singletons C, D score 1.0 with an empty prediction.
    r = eval_predictions(PRED[PRED["s1_id"] == "A"], 0.5, **KW)
    assert r["macro_f05"] == pytest.approx((5 / 7 + 0 + 1 + 1 + 0) / 5, abs=1e-6)


def test_best_threshold_hand_computed():
    # Best band is t in (0.55, 0.60]: A 5/7, B 1, C 1, D 0, E (P=1, R=1/2 -> 0.625/0.75 = 5/6).
    # Every grid value 0.56..0.60 ties; the lowest (0.56) is returned.
    t, f = best_threshold(PRED, s1_ids=S1_IDS, truth=TRUTH)
    assert t == pytest.approx(0.56)
    assert f == pytest.approx((5 / 7 + 1 + 1 + 0 + 5 / 6) / 5, abs=1e-9)


def test_threshold_curve_equals_eval_predictions_on_random_data():
    rng = np.random.default_rng(0)
    s1_ids = [f"S1-{i}" for i in range(300)]
    truth = {s: {f"S2-{s}-{j}" for j in range(rng.integers(0, 4))} for s in s1_ids}
    rows = []
    for s in s1_ids[:250]:  # the last 50 S1 have no predictions at all
        for o in list(truth[s]) + [f"S3-{s}-x{j}" for j in range(rng.integers(0, 3))]:
            if rng.random() < 0.8:
                rows.append((s, o, float(rng.random())))
    pred = pd.DataFrame(rows, columns=["s1_id", "other_id", "prob"])
    curve = threshold_curve(pred, s1_ids=s1_ids, truth=truth)
    for t in curve.index[::7]:
        r = eval_predictions(pred, t, s1_ids=s1_ids, truth=truth, country={})
        assert curve[t] == pytest.approx(r["macro_f05"], abs=1e-6)


def test_f05_counts_equals_f05_entity():
    for n_true in range(4):
        for n_pred in range(4):
            for tp in range(min(n_true, n_pred) + 1):
                truth = {f"t{i}" for i in range(n_true)}
                pred = {f"t{i}" for i in range(tp)} | {f"f{i}" for i in range(n_pred - tp)}
                assert _f05_counts([tp], [n_pred], [n_true])[0] == pytest.approx(f05_entity(pred, truth))


def test_candidate_stats():
    cand = pd.DataFrame(
        [("A", "S2-1", 0b01, .9), ("A", "S2-9", 0b10, .8), ("A", "S3-1", 0b11, .1),
         ("B", "S2-2", 0b10, .5), ("D", "S3-4", 0b01, .7), ("E", "S2-5", 0b01, .6),
         ("Z", "S2-1", 0b01, .9)],  # Z is not in the fold: ignored
        columns=["s1_id", "other_id", "key_hits", "cheap_score"])
    r = candidate_stats(cand, S1_IDS, TRUTH, COUNTRY)
    # true pairs = 5 (A:2, B:1, E:2); found = S2-1, S3-1, S2-2, S2-5 -> 4/5
    assert r["n_true_pairs"] == 5 and r["n_cand_pairs"] == 6
    assert r["recall_all"] == pytest.approx(0.8)
    assert r["recall_by_country"] == {"US": 1.0, "India": 0.5}
    # C has zero candidates -> 1/5; per-S1 counts A3 B1 C0 D1 E1
    assert r["s1_with_zero_candidates"] == pytest.approx(0.2)
    assert r["pairs_per_s1"]["max"] == 3 and r["pairs_per_s1"]["mean"] == pytest.approx(1.2)
    # K1 (bit 0) hits S2-1, S3-1, S2-5 = 3/5; K2 (bit 1) hits S3-1, S2-2 = 2/5
    assert r["per_key"]["K1"]["recall"] == pytest.approx(0.6)
    assert r["per_key"]["K2"]["recall"] == pytest.approx(0.4)
    # oracle: A {S2-1,S3-1} 1.0, B 1.0, C empty 1.0, D empty 1.0, E P=1 R=1/2 -> 5/6
    assert r["oracle_macro_f05"] == pytest.approx((4 + 5 / 6) / 5, abs=1e-6)
    # S3-1 is A's 3rd candidate by cheap_score, so recall@10 still finds all 4
    assert r["recall_at_k"][10] == pytest.approx(0.8)
