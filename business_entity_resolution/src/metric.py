"""Reference implementation of the official metric (problem statement, 'Evaluation Criteria').

Macro F_0.5 over ALL Source-1 entities in the evaluation set, singletons included:
  - truth empty & pred empty      -> 1.0
  - truth empty & pred non-empty  -> 0.0
  - truth non-empty & pred empty  -> 0.0
  - otherwise P = |pred & truth| / |pred|, R = |pred & truth| / |truth|,
    F = 1.25*P*R / (0.25*P + R)   (0.0 when P == 0 or R == 0)
Verified against the worked example in the PDF (expected 0.714).
Copy this file to business_entity_resolution/src/metric.py; do not re-derive it.
"""
from typing import Dict, Iterable, Set


def f05_entity(pred: Set[str], truth: Set[str]) -> float:
    if not truth:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    tp = len(pred & truth)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(truth)
    return 1.25 * p * r / (0.25 * p + r)


def macro_f05(pred: Dict[str, Set[str]], truth: Dict[str, Set[str]],
              s1_ids: Iterable[str]) -> float:
    """s1_ids = every S1 entity in the evaluation fold. Missing keys mean empty sets."""
    ids = list(s1_ids)
    return sum(f05_entity(pred.get(i, set()), truth.get(i, set())) for i in ids) / len(ids)


if __name__ == "__main__":
    ex = f05_entity({"S2-00047", "S2-00193", "S3-00812"}, {"S2-00047", "S3-00812"})
    assert round(ex, 3) == 0.714, ex
    assert f05_entity(set(), set()) == 1.0
    assert f05_entity({"S2-1"}, set()) == 0.0
    assert f05_entity(set(), {"S2-1"}) == 0.0
    assert macro_f05({"a": {"x"}}, {"a": {"x"}, "b": set()}, ["a", "b"]) == 1.0
    print("metric self-test PASS, PDF example =", round(ex, 4))
