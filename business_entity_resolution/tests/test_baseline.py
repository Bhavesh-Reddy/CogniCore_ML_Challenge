import pandas as pd

from business_entity_resolution.src.baseline import one_to_one


def test_one_to_one_keeps_best_s1_per_other_id():
    df = pd.DataFrame({"s1_id": ["S1-a", "S1-b", "S1-c", "S1-a", "S1-b"],
                       "other_id": ["S2-1", "S2-1", "S2-1", "S3-2", "S3-2"],
                       "prob": [0.4, 0.9, 0.9, 0.7, 0.2]})
    out = one_to_one(df).set_index("other_id")
    # S2-1: b and c tie at 0.9 -> smallest s1_id (S1-b); S3-2: S1-a has the higher score
    assert out.loc["S2-1", "s1_id"] == "S1-b" and out.loc["S3-2", "s1_id"] == "S1-a"
    assert len(out) == 2
