import numpy as np
import pandas as pd

from business_entity_resolution.src.features import FEATURES, group_features, pair_features


def _frame():
    q = {"name_norm": "kochar gold", "name_core": "kochar gold", "name_nospace": "kochargold",
         "name_has_url": 0, "name_nonlatin": 0, "addr_norm": "12 main road pune mh", "addr_core": "12 main road pune",
         "addr_nums": "12", "addr_first_num": "12", "addr_admin": "maharashtra", "source": 1}
    others = [
        dict(q, source=2),                                                        # identical
        dict(q, name_core="gold traders", name_nospace="goldtraders", name_norm="gold traders", addr_norm="",
             addr_core="", addr_nums="", addr_first_num="", addr_admin="", source=3, name_has_url=1),
        dict(q, addr_nums="99 12", addr_first_num="99", addr_admin="delhi", source=2),
    ]
    rows = []
    for i, o in enumerate(others):
        r = {"s1_id": "S1-a", "other_id": f"S2-{i}", "key_hits": 1 | 8, "cheap_score": [0.9, 0.3, 0.6][i],
             "key_weight": 0.5}
        r.update({f"q_{k}": v for k, v in q.items()})
        r.update({f"o_{k}": v for k, v in o.items()})
        rows.append(r)
    df = pd.DataFrame(rows)
    for side in ("q", "o"):
        for c in ("name_has_url", "name_nonlatin", "source"):
            df[f"{side}_{c}"] = df[f"{side}_{c}"].astype("int8")
    return df


def test_pair_features():
    f = pair_features(_frame())
    assert f.loc[0, "name_tset"] == 100 and f.loc[0, "addr_tset"] == 100 and f.loc[0, "first_num_eq"] == 1
    assert f.loc[0, "num_jacc"] == 1 and f.loc[0, "admin_eq"] == 1 and f.loc[0, "name_jacc"] == 1
    # empty other address: address similarities and number/admin features are -1, not a fake 100
    assert (f.loc[1, ["addr_tset", "addr_partial", "addrnorm_tsort", "addr_jacc", "num_jacc",
                      "first_num_eq", "admin_eq"]] == -1).all()
    assert f.loc[1, "other_addr_empty"] == 1 and f.loc[1, "other_name_has_url"] == 1 and f.loc[1, "other_source"] == 3
    assert f.loc[1, "s1_num_share"] == 0
    # different first number, shared "12", different admin region
    assert f.loc[2, "first_num_eq"] == 0 and f.loc[2, "num_jacc"] == 0.5 and f.loc[2, "admin_eq"] == 0
    assert f.loc[2, "s1_num_share"] == 1
    assert (f[["key_k1", "key_k4"]] == 1).all().all() and (f[["key_k2", "key_k3"]] == 0).all().all()
    assert not f.isna().any().any()


def test_group_features_and_feature_list():
    df = _frame()
    f = pair_features(df)
    g = group_features(df["s1_id"], f)
    assert g["n_candidates"].tolist() == [3, 3, 3]
    assert g["cs_rank"].tolist() == [1, 3, 2]
    assert np.allclose(g["cs_gap"], [0, -0.6, -0.3])
    f = pd.concat([f, g], axis=1).assign(n_s1_claiming=1.0, claim_rank=1.0)
    assert set(FEATURES) == set(f.columns) and len(FEATURES) == len(set(FEATURES))
    assert "country" not in " ".join(FEATURES)
