"""Negative tests: each classic format mistake fails the official validator when written by hand,
and write_id_list_tsv / write_submission prevent it."""
import pandas as pd
import pytest

from business_entity_resolution.src.io_utils import write_id_list_tsv
from business_entity_resolution.src.write_outputs import (
    MATCH_HEADER, JoinedLists, run_validator, write_submission)

S1_IDS = ["S1-a", "S1-b", "S1-c"]


@pytest.fixture
def test_dir(tmp_path):
    d = tmp_path / "test"
    d.mkdir()
    rows = "".join(f"{s}\tName {s}\tAddr\tUS\n" for s in S1_IDS)
    (d / "test_source1.tsv").write_text("entity_id\tbusiness_name\tbusiness_address\tcountry\n" + rows,
                                        encoding="utf-8")
    return d


def _hand_written(out, text):
    out.mkdir()
    (out / "matching_results.tsv").write_text(text, encoding="utf-8", newline="\n")
    return out


def _via_writer(out, mapping):
    out.mkdir()
    write_id_list_tsv(out / "matching_results.tsv", MATCH_HEADER, S1_IDS, mapping)
    return out


HEADER = "source1_entity_id\tmatched_entity_ids\n"
CASES = {
    # name: (hand-written bad file, the same intent as a mapping passed to the writer)
    "duplicate id in a list": (HEADER + "S1-a\tS2-1,S2-1\nS1-b\t\nS1-c\t\n", {"S1-a": ["S2-1", "S2-1"]}),
    "missing S1 row": (HEADER + "S1-a\tS2-1\nS1-c\t\n", {"S1-a": ["S2-1"]}),
    "S1- id in a list": (HEADER + "S1-a\tS1-c,S2-1\nS1-b\t\nS1-c\t\n", {"S1-a": ["S1-c", "S2-1"]}),
    "comma-separated header": ("source1_entity_id,matched_entity_ids\nS1-a\tS2-1\nS1-b\t\nS1-c\t\n",
                               {"S1-a": ["S2-1"]}),
}


@pytest.mark.parametrize("case", list(CASES))
def test_writer_prevents_format_error(case, tmp_path, test_dir):
    bad_text, mapping = CASES[case]
    assert run_validator(_hand_written(tmp_path / "bad", bad_text), test_dir) == 1
    good = _via_writer(tmp_path / "good", mapping)
    assert run_validator(good, test_dir) == 0
    lines = (good / "matching_results.tsv").read_text(encoding="utf-8").splitlines()
    assert lines == ["source1_entity_id\tmatched_entity_ids", "S1-a\tS2-1", "S1-b\t", "S1-c\t"]


def test_write_submission_dry_run_passes(tmp_path, test_dir):
    assert write_submission({}, {}, out_dir=tmp_path, s1_ids=S1_IDS, test_dir=test_dir) == 0
    for f in ("matching_results.tsv", "candidate_pairs.tsv"):
        assert (tmp_path / f).read_bytes().count(b"\t\n") == len(S1_IDS)


def test_write_submission_rejects_match_outside_candidates(tmp_path, test_dir):
    with pytest.raises(ValueError, match="not candidates.*S1-a"):
        write_submission({"S1-a": ["S2-1"]}, {"S1-a": ["S2-1", "S3-9"]}, out_dir=tmp_path, s1_ids=S1_IDS,
                         test_dir=test_dir)


def test_write_submission_rejects_unknown_s1(tmp_path, test_dir):
    with pytest.raises(ValueError, match="not in the test S1 list"):
        write_submission({"S1-zzz": ["S2-1"]}, {}, out_dir=tmp_path, s1_ids=S1_IDS, test_dir=test_dir)


def test_joined_lists_round_trip(tmp_path, test_dir):
    cand = JoinedLists.from_pairs(pd.DataFrame({"s1_id": ["S1-a", "S1-a", "S1-c"],
                                                "other_id": ["S2-1", "S3-2", "S2-7"]}))
    match = JoinedLists.from_pairs(pd.DataFrame({"s1_id": ["S1-a"], "other_id": ["S3-2"]}))
    assert cand["S1-a"] == ["S2-1", "S3-2"] and cand.get("S1-b") is None
    assert write_submission(cand, match, out_dir=tmp_path, s1_ids=S1_IDS, test_dir=test_dir) == 0
    assert (tmp_path / "candidate_pairs.tsv").read_text(encoding="utf-8").splitlines()[1:] == [
        "S1-a\tS2-1,S3-2", "S1-b\t", "S1-c\tS2-7"]
