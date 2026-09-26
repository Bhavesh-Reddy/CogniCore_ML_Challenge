from business_entity_resolution.src.io_utils import read_tsv, stable_hash, write_id_list_tsv

HEADER = "matched_entity_ids"


def _write(tmp_path, s1_ids, mapping):
    path = tmp_path / "out.tsv"
    write_id_list_tsv(path, HEADER, s1_ids, mapping)
    return path


def test_read_tsv_keeps_literal_null(tmp_path):
    path = tmp_path / "in.tsv"
    path.write_text("entity_id\tbusiness_address\nS1-1\tnull\nS1-2\t\n", encoding="utf-8")
    df = read_tsv(path)
    assert df["business_address"].tolist() == ["null", ""]


def test_round_trip(tmp_path):
    s1_ids = ["S1-a", "S1-b", "S1-c"]
    mapping = {"S1-a": ["S2-1", "S3-2"], "S1-c": ["S3-9"]}
    df = read_tsv(_write(tmp_path, s1_ids, mapping))
    assert df.columns.tolist() == ["source1_entity_id", HEADER]
    assert df["source1_entity_id"].tolist() == s1_ids
    assert df[HEADER].tolist() == ["S2-1,S3-2", "", "S3-9"]


def test_dedupe_keeps_order(tmp_path):
    path = _write(tmp_path, ["S1-a"], {"S1-a": ["S3-2", "S2-1", "S3-2", "S2-1"]})
    assert path.read_text(encoding="utf-8").splitlines()[1] == "S1-a\tS3-2,S2-1"


def test_drops_non_s2_s3_ids(tmp_path):
    path = _write(tmp_path, ["S1-a"], {"S1-a": ["S1-z", "S2-1", "X-3", ""]})
    assert path.read_text(encoding="utf-8").splitlines()[1] == "S1-a\tS2-1"


def test_empty_list_line(tmp_path):
    path = _write(tmp_path, ["S1-x", "S1-y"], {"S1-y": []})
    raw = path.read_bytes().decode("utf-8")
    assert raw == f"source1_entity_id\t{HEADER}\nS1-x\t\nS1-y\t\n"


def test_stable_hash_is_crc32():
    assert stable_hash("abc") == 891568578
