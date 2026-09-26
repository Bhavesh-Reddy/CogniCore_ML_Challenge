"""TSV reading/writing and stable hashing (contracts.md sections 2 and 5)."""
import csv
import zlib
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

from . import config

VALID_PREFIXES = ("S2-", "S3-")


def read_tsv(path, nrows: Optional[int] = None) -> pd.DataFrame:
    """Read a TSV with every value as a literal string ("null" stays "null")."""
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                       quoting=csv.QUOTE_NONE, nrows=nrows)


def stable_hash(s: str) -> int:
    """Process-independent hash. Never use the built-in hash(), which is salted."""
    return zlib.crc32(s.encode("utf-8"))


def write_id_list_tsv(path, header_col: str, s1_ids_in_order: Iterable[str],
                      mapping: Dict[str, Iterable[str]]) -> None:
    """Write one row per S1 in the given order: `s1<TAB>id1,id2,...`.

    Lists are deduplicated keeping order; only IDs starting with S2-/S3- are kept.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"source1_entity_id\t{header_col}\n")
        for s1 in s1_ids_in_order:
            ids = [i for i in dict.fromkeys(mapping.get(s1, ()))
                   if i.startswith(VALID_PREFIXES)]
            f.write(f"{s1}\t{','.join(ids)}\n")


def load_test_s1_ids() -> List[str]:
    """All test S1 IDs in test_source1.tsv file order."""
    path = config.DATA_DIR / "test" / "test_source1.tsv"
    return read_tsv(path).iloc[:, 0].tolist()
