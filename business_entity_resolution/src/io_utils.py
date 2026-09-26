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


def peak_mem_gb() -> Optional[float]:
    """Peak resident memory of this process in GB, or None if psutil is unavailable."""
    try:
        import psutil
    except ImportError:
        return None
    info = psutil.Process().memory_info()
    peak = getattr(info, "peak_wset", None) or getattr(info, "peak_rss", None)
    if peak is None:  # Linux/macOS: ru_maxrss is KB on Linux, bytes on macOS
        import resource
        import sys
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
    return round(peak / 2**30, 2)


class TreeMemWatch:
    """Samples the RSS of this process + all child processes (e.g. a multiprocessing Pool) in a thread.

    Use as a context manager; .peak_gb is the highest total seen, .total_gb the machine's RAM.
    Without psutil both stay None.
    """

    def __init__(self, interval: float = 0.5):
        self.interval, self.peak_gb, self.total_gb = interval, None, None
        self._stop = None

    def _sample(self, psutil, proc) -> float:
        total = 0
        for p in [proc] + proc.children(recursive=True):
            try:
                total += p.memory_info().rss
            except psutil.Error:
                pass
        return total / 2**30

    def __enter__(self):
        try:
            import psutil
        except ImportError:
            return self
        import threading
        proc = psutil.Process()
        self.total_gb = round(psutil.virtual_memory().total / 2**30, 2)
        self.peak_gb = 0.0
        self._stop = threading.Event()

        def run():
            while not self._stop.is_set():
                self.peak_gb = max(self.peak_gb, self._sample(psutil, proc))
                self._stop.wait(self.interval)

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        if self._stop is not None:
            self._stop.set()
            self._thread.join()
            self.peak_gb = round(self.peak_gb, 2)
        return False


def load_test_s1_ids() -> List[str]:
    """All test S1 IDs in test_source1.tsv file order."""
    path = config.DATA_DIR / "test" / "test_source1.tsv"
    return read_tsv(path).iloc[:, 0].tolist()
