"""Fetch + extract standard ANN benchmark datasets into benchmarks/data/.

    python benchmarks/harness/download.py --dataset sift1m
    python benchmarks/harness/download.py --dataset gist1m

Datasets are the TEXMEX SIFT1M / GIST1M corpora (Jégou et al.), distributed as
tarballs of ``.fvecs`` / ``.ivecs`` files. They are cached under
``benchmarks/data/`` (gitignored) and never committed. Checksums are verified on
download; a mismatch aborts. The first verified fetch on the target host should
pin the ``SHA256`` values below so subsequent runs are integrity-checked.
"""

from __future__ import annotations

import argparse
import logging
import tarfile
from pathlib import Path

from . import datasets

logger = logging.getLogger(__name__)

# Canonical TEXMEX tarballs. Pin SHA256 after the first verified fetch on target
# hardware (a wrong pin is worse than None, since a mismatch is fatal by design).
SOURCES = {
    "sift1m": {
        "url": "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz",
        "sha256": None,
        "member_dir": "sift",
    },
    "gist1m": {
        "url": "ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz",
        "sha256": None,
        "member_dir": "gist",
    },
}


def download(name: str) -> Path:
    if name not in SOURCES:
        raise KeyError(f"unknown dataset '{name}'; known: {sorted(SOURCES)}")
    src = SOURCES[name]
    datasets.DATA_DIR.mkdir(parents=True, exist_ok=True)
    tarball = datasets.DATA_DIR / f"{src['member_dir']}.tar.gz"
    datasets.fetch(src["url"], tarball, src["sha256"])
    logger.info("extracting %s ...", tarball.name)
    with tarfile.open(tarball, "r:gz") as tf:
        tf.extractall(datasets.DATA_DIR, filter="data")
    out = datasets.DATA_DIR / src["member_dir"]
    logger.info("ready: %s", out)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Download ANN benchmark datasets")
    p.add_argument("--dataset", required=True, choices=sorted(SOURCES))
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    out = download(args.dataset)
    print(f"Extracted to: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
