"""Fetch + extract standard ANN benchmark datasets into benchmarks/data/.

    python benchmarks/harness/download.py --dataset sift1m
    python benchmarks/harness/download.py --dataset gist1m

Datasets are the TEXMEX SIFT1M / GIST1M corpora (Jégou et al.), distributed as
tarballs of ``.fvecs`` / ``.ivecs`` files. They are cached under
``benchmarks/data/`` (gitignored) and never committed. Checksums are verified on
download; a mismatch aborts. The first verified fetch on the target host should
pin the ``SHA256`` values below so subsequent runs are integrity-checked.

The fetch/extract steps are checkpointed (``--resume`` / ``--fresh``) so a large
download interrupted mid-extract resumes instead of restarting.
"""

from __future__ import annotations

import argparse
import logging
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from . import _resume, datasets

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Source:
    """A downloadable dataset tarball. ``sha256`` is pinned after first verified fetch."""

    url: str
    sha256: Optional[str]
    member_dir: str


# Canonical TEXMEX tarballs. Pin SHA256 after the first verified fetch on target
# hardware (a wrong pin is worse than None, since a mismatch is fatal by design).
SOURCES: dict[str, Source] = {
    "sift1m": Source("ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz", None, "sift"),
    "gist1m": Source("ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz", None, "gist"),
}


def download(name: str, ckpt: Optional[object] = None) -> Path:
    """Fetch + extract dataset ``name``; ``ckpt`` (a Checkpoint) skips finished steps."""
    if name not in SOURCES:
        raise KeyError(f"unknown dataset '{name}'; known: {sorted(SOURCES)}")
    src = SOURCES[name]
    datasets.DATA_DIR.mkdir(parents=True, exist_ok=True)
    tarball = datasets.DATA_DIR / f"{src.member_dir}.tar.gz"

    if ckpt is None or not ckpt.done(f"fetch:{name}"):  # type: ignore[attr-defined]
        datasets.fetch(src.url, tarball, src.sha256)
        if ckpt is not None:
            ckpt.mark(f"fetch:{name}")  # type: ignore[attr-defined]

    if ckpt is None or not ckpt.done(f"extract:{name}"):  # type: ignore[attr-defined]
        logger.info("extracting %s ...", tarball.name)
        with tarfile.open(tarball, "r:gz") as tf:
            tf.extractall(datasets.DATA_DIR, filter="data")
        if ckpt is not None:
            ckpt.mark(f"extract:{name}")  # type: ignore[attr-defined]

    out = datasets.DATA_DIR / src.member_dir
    logger.info("ready: %s", out)
    return out


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Download ANN benchmark datasets")
    p.add_argument("--dataset", required=True, choices=sorted(SOURCES))
    p.add_argument("--verbose", action="store_true")
    _resume.add_resume_args(p, default_fresh=False)  # resume is the default
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    ckpt = _resume.make_checkpoint(
        _resume.checkpoint_path(f"download_{args.dataset}"), fresh=_resume.is_fresh(args)
    )
    out = download(args.dataset, ckpt=ckpt)
    print(f"Extracted to: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
