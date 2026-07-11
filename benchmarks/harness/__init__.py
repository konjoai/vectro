"""VECTRO recall-matched benchmark harness (audit item 5.2).

The measurement gate every later optimization sprint merges through: real
datasets (SIFT1M / GIST1M + a pluggable embedding-set loader), recall-matched
QPS comparison, 30-run percentile statistics with a paired Wilcoxon test and a
coefficient-of-variation gate, and JSON + Markdown results carrying a full scope
line. Entry point: ``python benchmarks/harness/run.py --suite core --dataset sift1m``.
"""

from __future__ import annotations

__all__ = ["datasets", "engines", "protocol", "report", "stats"]
