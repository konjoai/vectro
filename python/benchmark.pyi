import numpy as np
import platform
from _typeshed import Incomplete
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class BenchmarkEntry:
    profile: str
    n_vectors: int
    vector_dim: int
    precision_mode: str
    backend: str
    throughput_vps: float
    throughput_mbs: float
    compression_ratio: float
    compressed_mb: float
    original_mb: float
    mean_cosine_sim: float
    mean_absolute_error: float
    median_latency_ms: float
    p95_latency_ms: float
    python_version: str = field(default_factory=Incomplete)
    platform: str = field(default_factory=platform.platform)
    numpy_version: str = field(default_factory=Incomplete)

@dataclass
class BenchmarkReport:
    entries: list[BenchmarkEntry]
    generated_at: str = field(default_factory=Incomplete)
    vectro_version: str = ...
    def to_dict(self) -> dict: ...
    def save(self, path: str | Path, fmt: str | None = None) -> None: ...
    def _save_csv(self, path: Path) -> None: ...
    def to_csv_string(self) -> str: ...
    def print_summary(self) -> None: ...

class BenchmarkSuite:
    n: Incomplete
    dim: Incomplete
    profiles: Incomplete
    trials: Incomplete
    seed: Incomplete
    backend: Incomplete
    def __init__(
        self,
        n: int = 2000,
        dim: int = 384,
        profiles: list[str] | None = None,
        trials: int = 5,
        seed: int = 42,
        backend: str = "auto",
    ) -> None: ...
    def _make_data(self) -> np.ndarray: ...
    def _run_one(self, vectors: np.ndarray, profile: str) -> BenchmarkEntry: ...
    def run(self) -> BenchmarkReport: ...

def _main(argv: list[str] | None = None) -> None: ...
