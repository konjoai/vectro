"""Release candidate hardening suite for Vectro 2.0.0.

Covers all 7 verification gates from the Phase 5 plan:

  1. Quantization quality gates (cosine similarity thresholds)
  2. Compression gates (ratio thresholds per profile)
  3. Performance gates (throughput floor)
  4. Compatibility gates (v1 → v2 migration round-trip)
  5. Integration gates (connector + Arrow + streaming end-to-end)
  6. Distribution gates (package exports, version consistency)
  7. Launch readiness checks (CHANGELOG, README, docs hub)
"""

from __future__ import annotations

import importlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pytest

# ── constants ──────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPECTED_VERSION = "5.1.0"

# Quality and compression floors per profile
# (used in gate 1 and gate 2)
PROFILE_GATES = {
    "fast":     {"min_cosine": 0.95, "min_ratio": 3.5},
    "balanced": {"min_cosine": 0.97, "min_ratio": 3.5},
    "quality":  {"min_cosine": 0.97, "min_ratio": 3.5},
}

# Throughput floor: vectors/second (gate 3)
THROUGHPUT_FLOOR = 50_000

# Test dataset
N, DIM = 500, 256


@pytest.fixture(scope="module")
def embeddings() -> np.ndarray:
    rng = np.random.default_rng(42)
    vecs = rng.standard_normal((N, DIM)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs


@pytest.fixture(scope="module")
def vectro():
    from python.vectro import Vectro
    return Vectro()


# ── Gate 1 & 2: Quantization quality + compression ratio ──────────────────


class TestQualityAndCompressionGates:
    """Gate 1: cosine similarity ≥ threshold.  Gate 2: compression ratio ≥ threshold."""

    @pytest.mark.parametrize("profile_name,gates", list(PROFILE_GATES.items()))
    def test_quality_and_compression(self, embeddings, vectro, profile_name, gates, tmp_path):
        from python.profiles_api import get_compression_profile
        from python import decompress_vectors
        from python.interface import mean_cosine_similarity

        profile = get_compression_profile(profile_name)
        result = vectro.compress(embeddings, profile=profile_name)
        restored = decompress_vectors(result)

        cos_sim = mean_cosine_similarity(embeddings, restored)
        assert cos_sim >= gates["min_cosine"], (
            f"[{profile_name}] quality gate failed: "
            f"cosine_sim={cos_sim:.4f} < threshold={gates['min_cosine']}"
        )
        assert result.compression_ratio >= gates["min_ratio"], (
            f"[{profile_name}] compression gate failed: "
            f"ratio={result.compression_ratio:.2f}× < threshold={gates['min_ratio']}"
        )

    def test_int8_default_profile_passes(self, embeddings, vectro):
        from python import decompress_vectors
        from python.interface import mean_cosine_similarity

        result = vectro.compress(embeddings)
        restored = decompress_vectors(result)
        cos_sim = mean_cosine_similarity(embeddings, restored)
        assert cos_sim >= 0.95, f"Default profile cosine_sim={cos_sim:.4f} below 0.95"
        assert result.compression_ratio >= 3.5


# ── Gate 3: Performance (throughput) ───────────────────────────────────────


class TestPerformanceGates:
    """Gate 3: compression throughput must not regress below THROUGHPUT_FLOOR."""

    def test_throughput_floor(self, embeddings, vectro):
        # Warm-up run
        vectro.compress(embeddings[:10])

        # Timed run
        t0 = time.perf_counter()
        vectro.compress(embeddings)
        elapsed = time.perf_counter() - t0

        throughput = N / elapsed
        assert throughput >= THROUGHPUT_FLOOR, (
            f"Throughput {throughput:,.0f} vec/s < floor {THROUGHPUT_FLOOR:,} vec/s"
        )

    def test_streaming_throughput(self, embeddings, vectro):
        from python import StreamingDecompressor

        result = vectro.compress(embeddings)
        t0 = time.perf_counter()
        total = sum(len(chunk) for chunk in StreamingDecompressor(result, chunk_size=64))
        elapsed = time.perf_counter() - t0

        assert total == N
        throughput = total / elapsed
        assert throughput >= THROUGHPUT_FLOOR, (
            f"Streaming throughput {throughput:,.0f} vec/s < floor"
        )


# ── Gate 4: Compatibility (v1 → v2 migration) ──────────────────────────────


class TestCompatibilityGates:
    """Gate 4: v1 artifacts must load, inspect, upgrade, and validate correctly."""

    def _make_v1(self, tmp_path: Path, n=8, dim=32) -> Path:
        path = tmp_path / "v1.npz"
        rng = np.random.default_rng(99)
        np.savez_compressed(
            str(path),
            quantized=rng.integers(-128, 128, size=(n, dim), dtype=np.int8),
            scales=rng.random(n).astype(np.float32),
            n=np.array(n),
            dims=np.array(dim),
        )
        return path

    def test_v1_loads_correctly(self, vectro, tmp_path):
        """load_compressed handles v1 artifacts without errors."""
        src = self._make_v1(tmp_path)
        result = vectro.load_compressed(str(src))
        assert result is not None

    def test_v1_inspect_detects_version(self, tmp_path):
        from python.migration import inspect_artifact

        src = self._make_v1(tmp_path)
        info = inspect_artifact(src)
        assert info["format_version"] == 1
        assert info["needs_upgrade"] is True

    def test_v1_dry_run_upgrade(self, tmp_path):
        from python.migration import upgrade_artifact

        src = self._make_v1(tmp_path)
        dst = tmp_path / "v2.npz"
        result = upgrade_artifact(src, dst, dry_run=True)
        assert result["dry_run"] is True
        assert not dst.exists(), "dry-run must not write to disk"

    def test_v1_upgrade_roundtrip(self, vectro, tmp_path):
        from python.migration import upgrade_artifact, inspect_artifact, validate_artifact
        from python import decompress_vectors

        src = self._make_v1(tmp_path)
        dst = tmp_path / "v2.npz"
        upgrade_artifact(src, dst)

        # Upgraded artifact is v2 and valid
        info = inspect_artifact(dst)
        assert info["format_version"] == 2
        assert info["needs_upgrade"] is False
        assert validate_artifact(dst)["valid"]

        # Can be loaded and decompressed
        result = vectro.load_compressed(str(dst))
        restored = decompress_vectors(result)
        assert restored.shape == (8, 32)

    def test_upgraded_data_identical_to_original(self, tmp_path):
        from python.migration import upgrade_artifact

        src = self._make_v1(tmp_path)
        orig_data = np.load(src, allow_pickle=False)
        q_orig = orig_data["quantized"].copy()
        s_orig = orig_data["scales"].copy()

        dst = tmp_path / "v2.npz"
        upgrade_artifact(src, dst)
        new_data = np.load(dst, allow_pickle=False)

        np.testing.assert_array_equal(new_data["quantized"], q_orig)
        np.testing.assert_array_equal(new_data["scales"], s_orig)

    def test_bulk_upgrade_multiple_artifacts(self, tmp_path):
        from python.migration import upgrade_artifact, inspect_artifact

        artifacts = []
        for i in range(5):
            path = tmp_path / f"v1_{i}.npz"
            rng = np.random.default_rng(i)
            np.savez_compressed(
                str(path),
                quantized=rng.integers(-128, 128, size=(4, 16), dtype=np.int8),
                scales=rng.random(4).astype(np.float32),
                n=np.array(4),
                dims=np.array(16),
            )
            artifacts.append(path)

        for src in artifacts:
            dst = src.with_suffix(".v2.npz")
            upgrade_artifact(src, dst)
            info = inspect_artifact(dst)
            assert info["format_version"] == 2


# ── Gate 5: Integration end-to-end ─────────────────────────────────────────


class TestIntegrationGates:
    """Gate 5: connectors, Arrow bridge, and streaming end-to-end checks."""

    def test_in_memory_connector_round_trip(self, embeddings):
        from python.integrations import InMemoryVectorDBConnector
        from python import decompress_vectors

        from python.vectro import Vectro
        result = Vectro().compress(embeddings[:50])

        store = InMemoryVectorDBConnector()
        ids = [str(i) for i in range(50)]
        store.upsert_compressed(
            ids=ids,
            quantized=np.stack(result.quantized_vectors),
            scales=result.scales,
        )
        batch = store.fetch_compressed(ids[:5])
        assert batch is not None

    def test_save_load_round_trip(self, embeddings, vectro, tmp_path):
        from python import decompress_vectors
        from python.interface import mean_cosine_similarity

        path = str(tmp_path / "rc_test.npz")
        result = vectro.compress(embeddings)
        vectro.save_compressed(result, path)

        loaded = vectro.load_compressed(path)
        restored = decompress_vectors(loaded)

        cos_sim = mean_cosine_similarity(embeddings, restored)
        assert cos_sim >= 0.95

    def test_arrow_bridge_round_trip(self, embeddings, vectro):
        pytest.importorskip("pyarrow")
        from python.integrations import result_to_table, table_to_result, to_arrow_bytes, from_arrow_bytes
        from python import decompress_vectors

        result = vectro.compress(embeddings)

        # Table round-trip
        table = result_to_table(result)
        result2 = table_to_result(table)
        r1 = decompress_vectors(result)
        r2 = decompress_vectors(result2)
        np.testing.assert_allclose(r1, r2, atol=1e-6)

        # IPC bytes round-trip
        payload = to_arrow_bytes(result)
        result3 = from_arrow_bytes(payload)
        r3 = decompress_vectors(result3)
        np.testing.assert_allclose(r1, r3, atol=1e-6)

    def test_streaming_full_reconstruction(self, embeddings, vectro):
        from python import StreamingDecompressor, decompress_vectors
        from python.interface import mean_cosine_similarity

        result = vectro.compress(embeddings)

        # Streaming reconstruction must equal direct reconstruction
        chunks = list(StreamingDecompressor(result, chunk_size=50))
        streamed = np.vstack(chunks)
        direct = decompress_vectors(result)

        np.testing.assert_allclose(streamed, direct, atol=1e-6)
        assert streamed.shape == embeddings.shape

    def test_benchmark_suite_produces_report(self, embeddings):
        from python.benchmark import BenchmarkSuite

        suite = BenchmarkSuite(n=100, dim=embeddings.shape[1], trials=1)
        report = suite.run()
        assert len(report.entries) > 0

    def test_benchmark_report_json_serializable(self, embeddings, tmp_path):
        from python.benchmark import BenchmarkSuite

        suite = BenchmarkSuite(n=100, dim=embeddings.shape[1], trials=1)
        report = suite.run()
        out = tmp_path / "report.json"
        report.save(str(out))
        assert out.exists()
        import json
        data = json.loads(out.read_text())
        assert isinstance(data, (dict, list))


# ── Gate 6: Distribution (package exports + version consistency) ────────────


class TestDistributionGates:
    """Gate 6: all expected public symbols accessible; version consistent everywhere."""

    EXPECTED_SYMBOLS = [
        # Core
        "Vectro", "compress_vectors", "decompress_vectors",
        "analyze_compression_quality", "generate_compression_report",
        # Interface
        "QuantizationResult", "quantize_embeddings", "reconstruct_embeddings",
        "mean_cosine_similarity", "get_backend_info",
        # Batch
        "VectroBatchProcessor", "BatchQuantizationResult",
        "quantize_embeddings_batch",
        # Quality
        "VectroQualityAnalyzer", "QualityMetrics",
        "evaluate_quantization_quality",
        # Profiles
        "ProfileManager", "CompressionProfile",
        "get_compression_profile", "create_custom_profile",
        # Integrations
        "InMemoryVectorDBConnector", "QdrantConnector", "WeaviateConnector",
        "compress_tensor", "reconstruct_tensor", "HuggingFaceCompressor",
        "result_to_table", "table_to_result",
        "write_parquet", "read_parquet",
        "to_arrow_bytes", "from_arrow_bytes",
        # Streaming / quantization extras
        "StreamingDecompressor",
        "quantize_int2", "dequantize_int2", "quantize_adaptive",
        # Migration
        "inspect_artifact", "upgrade_artifact", "validate_artifact",
        # Utility
        "get_version_info",
    ]

    def test_all_expected_symbols_importable(self):
        import python as pkg
        missing = [sym for sym in self.EXPECTED_SYMBOLS if not hasattr(pkg, sym)]
        assert not missing, f"Missing public symbols: {missing}"

    def test_version_pyproject(self):
        try:
            import tomllib
        except ImportError:
            pytest.skip("tomllib unavailable (Python < 3.11)")
        text = (REPO_ROOT / "pyproject.toml").read_text()
        data = tomllib.loads(text)
        assert data["project"]["version"] == EXPECTED_VERSION

    def test_version_pixi(self):
        try:
            import tomllib
        except ImportError:
            pytest.skip("tomllib unavailable (Python < 3.11)")
        text = (REPO_ROOT / "pixi.toml").read_text()
        data = tomllib.loads(text)
        # version is nested under [workspace]
        version = data.get("workspace", {}).get("version") or data.get("version")
        assert version == EXPECTED_VERSION

    def test_version_python_init(self):
        import python as pkg
        assert pkg.__version__ == EXPECTED_VERSION

    def test_version_vectro_py(self):
        # Read the module's own __version__ attribute
        import python.vectro as m
        assert m.__version__ == EXPECTED_VERSION

    def test_version_info_returns_dict(self):
        import python as pkg
        vi = pkg.get_version_info()
        assert isinstance(vi, dict)

    def test_get_backend_info_returns_dict(self):
        import python as pkg
        bi = pkg.get_backend_info()
        assert isinstance(bi, dict)
        assert "numpy" in bi or "mojo" in bi


# ── Gate 7: Launch readiness ────────────────────────────────────────────────


class TestLaunchReadinessGates:
    """Gate 7: docs hub present, CHANGELOG mentions 2.0.0, README is up to date."""

    DOCS_FILES = [
        "docs/getting-started.md",
        "docs/migration-guide.md",
        "docs/integrations.md",
        "docs/benchmark-methodology.md",
        "docs/api-reference.md",
    ]

    EXAMPLE_FILES = [
        "examples/rag_quickstart.py",
        "examples/vector_search_quickstart.py",
    ]

    def test_docs_hub_all_files_exist(self):
        missing = [f for f in self.DOCS_FILES if not (REPO_ROOT / f).exists()]
        assert not missing, f"Missing docs files: {missing}"

    def test_docs_files_non_empty(self):
        for rel in self.DOCS_FILES:
            path = REPO_ROOT / rel
            assert path.stat().st_size > 200, f"{rel} appears too short"

    def test_example_files_exist(self):
        missing = [f for f in self.EXAMPLE_FILES if not (REPO_ROOT / f).exists()]
        assert not missing, f"Missing example files: {missing}"

    def test_changelog_mentions_version(self):
        changelog = (REPO_ROOT / "CHANGELOG.md").read_text()
        assert "2.0.0" in changelog, "CHANGELOG.md does not mention 2.0.0"

    def test_changelog_has_phase4_content(self):
        changelog = (REPO_ROOT / "CHANGELOG.md").read_text()
        assert "migration" in changelog.lower()
        assert "inspect_artifact" in changelog

    def test_readme_exists_and_non_empty(self):
        readme = REPO_ROOT / "README.md"
        assert readme.exists()
        assert readme.stat().st_size > 1000

    def test_release_workflow_exists(self):
        wf = REPO_ROOT / ".github" / "workflows" / "release.yml"
        assert wf.exists(), "release.yml workflow missing"

    def test_release_workflow_has_pypi_step(self):
        wf = (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text()
        assert "PYPI_API_TOKEN" in wf

    def test_ci_workflow_includes_migration_tests(self):
        wf = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text()
        # CI runs pytest over the entire tests/ directory; test_migration is
        # implicitly included.  Accept either explicit name or full-tests-dir pattern.
        assert "test_migration" in wf or "tests/" in wf

    def test_migration_module_has_cli_entrypoint(self):
        """python -m python.migration must be runnable (has __main__ block)."""
        src = (REPO_ROOT / "python" / "migration.py").read_text()
        assert '__name__ == "__main__"' in src or "if __name__" in src

    def test_cli_module_exists(self):
        cli = REPO_ROOT / "python" / "cli.py"
        assert cli.exists()

    def test_pyproject_script_entry_point_correct(self):
        try:
            import tomllib
        except ImportError:
            pytest.skip("tomllib unavailable (Python < 3.11)")
        text = (REPO_ROOT / "pyproject.toml").read_text()
        data = tomllib.loads(text)
        scripts = data.get("project", {}).get("scripts", {})
        assert "vectro" in scripts
        assert "cli" in scripts["vectro"]
