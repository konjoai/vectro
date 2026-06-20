"""Tests for python.migration — inspect, validate, and upgrade artifact versions."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from python.migration import (
    inspect_artifact,
    upgrade_artifact,
    validate_artifact,
    _CURRENT_FORMAT_VERSION,
    _STORAGE_FORMAT_NAME,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_v1_single(tmp_path: Path, n: int = 4, dim: int = 8) -> Path:
    """Create a v1-style single artifact (no storage_format_version field)."""
    path = tmp_path / "v1_single.npz"
    rng = np.random.default_rng(0)
    quantized = rng.integers(-128, 128, size=(n, dim), dtype=np.int8)
    scales = rng.random(n).astype(np.float32)
    np.savez_compressed(
        str(path),
        quantized=quantized,
        scales=scales,
        dims=np.array(dim),
        n=np.array(n),
        # NOTE: NO storage_format_version — this is the v1 signal
    )
    return path


def _make_v1_batch(tmp_path: Path, batch: int = 3, dim: int = 8) -> Path:
    """Create a v1-style batch artifact (has batch_size, no version field)."""
    path = tmp_path / "v1_batch.npz"
    rng = np.random.default_rng(1)
    quantized = rng.integers(-128, 128, size=(batch, dim), dtype=np.int8)
    scales = rng.random(batch).astype(np.float32)
    np.savez_compressed(
        str(path),
        quantized=quantized,
        scales=scales,
        batch_size=np.array(batch),
        vector_dim=np.array(dim),
        compression_ratio=np.array(4.0),
        total_original_bytes=np.array(batch * dim * 4),
        total_compressed_bytes=np.array(batch * dim),
    )
    return path


def _make_v2_single(tmp_path: Path, n: int = 4, dim: int = 8) -> Path:
    """Create a v2-style single artifact (all fields present)."""
    path = tmp_path / "v2_single.npz"
    rng = np.random.default_rng(2)
    quantized = rng.integers(-128, 128, size=(n, dim), dtype=np.int8)
    scales = rng.random(n).astype(np.float32)
    meta = json.dumps({"vectro_version": "1.2.0"})
    np.savez_compressed(
        str(path),
        quantized=quantized,
        scales=scales,
        dims=np.array(dim),
        n=np.array(n),
        artifact_type=np.array("single"),
        precision_mode=np.array("int8"),
        group_size=np.array(0),
        storage_format=np.array(_STORAGE_FORMAT_NAME),
        storage_format_version=np.array(_CURRENT_FORMAT_VERSION),
        metadata_json=np.array(meta),
    )
    return path


# ---------------------------------------------------------------------------
# inspect_artifact tests
# ---------------------------------------------------------------------------


class TestInspectArtifact(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_detect_v1_single(self):
        path = _make_v1_single(self.tmp_path)
        info = inspect_artifact(path)
        self.assertEqual(info["format_version"], 1)
        self.assertTrue(info["needs_upgrade"])
        self.assertEqual(info["artifact_type"], "single")
        self.assertEqual(info["n_vectors"], 4)
        self.assertEqual(info["vector_dim"], 8)

    def test_detect_v1_batch(self):
        path = _make_v1_batch(self.tmp_path)
        info = inspect_artifact(path)
        self.assertEqual(info["format_version"], 1)
        self.assertTrue(info["needs_upgrade"])
        self.assertEqual(info["artifact_type"], "batch")
        self.assertEqual(info["n_vectors"], 3)

    def test_detect_v2_current(self):
        path = _make_v2_single(self.tmp_path)
        info = inspect_artifact(path)
        self.assertEqual(info["format_version"], _CURRENT_FORMAT_VERSION)
        self.assertFalse(info["needs_upgrade"])

    def test_file_size_populated(self):
        path = _make_v1_single(self.tmp_path)
        info = inspect_artifact(path)
        self.assertGreater(info["file_size_bytes"], 0)

    def test_compression_ratio_single(self):
        path = _make_v1_single(self.tmp_path)
        info = inspect_artifact(path)
        self.assertGreater(info["compression_ratio"], 0.0)

    def test_compression_ratio_batch(self):
        path = _make_v1_batch(self.tmp_path)
        info = inspect_artifact(path)
        self.assertAlmostEqual(info["compression_ratio"], 4.0, places=5)

    def test_fields_list_returned(self):
        path = _make_v1_single(self.tmp_path)
        info = inspect_artifact(path)
        self.assertIn("quantized", info["fields"])
        self.assertIn("scales", info["fields"])

    def test_precision_mode_default_for_v1(self):
        path = _make_v1_single(self.tmp_path)
        info = inspect_artifact(path)
        self.assertEqual(info["precision_mode"], "int8")

    def test_group_size_default_for_v1(self):
        path = _make_v1_single(self.tmp_path)
        info = inspect_artifact(path)
        self.assertEqual(info["group_size"], 0)

    def test_metadata_parsed_when_present(self):
        path = _make_v2_single(self.tmp_path)
        info = inspect_artifact(path)
        self.assertIsNotNone(info["metadata"])
        self.assertEqual(info["metadata"].get("vectro_version"), "1.2.0")

    def test_metadata_none_when_absent(self):
        path = _make_v1_single(self.tmp_path)
        info = inspect_artifact(path)
        self.assertIsNone(info["metadata"])

    def test_file_not_found_raises(self):
        with self.assertRaises(FileNotFoundError):
            inspect_artifact(self.tmp_path / "nonexistent.npz")

    def test_path_string_accepted(self):
        path = _make_v1_single(self.tmp_path)
        info = inspect_artifact(str(path))
        self.assertEqual(info["format_version"], 1)


# ---------------------------------------------------------------------------
# validate_artifact tests
# ---------------------------------------------------------------------------


class TestValidateArtifact(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_valid_v1_single(self):
        path = _make_v1_single(self.tmp_path)
        result = validate_artifact(path)
        self.assertTrue(result["valid"])
        self.assertEqual(result["errors"], [])

    def test_valid_v2_single(self):
        path = _make_v2_single(self.tmp_path)
        result = validate_artifact(path)
        self.assertTrue(result["valid"])

    def test_valid_v1_batch(self):
        path = _make_v1_batch(self.tmp_path)
        result = validate_artifact(path)
        self.assertTrue(result["valid"])

    def test_missing_quantized_field(self):
        path = self.tmp_path / "bad.npz"
        np.savez_compressed(str(path), scales=np.ones(4, dtype=np.float32))
        result = validate_artifact(path)
        self.assertFalse(result["valid"])
        self.assertTrue(any("quantized" in e for e in result["errors"]))

    def test_missing_scales_field(self):
        path = self.tmp_path / "bad.npz"
        rng = np.random.default_rng(0)
        np.savez_compressed(
            str(path),
            quantized=rng.integers(-128, 128, size=(4, 8), dtype=np.int8),
            n=np.array(4),
            dims=np.array(8),
        )
        result = validate_artifact(path)
        self.assertFalse(result["valid"])
        self.assertTrue(any("scales" in e for e in result["errors"]))

    def test_nonexistent_file(self):
        result = validate_artifact(self.tmp_path / "missing.npz")
        self.assertFalse(result["valid"])

    def test_shape_mismatch_single(self):
        path = self.tmp_path / "mismatch.npz"
        rng = np.random.default_rng(0)
        np.savez_compressed(
            str(path),
            quantized=rng.integers(-128, 128, size=(5, 8), dtype=np.int8),
            scales=rng.random(4).astype(np.float32),
            n=np.array(4),
            dims=np.array(8),
        )
        result = validate_artifact(path)
        self.assertFalse(result["valid"])


# ---------------------------------------------------------------------------
# upgrade_artifact tests
# ---------------------------------------------------------------------------


class TestUpgradeArtifact(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_upgrades_v1_single_to_v2(self):
        src = _make_v1_single(self.tmp_path)
        dst = self.tmp_path / "upgraded.npz"
        result = upgrade_artifact(src, dst)
        self.assertTrue(result["upgraded"])
        self.assertEqual(result["src_version"], 1)
        self.assertEqual(result["dst_version"], _CURRENT_FORMAT_VERSION)
        info = inspect_artifact(dst)
        self.assertEqual(info["format_version"], _CURRENT_FORMAT_VERSION)
        self.assertFalse(info["needs_upgrade"])

    def test_upgrades_v1_batch_to_v2(self):
        src = _make_v1_batch(self.tmp_path)
        dst = self.tmp_path / "upgraded_batch.npz"
        upgrade_artifact(src, dst)
        info = inspect_artifact(dst)
        self.assertEqual(info["format_version"], _CURRENT_FORMAT_VERSION)
        self.assertEqual(info["artifact_type"], "batch")

    def test_upgrade_preserves_quantized_arrays(self):
        src = _make_v1_single(self.tmp_path)
        orig = np.load(src, allow_pickle=False)
        q_orig = orig["quantized"].copy()
        s_orig = orig["scales"].copy()
        dst = self.tmp_path / "upgraded.npz"
        upgrade_artifact(src, dst)
        upgraded = np.load(dst, allow_pickle=False)
        np.testing.assert_array_equal(upgraded["quantized"], q_orig)
        np.testing.assert_array_equal(upgraded["scales"], s_orig)

    def test_upgrade_adds_precision_mode(self):
        src = _make_v1_single(self.tmp_path)
        dst = self.tmp_path / "upgraded.npz"
        upgrade_artifact(src, dst)
        data = np.load(dst, allow_pickle=False)
        self.assertIn("precision_mode", data.files)
        self.assertEqual(str(data["precision_mode"]), "int8")

    def test_upgrade_adds_group_size(self):
        src = _make_v1_single(self.tmp_path)
        dst = self.tmp_path / "upgraded.npz"
        upgrade_artifact(src, dst)
        data = np.load(dst, allow_pickle=False)
        self.assertIn("group_size", data.files)
        self.assertEqual(int(data["group_size"]), 0)

    def test_upgrade_adds_metadata_with_migration_record(self):
        src = _make_v1_single(self.tmp_path)
        dst = self.tmp_path / "upgraded.npz"
        upgrade_artifact(src, dst)
        data = np.load(dst, allow_pickle=False)
        self.assertIn("metadata_json", data.files)
        meta = json.loads(str(data["metadata_json"]))
        self.assertIn("migration", meta)
        self.assertEqual(meta["migration"]["migrated_from_version"], 1)
        self.assertEqual(meta["migration"]["migrated_to_version"], _CURRENT_FORMAT_VERSION)

    def test_no_upgrade_needed_for_v2(self):
        src = _make_v2_single(self.tmp_path)
        dst = self.tmp_path / "copy.npz"
        result = upgrade_artifact(src, dst)
        self.assertFalse(result["upgraded"])
        self.assertEqual(result["src_version"], _CURRENT_FORMAT_VERSION)
        self.assertTrue(dst.exists())

    def test_dry_run_does_not_write_file(self):
        src = _make_v1_single(self.tmp_path)
        dst = self.tmp_path / "should_not_exist.npz"
        result = upgrade_artifact(src, dst, dry_run=True)
        self.assertTrue(result["dry_run"])
        self.assertFalse(dst.exists())

    def test_upgrade_artifact_type_single_inferred(self):
        src = _make_v1_single(self.tmp_path)
        dst = self.tmp_path / "upgraded.npz"
        upgrade_artifact(src, dst)
        data = np.load(dst, allow_pickle=False)
        self.assertEqual(str(data["artifact_type"]), "single")

    def test_upgrade_artifact_type_batch_inferred(self):
        src = _make_v1_batch(self.tmp_path)
        dst = self.tmp_path / "upgraded.npz"
        upgrade_artifact(src, dst)
        data = np.load(dst, allow_pickle=False)
        self.assertEqual(str(data["artifact_type"]), "batch")

    def test_upgrade_creates_parent_directory(self):
        src = _make_v1_single(self.tmp_path)
        dst = self.tmp_path / "subdir" / "nested" / "upgraded.npz"
        upgrade_artifact(src, dst)
        self.assertTrue(dst.exists())

    def test_upgraded_artifact_passes_validation(self):
        src = _make_v1_single(self.tmp_path)
        dst = self.tmp_path / "upgraded.npz"
        upgrade_artifact(src, dst)
        result = validate_artifact(dst)
        self.assertTrue(result["valid"])
        self.assertEqual(result["errors"], [])


if __name__ == "__main__":
    unittest.main()
