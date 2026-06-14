"""Tests for pipeline_checkpoint — python/pipeline_checkpoint.py (v5.4.0).

Covers:
- save_pipeline creates a file
- saved file is valid JSON
- load_pipeline round-trips stage names
- load_pipeline returns CompressionPipeline
- checkpoint_info returns the expected top-level keys
- checkpoint_info version matches __version__
- checkpoint_info metadata round-trips
- atomic write (tmp file absent after success)
- parent directories created automatically
- TypeError raised on non-pipeline argument
- FileNotFoundError on missing path
- ValueError on invalid JSON schema
- round-trip with zero stages (no crash)
- round-trip with three stages
- None metadata handled gracefully
- nested metadata dict preserved
- PipelineStage.to_config() returns dict with "name"
- PipelineStage.from_config() reconstructs the stage
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

try:
    from tests._path_setup import ensure_repo_root_on_path
except ModuleNotFoundError:
    from _path_setup import ensure_repo_root_on_path

ensure_repo_root_on_path()

from python.async_pipeline import CompressionPipeline, PipelineStage
from python.pipeline_checkpoint import (
    checkpoint_info,
    load_pipeline,
    save_pipeline,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pipeline(*modes: str) -> CompressionPipeline:
    """Build a CompressionPipeline with the given mode strings."""
    stages = [PipelineStage(mode=m) for m in modes]
    return CompressionPipeline(stages)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSavePipelineCreatesFile(unittest.TestCase):
    """save_pipeline creates the destination file."""

    def test_save_creates_file(self) -> None:
        """File must exist after a successful save."""
        pipeline = _make_pipeline("int8")
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "cp.json"
            save_pipeline(pipeline, dest)
            self.assertTrue(dest.exists(), "checkpoint file was not created")


class TestSaveIsValidJson(unittest.TestCase):
    """Saved file must be parseable JSON."""

    def test_save_is_valid_json(self) -> None:
        """JSON.loads must not raise on the saved file."""
        pipeline = _make_pipeline("int8")
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "cp.json"
            save_pipeline(pipeline, dest)
            raw = dest.read_text(encoding="utf-8")
            payload = json.loads(raw)  # raises on bad JSON
            self.assertIsInstance(payload, dict)


class TestLoadRoundtripStageNames(unittest.TestCase):
    """Loaded pipeline must preserve the stage names in order."""

    def test_load_roundtrip_stage_names(self) -> None:
        """Stage mode sequence must survive a save/load cycle."""
        modes = ["int8", "nf4", "binary"]
        pipeline = _make_pipeline(*modes)
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "cp.json"
            save_pipeline(pipeline, dest)
            loaded = load_pipeline(dest)
            loaded_modes = [s.mode for s in loaded.stages]
            self.assertEqual(loaded_modes, modes)


class TestLoadProducesCompressionPipeline(unittest.TestCase):
    """load_pipeline must return a CompressionPipeline."""

    def test_load_produces_compression_pipeline(self) -> None:
        """Returned object must be an instance of CompressionPipeline."""
        pipeline = _make_pipeline("int8")
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "cp.json"
            save_pipeline(pipeline, dest)
            loaded = load_pipeline(dest)
            self.assertIsInstance(loaded, CompressionPipeline)


class TestCheckpointInfoReturnsDict(unittest.TestCase):
    """checkpoint_info must return all required top-level keys."""

    def test_checkpoint_info_returns_dict(self) -> None:
        """Keys version, created_at, stage_configs, metadata must be present."""
        pipeline = _make_pipeline("int8")
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "cp.json"
            save_pipeline(pipeline, dest)
            info = checkpoint_info(dest)
            for key in ("version", "created_at", "stage_configs", "metadata"):
                self.assertIn(key, info, f"Missing key: {key}")


class TestCheckpointInfoVersion(unittest.TestCase):
    """Checkpoint version must match the package __version__."""

    def test_checkpoint_info_version(self) -> None:
        """version field must equal vectro.__version__."""
        from python import __version__

        pipeline = _make_pipeline("int8")
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "cp.json"
            save_pipeline(pipeline, dest)
            info = checkpoint_info(dest)
            self.assertEqual(info["version"], __version__)


class TestCheckpointInfoMetadata(unittest.TestCase):
    """User metadata must round-trip through checkpoint_info."""

    def test_checkpoint_info_metadata(self) -> None:
        """Metadata dict must be preserved verbatim."""
        pipeline = _make_pipeline("int8")
        user_meta = {"experiment": "run-01", "seed": 42}
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "cp.json"
            save_pipeline(pipeline, dest, metadata=user_meta)
            info = checkpoint_info(dest)
            self.assertEqual(info["metadata"], user_meta)


class TestSaveAtomicWrite(unittest.TestCase):
    """Tmp file must not be present after a successful save."""

    def test_save_atomic_write(self) -> None:
        """No *.tmp file must remain next to the checkpoint after save."""
        pipeline = _make_pipeline("int8")
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "cp.json"
            save_pipeline(pipeline, dest)
            tmp_files = list(Path(tmp).glob("*.tmp"))
            self.assertEqual(tmp_files, [], f"Leftover tmp files: {tmp_files}")


class TestSaveCreatesParentDirs(unittest.TestCase):
    """save_pipeline must create missing parent directories."""

    def test_save_creates_parent_dirs(self) -> None:
        """Nested directories must be created automatically."""
        pipeline = _make_pipeline("int8")
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "a" / "b" / "c" / "cp.json"
            save_pipeline(pipeline, dest)
            self.assertTrue(dest.exists())


class TestSaveTypeErrorOnNonPipeline(unittest.TestCase):
    """save_pipeline must raise TypeError on a non-pipeline argument."""

    def test_save_type_error_on_non_pipeline(self) -> None:
        """Passing a plain dict must raise TypeError."""
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "cp.json"
            with self.assertRaises(TypeError):
                save_pipeline({"stages": []}, dest)  # type: ignore[arg-type]


class TestLoadFileNotFound(unittest.TestCase):
    """load_pipeline must raise FileNotFoundError on a missing path."""

    def test_load_file_not_found(self) -> None:
        """Referencing a non-existent path must raise FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            load_pipeline("/tmp/vectro_nonexistent_checkpoint_xyz.json")


class TestLoadValueErrorOnInvalidSchema(unittest.TestCase):
    """load_pipeline must raise ValueError on malformed JSON schema."""

    def test_load_value_error_on_invalid_schema(self) -> None:
        """JSON missing required keys must raise ValueError."""
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.json"
            bad.write_text(json.dumps({"junk": True}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_pipeline(bad)


class TestRoundtripEmptyPipeline(unittest.TestCase):
    """A zero-stage checkpoint must save and load without crashing."""

    def test_roundtrip_empty_pipeline(self) -> None:
        """Empty pipeline (0 stages) must survive a save/load cycle."""
        # We need to bypass CompressionPipeline's guard, so patch stages directly.
        pipeline = _make_pipeline("int8")
        # Temporarily replace stages with empty list for serialisation only.
        object.__setattr__(pipeline, "_stages", [])

        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "empty.json"
            save_pipeline(pipeline, dest)
            loaded = load_pipeline(dest)
            self.assertEqual(loaded.stages, [])


class TestRoundtripMultiStage(unittest.TestCase):
    """A three-stage pipeline must survive a complete round-trip."""

    def test_roundtrip_multi_stage(self) -> None:
        """Stage names and count must be identical after save/load."""
        modes = ["int8", "nf4", "pq"]
        pipeline = _make_pipeline(*modes)
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "multi.json"
            save_pipeline(pipeline, dest)
            loaded = load_pipeline(dest)
            self.assertEqual([s.mode for s in loaded.stages], modes)


class TestRoundtripMetadataNone(unittest.TestCase):
    """Passing metadata=None must not crash and must store an empty dict."""

    def test_roundtrip_metadata_none(self) -> None:
        """metadata=None must be stored as an empty dict, not null."""
        pipeline = _make_pipeline("int8")
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "cp.json"
            save_pipeline(pipeline, dest, metadata=None)
            info = checkpoint_info(dest)
            self.assertIsInstance(info["metadata"], dict)


class TestRoundtripMetadataNested(unittest.TestCase):
    """Deeply nested metadata must round-trip without loss."""

    def test_roundtrip_metadata_nested(self) -> None:
        """Nested dicts and lists inside metadata must be preserved."""
        nested = {
            "config": {"lr": 1e-3, "batch": 64},
            "tags": ["prod", "v5"],
            "deep": {"a": {"b": {"c": 42}}},
        }
        pipeline = _make_pipeline("int8")
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "cp.json"
            save_pipeline(pipeline, dest, metadata=nested)
            info = checkpoint_info(dest)
            self.assertEqual(info["metadata"], nested)


class TestStageToConfig(unittest.TestCase):
    """PipelineStage.to_config() must return a dict containing 'name'."""

    def test_stage_to_config(self) -> None:
        """to_config() result must be a dict with at least a 'name' key."""
        stage = PipelineStage(mode="int8", profile="speed", group_size=128)
        cfg = stage.to_config()
        self.assertIsInstance(cfg, dict)
        self.assertIn("name", cfg)
        self.assertEqual(cfg["name"], "int8")
        self.assertEqual(cfg.get("profile"), "speed")
        self.assertEqual(cfg.get("group_size"), 128)


class TestStageFromConfig(unittest.TestCase):
    """PipelineStage.from_config() must reconstruct the stage from a config dict."""

    def test_stage_from_config(self) -> None:
        """from_config(to_config()) must produce an equivalent stage."""
        original = PipelineStage(mode="nf4", profile="quality", group_size=64)
        cfg = original.to_config()
        reconstructed = PipelineStage.from_config(cfg)
        self.assertIsInstance(reconstructed, PipelineStage)
        self.assertEqual(reconstructed.mode, original.mode)
        self.assertEqual(reconstructed.profile, original.profile)
        self.assertEqual(reconstructed.group_size, original.group_size)


if __name__ == "__main__":
    unittest.main()
