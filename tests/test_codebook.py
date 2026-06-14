"""Tests for Phase 7b — Autoencoder Codebook."""

import os
import tempfile
import unittest
import numpy as np

try:
    from tests._path_setup import ensure_repo_root_on_path
except ModuleNotFoundError:
    from _path_setup import ensure_repo_root_on_path

ensure_repo_root_on_path()

from python.codebook_api import Codebook  # noqa: E402


class TestCodebookBasic(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.data = rng.standard_normal((150, 64)).astype(np.float32)

    def test_train_returns_self(self):
        cb = Codebook(target_dim=16, seed=0)
        result = cb.train(self.data, n_epochs=5)
        self.assertIs(result, cb)

    def test_is_trained_after_train(self):
        cb = Codebook(target_dim=16, seed=0)
        cb.train(self.data, n_epochs=5)
        self.assertTrue(cb.is_trained)

    def test_untrained_encode_raises(self):
        cb = Codebook(target_dim=16)
        with self.assertRaises(RuntimeError):
            cb.encode(self.data[:5])

    def test_untrained_decode_raises(self):
        cb = Codebook(target_dim=16)
        with self.assertRaises(RuntimeError):
            cb.decode(np.zeros((5, 16), dtype=np.int8))

    def test_encode_output_shape(self):
        cb = Codebook(target_dim=16, seed=0)
        cb.train(self.data, n_epochs=5)
        codes = cb.encode(self.data[:20])
        self.assertEqual(codes.shape, (20, 16))

    def test_encode_output_dtype(self):
        cb = Codebook(target_dim=16, seed=0)
        cb.train(self.data, n_epochs=5)
        codes = cb.encode(self.data[:20])
        self.assertEqual(codes.dtype, np.int8)

    def test_encode_int8_range(self):
        cb = Codebook(target_dim=16, seed=0)
        cb.train(self.data, n_epochs=5)
        codes = cb.encode(self.data)
        self.assertTrue((codes >= -127).all())
        self.assertTrue((codes <= 127).all())

    def test_decode_output_shape(self):
        cb = Codebook(target_dim=16, seed=0)
        cb.train(self.data, n_epochs=5)
        codes = cb.encode(self.data[:20])
        recon = cb.decode(codes)
        self.assertEqual(recon.shape, (20, 64))

    def test_decode_output_dtype(self):
        cb = Codebook(target_dim=16, seed=0)
        cb.train(self.data, n_epochs=5)
        codes = cb.encode(self.data[:20])
        recon = cb.decode(codes)
        self.assertEqual(recon.dtype, np.float32)


class TestCodebookQuality(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(13)
        self.data = rng.standard_normal((300, 64)).astype(np.float32)

    def test_no_nan_in_reconstruction(self):
        cb = Codebook(target_dim=32, seed=0)
        cb.train(self.data, n_epochs=50)
        codes = cb.encode(self.data[:50])
        recon = cb.decode(codes)
        self.assertFalse(np.isnan(recon).any())
        self.assertFalse(np.isinf(recon).any())

    def test_cosine_quality_threshold(self):
        """After 100 epochs on d=64 → target_dim=32, cosine ≥ 0.75."""
        cb = Codebook(target_dim=32, seed=0)
        cb.train(self.data, n_epochs=100)
        codes = cb.encode(self.data)
        recon = cb.decode(codes)
        sim = cb.mean_cosine(self.data, recon)
        self.assertGreater(sim, 0.75, f"Cosine sim {sim:.4f} < 0.75 threshold")

    def test_more_epochs_better_or_equal(self):
        """150-epoch model should not perform drastically worse than 50-epoch."""

        def run(epochs):
            cb = Codebook(target_dim=16, seed=42)
            cb.train(self.data, n_epochs=epochs)
            codes = cb.encode(self.data)
            recon = cb.decode(codes)
            return cb.mean_cosine(self.data, recon)

        sim50 = run(50)
        sim150 = run(150)
        # 150 epochs should generally be at least as good; allow small tolerance
        self.assertGreaterEqual(sim150, sim50 - 0.10)


class TestCodebookCompression(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(55)
        self.data = rng.standard_normal((100, 64)).astype(np.float32)

    def test_compression_ratio_positive(self):
        cb = Codebook(target_dim=16, seed=0)
        cb.train(self.data, n_epochs=5)
        self.assertGreater(cb.compression_ratio(), 0)

    def test_compression_ratio_formula(self):
        """d=64, target_dim=16 → ratio = 64*4 / 16 = 16."""
        cb = Codebook(target_dim=16, seed=0)
        cb.train(self.data, n_epochs=5)
        self.assertAlmostEqual(cb.compression_ratio(), 16.0, places=5)

    def test_higher_compression_with_smaller_target_dim(self):
        cb8 = Codebook(target_dim=8, seed=0)
        cb8.train(self.data, n_epochs=5)
        cb32 = Codebook(target_dim=32, seed=0)
        cb32.train(self.data, n_epochs=5)
        self.assertGreater(cb8.compression_ratio(), cb32.compression_ratio())


class TestCodebookPersistence(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(77)
        self.data = rng.standard_normal((100, 32)).astype(np.float32)

    def test_save_and_load_roundtrip(self):
        cb = Codebook(target_dim=16, seed=0)
        cb.train(self.data, n_epochs=20)
        codes_before = cb.encode(self.data[:10])
        recon_before = cb.decode(codes_before)

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name

        try:
            cb.save(path)
            cb2 = Codebook.load(path)
            codes_after = cb2.encode(self.data[:10])
            recon_after = cb2.decode(codes_after)
            self.assertTrue(np.allclose(recon_before, recon_after, atol=1e-5))
        finally:
            os.unlink(path)

    def test_load_preserves_metadata(self):
        cb = Codebook(target_dim=16, seed=5)
        cb.train(self.data, n_epochs=10)

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            cb.save(path)
            cb2 = Codebook.load(path)
            self.assertEqual(cb2.target_dim, 16)
            self.assertEqual(cb2._d, 32)
            self.assertTrue(cb2.is_trained)
        finally:
            os.unlink(path)

    def test_save_untrained_raises(self):
        cb = Codebook(target_dim=16)
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            with self.assertRaises(RuntimeError):
                cb.save(path)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
