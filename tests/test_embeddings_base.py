"""Tests for BaseEmbeddingProvider — batching, caching, and protocol surface."""
from __future__ import annotations

import asyncio
import os
import tempfile
import threading
import unittest
from typing import List

import numpy as np

try:
    from tests._path_setup import ensure_repo_root_on_path
except ModuleNotFoundError:
    from _path_setup import ensure_repo_root_on_path

ensure_repo_root_on_path()

from python.embeddings.base import BaseEmbeddingProvider  # noqa: E402


# ---------------------------------------------------------------------------
# Deterministic stub provider — concatenates text to a tiny vocabulary vector.
# ---------------------------------------------------------------------------

class _StubProvider(BaseEmbeddingProvider):
    provider_name = "stub"

    def __init__(self, *args, dim: int = 8, **kwargs):
        super().__init__(*args, **kwargs)
        self._dim = dim
        self.batch_calls: List[List[str]] = []

    def _embed_batch(self, texts: List[str]) -> np.ndarray:
        self.batch_calls.append(list(texts))
        # Deterministic: hash text to a fixed-dim float32 vector.
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, t in enumerate(texts):
            h = abs(hash(t))
            for j in range(self._dim):
                out[i, j] = ((h >> (j * 4)) & 0xFF) / 255.0
        return out


class _BadShapeProvider(BaseEmbeddingProvider):
    provider_name = "bad-shape"

    def _embed_batch(self, texts: List[str]) -> np.ndarray:
        return np.zeros((len(texts) + 1, 4), dtype=np.float32)  # wrong row count


class _BadDimProvider(BaseEmbeddingProvider):
    provider_name = "bad-dim"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._call = 0

    def _embed_batch(self, texts: List[str]) -> np.ndarray:
        self._call += 1
        d = 4 if self._call == 1 else 8  # dim drift after first call
        return np.ones((len(texts), d), dtype=np.float32) * self._call


# ---------------------------------------------------------------------------
# Construction / config
# ---------------------------------------------------------------------------

class TestConstruction(unittest.TestCase):
    def test_batch_size_must_be_positive(self):
        with self.assertRaises(ValueError):
            _StubProvider(model="m", batch_size=0)

    def test_repr(self):
        p = _StubProvider(model="my-model", batch_size=4)
        s = repr(p)
        self.assertIn("model='my-model'", s)
        self.assertIn("batch_size=4", s)

    def test_no_cache_dir_skips_db(self):
        p = _StubProvider(model="m", cache_dir=None)
        self.assertEqual(p.cache_stats(), {"hits": 0, "misses": 0, "size": 0})


# ---------------------------------------------------------------------------
# Auto-batching
# ---------------------------------------------------------------------------

class TestBatching(unittest.TestCase):
    def test_splits_into_batch_size(self):
        p = _StubProvider(model="m", batch_size=3)
        out = p(["a", "b", "c", "d", "e", "f", "g"])
        self.assertEqual(out.shape, (7, 8))
        sizes = [len(c) for c in p.batch_calls]
        self.assertEqual(sizes, [3, 3, 1])

    def test_single_string_returns_1d(self):
        p = _StubProvider(model="m")
        out = p("hello")
        self.assertEqual(out.shape, (8,))

    def test_empty_list_returns_zero_rows(self):
        p = _StubProvider(model="m")
        out = p([])
        self.assertEqual(out.shape[0], 0)

    def test_bad_shape_raises(self):
        p = _BadShapeProvider(model="m")
        with self.assertRaises(ValueError):
            p(["a", "b"])

    def test_dimension_drift_raises(self):
        p = _BadDimProvider(model="m", batch_size=2)
        # First call seeds dimension=4
        p(["a", "b"])
        # Second call returns dim=8 — must raise
        with self.assertRaises(ValueError):
            p(["c", "d"])


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

class TestCache(unittest.TestCase):
    def test_cache_hit_avoids_embed(self):
        with tempfile.TemporaryDirectory() as td:
            p = _StubProvider(model="m", cache_dir=td)
            p(["x", "y", "z"])
            self.assertEqual(len(p.batch_calls), 1)
            # Re-embed exact same texts → no new batch call
            p(["x", "y", "z"])
            self.assertEqual(len(p.batch_calls), 1)
            stats = p.cache_stats()
            self.assertEqual(stats["hits"], 3)
            self.assertEqual(stats["misses"], 3)
            self.assertEqual(stats["size"], 3)

    def test_partial_cache_hit_only_misses_embedded(self):
        with tempfile.TemporaryDirectory() as td:
            p = _StubProvider(model="m", batch_size=10, cache_dir=td)
            p(["a", "b"])
            p.batch_calls.clear()
            p(["a", "c", "b", "d"])
            # Only "c" and "d" should be in the new batch call
            self.assertEqual(len(p.batch_calls), 1)
            self.assertEqual(set(p.batch_calls[0]), {"c", "d"})

    def test_cache_persists_across_instances(self):
        with tempfile.TemporaryDirectory() as td:
            p1 = _StubProvider(model="m", cache_dir=td)
            v1 = p1("hello")
            p1.close()

            p2 = _StubProvider(model="m", cache_dir=td)
            v2 = p2("hello")
            self.assertEqual(p2.batch_calls, [],
                             "expected zero batch calls on cached re-load")
            np.testing.assert_array_equal(v1, v2)

    def test_clear_cache(self):
        with tempfile.TemporaryDirectory() as td:
            p = _StubProvider(model="m", cache_dir=td)
            p(["x", "y"])
            self.assertEqual(p.cache_stats()["size"], 2)
            p.clear_cache()
            self.assertEqual(p.cache_stats()["size"], 0)
            self.assertEqual(p.cache_stats()["hits"], 0)
            self.assertEqual(p.cache_stats()["misses"], 0)

    def test_cache_keyed_by_model(self):
        with tempfile.TemporaryDirectory() as td:
            p1 = _StubProvider(model="model-a", cache_dir=td)
            p1("shared text")
            p2 = _StubProvider(model="model-b", cache_dir=td)
            p2("shared text")
            # Separate model namespaces — both miss
            self.assertEqual(len(p1.batch_calls), 1)
            self.assertEqual(len(p2.batch_calls), 1)

    def test_cache_keyed_by_provider(self):
        class _OtherStub(_StubProvider):
            provider_name = "other-stub"
        with tempfile.TemporaryDirectory() as td:
            a = _StubProvider(model="m", cache_dir=td)
            a("text")
            b = _OtherStub(model="m", cache_dir=td)
            b("text")
            self.assertEqual(len(b.batch_calls), 1,
                             "different provider must not collide")

    def test_concurrent_calls_are_safe(self):
        with tempfile.TemporaryDirectory() as td:
            p = _StubProvider(model="m", cache_dir=td)
            errors: List[BaseException] = []

            def worker(i):
                try:
                    out = p([f"text-{i}-{j}" for j in range(5)])
                    self.assertEqual(out.shape, (5, 8))
                except BaseException as e:
                    errors.append(e)

            threads = [threading.Thread(target=worker, args=(i,))
                       for i in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            self.assertFalse(errors, errors)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

class TestNormalize(unittest.TestCase):
    def test_normalize_unit_norm(self):
        p = _StubProvider(model="m", normalize=True)
        out = p(["one", "two", "three"])
        norms = np.linalg.norm(out, axis=1)
        np.testing.assert_allclose(norms, np.ones_like(norms), atol=1e-5)

    def test_normalize_handles_zero_vector(self):
        # Provider returning all zeros must not divide-by-zero
        class _ZeroProvider(BaseEmbeddingProvider):
            provider_name = "zero"
            def _embed_batch(self, texts):
                return np.zeros((len(texts), 4), dtype=np.float32)

        p = _ZeroProvider(model="m", normalize=True)
        out = p(["x"])
        self.assertEqual(out.shape, (1, 4))
        self.assertTrue(np.all(np.isfinite(out)))


# ---------------------------------------------------------------------------
# Protocol surfaces (LangChain + LlamaIndex)
# ---------------------------------------------------------------------------

class TestProtocols(unittest.TestCase):
    def test_langchain_embed_query_returns_list(self):
        p = _StubProvider(model="m")
        out = p.embed_query("hi")
        self.assertIsInstance(out, list)
        self.assertEqual(len(out), 8)
        self.assertTrue(all(isinstance(x, float) for x in out))

    def test_langchain_embed_documents_returns_list_of_list(self):
        p = _StubProvider(model="m")
        out = p.embed_documents(["a", "b"])
        self.assertIsInstance(out, list)
        self.assertEqual(len(out), 2)
        self.assertEqual(len(out[0]), 8)

    def test_llamaindex_methods_match_langchain(self):
        p = _StubProvider(model="m")
        self.assertEqual(p._get_query_embedding("q"), p.embed_query("q"))
        self.assertEqual(p._get_text_embedding("t"), p.embed_query("t"))
        self.assertEqual(p._get_text_embeddings(["a", "b"]),
                         p.embed_documents(["a", "b"]))

    def test_async_methods(self):
        p = _StubProvider(model="m")

        async def go():
            q = await p.aembed_query("q")
            d = await p.aembed_documents(["a", "b"])
            qq = await p._aget_query_embedding("q")
            tt = await p._aget_text_embedding("t")
            return q, d, qq, tt

        q, d, qq, tt = asyncio.run(go())
        self.assertEqual(q, p.embed_query("q"))
        self.assertEqual(d, p.embed_documents(["a", "b"]))
        self.assertEqual(qq, p.embed_query("q"))
        self.assertEqual(tt, p.embed_query("t"))


# ---------------------------------------------------------------------------
# Vectro embed_fn contract — drop-in for VectroDSPyRetriever
# ---------------------------------------------------------------------------

class TestEmbedFnContract(unittest.TestCase):
    def test_used_as_dspy_embed_fn(self):
        # Ensure dspy fallback works (no real dspy installed during test)
        from python.integrations import VectroDSPyRetriever
        p = _StubProvider(model="m", batch_size=2)
        rm = VectroDSPyRetriever(embed_fn=p, k=2)
        rm.add_texts(
            ["paris france", "berlin germany", "tokyo japan", "machine learning"]
        )
        out = rm("paris france")
        self.assertEqual(len(out.passages), 2)
        self.assertIn("paris france", out.passages)


if __name__ == "__main__":
    unittest.main()
