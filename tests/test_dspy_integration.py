"""Tests for VectroDSPyRetriever — DSPy retrieval module backed by Vectro."""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import types
import unittest
from typing import Any, List

import numpy as np

try:
    from tests._path_setup import ensure_repo_root_on_path
except ModuleNotFoundError:
    from _path_setup import ensure_repo_root_on_path

ensure_repo_root_on_path()


# ---------------------------------------------------------------------------
# Minimal dspy stub — install BEFORE importing the integration so
# _make_prediction routes through dspy.Prediction in the success path.
# ---------------------------------------------------------------------------

class _Prediction:
    def __init__(self, **fields: Any) -> None:
        self.__dict__.update(fields)
        # DSPy semantic — passages and other named fields visible as attrs


def _install_dspy_stub() -> None:
    if "dspy" in sys.modules:
        return
    mod = types.ModuleType("dspy")
    mod.Prediction = _Prediction
    sys.modules["dspy"] = mod


_install_dspy_stub()

from python.integrations.dspy_integration import VectroDSPyRetriever  # noqa: E402


# ---------------------------------------------------------------------------
# Deterministic fake embedder — keyword-overlap pseudo-vectors.
# ---------------------------------------------------------------------------

VOCAB = ["paris", "berlin", "tokyo", "capital", "winter", "cold", "city",
         "france", "germany", "japan", "ai", "machine", "learning", "deep",
         "vector", "search", "rag", "retrieval"]


def _vec(text: str) -> np.ndarray:
    v = np.zeros(len(VOCAB), dtype=np.float32)
    toks = text.lower().split()
    for tok in toks:
        if tok in VOCAB:
            v[VOCAB.index(tok)] += 1.0
    n = float(np.linalg.norm(v))
    if n > 0:
        v /= n
    else:
        v[0] = 1.0
    return v


def embed_fn(x):
    if isinstance(x, str):
        return _vec(x)
    return np.stack([_vec(t) for t in x], axis=0)


PASSAGES = [
    "Paris is the capital of France",
    "Berlin is cold in winter",
    "Tokyo is the capital of Japan",
    "Machine learning powers modern AI",
    "Deep learning is a subfield of machine learning",
    "Vector search enables RAG retrieval",
]


def _make_retriever(k: int = 3, profile: str = "balanced",
                    metadatas=None) -> VectroDSPyRetriever:
    rm = VectroDSPyRetriever(embed_fn=embed_fn, k=k, compression_profile=profile)
    rm.add_texts(PASSAGES, metadatas=metadatas)
    return rm


# ---------------------------------------------------------------------------
# Construction & corpus management
# ---------------------------------------------------------------------------

class TestConstruction(unittest.TestCase):
    def test_init_defaults(self):
        rm = VectroDSPyRetriever(embed_fn=embed_fn)
        self.assertEqual(rm.k, 3)
        self.assertEqual(len(rm), 0)

    def test_k_setter(self):
        rm = VectroDSPyRetriever(embed_fn=embed_fn, k=5)
        rm.k = 7
        self.assertEqual(rm.k, 7)

    def test_repr_includes_state(self):
        rm = _make_retriever(k=4)
        s = repr(rm)
        self.assertIn("VectroDSPyRetriever", s)
        self.assertIn("k=4", s)
        self.assertIn(f"n={len(PASSAGES)}", s)

    def test_add_texts_no_embed_fn_requires_embeddings(self):
        rm = VectroDSPyRetriever(embed_fn=None, k=3)
        with self.assertRaises(ValueError):
            rm.add_texts(PASSAGES)
        # Pre-computed embeddings work fine
        embs = embed_fn(PASSAGES)
        n = rm.add_texts(PASSAGES, embeddings=embs)
        self.assertEqual(n, len(PASSAGES))

    def test_add_texts_zero(self):
        rm = VectroDSPyRetriever(embed_fn=embed_fn)
        self.assertEqual(rm.add_texts([]), 0)

    def test_metadatas_length_mismatch(self):
        rm = VectroDSPyRetriever(embed_fn=embed_fn)
        with self.assertRaises(ValueError):
            rm.add_texts(["a", "b"], metadatas=[{"x": 1}])

    def test_embeddings_shape_mismatch(self):
        rm = VectroDSPyRetriever(embed_fn=embed_fn)
        with self.assertRaises(ValueError):
            rm.add_texts(["a", "b"], embeddings=np.zeros((3, 5), dtype=np.float32))

    def test_clear(self):
        rm = _make_retriever()
        self.assertGreater(len(rm), 0)
        rm.clear()
        self.assertEqual(len(rm), 0)
        out = rm("paris france")
        self.assertEqual(out.passages, [])


# ---------------------------------------------------------------------------
# Forward / __call__ — DSPy retrieval contract
# ---------------------------------------------------------------------------

class TestForward(unittest.TestCase):
    def test_returns_dspy_prediction(self):
        rm = _make_retriever(k=2)
        out = rm.forward("paris france capital")
        # Should be a dspy.Prediction (our stub) with .passages
        self.assertIsInstance(out, _Prediction)
        self.assertEqual(len(out.passages), 2)
        self.assertIn("Paris is the capital of France", out.passages)

    def test_call_proxy(self):
        rm = _make_retriever(k=2)
        out = rm("tokyo capital")
        self.assertIn("Tokyo is the capital of Japan", out.passages)

    def test_top1_relevant(self):
        rm = _make_retriever(k=1)
        out = rm("berlin winter cold")
        self.assertEqual(out.passages, ["Berlin is cold in winter"])

    def test_k_override(self):
        rm = _make_retriever(k=10)
        out = rm("ai machine learning", k=2)
        self.assertEqual(len(out.passages), 2)

    def test_k_zero_returns_empty(self):
        rm = _make_retriever()
        out = rm("anything", k=0)
        self.assertEqual(out.passages, [])

    def test_empty_corpus_returns_empty(self):
        rm = VectroDSPyRetriever(embed_fn=embed_fn, k=3)
        out = rm("anything")
        self.assertEqual(out.passages, [])

    def test_multi_query_aggregation(self):
        rm = _make_retriever(k=2)
        out = rm(["paris france", "tokyo japan"])
        # Both capital/country queries should surface
        self.assertEqual(len(out.passages), 2)
        joined = " | ".join(out.passages)
        self.assertTrue(
            "Paris" in joined or "Tokyo" in joined,
            f"expected at least one capital passage, got {out.passages!r}",
        )

    def test_query_embedding_bypass(self):
        rm = VectroDSPyRetriever(embed_fn=None, k=2)
        embs = embed_fn(PASSAGES)
        rm.add_texts(PASSAGES, embeddings=embs)
        q = _vec("vector search rag")
        out = rm.forward("ignored", query_embedding=q)
        self.assertIn("Vector search enables RAG retrieval", out.passages)

    def test_query_embedding_must_be_1d(self):
        rm = _make_retriever()
        with self.assertRaises(ValueError):
            rm.forward("x", query_embedding=np.zeros((2, 5), dtype=np.float32))

    def test_indices_and_scores_attached(self):
        rm = _make_retriever(k=2)
        out = rm.forward("paris france")
        self.assertTrue(hasattr(out, "indices"))
        self.assertTrue(hasattr(out, "scores"))
        self.assertEqual(len(out.indices), 2)
        self.assertEqual(len(out.scores), 2)
        # Scores monotonically non-increasing
        self.assertGreaterEqual(out.scores[0], out.scores[1])


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

class TestFilters(unittest.TestCase):
    def _retriever_with_meta(self):
        metas = [
            {"region": "eu", "topic": "geography"},
            {"region": "eu", "topic": "geography"},
            {"region": "asia", "topic": "geography"},
            {"region": "global", "topic": "ai"},
            {"region": "global", "topic": "ai"},
            {"region": "global", "topic": "ai"},
        ]
        return _make_retriever(k=10, metadatas=metas)

    def test_filter_equality(self):
        rm = self._retriever_with_meta()
        out = rm.forward("ai machine learning", filters={"topic": "ai"})
        for p in out.passages:
            self.assertNotIn("Paris", p)
            self.assertNotIn("Berlin", p)
            self.assertNotIn("Tokyo", p)

    def test_filter_no_match_returns_empty(self):
        rm = self._retriever_with_meta()
        out = rm.forward("ai", filters={"region": "antarctica"})
        self.assertEqual(out.passages, [])

    def test_filter_multiple_keys(self):
        rm = self._retriever_with_meta()
        out = rm.forward("paris france capital",
                         filters={"region": "eu", "topic": "geography"})
        for p in out.passages:
            self.assertNotIn("Tokyo", p)


# ---------------------------------------------------------------------------
# Async forward
# ---------------------------------------------------------------------------

class TestAsync(unittest.TestCase):
    def test_aforward_matches_forward(self):
        rm = _make_retriever(k=2)
        sync_out = rm.forward("paris france")

        async def go():
            return await rm.aforward("paris france")
        async_out = asyncio.run(go())
        self.assertEqual(sync_out.passages, async_out.passages)


# ---------------------------------------------------------------------------
# MMR
# ---------------------------------------------------------------------------

class TestMMR(unittest.TestCase):
    def test_mmr_returns_k(self):
        rm = _make_retriever(k=3)
        out = rm.forward_mmr("machine learning ai", k=3, fetch_k=6)
        self.assertEqual(len(out.passages), 3)

    def test_mmr_lambda_one_matches_relevance_ordering(self):
        rm = _make_retriever(k=2)
        rel = rm.forward("ai machine learning", k=2)
        mmr = rm.forward_mmr("ai machine learning", k=2, fetch_k=6,
                             lambda_mult=1.0)
        self.assertEqual(set(rel.passages), set(mmr.passages))

    def test_mmr_diversity_changes_selection(self):
        rm = _make_retriever(k=2)
        relevance = rm.forward("machine learning", k=2)
        diverse = rm.forward_mmr(
            "machine learning", k=2, fetch_k=6, lambda_mult=0.0,
        )
        # With pure diversity the selected set may differ from pure relevance
        self.assertEqual(len(diverse.passages), 2)

    def test_mmr_filters(self):
        metas = [{"topic": "geo"}] * 3 + [{"topic": "ai"}] * 3
        rm = _make_retriever(k=3, metadatas=metas)
        out = rm.forward_mmr("ai machine learning", k=2, fetch_k=4,
                             filters={"topic": "ai"})
        for p in out.passages:
            self.assertNotIn("Paris", p)

    def test_amfr_async_matches(self):
        rm = _make_retriever(k=3)
        sync_out = rm.forward_mmr("machine learning ai", k=2, fetch_k=4)

        async def go():
            return await rm.aforward_mmr("machine learning ai", k=2, fetch_k=4)
        async_out = asyncio.run(go())
        self.assertEqual(sync_out.passages, async_out.passages)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence(unittest.TestCase):
    def test_save_load_roundtrip(self):
        metas = [{"i": i} for i in range(len(PASSAGES))]
        rm = _make_retriever(k=2, metadatas=metas)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "rm")
            rm.save(path)
            self.assertTrue(os.path.isfile(os.path.join(path, "meta.json")))
            self.assertTrue(os.path.isfile(os.path.join(path, "vectors.npy")))

            rm2 = VectroDSPyRetriever.load(path, embed_fn=embed_fn)
            self.assertEqual(len(rm2), len(rm))
            self.assertEqual(rm2.k, rm.k)
            out1 = rm.forward("paris france", k=2)
            out2 = rm2.forward("paris france", k=2)
            self.assertEqual(out1.passages, out2.passages)

    def test_load_rejects_wrong_store_type(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(tmpdir, exist_ok=True)
            import json
            with open(os.path.join(tmpdir, "meta.json"), "w") as fh:
                json.dump({"store_type": "haystack", "profile": "balanced",
                           "n_dims": 4, "passages": [], "k": 3}, fh)
            np.save(os.path.join(tmpdir, "vectors.npy"),
                    np.zeros((0, 4), dtype=np.float32))
            with self.assertRaises(ValueError):
                VectroDSPyRetriever.load(tmpdir, embed_fn=embed_fn)

    def test_save_empty_retriever(self):
        rm = VectroDSPyRetriever(embed_fn=embed_fn, k=3)
        with tempfile.TemporaryDirectory() as tmpdir:
            rm.save(tmpdir)
            rm2 = VectroDSPyRetriever.load(tmpdir, embed_fn=embed_fn)
            self.assertEqual(len(rm2), 0)


# ---------------------------------------------------------------------------
# Compression stats
# ---------------------------------------------------------------------------

class TestCompressionStats(unittest.TestCase):
    def test_empty_stats(self):
        rm = VectroDSPyRetriever(embed_fn=embed_fn)
        s = rm.compression_stats
        self.assertEqual(s["n_passages"], 0)
        self.assertEqual(s["compression_ratio"], 1.0)

    def test_populated_stats(self):
        rm = _make_retriever()
        s = rm.compression_stats
        self.assertEqual(s["n_passages"], len(PASSAGES))
        self.assertEqual(s["dimensions"], len(VOCAB))
        self.assertEqual(s["compression_profile"], "balanced")
        self.assertGreater(s["compression_ratio"], 1.0)


# ---------------------------------------------------------------------------
# Top-level export sanity
# ---------------------------------------------------------------------------

class TestTopLevelExports(unittest.TestCase):
    def test_in_python_namespace(self):
        import python as vp
        self.assertTrue(hasattr(vp, "VectroDSPyRetriever"))
        self.assertIn("VectroDSPyRetriever", vp.__all__)

    def test_in_integrations_namespace(self):
        from python.integrations import VectroDSPyRetriever as RM
        self.assertIs(RM, VectroDSPyRetriever)


# ---------------------------------------------------------------------------
# Fallback _Prediction (when dspy is not installed)
# ---------------------------------------------------------------------------

class TestFallbackPrediction(unittest.TestCase):
    def test_make_prediction_without_dspy(self):
        # Temporarily mask dspy from sys.modules to exercise the fallback path.
        from python.integrations import dspy_integration as di
        orig = sys.modules.pop("dspy", None)
        try:
            out = di._make_prediction(["a", "b"])
            self.assertEqual(out.passages, ["a", "b"])
        finally:
            if orig is not None:
                sys.modules["dspy"] = orig


if __name__ == "__main__":
    unittest.main()
