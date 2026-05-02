"""Tests for OpenAI / Voyage / Cohere / SentenceTransformers embedding providers.

Each provider is exercised with an in-process stub client (no network calls),
verifying request shape, response decoding, asymmetric query input_type
handling, and cache-key separation between document and query embeddings.
"""
from __future__ import annotations

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

from python.embeddings import (  # noqa: E402
    OpenAIEmbeddings,
    VoyageEmbeddings,
    CohereEmbeddings,
    SentenceTransformersEmbeddings,
)


def _vec(text: str, dim: int = 8, seed: int = 0) -> List[float]:
    """Deterministic per-text vector — sum of (text + seed) char codes per dim."""
    base = sum(ord(c) for c in text) + seed
    return [float(((base + i * 13) % 251) / 251.0) for i in range(dim)]


# ---------------------------------------------------------------------------
# OpenAI stub
# ---------------------------------------------------------------------------

class _OpenAIData:
    def __init__(self, embedding):
        self.embedding = embedding


class _OpenAIResp:
    def __init__(self, data):
        self.data = data


class _OpenAIEmbeddingsApi:
    def __init__(self, dim=8, log=None):
        self._dim = dim
        self.log = log if log is not None else []

    def create(self, *, model, input):
        self.log.append({"model": model, "input": list(input)})
        rows = [_OpenAIData(_vec(t, dim=self._dim)) for t in input]
        return _OpenAIResp(rows)


class _OpenAIClient:
    def __init__(self, dim=8):
        self.log: list = []
        self.embeddings = _OpenAIEmbeddingsApi(dim=dim, log=self.log)


class TestOpenAIEmbeddings(unittest.TestCase):
    def test_basic_embedding(self):
        client = _OpenAIClient(dim=8)
        p = OpenAIEmbeddings(model="text-embedding-3-small", client=client,
                             batch_size=2)
        out = p(["a", "b", "c"])
        self.assertEqual(out.shape, (3, 8))
        self.assertEqual([len(c["input"]) for c in client.log], [2, 1])
        self.assertEqual(client.log[0]["model"], "text-embedding-3-small")

    def test_dict_response_shape_supported(self):
        # Some openai-compatible servers return dicts, not objects.
        class _DictApi:
            def create(self, *, model, input):
                return {"data": [{"embedding": _vec(t, 4)} for t in input]}

        class _DictClient:
            embeddings = _DictApi()

        p = OpenAIEmbeddings(model="any", client=_DictClient(), batch_size=2)
        out = p(["x", "y"])
        self.assertEqual(out.shape, (2, 4))

    def test_cache_hits_skip_api(self):
        with tempfile.TemporaryDirectory() as td:
            client = _OpenAIClient(dim=4)
            p = OpenAIEmbeddings(model="m", client=client, batch_size=10,
                                 cache_dir=td)
            p(["a", "b"])
            calls_after_first = len(client.log)
            p(["a", "b"])
            self.assertEqual(len(client.log), calls_after_first)

    def test_missing_sdk_raises(self):
        # No client passed and openai not importable → ImportError.
        original = sys.modules.pop("openai", None)
        sys.modules["openai"] = None  # type: ignore — force ImportError
        try:
            p = OpenAIEmbeddings(model="m", client=None)
            with self.assertRaises(ImportError):
                p._ensure_client()
        finally:
            sys.modules.pop("openai", None)
            if original is not None:
                sys.modules["openai"] = original


# ---------------------------------------------------------------------------
# Voyage stub
# ---------------------------------------------------------------------------

class _VoyageResp:
    def __init__(self, embeddings):
        self.embeddings = embeddings


class _VoyageClient:
    def __init__(self, dim=8):
        self._dim = dim
        self.log: list = []

    def embed(self, texts, *, model, input_type):
        self.log.append({"texts": list(texts), "model": model,
                         "input_type": input_type})
        return _VoyageResp([_vec(t, dim=self._dim) for t in texts])


class TestVoyageEmbeddings(unittest.TestCase):
    def test_default_input_type_is_document(self):
        client = _VoyageClient(dim=4)
        p = VoyageEmbeddings(model="voyage-3", client=client, batch_size=2)
        p(["a", "b"])
        self.assertEqual(client.log[0]["input_type"], "document")

    def test_embed_query_uses_query_input_type(self):
        client = _VoyageClient(dim=4)
        p = VoyageEmbeddings(model="voyage-3", client=client, batch_size=2)
        # First seed corpus
        p(["doc"])
        client.log.clear()
        p.embed_query("question")
        self.assertEqual(client.log[0]["input_type"], "query")

    def test_query_and_document_embeddings_use_separate_cache(self):
        with tempfile.TemporaryDirectory() as td:
            client = _VoyageClient(dim=4)
            p = VoyageEmbeddings(model="voyage-3", client=client,
                                 batch_size=4, cache_dir=td)
            p(["text"])  # document path
            p.embed_query("text")  # query path — must miss
            self.assertEqual(len(client.log), 2)
            self.assertEqual(client.log[0]["input_type"], "document")
            self.assertEqual(client.log[1]["input_type"], "query")
            # Repeat — both should hit cache now
            p(["text"])
            p.embed_query("text")
            self.assertEqual(len(client.log), 2)

    def test_dict_response_shape(self):
        class _DictClient:
            def embed(self, texts, *, model, input_type):
                return {"embeddings": [_vec(t, 4) for t in texts]}
        p = VoyageEmbeddings(model="m", client=_DictClient(), batch_size=4)
        out = p(["a", "b"])
        self.assertEqual(out.shape, (2, 4))


# ---------------------------------------------------------------------------
# Cohere stub
# ---------------------------------------------------------------------------

class _CohereResp:
    def __init__(self, embeddings):
        self.embeddings = embeddings


class _CohereClient:
    def __init__(self, dim=8):
        self._dim = dim
        self.log: list = []

    def embed(self, *, texts, model, input_type):
        self.log.append({"texts": list(texts), "model": model,
                         "input_type": input_type})
        return _CohereResp([_vec(t, dim=self._dim) for t in texts])


class _CohereV2Resp:
    """Mimics the embed-by-type response shape with .embeddings.float_."""
    class _ByType:
        def __init__(self, rows):
            self.float_ = rows

    def __init__(self, rows):
        self.embeddings = self._ByType(rows)


class _CohereV2Client:
    def __init__(self, dim=8):
        self._dim = dim
        self.log: list = []

    def embed(self, *, texts, model, input_type):
        self.log.append({"texts": list(texts), "model": model,
                         "input_type": input_type})
        return _CohereV2Resp([_vec(t, dim=self._dim) for t in texts])


class TestCohereEmbeddings(unittest.TestCase):
    def test_default_input_type_is_search_document(self):
        client = _CohereClient(dim=4)
        p = CohereEmbeddings(model="embed-english-v3.0", client=client,
                             batch_size=2)
        p(["a", "b"])
        self.assertEqual(client.log[0]["input_type"], "search_document")

    def test_embed_query_uses_search_query_input_type(self):
        client = _CohereClient(dim=4)
        p = CohereEmbeddings(model="m", client=client, batch_size=2)
        p(["doc"])
        client.log.clear()
        p.embed_query("question")
        self.assertEqual(client.log[0]["input_type"], "search_query")

    def test_v2_by_type_response_unwrapped(self):
        client = _CohereV2Client(dim=4)
        p = CohereEmbeddings(model="m", client=client, batch_size=4)
        out = p(["a", "b"])
        self.assertEqual(out.shape, (2, 4))

    def test_query_document_separate_cache(self):
        with tempfile.TemporaryDirectory() as td:
            client = _CohereClient(dim=4)
            p = CohereEmbeddings(model="m", client=client, batch_size=4,
                                 cache_dir=td)
            p(["text"])
            p.embed_query("text")
            self.assertEqual(len(client.log), 2)
            # Cache hits
            p(["text"])
            p.embed_query("text")
            self.assertEqual(len(client.log), 2)


# ---------------------------------------------------------------------------
# SentenceTransformers stub
# ---------------------------------------------------------------------------

class _StModel:
    def __init__(self, dim=8):
        self._dim = dim
        self.log: list = []

    def encode(self, texts, batch_size, convert_to_numpy,
               normalize_embeddings, show_progress_bar):
        self.log.append({"n": len(texts), "batch_size": batch_size,
                         "convert_to_numpy": convert_to_numpy,
                         "normalize_embeddings": normalize_embeddings,
                         "show_progress_bar": show_progress_bar})
        rows = [_vec(t, dim=self._dim) for t in texts]
        return np.asarray(rows, dtype=np.float32)


class TestSentenceTransformersEmbeddings(unittest.TestCase):
    def test_basic_encode(self):
        m = _StModel(dim=4)
        p = SentenceTransformersEmbeddings(model="x", model_obj=m,
                                           batch_size=2)
        out = p(["a", "b", "c"])
        self.assertEqual(out.shape, (3, 4))
        # Two encode calls (batch_size=2 → 2+1)
        self.assertEqual([c["n"] for c in m.log], [2, 1])
        self.assertFalse(m.log[0]["normalize_embeddings"])
        self.assertFalse(m.log[0]["show_progress_bar"])

    def test_normalize_applied_post_encode(self):
        m = _StModel(dim=4)
        p = SentenceTransformersEmbeddings(model="x", model_obj=m,
                                           normalize=True, batch_size=4)
        out = p(["a", "b"])
        norms = np.linalg.norm(out, axis=1)
        np.testing.assert_allclose(norms, np.ones_like(norms), atol=1e-5)

    def test_cache_hits_skip_encode(self):
        with tempfile.TemporaryDirectory() as td:
            m = _StModel(dim=4)
            p = SentenceTransformersEmbeddings(model="x", model_obj=m,
                                               batch_size=4, cache_dir=td)
            p(["a", "b"])
            n_calls = len(m.log)
            p(["a", "b"])
            self.assertEqual(len(m.log), n_calls)


# ---------------------------------------------------------------------------
# Top-level export + integration sanity
# ---------------------------------------------------------------------------

class TestExportsAndIntegration(unittest.TestCase):
    def test_top_level_exports(self):
        import python as vp
        for name in ("BaseEmbeddingProvider", "OpenAIEmbeddings",
                     "VoyageEmbeddings", "CohereEmbeddings",
                     "SentenceTransformersEmbeddings"):
            self.assertTrue(hasattr(vp, name), f"missing top-level export: {name}")
            self.assertIn(name, vp.__all__)

    def test_provider_works_with_dspy_retriever(self):
        # Install dspy stub if needed
        if "dspy" not in sys.modules:
            mod = types.ModuleType("dspy")
            class _Pred:
                def __init__(self, **kw):
                    self.__dict__.update(kw)
            mod.Prediction = _Pred
            sys.modules["dspy"] = mod

        from python.integrations import VectroDSPyRetriever
        client = _OpenAIClient(dim=8)
        p = OpenAIEmbeddings(model="text-embedding-3-small", client=client,
                             batch_size=8)
        rm = VectroDSPyRetriever(embed_fn=p, k=1)
        rm.add_texts(["paris france", "tokyo japan"])
        out = rm("paris france")
        self.assertEqual(len(out.passages), 1)


if __name__ == "__main__":
    unittest.main()
