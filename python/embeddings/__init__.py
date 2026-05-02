"""Embedding-provider bridges — drop-in callable embedders for Vectro.

Each provider:

- Returns ``np.ndarray`` from ``__call__(str | List[str])`` so it can be used
  as a Vectro ``embed_fn`` (e.g. for :class:`VectroDSPyRetriever`).
- Implements the LangChain ``Embeddings`` protocol (``embed_query``,
  ``embed_documents``, plus async variants) for use with
  :class:`LangChainVectorStore`.
- Implements the LlamaIndex ``BaseEmbedding`` private protocol
  (``_get_query_embedding``, ``_get_text_embedding``,
  ``_get_text_embeddings``) for use with :class:`LlamaIndexVectorStore`.
- Auto-batches large inputs to fit each provider's request size.
- Persists embeddings to a SQLite-backed on-disk cache, keyed by
  ``provider:model:text``, when ``cache_dir`` is set.
"""

from .base import BaseEmbeddingProvider, TextLike
from .openai import OpenAIEmbeddings
from .voyage import VoyageEmbeddings
from .cohere import CohereEmbeddings
from .sentence_transformers import SentenceTransformersEmbeddings

__all__ = [
    "BaseEmbeddingProvider",
    "TextLike",
    "OpenAIEmbeddings",
    "VoyageEmbeddings",
    "CohereEmbeddings",
    "SentenceTransformersEmbeddings",
]
