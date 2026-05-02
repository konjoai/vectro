"""Integration adapters for external vector and ML ecosystems."""

from .vector_db import (
    StoredVectorBatch,
    VectorDBConnector,
    InMemoryVectorDBConnector,
)
from .qdrant_connector import QdrantConnector
from .weaviate_connector import WeaviateConnector
from .milvus_connector import MilvusConnector
from .chroma_connector import ChromaConnector
from .pinecone_connector import PineconeConnector
from .torch_bridge import compress_tensor, reconstruct_tensor, HuggingFaceCompressor
from .arrow_bridge import (
    result_to_table,
    table_to_result,
    write_parquet,
    read_parquet,
    to_arrow_bytes,
    from_arrow_bytes,
)
from .langchain_integration import VectroVectorStore as LangChainVectorStore
from .llamaindex_integration import VectroVectorStore as LlamaIndexVectorStore
from .haystack_integration import VectroDocumentStore as HaystackDocumentStore
from .dspy_integration import VectroDSPyRetriever

__all__ = [
    "StoredVectorBatch",
    "VectorDBConnector",
    "InMemoryVectorDBConnector",
    "QdrantConnector",
    "WeaviateConnector",
    "MilvusConnector",
    "ChromaConnector",
    "PineconeConnector",
    "compress_tensor",
    "reconstruct_tensor",
    "HuggingFaceCompressor",
    # Arrow / Parquet bridge
    "result_to_table",
    "table_to_result",
    "write_parquet",
    "read_parquet",
    "to_arrow_bytes",
    "from_arrow_bytes",
    # Framework integrations
    "LangChainVectorStore",
    "LlamaIndexVectorStore",
    "HaystackDocumentStore",
    "VectroDSPyRetriever",
]
