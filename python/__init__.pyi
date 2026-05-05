"""Type stubs for the vectro top-level Python package."""

from .auto_quantize_api import auto_quantize as auto_quantize
from .batch_api import (
    BatchCompressionAnalyzer as BatchCompressionAnalyzer,
    BatchQuantizationResult as BatchQuantizationResult,
    VectroBatchProcessor as VectroBatchProcessor,
    benchmark_batch_compression as benchmark_batch_compression,
    quantize_embeddings_batch as quantize_embeddings_batch,
)
from .bf16_api import Bf16Encoder as Bf16Encoder
from .binary_api import (
    dequantize_binary as dequantize_binary,
    quantize_binary as quantize_binary,
)
from .codebook_api import Codebook as Codebook
from .embeddings import (
    BaseEmbeddingProvider as BaseEmbeddingProvider,
    CohereEmbeddings as CohereEmbeddings,
    OpenAIEmbeddings as OpenAIEmbeddings,
    SentenceTransformersEmbeddings as SentenceTransformersEmbeddings,
    VoyageEmbeddings as VoyageEmbeddings,
)
from .integrations import (
    ChromaConnector as ChromaConnector,
    HuggingFaceCompressor as HuggingFaceCompressor,
    InMemoryVectorDBConnector as InMemoryVectorDBConnector,
    MilvusConnector as MilvusConnector,
    PineconeConnector as PineconeConnector,
    QdrantConnector as QdrantConnector,
    StoredVectorBatch as StoredVectorBatch,
    VectorDBConnector as VectorDBConnector,
    WeaviateConnector as WeaviateConnector,
    compress_tensor as compress_tensor,
    from_arrow_bytes as from_arrow_bytes,
    read_parquet as read_parquet,
    reconstruct_tensor as reconstruct_tensor,
    result_to_table as result_to_table,
    table_to_result as table_to_result,
    to_arrow_bytes as to_arrow_bytes,
    write_parquet as write_parquet,
)
from .interface import (
    QuantizationResult as QuantizationResult,
    get_backend_info as get_backend_info,
    mean_cosine_similarity as mean_cosine_similarity,
    quantize_embeddings as quantize_embeddings,
    reconstruct_embeddings as reconstruct_embeddings,
)
from .ivf_api import IVFIndex as IVFIndex, IVFPQIndex as IVFPQIndex
from .lora_api import (
    LoRAResult as LoRAResult,
    compress_lora as compress_lora,
    compress_lora_adapter as compress_lora_adapter,
    decompress_lora as decompress_lora,
)
from .migration import (
    inspect_artifact as inspect_artifact,
    upgrade_artifact as upgrade_artifact,
    validate_artifact as validate_artifact,
)
from .nf4_api import (
    dequantize_mixed as dequantize_mixed,
    dequantize_nf4 as dequantize_nf4,
    quantize_mixed as quantize_mixed,
    quantize_nf4 as quantize_nf4,
    select_outlier_dims as select_outlier_dims,
)
from .onnx_export import export_onnx as export_onnx, to_onnx_model as to_onnx_model
from .profiles import get_profile as get_profile, QuantProfile as QuantProfile
from .profiles_api import (
    CompressionOptimizer as CompressionOptimizer,
    CompressionProfile as CompressionProfile,
    CompressionStrategy as CompressionStrategy,
    ProfileComparison as ProfileComparison,
    ProfileManager as ProfileManager,
    create_custom_profile as create_custom_profile,
    get_compression_profile as get_compression_profile,
)
from .quality_api import (
    QualityBenchmark as QualityBenchmark,
    QualityMetrics as QualityMetrics,
    QualityReport as QualityReport,
    VectroQualityAnalyzer as VectroQualityAnalyzer,
    evaluate_quantization_quality as evaluate_quantization_quality,
    generate_quality_report as generate_quality_report,
)
from .quantization_extra import (
    dequantize_int2 as dequantize_int2,
    quantize_adaptive as quantize_adaptive,
    quantize_int2 as quantize_int2,
)
from .retrieval import (
    HaystackReranker as HaystackReranker,
    LangChainRRFRetriever as LangChainRRFRetriever,
    LangChainReranker as LangChainReranker,
    RRFRetriever as RRFRetriever,
    VectroReranker as VectroReranker,
    reciprocal_rank_fusion as reciprocal_rank_fusion,
    rrf_top_k as rrf_top_k,
)
from .retriever import (
    RetrievalResult as RetrievalResult,
    RetrieverProtocol as RetrieverProtocol,
    VectroRetriever as VectroRetriever,
)
from .rq_api import ResidualQuantizer as ResidualQuantizer
from .storage_v3 import (
    VQZResult as VQZResult,
    load_compressed as load_compressed,
    load_vqz as load_vqz,
    save_compressed as save_compressed,
    save_vqz as save_vqz,
)
from .streaming import (
    AsyncStreamingDecompressor as AsyncStreamingDecompressor,
    StreamingDecompressor as StreamingDecompressor,
)
from .v3_api import (
    HNSWIndex as HNSWIndex,
    PQCodebook as PQCodebook,
    V3Result as V3Result,
    VectroV3 as VectroV3,
)
from .vectro import (
    QuantizationConfig as QuantizationConfig,
    Vectro as Vectro,
    analyze_compression_quality as analyze_compression_quality,
    compress_vectors as compress_vectors,
    decompress_vectors as decompress_vectors,
    generate_compression_report as generate_compression_report,
    get_version_info as get_version_info,
)

__version__: str
__author__: str
__license__: str
__description__: str

__all__ = [
    # Main API
    "Vectro",
    "QuantizationConfig",
    "compress_vectors",
    "decompress_vectors",
    "analyze_compression_quality",
    "generate_compression_report",
    # Core interface
    "QuantizationResult",
    "quantize_embeddings",
    "reconstruct_embeddings",
    "mean_cosine_similarity",
    "get_backend_info",
    # Batch processing
    "VectroBatchProcessor",
    "BatchQuantizationResult",
    "BatchCompressionAnalyzer",
    "quantize_embeddings_batch",
    "benchmark_batch_compression",
    # Quality analysis
    "VectroQualityAnalyzer",
    "QualityMetrics",
    "QualityBenchmark",
    "QualityReport",
    "evaluate_quantization_quality",
    "generate_quality_report",
    # Compression profiles
    "ProfileManager",
    "CompressionProfile",
    "CompressionStrategy",
    "CompressionOptimizer",
    "ProfileComparison",
    "get_compression_profile",
    "create_custom_profile",
    # Model-family profile registry
    "get_profile",
    "QuantProfile",
    # Integrations
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
    # Embedding-provider bridges
    "BaseEmbeddingProvider",
    "OpenAIEmbeddings",
    "VoyageEmbeddings",
    "CohereEmbeddings",
    "SentenceTransformersEmbeddings",
    # Streaming
    "StreamingDecompressor",
    "AsyncStreamingDecompressor",
    # Advanced quantization
    "quantize_int2",
    "dequantize_int2",
    "quantize_adaptive",
    # Migration tooling
    "inspect_artifact",
    "upgrade_artifact",
    "validate_artifact",
    # v3 unified API
    "PQCodebook",
    "HNSWIndex",
    "V3Result",
    "VectroV3",
    # v3 quantization
    "quantize_nf4",
    "dequantize_nf4",
    "quantize_mixed",
    "dequantize_mixed",
    "select_outlier_dims",
    "quantize_binary",
    "dequantize_binary",
    "ResidualQuantizer",
    "Codebook",
    # LoRA adapter compression
    "compress_lora",
    "decompress_lora",
    "compress_lora_adapter",
    "LoRAResult",
    "auto_quantize",
    # v3 storage
    "save_vqz",
    "load_vqz",
    "save_compressed",
    "load_compressed",
    "VQZResult",
    # ONNX export
    "to_onnx_model",
    "export_onnx",
    # Hybrid retrieval
    "VectroRetriever",
    "RetrieverProtocol",
    "RetrievalResult",
    # RRF hybrid fusion + re-ranking
    "reciprocal_rank_fusion",
    "rrf_top_k",
    "RRFRetriever",
    "LangChainRRFRetriever",
    "VectroReranker",
    "LangChainReranker",
    "HaystackReranker",
    # IVF approximate nearest-neighbour indices
    "IVFIndex",
    "IVFPQIndex",
    # BFloat16 vector store
    "Bf16Encoder",
    # Utilities
    "get_version_info",
]
