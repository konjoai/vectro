"""v3_api.py — Unified v3 high-level API (VECTRO_V3_PLAN.md § 6b).

Exposes every quantization mode added in Phases 2-8 through a single,
ergonomic interface that matches the v3 plan:

    PQCodebook.train(training_vectors, n_subspaces=96)
    HNSWIndex(dim=768, quantization="int8", M=16)
    VectroV3(profile="pq-96", codebook=codebook)

Profiles supported by VectroV3
--------------------------------
"int8"      — symmetric INT8, 4× compression  (Phase 1)
"nf4"       — Normal Float 4-bit, 8× compression  (Phase 2)
"nf4-mixed" — NF4 + FP16 outlier dims, ~7.5× compression  (Phase 2)
"pq-96"     — Product Quantisation, 96 sub-spaces, 32× compression (Phase 3)
"pq-48"     — Product Quantisation, 48 sub-spaces, 16× compression (Phase 3)
"binary"    — 1-bit sign quantisation, 32× compression  (Phase 4)
"rq-3pass"  — Residual Quantisation (3 passes of PQ), ~10× compression (Phase 7)
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .pq_api import PQCodebook as _PQCodebook, train_pq_codebook, pq_encode, pq_decode
from .hnsw_api import HNSWIndex as _HNSWBase
from .nf4_api import (
    quantize_nf4,
    dequantize_nf4,
    quantize_mixed,
    dequantize_mixed,
    select_outlier_dims,
)
from .binary_api import quantize_binary, dequantize_binary
from .rq_api import ResidualQuantizer
from .auto_quantize_api import auto_quantize
from .interface import quantize_embeddings
from .storage_v3 import save_vqz, load_vqz

_SAVE_VERSION = 3


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class V3Result:
    """Holds the output of VectroV3.compress() for any profile.

    Attributes
    ----------
    profile    : profile name, e.g. "int8", "nf4", "pq-96"
    n_vectors  : number of compressed vectors
    dims       : original embedding dimensionality
    data       : dict of numpy arrays whose keys depend on the profile:
        int8       → {"quantized": int8[n,d], "scales": float32[n]}
        nf4        → {"packed": uint8[n,d//2], "scales": float32[n]}
        nf4-mixed  → {"fp16_vals": float16, "nf4_packed": uint8,
                       "nf4_scales": float32, "out_dims": int[k]}
        pq-*       → {"codes": uint8[n,M]}
        binary     → {"packed": uint8[n,ceil(d/8)]}
        rq-3pass   → {"codes": List[uint8[n,M]], "n_passes": 3}
    """

    profile: str
    n_vectors: int
    dims: int
    data: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# PQCodebook wrapper
# ---------------------------------------------------------------------------


class PQCodebook:
    """Product-Quantisation codebook with train / encode / decode / save / load.

    Parameters
    ----------
    n_subspaces : int
        Number of PQ sub-spaces M (e.g. 96 for d=768).  d must be divisible.
    n_centroids : int
        Centroids per sub-space K (default 256 → 1 byte per sub-space).

    Usage
    -----
    >>> cb = PQCodebook.train(training_vectors, n_subspaces=96)
    >>> codes = cb.encode(database_vectors)   # uint8 [n, 96]
    >>> reconstructed = cb.decode(codes)      # float32 [n, 768]
    >>> cb.save("codebook.pqcb.npz")
    >>> cb2 = PQCodebook.load("codebook.pqcb.npz")
    """

    def __init__(self, _internal: _PQCodebook) -> None:
        self._cb = _internal

    # -- factory --

    @classmethod
    def train(
        cls,
        training_data: np.ndarray,
        n_subspaces: int = 96,
        n_centroids: int = 256,
        max_iter: int = 25,
        random_state: int = 0,
    ) -> "PQCodebook":
        """Train a PQ codebook via per-sub-space K-means.

        Parameters
        ----------
        training_data : float32 array shape (n_train, d)
        n_subspaces   : number of sub-spaces M (d must be divisible by M)
        n_centroids   : centroids per sub-space (default 256)
        """
        cb = train_pq_codebook(
            training_data,
            n_subspaces=n_subspaces,
            n_centroids=n_centroids,
            max_iter=max_iter,
            random_state=random_state,
        )
        return cls(cb)

    # -- encode / decode --

    def encode(self, vectors: np.ndarray) -> np.ndarray:
        """PQ-encode vectors. Returns uint8 code array of shape (n, M)."""
        return pq_encode(vectors, self._cb)

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """PQ-decode codes back to approximate float32 vectors (n, d)."""
        return pq_decode(codes, self._cb)

    # -- persistence --

    def save(self, path: str) -> None:
        """Save codebook centroids to a .npz file."""
        np.savez_compressed(
            path,
            centroids=self._cb.centroids,
            n_subspaces=np.array(self._cb.n_subspaces),
            n_centroids=np.array(self._cb.n_centroids),
            sub_dim=np.array(self._cb.sub_dim),
        )

    @classmethod
    def load(cls, path: str) -> "PQCodebook":
        """Load codebook from a .npz file saved with save()."""
        arc = np.load(path, allow_pickle=False)
        cb = _PQCodebook(
            n_subspaces=int(arc["n_subspaces"]),
            n_centroids=int(arc["n_centroids"]),
            sub_dim=int(arc["sub_dim"]),
            centroids=arc["centroids"],
        )
        return cls(cb)

    @property
    def n_subspaces(self) -> int:
        return self._cb.n_subspaces

    @property
    def n_centroids(self) -> int:
        return self._cb.n_centroids

    @property
    def sub_dim(self) -> int:
        return self._cb.sub_dim

    def compression_ratio(self, d: int) -> float:
        """Compression ratio vs FP32: d*4 bytes / (n_subspaces bytes)."""
        return float(d * 4) / float(self._cb.n_subspaces)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"PQCodebook(n_subspaces={self._cb.n_subspaces}, "
            f"n_centroids={self._cb.n_centroids}, "
            f"sub_dim={self._cb.sub_dim})"
        )


# ---------------------------------------------------------------------------
# HNSWIndex wrapper
# ---------------------------------------------------------------------------


class HNSWIndex:
    """HNSW approximate nearest-neighbour index with batch insert / search.

    Parameters
    ----------
    dim           : embedding dimensionality (informational only; not enforced
                    at insert time to allow flexible batched inserts)
    quantization  : "int8" | "cosine"  (currently: only cosine HNSW is used; the
                    quantization tag is stored for future Mojo INT8 backend)
    M             : max bidirectional links per node (typical: 8..32)
    ef_build      : beam width during construction (typical: 100..400)
    space         : "cosine" (default) or "l2"

    Usage
    -----
    >>> idx = HNSWIndex(dim=768, M=16, ef_build=200)
    >>> idx.add_batch(vectors, ids=["doc_0", "doc_1", ...])
    >>> user_ids, distances = idx.search(query, top_k=10, ef=64)
    >>> idx.save("index.hnsw")
    >>> idx2 = HNSWIndex.load("index.hnsw")
    """

    def __init__(
        self,
        dim: int,
        quantization: str = "int8",
        M: int = 16,
        ef_build: int = 200,
        space: str = "cosine",
    ) -> None:
        self._dim = dim
        self._quantization = quantization
        self._index = _HNSWBase(M=M, ef_construction=ef_build, space=space)
        self._user_ids: List[Any] = []

    # -- population --

    def add_batch(
        self,
        vectors: np.ndarray,
        ids: Optional[Sequence] = None,
    ) -> None:
        """Add one or more vectors to the index.

        Parameters
        ----------
        vectors : float32 array shape (n, d) or (d,)
        ids     : optional user IDs (any hashable); defaults to sequential ints
        """
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors[np.newaxis, :]
        n = vectors.shape[0]

        if ids is None:
            base = len(self._user_ids)
            ids = list(range(base, base + n))
        else:
            ids = list(ids)
            if len(ids) != n:
                raise ValueError(f"ids length ({len(ids)}) must match vectors length ({n}).")

        self._user_ids.extend(ids)
        self._index.add(vectors)

    # -- search --

    def search(
        self,
        query: np.ndarray,
        top_k: int = 10,
        ef: int = 64,
    ) -> Tuple[List[Any], np.ndarray]:
        """Find approximate nearest neighbours of query.

        Parameters
        ----------
        query  : float32 array shape (d,)
        top_k  : number of neighbours to return
        ef     : search beam width (≥ top_k)

        Returns
        -------
        (user_ids, distances) — user_ids is a list of user-supplied IDs;
        distances is a float32 array of corresponding cosine distances.
        """
        indices, distances = self._index.search(query, k=top_k, ef=max(ef, top_k))
        result_ids = [self._user_ids[i] for i in indices if i < len(self._user_ids)]
        return result_ids, distances[: len(result_ids)]

    # -- persistence --

    def save(self, path: str) -> None:
        """Save index (HNSW graph + user IDs) to path via pickle."""
        import pickle

        with open(path, "wb") as fh:
            pickle.dump(
                {
                    "dim": self._dim,
                    "quantization": self._quantization,
                    "user_ids": self._user_ids,
                    # serialise the inner graph via its own save/load
                    "graph": {
                        "M": self._index.M,
                        "ef_construction": self._index.ef_construction,
                        "space": self._index.space,
                        "vectors": self._index._vectors,
                        "neighbors": self._index._neighbors,
                        "levels": self._index._levels,
                        "entry_point": self._index._entry_point,
                        "max_level": self._index._max_level,
                    },
                },
                fh,
            )

    @classmethod
    def load(cls, path: str) -> "HNSWIndex":
        """Load index from a file saved with save()."""
        import pickle

        with open(path, "rb") as fh:
            state = pickle.load(fh)

        g = state["graph"]
        inner = _HNSWBase(M=g["M"], ef_construction=g["ef_construction"], space=g["space"])
        inner._vectors = g["vectors"]
        inner._neighbors = g["neighbors"]
        inner._levels = g["levels"]
        inner._entry_point = g["entry_point"]
        inner._max_level = g["max_level"]

        obj = cls.__new__(cls)
        obj._dim = state["dim"]
        obj._quantization = state["quantization"]
        obj._index = inner
        obj._user_ids = state["user_ids"]
        return obj

    def __len__(self) -> int:
        return len(self._user_ids)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"HNSWIndex(dim={self._dim}, n={len(self)}, "
            f"M={self._index.M}, space={self._index.space!r})"
        )


# ---------------------------------------------------------------------------
# VectroV3 — unified v3 compressor
# ---------------------------------------------------------------------------

_VALID_PROFILES = frozenset(
    {
        "int8",
        "nf4",
        "nf4-mixed",
        "pq-96",
        "pq-48",
        "binary",
        "rq-3pass",
        "lora-nf4",
        "lora-int8",
        "lora-rq",
    }
)


class VectroV3:
    """Unified v3 compressor: all quantization modes behind one API.

    Parameters
    ----------
    profile    : compression profile (see _VALID_PROFILES)
    codebook   : PQCodebook instance — required for "pq-*" profiles
    rq         : pre-trained ResidualQuantizer — used for "rq-3pass";
                 alternatively call train_rq() after construction

    Usage
    -----
    # INT8 (default)
    >>> v = VectroV3()
    >>> result = v.compress(vectors)
    >>> recon  = v.decompress(result)

    # NF4
    >>> v = VectroV3(profile="nf4")

    # Product Quantisation
    >>> codebook = PQCodebook.train(training_data, n_subspaces=96)
    >>> v = VectroV3(profile="pq-96", codebook=codebook)

    # Binary
    >>> v = VectroV3(profile="binary")

    # Residual Quantisation
    >>> v = VectroV3(profile="rq-3pass")
    >>> v.train_rq(training_data)  # or pass rq= to __init__

    # Auto-select
    >>> result = VectroV3.auto_compress(vectors, target_cosine=0.97, target_compression=8.0)
    """

    def __init__(
        self,
        profile: str = "int8",
        codebook: Optional[PQCodebook] = None,
        rq: Optional[ResidualQuantizer] = None,
    ) -> None:
        if profile not in _VALID_PROFILES:
            raise ValueError(
                f"Unknown profile {profile!r}. Valid profiles: {sorted(_VALID_PROFILES)}"
            )
        self._profile = profile
        self._codebook = codebook
        self._rq = rq

    # -- training helpers --

    def train_rq(
        self,
        training_data: np.ndarray,
        n_subspaces: int = 96,
        n_passes: int = 3,
    ) -> None:
        """Train the ResidualQuantizer for 'rq-3pass' profile.

        Parameters
        ----------
        training_data : float32 array (n_train, d)
        n_subspaces   : PQ sub-spaces per pass
        n_passes      : number of residual passes (stored in rq_passes attribute)
        """
        self._rq = ResidualQuantizer(n_subspaces=n_subspaces, n_passes=n_passes)
        self._rq.train(training_data)

    # -- compress --

    def compress(self, vectors: np.ndarray) -> V3Result:
        """Compress vectors using the configured profile.

        Parameters
        ----------
        vectors : float32 array (n, d) or (d,)

        Returns
        -------
        V3Result carrying compressed data + profile metadata.
        """
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors[np.newaxis, :]
        n, d = vectors.shape
        profile = self._profile

        if profile == "int8":
            qr = quantize_embeddings(vectors)
            return V3Result(
                profile=profile,
                n_vectors=n,
                dims=d,
                data={"quantized": qr.quantized, "scales": qr.scales},
            )

        if profile == "nf4":
            packed, scales = quantize_nf4(vectors)
            return V3Result(
                profile=profile,
                n_vectors=n,
                dims=d,
                data={"packed": packed, "scales": scales},
            )

        if profile == "nf4-mixed":
            out_dims = select_outlier_dims(vectors)
            fp16_vals, nf4_packed, nf4_scales, out_dims = quantize_mixed(vectors, out_dims)
            return V3Result(
                profile=profile,
                n_vectors=n,
                dims=d,
                data={
                    "fp16_vals": fp16_vals,
                    "nf4_packed": nf4_packed,
                    "nf4_scales": nf4_scales,
                    "out_dims": out_dims,
                },
            )

        if profile.startswith("pq-"):
            if self._codebook is None:
                raise ValueError(
                    f"Profile '{profile}' requires a trained PQCodebook. "
                    "Pass codebook=PQCodebook.train(training_data) to VectroV3()."
                )
            codes = self._codebook.encode(vectors)
            return V3Result(
                profile=profile,
                n_vectors=n,
                dims=d,
                data={"codes": codes},
            )

        if profile == "binary":
            packed = quantize_binary(vectors)
            return V3Result(
                profile=profile,
                n_vectors=n,
                dims=d,
                data={"packed": packed},
            )

        if profile == "rq-3pass":
            if self._rq is None:
                raise ValueError(
                    "Profile 'rq-3pass' requires a trained ResidualQuantizer. "
                    "Call v.train_rq(training_data) first."
                )
            codes_list = self._rq.encode(vectors)
            return V3Result(
                profile=profile,
                n_vectors=n,
                dims=d,
                data={"codes": codes_list, "n_passes": len(codes_list)},
            )

        if profile.startswith("lora-"):
            raise ValueError(
                f"Profile {profile!r} requires (A, B) matrix inputs. "
                "Use compress_lora(A, B, profile=...) directly, or pass "
                "vectors as a dict {'A': np.ndarray, 'B': np.ndarray}."
            )

        raise ValueError(f"Unhandled profile: {profile}")  # pragma: no cover

    # -- decompress --

    def decompress(self, result: V3Result) -> np.ndarray:
        """Reconstruct float32 vectors from a V3Result.

        Parameters
        ----------
        result : V3Result from compress()

        Returns
        -------
        float32 array of shape (n, d)
        """
        profile = result.profile
        data = result.data
        d = result.dims

        if profile == "int8":
            from .interface import QuantizationResult, reconstruct_embeddings

            qr = QuantizationResult(
                quantized=data["quantized"],
                scales=data["scales"],
                dims=d,
                n=result.n_vectors,
                precision_mode="int8",
                group_size=64,
            )
            return reconstruct_embeddings(qr)

        if profile == "nf4":
            return dequantize_nf4(data["packed"], data["scales"], d)

        if profile == "nf4-mixed":
            return dequantize_mixed(
                data["fp16_vals"],
                data["nf4_packed"],
                data["nf4_scales"],
                data["out_dims"],
                d,
            )

        if profile.startswith("pq-"):
            if self._codebook is None:
                raise ValueError(
                    "Cannot decompress PQ result without the codebook. "
                    "Attach codebook to VectroV3 via the 'codebook' parameter."
                )
            return self._codebook.decode(data["codes"])

        if profile == "binary":
            return dequantize_binary(data["packed"], d)

        if profile == "rq-3pass":
            if self._rq is None:
                raise ValueError(
                    "Cannot decompress RQ result without the trained ResidualQuantizer."
                )
            return self._rq.decode(data["codes"])

        if profile.startswith("lora-"):
            raise ValueError(
                f"Profile {profile!r} is a LoRA result. "
                "Use decompress_lora(result) instead of VectroV3.decompress()."
            )

        raise ValueError(f"Unhandled profile: {profile}")  # pragma: no cover

    # -- auto compress --

    @classmethod
    def auto_compress(
        cls,
        vectors: np.ndarray,
        target_cosine: float = 0.97,
        target_compression: float = 8.0,
    ) -> dict:
        """Automatically select and apply the best quantization scheme.

        Delegates to auto_quantize_api.auto_quantize, which uses kurtosis-based
        routing to choose among NF4, NF4-mixed, PQ, binary, and INT8 fallback.

        Parameters
        ----------
        vectors            : float32 array (n, d)
        target_cosine      : minimum acceptable mean cosine similarity
        target_compression : minimum acceptable compression ratio vs FP32

        Returns
        -------
        dict from auto_quantize_api.auto_quantize — keys include:
            "profile", "cosine_sim", "compression_ratio", "quantized", "scales",
            "success", "strategy"
        """
        return auto_quantize(
            vectors,
            target_cosine=target_cosine,
            target_compression=target_compression,
        )

    # -- save / load compressed --

    def save_compressed(
        self,
        result: V3Result,
        path: str,
        compression: str = "zstd",
        level: int = 3,
    ) -> None:
        """Save a V3Result to a local file or cloud URI.

        For INT8 and NF4 profiles, writes a VQZ container.
        For all other profiles, writes a numpy npz archive.

        Cloud URIs (s3://, gs://, abfs://) are supported if the profile is
        INT8/NF4 — other profiles will raise NotImplementedError for cloud paths.

        Parameters
        ----------
        result      : V3Result from compress()
        path        : local file path or cloud URI
        compression : "zstd" | "zlib" | "none" (VQZ modes, for INT8/NF4 only)
        level       : compression level
        """
        is_cloud = _is_cloud_uri(path)

        if is_cloud:
            self._cloud_save(result, path, compression=compression, level=level)
            return

        profile = result.profile
        meta = json.dumps(
            {
                "profile": profile,
                "dims": result.dims,
                "n_vectors": result.n_vectors,
                "vectro_save_version": _SAVE_VERSION,
            }
        ).encode()

        if profile == "int8":
            save_vqz(
                result.data["quantized"],
                result.data["scales"],
                result.dims,
                path,
                compression=compression,
                metadata=meta,
                level=level,
            )

        elif profile == "nf4":
            # Store packed uint8 re-interpreted as int8
            # NF4 packs 2 values/byte, so packed shape is (n, dims//2)
            q_int8 = result.data["packed"].view(np.int8)
            packed_dims = q_int8.shape[1]  # dims//2, not real dims
            save_vqz(
                q_int8,
                result.data["scales"],
                packed_dims,
                path,
                compression=compression,
                metadata=meta,
                level=level,
            )

        else:
            # Profiles PQ / binary / rq-3pass / nf4-mixed — use npz
            arrs = {k: np.asarray(v) for k, v in result.data.items() if not isinstance(v, list)}
            # RQ codes_list is a Python list of arrays — expand per pass
            if "codes" in result.data and isinstance(result.data["codes"], list):
                for i, c in enumerate(result.data["codes"]):
                    arrs[f"codes_{i}"] = c
            meta_path = path + ".meta"
            with open(meta_path, "wb") as fh:
                fh.write(meta)
            np.savez_compressed(path, **arrs)

    def load_compressed(self, path: str) -> V3Result:
        """Load a V3Result from a local file or cloud URI.

        Parameters
        ----------
        path : local file path or cloud URI saved with save_compressed()

        Returns
        -------
        V3Result with the same profile and data as the original.
        """
        if _is_cloud_uri(path):
            return self._cloud_load(path)

        # Detect format: VQZ if starts with VECTRO magic, else npz
        if _is_vqz(path):
            raw = load_vqz(path)
            meta = json.loads(raw["metadata"])
            profile = meta["profile"]
            dims = meta["dims"]
            n = meta["n_vectors"]

            if profile == "int8":
                return V3Result(
                    profile=profile,
                    n_vectors=n,
                    dims=dims,
                    data={
                        "quantized": raw["quantized"],
                        "scales": raw["scales"],
                    },
                )
            if profile == "nf4":
                return V3Result(
                    profile=profile,
                    n_vectors=n,
                    dims=dims,
                    data={
                        "packed": raw["quantized"].view(np.uint8),
                        "scales": raw["scales"],
                    },
                )
            raise ValueError(f"Unexpected VQZ profile {profile!r} — cannot load.")

        # npz format (pq / binary / rq-3pass / nf4-mixed)
        meta_path = path + ".meta"
        if os.path.exists(meta_path):
            with open(meta_path, "rb") as fh:
                meta = json.loads(fh.read())
        else:
            # Attempt to read meta from a "meta" key inside the archive
            meta = {}

        arc = np.load(path, allow_pickle=False)
        if not meta:
            raise ValueError(f"Cannot load {path!r}: missing .meta sidecar file.")

        profile = meta["profile"]
        dims = meta["dims"]
        n = meta["n_vectors"]

        data: Dict[str, Any] = {}
        for key in arc.files:
            data[key] = arc[key]

        # Reconstruct RQ codes_list
        if profile == "rq-3pass":
            codes_list = []
            i = 0
            while f"codes_{i}" in data:
                codes_list.append(data.pop(f"codes_{i}"))
                i += 1
            data["codes"] = codes_list
            data["n_passes"] = len(codes_list)

        return V3Result(profile=profile, n_vectors=n, dims=dims, data=data)

    # -- cloud helpers (thin wrappers around storage_v3 backends) --

    def _cloud_save(
        self,
        result: V3Result,
        uri: str,
        compression: str = "zstd",
        level: int = 3,
    ) -> None:
        """Save V3Result to a cloud URI (INT8/NF4 only)."""
        from .storage_v3 import S3Backend, GCSBackend, AzureBlobBackend

        if result.profile not in ("int8", "nf4"):
            raise NotImplementedError(
                f"cloud save is only supported for 'int8' and 'nf4' profiles "
                f"(got {result.profile!r}). Save locally first for other profiles."
            )
        backend = _make_cloud_backend(uri, S3Backend, GCSBackend, AzureBlobBackend)

        q = (
            result.data["quantized"]
            if result.profile == "int8"
            else result.data["packed"].view(np.int8)
        )
        meta = json.dumps(
            {"profile": result.profile, "dims": result.dims, "n_vectors": result.n_vectors}
        ).encode()

        with tempfile.NamedTemporaryFile(suffix=".vqz", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            save_vqz(
                q,
                result.data.get("scales", np.zeros(result.n_vectors, dtype=np.float32)),
                result.dims,
                tmp_path,
                compression=compression,
                metadata=meta,
                level=level,
            )
            backend.upload(tmp_path, _uri_key(uri))
        finally:
            os.unlink(tmp_path)

    def _cloud_load(self, uri: str) -> V3Result:
        """Load V3Result from a cloud URI."""
        from .storage_v3 import S3Backend, GCSBackend, AzureBlobBackend

        backend = _make_cloud_backend(uri, S3Backend, GCSBackend, AzureBlobBackend)
        with tempfile.NamedTemporaryFile(suffix=".vqz", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            backend.download(_uri_key(uri), tmp_path)
            return self.load_compressed(tmp_path)
        finally:
            os.unlink(tmp_path)

    # -- convenience --

    @property
    def profile(self) -> str:
        return self._profile

    def __repr__(self) -> str:  # pragma: no cover
        return f"VectroV3(profile={self._profile!r})"


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------


def auto_compress(
    vectors: np.ndarray,
    target_cosine: float = 0.97,
    target_compression: float = 8.0,
) -> dict:
    """Automatically select and apply the best quantization scheme.

    A module-level wrapper around auto_quantize_api.auto_quantize that delegates
    to VectroV3.auto_compress under the hood. Provides an ergonomic top-level API
    for users who want automatic strategy selection without instantiating VectroV3.

    Parameters
    ----------
    vectors            : float32 array (n, d)
    target_cosine      : minimum acceptable mean cosine similarity (default 0.97)
    target_compression : minimum acceptable compression ratio vs FP32 (default 8.0)

    Returns
    -------
    dict from auto_quantize_api.auto_quantize — keys include:
        "profile", "cosine_sim", "compression_ratio", "quantized", "scales",
        "success", "strategy"

    Examples
    --------
    >>> from python.v3_api import auto_compress
    >>> result = auto_compress(vectors, target_cosine=0.97, target_compression=8.0)
    >>> print(f"Selected profile: {result['profile']}")
    >>> print(f"Achieved cosine similarity: {result['cosine_sim']:.5f}")
    """
    return auto_quantize(
        vectors,
        target_cosine=target_cosine,
        target_compression=target_compression,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_cloud_uri(path: str) -> bool:
    return path.startswith(("s3://", "gs://", "gcs://", "abfs://", "az://"))


def _is_vqz(path: str) -> bool:
    """Return True if the file starts with the VQZ magic bytes."""
    try:
        with open(path, "rb") as fh:
            header = fh.read(8)
        return header == b"VECTRO\x03\x00"
    except (OSError, IOError):
        return False


def _uri_key(uri: str) -> str:
    """Extract the object key from a cloud URI, ignoring the bucket prefix."""
    # e.g. s3://my-bucket/prefix/file.vqz  →  prefix/file.vqz
    parts = uri.split("/", 3)
    return parts[3] if len(parts) > 3 else parts[-1]


def _make_cloud_backend(uri: str, S3Backend, GCSBackend, AzureBlobBackend):
    """Instantiate the correct cloud backend for a given URI scheme."""
    if uri.startswith("s3://"):
        bucket = uri[5:].split("/")[0]
        prefix = "/".join(uri[5:].split("/")[1:-1])
        return S3Backend(bucket, prefix)
    if uri.startswith(("gs://", "gcs://")):
        scheme = "gcs://" if uri.startswith("gcs://") else "gs://"
        rest = uri[len(scheme) :]
        bucket = rest.split("/")[0]
        prefix = "/".join(rest.split("/")[1:-1])
        return GCSBackend(bucket, prefix)
    if uri.startswith(("abfs://", "az://")):
        scheme = "abfs://" if uri.startswith("abfs://") else "az://"
        rest = uri[len(scheme) :]
        bucket = rest.split("/")[0]
        prefix = "/".join(rest.split("/")[1:-1])
        return AzureBlobBackend(bucket, prefix)
    raise ValueError(f"Unsupported cloud URI scheme: {uri!r}")
