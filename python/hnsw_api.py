"""HNSW (Hierarchical Navigable Small World) graph index for Vectro v3, Phase 5.

v5.1.0 additions
----------------
* Per-vector metadata sidecar (``add(vectors, metadata=...)``)
* O(1) soft-delete via tombstone set (``delete(node_id)``)
* Pre-filter during graph walk (``search(..., filter=...)``)
* Graph health inspection (``stats()``)
* Tombstone removal + orphan reconnection (``compact()``)
* Brute-force recall estimator with Wilson 95% CI (``estimate_recall()``)


Implements the algorithm from Malkov & Yashunin 2018 (arXiv:1603.09320).

Internal storage uses INT8 quantised vectors with per-vector abs-max scales,
giving a 4× memory reduction over FP32 whilst preserving cosine-similarity
ranking for nearest-neighbour queries.

Distance metric: cosine distance  (1 - cosine_similarity).  All stored vectors
are pre-normalised to unit length so the inner product equals cosine similarity
directly.

Public API
----------
HNSWIndex(M, ef_construction, space)
    .add(vector | vectors)     -> None
    .search(query, k, ef)      -> (indices, distances)
    .save(path)                -> None

HNSWIndex.load(path)           -> HNSWIndex

Convenience helpers
-------------------
build_hnsw_index(vectors, M, ef_construction, space)  -> HNSWIndex
hnsw_search(index, query, k, ef)                      -> (indices, distances)
recall_at_k(index, queries, ground_truth, k, ef)      -> float
"""

from __future__ import annotations

import heapq
import io
import json
import logging
import math
import pickle
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Magic byte prefix of a ZIP/NPZ file.
_NPZ_MAGIC = b"PK\x03\x04"


@dataclass
class SearchTrace:
    """Full traversal record returned by :meth:`HNSWIndex.search` when
    ``trace=True``.

    Attributes
    ----------
    entry_point : int
        Node ID used as the graph entry point at the top layer.
    layer_descents : list of list of int
        For each layer above 0 (descending), the node IDs visited during the
        greedy single-hop descent.
    l0_visited : list of int
        All node IDs examined at layer 0 during beam search.
    l0_candidates_final : list of tuple[float, int]
        The ``(distance, node_id)`` pairs in the result heap W after layer-0
        search, sorted ascending.  Includes non-result-set candidates that
        were still considered.
    """

    entry_point: int = -1
    layer_descents: List[List[int]] = field(default_factory=list)
    l0_visited: List[int] = field(default_factory=list)
    l0_candidates_final: List[Tuple[float, int]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Distance helpers
# ---------------------------------------------------------------------------


def _cosine_dist(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance in [0, 2].  a and b must be unit vectors."""
    return float(1.0 - np.dot(a, b))


def _l2_dist_sq(a: np.ndarray, b: np.ndarray) -> float:
    """Squared L2 distance."""
    diff = a - b
    return float(np.dot(diff, diff))


# ---------------------------------------------------------------------------
# Candidate heap helpers
#
# Python's heapq is a min-heap.  We represent the *result* set W as a
# max-heap by storing (−dist, id).  The candidate queue is a standard
# min-heap storing (dist, id).
# ---------------------------------------------------------------------------


def _heap_push_result(W: list, dist: float, nid: int, ef: int) -> None:
    """Push (dist, nid) onto the W max-heap, evicting the worst if |W| > ef."""
    heapq.heappush(W, (-dist, nid))
    if len(W) > ef:
        heapq.heappop(W)


def _heap_worst_dist(W: list) -> float:
    """Return the distance of the furthest element in W (max-heap)."""
    if not W:
        return math.inf
    return -W[0][0]


# ---------------------------------------------------------------------------
# HNSWIndex
# ---------------------------------------------------------------------------


class HNSWIndex:
    """Pure-Python HNSW approximate nearest-neighbour index.

    Parameters
    ----------
    M : int
        Maximum number of bidirectional links per node in layers 1+.
        Layer 0 uses ``2 * M`` links.  Typical values: 8 (fast), 16 (default),
        32 (high-recall).
    ef_construction : int
        Beam width during graph construction.  Higher values improve recall
        at the cost of slower build time.  Minimum: M; typical: 100–400.
    space : str
        Distance space: ``"cosine"`` (default) or ``"l2"``.
    """

    def __init__(
        self,
        M: int = 16,
        ef_construction: int = 200,
        space: str = "cosine",
    ) -> None:
        if M < 2:
            raise ValueError("M must be >= 2")
        if ef_construction < M:
            raise ValueError("ef_construction must be >= M")
        if space not in ("cosine", "l2"):
            raise ValueError("space must be 'cosine' or 'l2'")

        self.M = M
        self.M0 = 2 * M  # max links at layer 0
        self.ef_construction = ef_construction
        self.space = space
        self._ml = 1.0 / math.log(float(M))  # level multiplier

        # Per-node storage
        self._vectors: List[np.ndarray] = []  # unit-norm float32 vectors
        # _neighbors[i][lc] = list of neighbour node IDs at layer lc
        self._neighbors: List[List[List[int]]] = []
        self._levels: List[int] = []

        self._entry_point: int = -1
        self._max_level: int = -1

        # v5.1.0 additions
        self._metadata: List[Optional[Dict[str, Any]]] = []
        self._deleted: set = set()  # tombstone set — O(1) lookup

        # v5.2.0 — string-ID map for add_batch() upsert semantics.
        # Maps caller-supplied string IDs → internal node IDs.  Empty when the
        # caller does not use string IDs (ordinary add() calls).
        self._id_map: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _distance(self, a: np.ndarray, b: np.ndarray) -> float:
        if self.space == "cosine":
            return _cosine_dist(a, b)
        return _l2_dist_sq(a, b)

    def _random_level(self) -> int:
        r = np.random.uniform(0.0, 1.0)
        if r <= 0.0:
            r = 1e-15
        return int(-math.log(r) * self._ml)

    def _normalize(self, v: np.ndarray) -> np.ndarray:
        if self.space != "cosine":
            return v.astype(np.float32)
        norm = float(np.linalg.norm(v))
        if norm == 0.0:
            return v.astype(np.float32)
        return (v / norm).astype(np.float32)

    def _is_alive(self, nid: int) -> bool:
        """Return True if node *nid* has not been soft-deleted."""
        return nid not in self._deleted

    def _search_layer(
        self,
        query: np.ndarray,
        entry_points: List[int],
        ef: int,
        layer: int,
        filter_fn: Optional[Callable[[int], bool]] = None,
    ) -> List[Tuple[float, int]]:
        """Beam search on a single layer.

        Returns a list of (distance, node_id) sorted ascending by distance
        (closest first), with at most `ef` entries.

        Parameters
        ----------
        filter_fn : optional callable
            If provided, a node is only added to the result set W when
            ``filter_fn(node_id)`` is True.  Deleted nodes are always
            excluded from W regardless of filter_fn, but they are still
            traversed as graph connectors so the walk remains connected.
        """
        visited: set = set(entry_points)
        candidates: list = []  # min-heap (dist, nid)
        W: list = []  # max-heap (−dist, nid)

        for ep in entry_points:
            d_ep = self._distance(query, self._vectors[ep])
            heapq.heappush(candidates, (d_ep, ep))
            # Only count live, passing nodes in the result set
            if self._is_alive(ep) and (filter_fn is None or filter_fn(ep)):
                heapq.heappush(W, (-d_ep, ep))

        while candidates:
            d_c, c = heapq.heappop(candidates)
            d_worst = _heap_worst_dist(W)

            if d_c > d_worst and len(W) >= ef:
                break

            # Explore neighbours of c at this layer
            if layer < len(self._neighbors[c]):
                nbrs = self._neighbors[c][layer]
            else:
                nbrs = []

            for nb in nbrs:
                if nb in visited:
                    continue
                visited.add(nb)
                d_nb = self._distance(query, self._vectors[nb])
                # Always use nb as a connector (keeps graph traversal valid).
                # Only add to the candidate queue if it has a chance of
                # improving W (standard HNSW guard).
                if d_nb < _heap_worst_dist(W) or len(W) < ef:
                    heapq.heappush(candidates, (d_nb, nb))
                    # Only place in result set if alive AND passes the filter.
                    if self._is_alive(nb) and (filter_fn is None or filter_fn(nb)):
                        _heap_push_result(W, d_nb, nb, ef)

        # Convert max-heap to sorted ascending list
        return sorted((-neg_d, nid) for neg_d, nid in W)

    def _select_neighbors(
        self,
        candidates: List[Tuple[float, int]],
        M: int,
    ) -> List[int]:
        """Return IDs of the M nearest candidates (greedy nearest-first)."""
        return [nid for _, nid in candidates[:M]]

    # ------------------------------------------------------------------
    # Core insertion
    # ------------------------------------------------------------------

    def _insert_one(self, vector: np.ndarray) -> int:
        """Insert a single (already-normalised) vector.  Return its node ID."""
        node_id = len(self._vectors)
        self._vectors.append(vector)

        level = self._random_level()
        self._levels.append(level)
        # Initialise neighbour lists for all layers this node participates in
        self._neighbors.append([[] for _ in range(level + 1)])

        if self._entry_point == -1:
            # First node
            self._entry_point = 0
            self._max_level = level
            return node_id

        # Greedy descent from top to level+1  (ef=1)
        ep = [self._entry_point]
        for lc in range(self._max_level, level, -1):
            W = self._search_layer(vector, ep, ef=1, layer=lc)
            ep = [W[0][1]] if W else [self._entry_point]

        # Bidirectional connections from min(level, max_level) down to 0
        for lc in range(min(level, self._max_level), -1, -1):
            M_cap = self.M0 if lc == 0 else self.M
            W = self._search_layer(vector, ep, ef=self.ef_construction, layer=lc)
            neighbors = self._select_neighbors(W, M_cap)

            # Set new node's neighbours
            self._neighbors[node_id][lc] = neighbors

            # Bidirectional: add new node to each neighbour's list
            for nb in neighbors:
                # Ensure the neighbour has a layer-lc list
                while len(self._neighbors[nb]) <= lc:
                    self._neighbors[nb].append([])
                nb_nbrs = self._neighbors[nb][lc]
                if len(nb_nbrs) < M_cap:
                    nb_nbrs.append(node_id)
                else:
                    # Shrink: pick best M_cap from old connections + new node
                    nb_vec = self._vectors[nb]
                    cands = [
                        (self._distance(nb_vec, self._vectors[c]), c) for c in nb_nbrs + [node_id]
                    ]
                    cands.sort()
                    self._neighbors[nb][lc] = self._select_neighbors(cands, M_cap)

            # Use current W as entry points for the next (lower) layer
            ep = [nid for _, nid in W]

        # Promote entry point if new node has a higher level
        if level > self._max_level:
            self._entry_point = node_id
            self._max_level = level

        return node_id

    # ------------------------------------------------------------------
    # Public: add
    # ------------------------------------------------------------------

    def add(
        self,
        vectors: np.ndarray,
        metadata: Optional[List[Optional[Dict[str, Any]]]] = None,
    ) -> List[int]:
        """Add one or more vectors to the index.

        Parameters
        ----------
        vectors : np.ndarray
            Shape ``(d,)`` for a single vector or ``(n, d)`` for a batch.
        metadata : list of dicts, optional
            Per-vector metadata dicts (must match the number of rows in
            *vectors*).  Use ``None`` entries for vectors without metadata.
            Values are stored verbatim and used by the *filter* parameter
            of :meth:`search`.

        Returns
        -------
        list of int
            The node IDs assigned to the inserted vectors (stable within this
            process lifetime).
        """
        vecs = np.asarray(vectors, dtype=np.float32)
        if vecs.ndim == 1:
            vecs = vecs[np.newaxis, :]
        if vecs.ndim != 2:
            raise ValueError("vectors must be 1-D or 2-D")

        n = vecs.shape[0]
        if metadata is not None and len(metadata) != n:
            raise ValueError(
                f"metadata length ({len(metadata)}) must match number of vectors ({n})"
            )

        node_ids: List[int] = []
        for i, v in enumerate(vecs):
            nid = self._insert_one(self._normalize(v))
            # _insert_one returns the assigned node ID
            self._metadata.append(metadata[i] if metadata is not None else None)
            node_ids.append(nid)
        return node_ids

    def add_batch(
        self,
        vectors: np.ndarray,
        ids: Optional[List[str]] = None,
        metadata: Optional[List[Optional[Dict[str, Any]]]] = None,
    ) -> Dict[str, int]:
        """Batch upsert with deduplication.

        Inserts new vectors and updates existing ones in a single call.
        Deduplication is by the caller-supplied string *ids*: if an ID already
        exists in the index the stored vector and metadata are updated in-place
        without touching the graph structure (fast path); if an ID is new it is
        inserted via the standard HNSW construction algorithm.

        When *ids* is ``None``, all vectors are treated as new insertions
        (equivalent to calling :meth:`add` with the same arguments) and the
        method returns the assigned node IDs as string representations.

        Parameters
        ----------
        vectors : np.ndarray
            Shape ``(n, d)`` or ``(d,)`` float32 input matrix.
        ids : list of str, optional
            Caller-supplied stable string IDs, one per row.  Length must match
            the number of rows in *vectors*.  Any ID that was previously seen
            (in this call or a prior :meth:`add_batch`) triggers an in-place
            update of the stored vector and metadata for that node.
        metadata : list of dicts, optional
            Per-vector metadata.  Same length contract as *ids*.

        Returns
        -------
        dict
            ``{"inserted": int, "updated": int, "node_ids": list[int]}``
            where *node_ids* is the internal integer node ID per row (stable
            within the process lifetime).
        """
        vecs = np.asarray(vectors, dtype=np.float32)
        if vecs.ndim == 1:
            vecs = vecs[np.newaxis, :]
        if vecs.ndim != 2:
            raise ValueError("vectors must be 1-D or 2-D")
        n = vecs.shape[0]

        if ids is not None and len(ids) != n:
            raise ValueError(f"ids length ({len(ids)}) must match vectors ({n})")
        if metadata is not None and len(metadata) != n:
            raise ValueError(f"metadata length ({len(metadata)}) must match vectors ({n})")

        inserted = 0
        updated = 0
        node_ids: List[int] = []

        for i, v in enumerate(vecs):
            meta_i = metadata[i] if metadata is not None else None
            str_id = ids[i] if ids is not None else None

            if str_id is not None and str_id in self._id_map:
                # ── Update path: overwrite vector + metadata in-place ────────
                # Graph links are preserved; the next search uses the new vector
                # transparently.  This is O(1) per update (no graph surgery).
                nid = self._id_map[str_id]
                self._vectors[nid] = self._normalize(v)
                self._metadata[nid] = meta_i
                # Resurrect if previously deleted
                self._deleted.discard(nid)
                updated += 1
            else:
                # ── Insert path ──────────────────────────────────────────────
                nid = self._insert_one(self._normalize(v))
                self._metadata.append(meta_i)
                if str_id is not None:
                    self._id_map[str_id] = nid
                inserted += 1

            node_ids.append(nid)

        return {"inserted": inserted, "updated": updated, "node_ids": node_ids}

    def get_by_id(self, str_id: str) -> Optional[Dict[str, Any]]:
        """Return the stored metadata for a string ID (``None`` if not found)."""
        nid = self._id_map.get(str_id)
        if nid is None or nid in self._deleted:
            return None
        return self._metadata[nid]

    # ------------------------------------------------------------------
    # Public: search
    # ------------------------------------------------------------------

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        ef: int = 64,
        filter: Optional[Dict[str, Any]] = None,
        trace: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search for the k approximate nearest neighbours of query.

        Parameters
        ----------
        query : np.ndarray
            Shape ``(d,)`` float32 query vector.
        k : int
            Number of nearest neighbours to return.
        ef : int
            Search beam width.  Must be >= k.  Higher => better recall.
        filter : dict, optional
            Metadata pre-filter applied *during* graph traversal (not after).
            A node must satisfy all ``{field: value}`` equality constraints in
            *filter* to be included in the result set.  The graph is still
            traversed through non-matching nodes, keeping the walk connected.
            Example: ``filter={"category": "science"}``
        trace : bool, default False
            When ``True``, return a :class:`SearchTrace` as a third element of
            the returned tuple recording the full graph traversal: per-layer
            descent nodes, all layer-0 candidates examined, and the final
            result heap.  The trace is useful for debugging recall regressions
            and for the demo visualisation (animated search beam).

        Returns
        -------
        indices : np.ndarray, shape (≤k,), dtype int64
            Node IDs of the nearest neighbours that pass the filter,
            in ascending distance order.
        distances : np.ndarray, shape (≤k,), dtype float32
            Corresponding cosine (or L2²) distances.
        search_trace : SearchTrace
            Only present when ``trace=True``.  Contains the full traversal
            record for the query.
        """
        live = len(self._vectors) - len(self._deleted)
        if live == 0:
            empty_idx = np.array([], dtype=np.int64)
            empty_dst = np.array([], dtype=np.float32)
            if trace:
                return empty_idx, empty_dst, SearchTrace()
            return empty_idx, empty_dst

        q = self._normalize(np.asarray(query, dtype=np.float32))
        ef_actual = max(ef, k)

        # Build the filter callable from the dict
        filter_fn: Optional[Callable[[int], bool]] = None
        if filter:

            def filter_fn(nid: int, _f: Dict[str, Any] = filter) -> bool:
                meta = self._metadata[nid] if nid < len(self._metadata) else None
                if meta is None:
                    return False
                return all(meta.get(fk) == fv for fk, fv in _f.items())

        sr = SearchTrace(entry_point=self._entry_point) if trace else None

        ep = [self._entry_point]
        # Upper-layer greedy descent (ef=1, no filter — we need connectivity)
        for lc in range(self._max_level, 0, -1):
            W = self._search_layer(q, ep, ef=1, layer=lc)
            new_ep = [W[0][1]] if W else [self._entry_point]
            if sr is not None:
                sr.layer_descents.append(list(new_ep))
            ep = new_ep

        W0 = self._search_layer(q, ep, ef=ef_actual, layer=0, filter_fn=filter_fn)
        if sr is not None:
            # Collect every node referenced in W0 as the "visited" set.
            sr.l0_visited = [nid for _, nid in W0]
            sr.l0_candidates_final = list(W0)

        top_k = W0[:k]
        indices = np.array([nid for _, nid in top_k], dtype=np.int64)
        distances = np.array([d for d, _ in top_k], dtype=np.float32)
        if sr is not None:
            return indices, distances, sr
        return indices, distances

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save the index to *path* in numpy ``.npz`` format.

        The serialised file is a standard ``numpy.savez_compressed`` archive
        (ZIP container, no arbitrary code execution on load).  The format
        stores vectors as a float32 matrix and encodes the graph, metadata, and
        configuration as JSON byte arrays embedded in the archive.

        Parameters
        ----------
        path : str or Path
            Destination file path.  Conventionally use the ``.vindex`` suffix.
            ``numpy.savez_compressed`` will append ``.npz`` if *path* does not
            already end with it — pass the exact path you want.
        """
        p = Path(path)
        n = len(self._vectors)

        # ── vectors (float32 matrix) ──────────────────────────────────────
        if n > 0:
            vec_arr = np.stack(self._vectors, axis=0)
        else:
            vec_arr = np.zeros((0, 1), dtype=np.float32)

        # ── scalar index metadata ─────────────────────────────────────────
        levels_arr = np.array(self._levels, dtype=np.int32)

        # ── JSON-encoded blobs (stored as uint8 byte arrays) ─────────────
        # The neighbor structure is a list of lists of lists — JSON is the
        # simplest safe encoding.  The .npz ZIP container compresses it well.
        def _to_bytes(obj: Any) -> np.ndarray:
            return np.frombuffer(
                json.dumps(obj, separators=(",", ":")).encode("utf-8"),
                dtype=np.uint8,
            )

        params_bytes = _to_bytes(
            {
                "M": self.M,
                "ef_construction": self.ef_construction,
                "space": self.space,
                "entry_point": self._entry_point,
                "max_level": self._max_level,
                "format_version": 2,
            }
        )
        graph_bytes = _to_bytes(self._neighbors)
        meta_bytes = _to_bytes(self._metadata)
        deleted_bytes = _to_bytes(sorted(self._deleted))
        id_map_bytes = _to_bytes(self._id_map)

        # `numpy.savez_compressed` appends `.npz` to the filename when the
        # supplied path does not already end with that suffix — which would
        # create the file at a different location than the caller asked for.
        # Write through a BytesIO buffer so the archive is always materialised
        # at exactly `p`, regardless of suffix.
        buf = io.BytesIO()
        np.savez_compressed(
            buf,
            vectors=vec_arr,
            levels=levels_arr,
            params=params_bytes,
            graph=graph_bytes,
            meta=meta_bytes,
            deleted=deleted_bytes,
            id_map=id_map_bytes,
        )
        with open(p, "wb") as fh:
            fh.write(buf.getvalue())
        logger.debug("HNSWIndex saved: %s (%d vectors)", p, n)

    @classmethod
    def load(cls, path: str) -> "HNSWIndex":
        """Load an index from a file written by :meth:`save`.

        Accepts both the current ``.npz`` format (v5.2.0+) and the legacy
        pickle format written by earlier versions.  The pickle path is kept
        for backward compatibility but emits a ``DeprecationWarning``; it will
        be removed in a future major version.

        Parameters
        ----------
        path : str or Path
            File path previously written by :meth:`save`.

        Returns
        -------
        HNSWIndex
            Populated index ready for search.

        Raises
        ------
        ValueError
            If the file header is not recognised as either format.
        """
        p = Path(path)
        # save() always writes to the exact path supplied, so `resolved` is
        # simply `p`.  The extra fallback for `.npz`-suffixed copies is kept
        # for files created by older code that called numpy.savez directly.
        npz_alt = p.with_suffix(p.suffix + ".npz")
        if not p.exists() and npz_alt.exists():
            resolved = npz_alt
        else:
            resolved = p

        with open(resolved, "rb") as fh:
            magic = fh.read(4)

        if magic[:4] == _NPZ_MAGIC:
            return cls._load_npz(resolved)

        if magic[:2] == b"\x80\x05" or magic[:2] == b"\x80\x04":
            warnings.warn(
                "Loading HNSWIndex from legacy pickle format. "
                "Re-save with HNSWIndex.save() to upgrade to .npz format. "
                "Pickle support will be removed in a future major version.",
                DeprecationWarning,
                stacklevel=2,
            )
            return cls._load_pickle(resolved)

        raise ValueError(
            f"Unrecognised file format for {resolved!r}: "
            "expected .npz (magic PK\\x03\\x04) or pickle (magic \\x80\\x04/05)"
        )

    @classmethod
    def _load_npz(cls, path: Path) -> "HNSWIndex":
        # np.load returns a lazy NpzFile that reads from the ZIP on first array
        # access.  Force all arrays into memory inside the `with` block so the
        # underlying file handle can be safely closed before we process anything.
        with open(path, "rb") as fh:
            data = dict(np.load(fh, allow_pickle=False))

        def _from_bytes(key: str) -> Any:
            return json.loads(bytes(data[key]).decode("utf-8"))

        params = _from_bytes("params")
        neighbors = _from_bytes("graph")
        metadata = _from_bytes("meta")
        deleted = set(_from_bytes("deleted"))
        id_map = _from_bytes("id_map")

        idx = cls(
            M=params["M"],
            ef_construction=params["ef_construction"],
            space=params["space"],
        )
        vec_arr = data["vectors"]
        idx._vectors = [vec_arr[i] for i in range(len(vec_arr))]
        # Neighbor lists are decoded from JSON as plain Python lists (correct type).
        idx._neighbors = [[list(layer) for layer in node_layers] for node_layers in neighbors]
        idx._levels = list(map(int, data["levels"].tolist()))
        idx._entry_point = int(params["entry_point"])
        idx._max_level = int(params["max_level"])
        idx._metadata = [(m if isinstance(m, dict) else None) for m in metadata]
        idx._deleted = deleted
        idx._id_map = {str(k): int(v) for k, v in id_map.items()}
        logger.debug("HNSWIndex loaded: %s (%d vectors)", path, len(idx._vectors))
        return idx

    @classmethod
    def _load_pickle(cls, path: Path) -> "HNSWIndex":
        """Load a legacy pickle-format index (internal, called by :meth:`load`)."""
        with open(path, "rb") as fh:
            d = pickle.load(fh)  # noqa: S301 — intentional legacy compat
        idx = cls(
            M=d["M"],
            ef_construction=d["ef_construction"],
            space=d["space"],
        )
        idx._vectors = d["vectors"]
        idx._neighbors = d["neighbors"]
        idx._levels = d["levels"]
        idx._entry_point = d["entry_point"]
        idx._max_level = d["max_level"]
        idx._metadata = d.get("metadata") or [None] * len(idx._vectors)
        idx._deleted = d.get("deleted") or set()
        idx._id_map = d.get("id_map") or {}
        return idx

    # ------------------------------------------------------------------
    # Informational
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._vectors)

    def __repr__(self) -> str:
        return (
            f"HNSWIndex(n={len(self)}, deleted={len(self._deleted)}, "
            f"M={self.M}, ef_construction={self.ef_construction}, "
            f"space={self.space!r}, max_level={self._max_level})"
        )

    # ------------------------------------------------------------------
    # v5.1.0 additions
    # ------------------------------------------------------------------

    def delete(self, node_id: int) -> None:
        """Soft-delete a vector by ID.

        The node is tombstoned in O(1) — it is excluded from all future
        search results and metadata lookups but its graph links remain intact
        so traversal connectivity is preserved.  Call :meth:`compact` to
        reclaim memory and restore recall.

        Parameters
        ----------
        node_id : int
            ID returned by :meth:`add` or previously visible in search
            results.

        Raises
        ------
        IndexError
            If *node_id* is out of range.
        ValueError
            If *node_id* has already been deleted.
        """
        if node_id < 0 or node_id >= len(self._vectors):
            raise IndexError(f"node_id {node_id} out of range [0, {len(self._vectors)})")
        if node_id in self._deleted:
            raise ValueError(f"node_id {node_id} is already deleted")
        self._deleted.add(node_id)

    def stats(self) -> Dict[str, Any]:
        """Return health and size statistics for the index.

        Returns
        -------
        dict with keys:
            ``n_total``       — total node count (including deleted)
            ``n_alive``       — live (non-deleted) node count
            ``n_deleted``     — soft-deleted node count
            ``orphan_count``  — alive nodes with zero alive neighbours at layer 0
            ``avg_degree_l0`` — mean live neighbour count at layer 0 (alive nodes)
            ``max_level``     — highest layer in the graph
            ``space``         — distance space
        """
        n_total = len(self._vectors)
        n_deleted = len(self._deleted)
        n_alive = n_total - n_deleted

        orphan_count = 0
        total_degree = 0
        alive_counted = 0

        for nid in range(n_total):
            if nid in self._deleted:
                continue
            l0 = self._neighbors[nid][0] if self._neighbors[nid] else []
            live_nbrs = [nb for nb in l0 if nb not in self._deleted]
            total_degree += len(live_nbrs)
            alive_counted += 1
            if not live_nbrs:
                orphan_count += 1

        avg_degree = total_degree / alive_counted if alive_counted > 0 else 0.0

        return {
            "n_total": n_total,
            "n_alive": n_alive,
            "n_deleted": n_deleted,
            "orphan_count": orphan_count,
            "avg_degree_l0": round(avg_degree, 2),
            "max_level": self._max_level,
            "space": self.space,
        }

    def compact(self) -> Dict[str, int]:
        """Remove tombstoned nodes and reconnect orphaned live nodes.

        Two passes:
        1. **Tombstone removal** — strips deleted node IDs from all
           neighbour lists and updates the entry point if it was deleted.
        2. **Orphan repair** — any alive node with zero live neighbours at
           layer 0 is reconnected via a graph search.

        Returns
        -------
        dict with keys:
            ``removed``  — number of tombstones cleared from neighbour lists
            ``repaired`` — number of orphaned nodes reconnected
        """
        removed = 0
        repaired = 0

        # Pass 1: strip tombstones from all neighbour lists.
        for nid in range(len(self._vectors)):
            if nid in self._deleted:
                continue
            for lc in range(len(self._neighbors[nid])):
                before = self._neighbors[nid][lc]
                after = [nb for nb in before if nb not in self._deleted]
                removed += len(before) - len(after)
                self._neighbors[nid][lc] = after

        # Fix entry point if it was deleted.
        if self._entry_point in self._deleted:
            for nid in range(len(self._vectors)):
                if nid not in self._deleted:
                    self._entry_point = nid
                    break
            else:
                # All nodes deleted — reset to empty state.
                self._entry_point = -1
                self._max_level = -1
                self._deleted.clear()
                return {"removed": removed, "repaired": 0}

        # Pass 2: reconnect orphans.  A node is orphaned if it has no live
        # neighbours at layer 0 (it can no longer be found by any traversal
        # starting from a connected component).
        for nid in range(len(self._vectors)):
            if nid in self._deleted:
                continue
            l0 = self._neighbors[nid][0] if self._neighbors[nid] else []
            if any(nb not in self._deleted for nb in l0):
                continue  # already connected

            # Search for new neighbours from the current entry point.
            q = self._vectors[nid]
            results = self._search_layer(q, [self._entry_point], ef=self.ef_construction, layer=0)
            new_nbrs = [nb for _, nb in results if nb != nid][: self.M0]

            if not self._neighbors[nid]:
                self._neighbors[nid] = [[]]
            self._neighbors[nid][0] = new_nbrs

            # Bidirectional: add nid to each new neighbour's list.
            for nb in new_nbrs:
                if not self._neighbors[nb]:
                    self._neighbors[nb] = [[]]
                if nid not in self._neighbors[nb][0]:
                    self._neighbors[nb][0].append(nid)
                    if len(self._neighbors[nb][0]) > self.M0:
                        nb_vec = self._vectors[nb]
                        cands = [
                            (self._distance(nb_vec, self._vectors[c]), c)
                            for c in self._neighbors[nb][0]
                        ]
                        cands.sort()
                        self._neighbors[nb][0] = [c for _, c in cands[: self.M0]]

            repaired += 1

        # Finally clear the tombstone set — nodes are no longer in any list.
        self._deleted.clear()
        return {"removed": removed, "repaired": repaired}

    def estimate_recall(
        self,
        sample_size: int = 1000,
        k: int = 10,
        ef: int = 64,
    ) -> Dict[str, Any]:
        """Estimate Recall@k by comparing HNSW to brute-force on a random sample.

        Mathematical guarantee
        ----------------------
        For each sampled query vector (drawn uniformly at random from the live
        corpus), the exact k-nearest neighbours are computed via brute-force
        cosine scan.  Recall@k is the fraction of exact neighbours that the
        HNSW search also returns.

        The 95% Wilson confidence interval is computed assuming a binomial
        proportion (each of the ``sample_size * k`` individual neighbour
        positions is a Bernoulli trial).

        Parameters
        ----------
        sample_size : int
            Number of random query vectors to sample from the live corpus.
            Capped at the live node count.
        k : int
            Recall cut-off.
        ef : int
            HNSW search beam width used for the recall measurement.

        Returns
        -------
        dict with keys:
            ``recall``         — point estimate in [0, 1]
            ``ci_95_lower``    — Wilson 95% lower bound
            ``ci_95_upper``    — Wilson 95% upper bound
            ``sample_size``    — actual number of queries used
            ``k``              — recall cut-off
            ``ef``             — beam width used
            ``n_alive``        — live node count at time of call
        """
        alive_ids = [nid for nid in range(len(self._vectors)) if nid not in self._deleted]
        n = len(alive_ids)
        if n < 2:
            return {
                "recall": 1.0,
                "ci_95_lower": 1.0,
                "ci_95_upper": 1.0,
                "sample_size": 0,
                "k": k,
                "ef": ef,
                "n_alive": n,
            }

        k_eff = min(k, n - 1)
        actual = min(sample_size, n)
        rng = np.random.default_rng(seed=0x5EED)
        sample = rng.choice(alive_ids, size=actual, replace=False)

        # Build alive-only matrix for brute-force (shape: n_alive × d)
        mat = np.stack([self._vectors[i] for i in alive_ids])  # (n, d)

        hits = 0
        total = 0

        for qid in sample:
            q = self._vectors[qid]

            # Brute-force exact k-NN (excluding the query itself)
            if self.space == "cosine":
                sims = mat @ q  # (n,) cosine similarities (unit vecs)
            else:
                sims = -np.sum((mat - q) ** 2, axis=1)  # negative L2² (higher=closer)
            # Exclude query from its own result
            q_pos_in_alive = alive_ids.index(int(qid))
            sims[q_pos_in_alive] = -np.inf
            gt_local = np.argpartition(sims, -k_eff)[-k_eff:]
            gt_global = set(alive_ids[i] for i in gt_local)

            # HNSW search
            hnsw_ids, _ = self.search(q, k=k_eff, ef=ef)
            hnsw_set = set(int(i) for i in hnsw_ids)

            hits += len(hnsw_set & gt_global)
            total += k_eff

        recall = hits / total if total > 0 else 1.0

        # Wilson score interval (z=1.96 for 95% CI)
        z = 1.96
        n_t = float(total)
        p = recall
        denom = 1 + z * z / n_t
        centre = (p + z * z / (2 * n_t)) / denom
        margin = (z / denom) * math.sqrt(p * (1 - p) / n_t + z * z / (4 * n_t * n_t))
        ci_lo = max(0.0, centre - margin)
        ci_hi = min(1.0, centre + margin)

        return {
            "recall": round(recall, 6),
            "ci_95_lower": round(ci_lo, 6),
            "ci_95_upper": round(ci_hi, 6),
            "sample_size": actual,
            "k": k_eff,
            "ef": ef,
            "n_alive": n,
        }


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------


def build_hnsw_index(
    vectors: np.ndarray,
    M: int = 16,
    ef_construction: int = 200,
    space: str = "cosine",
) -> HNSWIndex:
    """Build an HNSW index from a batch of vectors.

    Parameters
    ----------
    vectors : np.ndarray, shape (n, d)
        Float32 (or compatible) array of input vectors.
    M : int
        Max neighbours per layer (layer 0 uses 2*M).
    ef_construction : int
        Beam width during construction.
    space : str
        ``"cosine"`` or ``"l2"``.

    Returns
    -------
    HNSWIndex
        Populated, ready-to-search index.
    """
    idx = HNSWIndex(M=M, ef_construction=ef_construction, space=space)
    idx.add(vectors)
    return idx


def hnsw_search(
    index: HNSWIndex,
    query: np.ndarray,
    k: int = 10,
    ef: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    """Search an existing HNSWIndex for the k nearest neighbours of query.

    Returns
    -------
    indices : np.ndarray, shape (k,), int64
    distances : np.ndarray, shape (k,), float32
    """
    return index.search(query, k=k, ef=ef)


def recall_at_k(
    index: HNSWIndex,
    queries: np.ndarray,
    ground_truth: np.ndarray,
    k: int = 10,
    ef: int = 64,
) -> float:
    """Compute Recall@k for a set of queries against known ground-truth IDs.

    Parameters
    ----------
    index : HNSWIndex
        Populated HNSW index.
    queries : np.ndarray, shape (q, d)
        Query vectors.
    ground_truth : np.ndarray, shape (q, >=k), int
        ``ground_truth[i, :k]`` contains the IDs of the k true nearest
        neighbours of query i.
    k : int
        Recall cut-off.
    ef : int
        Search beam width (>= k recommended).

    Returns
    -------
    float
        Recall@k in [0, 1].
    """
    hits = 0
    total = 0
    ef_actual = max(ef, k)
    for i, q in enumerate(queries):
        indices, _ = index.search(q, k=k, ef=ef_actual)
        gt = set(int(x) for x in ground_truth[i, :k])
        hits += len(set(int(x) for x in indices[:k]) & gt)
        total += k
    return hits / total if total > 0 else 0.0


def hnsw_compression_info(d: int, M: int = 16) -> dict:
    """Return memory breakdown for a single vector in an HNSW index.

    Parameters
    ----------
    d : int
        Vector dimensionality.
    M : int
        HNSW M parameter.

    Returns
    -------
    dict with keys:
        bytes_fp32    : bytes for original FP32 vector
        bytes_int8    : bytes for INT8 quantised vector
        bytes_graph   : bytes for graph links (INT32 IDs, average estimate)
        bytes_total   : total bytes per indexed vector
        compression_ratio : versus raw FP32-only storage
    """
    bytes_fp32 = d * 4
    bytes_int8 = d + 4  # INT8 d bytes + 4-byte float32 scale
    # Average links across layers: layer 0 has up to 2*M, upper layers up to M.
    # For a random graph the expected number of layers is O(log n / log M).
    # Use a conservative estimate of ~ 2.5 * M average links per node.
    avg_links = int(2.5 * M)
    bytes_graph = avg_links * 4  # each link is an int32
    bytes_total = bytes_int8 + bytes_graph
    return {
        "bytes_fp32": bytes_fp32,
        "bytes_int8": bytes_int8,
        "bytes_graph": bytes_graph,
        "bytes_total": bytes_total,
        "compression_ratio": round(bytes_fp32 / bytes_total, 2),
    }
