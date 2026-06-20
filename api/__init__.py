"""vectro.api — FastAPI server hosting an in-memory vector index.

V7 (2026-05-09) introduces visualization-friendly endpoints atop the
existing index CRUD: PCA-2D projection and k-means clustering, both
implemented directly on numpy (no sklearn required).

Public surface:

    from api import app, STORE
    from api.store import IndexStore, pca_2d, kmeans

The demo at ``demo/viz.html`` talks to the matching routes mounted in
``demo/server.py``.  Both servers share the helpers in :mod:`api.store`,
so behaviour is identical regardless of which entrypoint is running.
"""

from .app import STORE, app
from .store import Index, IndexStore, kmeans, pca_2d

__all__ = ["app", "STORE", "Index", "IndexStore", "pca_2d", "kmeans"]
