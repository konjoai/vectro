"""Cosine scoring + Maximal Marginal Relevance — shared across framework adapters."""

from __future__ import annotations

from typing import List

import numpy as np


def cosine_scores(query_vec: np.ndarray, mat: np.ndarray) -> np.ndarray:
    """Cosine similarity of *query_vec* against each row of *mat*.

    Args:
        query_vec: 1-D query vector, shape (d,).
        mat: 2-D matrix of stored vectors, shape (n, d).

    Returns:
        1-D array of cosine similarities, shape (n,).
    """
    q = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-10
    return (mat / norms) @ q


def mmr_select(
    embeddings: np.ndarray,
    query_vec: np.ndarray,
    k: int,
    fetch_k: int,
    lambda_mult: float = 0.5,
) -> np.ndarray:
    """Return indices of *k* documents chosen by Maximal Marginal Relevance.

    Algorithm:
        1. Fetch ``fetch_k`` most relevant candidates by cosine similarity.
        2. Greedily add the document that maximises:
               lambda_mult * sim(doc, query)
             - (1 - lambda_mult) * max_sim(doc, already_selected)

    Args:
        embeddings: All stored vectors, shape (n, d).
        query_vec: Query embedding, shape (d,).
        k: Number of documents to select.
        fetch_k: Candidate pool size (≥ k).
        lambda_mult: Relevance weight in [0, 1].

    Returns:
        Index array of length ≤ *k* in selection order (global indices into *embeddings*).
    """
    n = len(embeddings)
    k = min(k, n)
    fetch_k = min(fetch_k, n)

    q = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
    emb_norm = embeddings / norms  # (n, d)

    rel_scores = emb_norm @ q  # (n,)
    cand_idx = np.argpartition(rel_scores, -fetch_k)[-fetch_k:]
    cand_idx = cand_idx[np.argsort(rel_scores[cand_idx])[::-1]]  # descending

    cand_embs = emb_norm[cand_idx]  # (fetch_k, d)

    selected_local: List[int] = []
    selected_global: List[int] = []
    remaining = list(range(len(cand_idx)))

    for _ in range(k):
        if not selected_local:
            best_local = 0
        else:
            sel_embs = cand_embs[np.array(selected_local)]  # (s, d)
            sim_to_sel = cand_embs[remaining] @ sel_embs.T  # (r, s)
            max_sim_to_sel = sim_to_sel.max(axis=1)  # (r,)
            rel = rel_scores[cand_idx[remaining]]  # (r,)
            mmr = lambda_mult * rel - (1.0 - lambda_mult) * max_sim_to_sel
            best_local = remaining[int(np.argmax(mmr))]

        selected_local.append(best_local)
        selected_global.append(int(cand_idx[best_local]))
        remaining = [i for i in remaining if i != best_local]
        if not remaining:
            break

    return np.array(selected_global)
