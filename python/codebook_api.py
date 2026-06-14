"""Autoencoder Codebook — Phase 7b Learned Quantization.

A lightweight neural autoencoder (pure NumPy, no PyTorch) that learns a
low-dimensional INT8 code space for embedding vectors.

Architecture
    Encoder:  d  → hidden  →  target_dim   (Linear+ReLU+Linear)
    Decoder:  target_dim → hidden  →  d    (Linear+ReLU+Linear)
    Storage:  encoder output is scaled and rounded to int8

Training uses mini-batch SGD with cosine loss and L2 regularisation.
Weights are Xavier-initialised.

Typical use
-----------
>>> cb = Codebook(target_dim=32, l2_reg=1e-4)
>>> cb.train(train_vecs, n_epochs=100, lr=0.01, batch_size=64)
>>> codes = cb.encode(vecs)          # int8 (n, target_dim)
>>> recon  = cb.decode(codes)        # float32 (n, d)
>>> sim    = cb.mean_cosine(vecs, recon)   # ≥ 0.75 on small d=64
"""

from __future__ import annotations

from typing import Optional
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _xavier(fan_in: int, fan_out: int, rng: np.random.Generator) -> np.ndarray:
    scale = np.sqrt(2.0 / (fan_in + fan_out))
    return rng.standard_normal((fan_in, fan_out)).astype(np.float32) * scale


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(np.float32)


def _cosine_loss_and_grad(
    recon: np.ndarray,
    target: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Return (mean cosine loss, grad w.r.t. recon).

    Loss = 1 - cosine_similarity, so minimising drives reconstruction toward target.
    """
    eps = 1e-8
    r_norm = np.linalg.norm(recon, axis=1, keepdims=True) + eps
    t_norm = np.linalg.norm(target, axis=1, keepdims=True) + eps
    r_hat = recon / r_norm
    t_hat = target / t_norm
    cos_sim = (r_hat * t_hat).sum(axis=1)
    loss = float((1.0 - cos_sim).mean())

    # Gradient: d(1 - cos)/d(recon) = -( t_hat/r_norm - r_hat * cos[:, None] / r_norm )
    # = -(t_hat - r_hat * cos[:, None]) / r_norm
    grad = -(t_hat - r_hat * cos_sim[:, np.newaxis]) / r_norm
    grad /= recon.shape[0]  # mean over batch
    return loss, grad


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


class Codebook:
    """Learned INT8 autoencoder codebook.

    Parameters
    ----------
    target_dim : int   Compressed code dimensionality (default 64)
    l2_reg     : float L2 weight regularisation coefficient (default 1e-4)
    seed       : int   NumPy random seed
    """

    def __init__(
        self,
        target_dim: int = 64,
        l2_reg: float = 1e-4,
        seed: int = 42,
    ) -> None:
        self.target_dim = target_dim
        self.l2_reg = l2_reg
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self._d: int = 0
        self._hidden: int = 0
        # Encoder weights: W1 (d, hidden), W2 (hidden, target_dim)
        self.W1e: Optional[np.ndarray] = None
        self.W2e: Optional[np.ndarray] = None
        # Decoder weights: W1 (target_dim, hidden), W2 (hidden, d)
        self.W1d: Optional[np.ndarray] = None
        self.W2d: Optional[np.ndarray] = None
        # Adam moments
        self._m: dict = {}
        self._v: dict = {}
        self._t: int = 0
        # INT8 scale for codes
        self._code_scale: float = 1.0
        self.is_trained: bool = False

    # ------------------------------------------------------------------
    # Forward pass through encoder (returns float activations)
    # ------------------------------------------------------------------

    def _encode_float(self, x: np.ndarray) -> np.ndarray:
        """Return pre-quantisation encoder output (n, target_dim) float32.

        Encoder: Linear(d, hidden) → ReLU → Linear(hidden, target_dim)
        """
        h = _relu(x @ self.W1e)  # (n, hidden)
        return h @ self.W2e  # (n, target_dim)

    # ------------------------------------------------------------------
    # Forward pass through decoder
    # ------------------------------------------------------------------

    def _decode_float(self, z: np.ndarray) -> np.ndarray:
        """Return reconstructed float32 from float code z (n, target_dim)."""
        h = _relu(z @ self.W1d)  # (n, hidden)
        return h @ self.W2d  # (n, d)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        embeddings: np.ndarray,
        n_epochs: int = 100,
        lr: float = 0.01,
        batch_size: int = 64,
    ) -> "Codebook":
        """Train the autoencoder using mini-batch Adam + cosine loss.

        Parameters
        ----------
        embeddings : (n, d) float32
        n_epochs   : training epochs
        lr         : Adam learning rate
        batch_size : mini-batch size

        Returns
        -------
        self
        """
        data = np.ascontiguousarray(embeddings, dtype=np.float32)
        n, d = data.shape
        self._d = d
        self._hidden = max(self.target_dim * 2, min(d, 128))

        # Xavier init
        rng = self._rng
        self.W1e = _xavier(d, self._hidden, rng)
        self.W2e = _xavier(self._hidden, self.target_dim, rng)
        self.W1d = _xavier(self.target_dim, self._hidden, rng)
        self.W2d = _xavier(self._hidden, d, rng)

        params = ["W1e", "W2e", "W1d", "W2d"]
        self._m = {p: np.zeros_like(getattr(self, p)) for p in params}
        self._v = {p: np.zeros_like(getattr(self, p)) for p in params}
        self._t = 0

        # Normalise input norms to stabilise training
        norms = np.linalg.norm(data, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        data_n = data / norms

        beta1, beta2, eps_adam = 0.9, 0.999, 1e-8

        for epoch in range(n_epochs):
            idx = rng.permutation(n)
            for start in range(0, n, batch_size):
                b = data_n[idx[start : start + batch_size]]
                if b.shape[0] == 0:
                    continue
                self._t += 1

                # ----- Forward -----
                # Encoder
                H_e = b @ self.W1e  # (bs, hidden)
                A_e = _relu(H_e)
                Z = A_e @ self.W2e  # (bs, target_dim)

                # Decoder
                H_d = Z @ self.W1d  # (bs, hidden)
                A_d = _relu(H_d)
                recon = A_d @ self.W2d  # (bs, d)

                # ----- Loss + grad at output -----
                loss, dL_drecon = _cosine_loss_and_grad(recon, b)

                # ----- Backward through decoder -----
                dL_dW2d = A_d.T @ dL_drecon + self.l2_reg * self.W2d
                dL_dA_d = dL_drecon @ self.W2d.T
                dL_dH_d = dL_dA_d * _relu_grad(H_d)
                dL_dW1d = Z.T @ dL_dH_d + self.l2_reg * self.W1d
                dL_dZ = dL_dH_d @ self.W1d.T

                # ----- Backward through encoder -----
                dL_dW2e = A_e.T @ dL_dZ + self.l2_reg * self.W2e
                dL_dA_e = dL_dZ @ self.W2e.T
                dL_dH_e = dL_dA_e * _relu_grad(H_e)
                dL_dW1e = b.T @ dL_dH_e + self.l2_reg * self.W1e

                grads = {
                    "W1e": dL_dW1e,
                    "W2e": dL_dW2e,
                    "W1d": dL_dW1d,
                    "W2d": dL_dW2d,
                }

                # ----- Adam update -----
                bc1 = 1.0 - beta1**self._t
                bc2 = 1.0 - beta2**self._t
                for p in params:
                    g = grads[p]
                    self._m[p] = beta1 * self._m[p] + (1 - beta1) * g
                    self._v[p] = beta2 * self._v[p] + (1 - beta2) * g * g
                    m_hat = self._m[p] / bc1
                    v_hat = self._v[p] / bc2
                    delta = lr * m_hat / (np.sqrt(v_hat) + eps_adam)
                    setattr(self, p, (getattr(self, p) - delta).astype(np.float32))

        # Calibrate INT8 scale on training data
        z_all = self._encode_float(data_n[: min(n, 1000)])
        abs_max = np.abs(z_all).max()
        self._code_scale = float(abs_max / 127.0) if abs_max > 0 else 1.0

        self.is_trained = True
        return self

    # ------------------------------------------------------------------
    # Encode / decode
    # ------------------------------------------------------------------

    def encode(self, embeddings: np.ndarray) -> np.ndarray:
        """Encode float32 embeddings to INT8 codes.

        Parameters
        ----------
        embeddings : (n, d) float32

        Returns
        -------
        np.ndarray, (n, target_dim) int8
        """
        if not self.is_trained:
            raise RuntimeError("Codebook.train() must be called first.")
        data = np.ascontiguousarray(embeddings, dtype=np.float32)
        norms = np.linalg.norm(data, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        data_n = data / norms

        z = self._encode_float(data_n)
        codes = np.clip(np.round(z / self._code_scale), -127, 127).astype(np.int8)
        return codes

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """Decode INT8 codes to float32 embeddings.

        Parameters
        ----------
        codes : (n, target_dim) int8

        Returns
        -------
        np.ndarray, (n, d) float32
        """
        if not self.is_trained:
            raise RuntimeError("Codebook.train() must be called first.")
        # Use np.float32 for the scale to avoid Python-float upcasting to float64
        z = codes.astype(np.float32) * np.float32(self._code_scale)
        return self._decode_float(z).astype(np.float32)

    # ------------------------------------------------------------------
    # Quality + compression
    # ------------------------------------------------------------------

    def mean_cosine(
        self,
        original: np.ndarray,
        reconstructed: np.ndarray,
    ) -> float:
        """Mean per-vector cosine similarity between original and reconstructed."""
        o = np.ascontiguousarray(original, dtype=np.float32)
        r = np.ascontiguousarray(reconstructed, dtype=np.float32)
        dots = (o * r).sum(axis=1)
        norms = np.linalg.norm(o, axis=1) * np.linalg.norm(r, axis=1)
        norms = np.where(norms == 0, 1.0, norms)
        return float((dots / norms).mean())

    def compression_ratio(self) -> float:
        """Compression ratio: float32 bytes / int8 code bytes."""
        if self._d == 0:
            return 0.0
        float_bytes = self._d * 4
        code_bytes = self.target_dim  # int8 = 1 byte
        return float_bytes / max(code_bytes, 1)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save model weights to a compressed .npz file."""
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained Codebook.")
        np.savez_compressed(
            path,
            W1e=self.W1e,
            W2e=self.W2e,
            W1d=self.W1d,
            W2d=self.W2d,
            meta=np.array(
                [self._d, self._hidden, self.target_dim, self._t],
                dtype=np.int64,
            ),
            code_scale=np.array([self._code_scale], dtype=np.float32),
            l2_reg=np.array([self.l2_reg], dtype=np.float32),
        )

    @classmethod
    def load(cls, path: str) -> "Codebook":
        """Load a previously saved Codebook from a .npz file."""
        data = np.load(path)
        meta = data["meta"]
        d, hidden, target_dim, t = int(meta[0]), int(meta[1]), int(meta[2]), int(meta[3])
        cb = cls(target_dim=target_dim, l2_reg=float(data["l2_reg"][0]))
        cb._d = d
        cb._hidden = hidden
        cb._t = t
        cb.W1e = data["W1e"]
        cb.W2e = data["W2e"]
        cb.W1d = data["W1d"]
        cb.W2d = data["W2d"]
        cb._code_scale = float(data["code_scale"][0])
        cb.is_trained = True
        return cb
