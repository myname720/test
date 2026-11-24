"""Token embedding lookup and sinusoidal position encoding.

The token embedding is a learned `(vocab_size, d_model)` lookup table; the
position encoding is the closed-form sin/cos formula from Vaswani et al.
(2017) §3.5 — see notes/attention.md §7 for the derivation. Position
encodings are constants (not learned) and are added to the token embeddings
elementwise before the first Transformer block.
"""
from __future__ import annotations

import numpy as np


class TokenEmbedding:
    """Learned (vocab_size, d_model) lookup table."""

    def __init__(self, vocab_size: int, d_model: int, *, rng: np.random.Generator) -> None:
        # Small Gaussian init (std=0.02) -- standard for transformer LMs (GPT-2 default).
        self.W = (rng.standard_normal((vocab_size, d_model)) * 0.02).astype(np.float32)
        self._idx_cache: np.ndarray | None = None
        self.dW: np.ndarray | None = None

    def forward(self, idx: np.ndarray) -> np.ndarray:
        """idx: (B, T) integer tokens -> (B, T, d_model)."""
        self._idx_cache = idx
        return self.W[idx]

    def backward(self, grad_y: np.ndarray) -> None:
        """Accumulate gradients into self.dW. Returns None since input is integer-indexed."""
        idx = self._idx_cache
        assert idx is not None, "TokenEmbedding.backward called before forward"
        self.dW = np.zeros_like(self.W)
        # np.add.at handles repeated indices correctly (a += grad does not).
        np.add.at(self.dW, idx, grad_y)
        return None


def sinusoidal_position_encoding(max_T: int, d_model: int) -> np.ndarray:
    """Return (max_T, d_model) sinusoidal position encodings.

    For position pos and dimension i:
        PE[pos, 2i]   = sin(pos / 10000^(2i/d_model))
        PE[pos, 2i+1] = cos(pos / 10000^(2i/d_model))

    Constant — no learnable parameters. See notes/attention.md §7 for the
    linear-shift property argument.
    """
    pos = np.arange(max_T)[:, None]                                  # (T, 1)
    i = np.arange(d_model // 2)[None, :]                              # (1, d/2)
    freq = np.exp(-np.log(10000.0) * (2 * i) / d_model)               # (1, d/2)
    angles = pos * freq                                               # (T, d/2)

    pe = np.zeros((max_T, d_model), dtype=np.float32)
    pe[:, 0::2] = np.sin(angles)
    pe[:, 1::2] = np.cos(angles)
    return pe


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    emb = TokenEmbedding(vocab_size=65, d_model=64, rng=rng)
    x = np.array([[0, 5, 12, 7], [1, 1, 9, 3]], dtype=np.int64)
    y = emb.forward(x)
    print(f"token embedding: x{x.shape} -> y{y.shape}")

    grad = rng.standard_normal(y.shape).astype(np.float32)
    emb.backward(grad)
    print(f"  dW shape: {emb.dW.shape}, ||dW||={np.linalg.norm(emb.dW):.4f}")

    pe = sinusoidal_position_encoding(max_T=64, d_model=64)
    print(f"\nposition encoding: shape {pe.shape}")
    print(f"  PE[0]    range [{pe[0].min():+.3f}, {pe[0].max():+.3f}]")
    print(f"  PE[1]    range [{pe[1].min():+.3f}, {pe[1].max():+.3f}]")
    print(f"  PE[63]   range [{pe[63].min():+.3f}, {pe[63].max():+.3f}]")
    # Adjacent positions should be very similar (low frequencies dominate)
    sim = np.dot(pe[0], pe[1]) / (np.linalg.norm(pe[0]) * np.linalg.norm(pe[1]))
    print(f"  cosine sim PE[0] vs PE[1]: {sim:.4f}")
    sim_far = np.dot(pe[0], pe[63]) / (np.linalg.norm(pe[0]) * np.linalg.norm(pe[63]))
    print(f"  cosine sim PE[0] vs PE[63]: {sim_far:.4f}")
