"""Cross-entropy loss with a numerically stable log-softmax.

Same pattern as `mnist-nn-from-scratch/src/losses.py`. The 1/B scaling is
applied here so every downstream `backward()` operates on the mean-batch
gradient. For the language-model task we flatten the (B, T) sequence
dimension into a single batch dimension before calling cross_entropy.
"""
from __future__ import annotations

import numpy as np


def log_softmax(z: np.ndarray) -> np.ndarray:
    """Numerically stable log-softmax along the last axis."""
    z_max = z.max(axis=-1, keepdims=True)
    z_shifted = z - z_max
    log_sum_exp = np.log(np.exp(z_shifted).sum(axis=-1, keepdims=True))
    return z_shifted - log_sum_exp


def softmax(z: np.ndarray) -> np.ndarray:
    return np.exp(log_softmax(z))


def cross_entropy_loss(
    logits: np.ndarray,
    y: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Mean-batch cross-entropy and gradient w.r.t. logits.

    Args:
        logits: shape (..., n_classes). Leading dims are flattened.
        y: integer labels of shape matching the leading dims.

    Returns:
        loss: scalar mean cross-entropy.
        grad_logits: same shape as logits, equal to (softmax - one_hot(y)) / N.
    """
    n_classes = logits.shape[-1]
    flat_logits = logits.reshape(-1, n_classes)
    flat_y = y.reshape(-1)
    N = flat_y.shape[0]

    log_p = log_softmax(flat_logits)
    loss = float(-log_p[np.arange(N), flat_y].mean())

    grad = np.exp(log_p)
    grad[np.arange(N), flat_y] -= 1
    grad /= N
    return loss, grad.reshape(logits.shape)
