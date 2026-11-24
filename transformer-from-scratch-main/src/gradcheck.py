"""Numerical gradient checking — finite-difference verification.

Catches backward-pass bugs before they corrupt training. Same pattern as
mnist-nn-from-scratch/src/gradcheck.py: parameters are upcast to float64
in-place during the check so that finite-difference cancellation noise
doesn't dominate the comparison.
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def numerical_gradient(
    loss_fn: Callable[[], float],
    param: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """Centered finite differences w.r.t. every entry of `param`.

    Cost: 2 × `param.size` calls to loss_fn. Use only for small tensors.
    """
    grad = np.zeros_like(param, dtype=np.float64)
    for idx in np.ndindex(*param.shape):
        original = float(param[idx])
        param[idx] = original + eps
        f_plus = float(loss_fn())
        param[idx] = original - eps
        f_minus = float(loss_fn())
        param[idx] = original
        grad[idx] = (f_plus - f_minus) / (2 * eps)
    return grad.astype(param.dtype)


def relative_error(a: np.ndarray, b: np.ndarray) -> float:
    """Per-element relative error, max over all elements.

    The denominator is clipped at 1e-4 of the global scale of the gradient
    so that components which are theoretically zero — for example the K-bias
    gradient is mathematically zero because softmax is shift-invariant
    (adding a constant to all keys leaves attention weights unchanged) —
    don't blow up the metric on finite-difference numerical noise.
    """
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    abs_floor = max(np.abs(a).max(), np.abs(b).max(), 1e-8) * 1e-4
    denom = np.maximum(np.abs(a) + np.abs(b), abs_floor)
    rel = np.abs(a - b) / denom
    return float(rel.max())


def upcast_module(module) -> None:
    """Cast all parameters of a module to float64 in-place. Used during gradcheck."""
    for attr in dir(module):
        val = getattr(module, attr, None)
        if isinstance(val, np.ndarray) and val.dtype == np.float32:
            setattr(module, attr, val.astype(np.float64))
