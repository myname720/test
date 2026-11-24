"""Layer primitives — Linear (affine) and ReLU.

Pattern matches the MLP-from-scratch repo: each layer caches what its
backward pass needs in `forward()` and exposes parameter gradients on
itself for the optimizer to consume.

Convention. `backward()` receives `grad_y = dL/d(output)` and returns
`dL/d(input)`. Parameter gradients (`dW`, `db`) are stored on the layer.
The loss layer is responsible for the 1/B scaling so all downstream
gradients are mean-batch.
"""
from __future__ import annotations

import numpy as np


class Linear:
    """Affine layer: ``y = x W + b``. Inputs are batched as the leading axes of x.

    Weights are He-initialised (``sqrt(2/in_features)``); biases zero.
    Works on any leading-batch shape — the linear is applied along the last axis.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        rng: np.random.Generator,
    ) -> None:
        scale = np.sqrt(2.0 / in_features)
        self.W = (rng.standard_normal((in_features, out_features)) * scale).astype(np.float32)
        self.b = np.zeros(out_features, dtype=np.float32)
        self._x_cache: np.ndarray | None = None
        self.dW: np.ndarray | None = None
        self.db: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x_cache = x
        return x @ self.W + self.b

    def backward(self, grad_y: np.ndarray) -> np.ndarray:
        x = self._x_cache
        assert x is not None, "Linear.backward called before forward"
        # Flatten leading axes so the matmul collapses them; reduce by sum.
        in_features = self.W.shape[0]
        out_features = self.W.shape[1]
        x_flat = x.reshape(-1, in_features)
        grad_y_flat = grad_y.reshape(-1, out_features)
        self.dW = x_flat.T @ grad_y_flat                    # (in, out)
        self.db = grad_y_flat.sum(axis=0)                   # (out,)
        return (grad_y_flat @ self.W.T).reshape(x.shape)    # back to original shape


class ReLU:
    """Elementwise ``y = max(0, x)``."""

    def __init__(self) -> None:
        self._mask_cache: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._mask_cache = x > 0
        return x * self._mask_cache

    def backward(self, grad_y: np.ndarray) -> np.ndarray:
        mask = self._mask_cache
        assert mask is not None, "ReLU.backward called before forward"
        return grad_y * mask
