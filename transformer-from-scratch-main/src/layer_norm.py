"""Layer normalisation.

Forward and backward passes correspond line-for-line to the derivation in
`notes/layer_norm.md`. Normalises along the last axis (the feature axis)
for each example independently — no running statistics, no batch-axis
coupling. The learnable affine `gamma * x_hat + beta` follows the
normalisation.
"""
from __future__ import annotations

import numpy as np


class LayerNorm:
    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        self.eps = eps
        self.gamma = np.ones(d_model, dtype=np.float32)   # initialise to identity transform
        self.beta = np.zeros(d_model, dtype=np.float32)
        self._cache: dict | None = None
        self.dgamma: np.ndarray | None = None
        self.dbeta: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (..., d_model). Normalise along the last axis."""
        mu = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        sigma = np.sqrt(var + self.eps)
        x_hat = (x - mu) / sigma
        y = self.gamma * x_hat + self.beta
        self._cache = {"x_hat": x_hat, "sigma": sigma}
        return y

    def backward(self, grad_y: np.ndarray) -> np.ndarray:
        """grad_y: same shape as forward output. Returns dL/dx."""
        c = self._cache
        assert c is not None, "LayerNorm.backward called before forward"
        x_hat, sigma = c["x_hat"], c["sigma"]
        d = x_hat.shape[-1]

        # 2. Affine: y = gamma * x_hat + beta. Sum gamma/beta grads across leading axes
        # (gamma and beta are shared across every position in the leading dims).
        leading_axes = tuple(range(grad_y.ndim - 1))
        self.dbeta = grad_y.sum(axis=leading_axes)
        self.dgamma = (grad_y * x_hat).sum(axis=leading_axes)
        dxhat = grad_y * self.gamma                                     # (..., d)

        # 3. Three-term backward through the normalisation:
        #    dx = (1/sigma) * (dxhat - mean(dxhat) - x_hat * mean(dxhat*x_hat))
        mean_dxhat = dxhat.mean(axis=-1, keepdims=True)
        mean_dxhat_xhat = (dxhat * x_hat).mean(axis=-1, keepdims=True)
        dx = (dxhat - mean_dxhat - x_hat * mean_dxhat_xhat) / sigma
        return dx

    def params_and_grads(self):
        yield self.gamma, self.dgamma
        yield self.beta, self.dbeta


if __name__ == "__main__":
    from src.gradcheck import numerical_gradient, relative_error, upcast_module

    rng = np.random.default_rng(0)
    ln = LayerNorm(d_model=8)
    # Vary gamma and beta from default to make the gradient interesting.
    ln.gamma = (rng.standard_normal(8) * 0.5 + 1).astype(np.float32)
    ln.beta = (rng.standard_normal(8) * 0.1).astype(np.float32)
    x = rng.standard_normal((2, 3, 8)).astype(np.float32)

    out = ln.forward(x)
    print(f"forward: x{x.shape} -> y{out.shape}")
    # Sanity: per-example mean ≈ beta, variance ≈ gamma^2 (after normalisation+affine).
    print(f"  pre-affine x_hat row mean ≈ 0:  max|mean| = {np.abs(ln._cache['x_hat'].mean(-1)).max():.2e}")
    print(f"  pre-affine x_hat row std  ≈ 1:  max|std-1| = {np.abs(ln._cache['x_hat'].std(-1) - 1).max():.2e}")

    upcast_module(ln)
    x = x.astype(np.float64)
    grad_out = rng.standard_normal((2, 3, 8))

    ln.forward(x)
    dx_analytic = ln.backward(grad_out)

    def loss_fn():
        return float((ln.forward(x) * grad_out).sum())

    print(f"\ngradient check (float64):")
    print(f"  {'param':<10}{'shape':<12}{'max rel err':>14}")
    for name, param, analytic in [
        ("gamma", ln.gamma, ln.dgamma),
        ("beta",  ln.beta,  ln.dbeta),
    ]:
        numerical = numerical_gradient(loss_fn, param, eps=1e-6)
        err = relative_error(analytic, numerical)
        flag = "PASS" if err < 1e-5 else "FAIL"
        print(f"  {name:<10}{str(param.shape):<12}{err:>14.2e}  {flag}")

    dx_num = numerical_gradient(loss_fn, x, eps=1e-6)
    err_x = relative_error(dx_analytic, dx_num)
    print(f"  {'dL/dx':<10}{str(x.shape):<12}{err_x:>14.2e}  {'PASS' if err_x < 1e-5 else 'FAIL'}")
