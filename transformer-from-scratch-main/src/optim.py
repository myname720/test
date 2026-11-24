"""AdamW optimiser — Adam with decoupled weight decay.

The update rule, for each parameter θ at step t:

    m_t = β₁ m_{t-1} + (1 - β₁) g_t
    v_t = β₂ v_{t-1} + (1 - β₂) g_t²
    m̂_t = m_t / (1 - β₁ᵗ)        (bias-corrected first moment)
    v̂_t = v_t / (1 - β₂ᵗ)        (bias-corrected second moment)
    θ_t = θ_{t-1} - lr · (m̂_t / (√v̂_t + ε) + λ θ_{t-1})

The decoupled weight-decay term (Loshchilov & Hutter, 2017) is what makes
this AdamW rather than plain Adam — `λ θ` is added to the update directly
rather than to the loss (which would have been multiplied by `1/√v̂_t`,
breaking the regularisation effect).

Convention: weight decay applies only to matrix-shaped parameters (ndim ≥ 2).
Biases, LayerNorm scales/shifts, and any other vector parameters are
exempt — this is the standard heuristic from GPT-2/3 and follow-ups.
"""
from __future__ import annotations

from typing import Callable, Iterator

import numpy as np


class AdamW:
    def __init__(
        self,
        params_and_grads_fn: Callable[[], Iterator[tuple[np.ndarray, np.ndarray]]],
        lr: float = 3e-4,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.1,
    ) -> None:
        self.params_and_grads_fn = params_and_grads_fn
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m: list[np.ndarray] = []
        self.v: list[np.ndarray] = []
        self._initialised = False

    def _lazy_init(self) -> None:
        if self._initialised:
            return
        for param, _ in self.params_and_grads_fn():
            self.m.append(np.zeros_like(param))
            self.v.append(np.zeros_like(param))
        self._initialised = True

    def step(self) -> None:
        self._lazy_init()
        self.t += 1
        bc1 = 1 - self.beta1**self.t
        bc2 = 1 - self.beta2**self.t

        for i, (param, grad) in enumerate(self.params_and_grads_fn()):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad * grad
            m_hat = self.m[i] / bc1
            v_hat = self.v[i] / bc2
            update = m_hat / (np.sqrt(v_hat) + self.eps)

            # Decoupled weight decay only for matrix-shaped params.
            if param.ndim >= 2 and self.weight_decay > 0:
                param -= self.lr * (update + self.weight_decay * param)
            else:
                param -= self.lr * update


if __name__ == "__main__":
    # Smoke test: optimise a simple quadratic to verify Adam descends.
    rng = np.random.default_rng(0)
    target = np.array([1.0, -2.0, 3.0], dtype=np.float32)
    x = rng.standard_normal(3).astype(np.float32)

    # Wrap (x, grad) in the iterator interface AdamW expects.
    grad_buffer = np.zeros_like(x)

    def params_and_grads():
        yield x, grad_buffer

    opt = AdamW(params_and_grads, lr=0.1, weight_decay=0.0)
    for step in range(50):
        loss = float(np.sum((x - target) ** 2))
        grad_buffer[:] = 2 * (x - target)                               # dL/dx
        opt.step()
        if step % 10 == 0:
            print(f"step {step:>2}  loss={loss:.6f}  x={x}")

    print(f"\nfinal x={x}, target={target}")
    print(f"converged: {np.allclose(x, target, atol=1e-3)}")
