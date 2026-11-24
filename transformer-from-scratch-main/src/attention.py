"""Multi-head self-attention — forward pass.

Implements the operation derived in `notes/attention.md`:

    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

with multi-head decomposition computed jointly via reshape rather than as
separate per-head matmuls. The H heads share a single combined QKV
projection of shape (d_model, 3 d_model), and a single output projection
W_o of shape (d_model, d_model).

Causal masking is hardcoded — this is a decoder-only model, so position i
must never see positions j > i. The mask is applied to the pre-softmax
scores by setting future-position entries to -1e9, which after softmax
gives ~0 attention weight there.

Backward pass derivation is in `notes/attention_backward.md` — the steps
in `backward()` are labelled to the corresponding section of those notes.
"""
from __future__ import annotations

import numpy as np


class MultiHeadSelfAttention:
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        *,
        rng: np.random.Generator,
    ) -> None:
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        scale = np.sqrt(2.0 / d_model)
        self.W_qkv = (rng.standard_normal((d_model, 3 * d_model)) * scale).astype(np.float32)
        self.b_qkv = np.zeros(3 * d_model, dtype=np.float32)
        self.W_o = (rng.standard_normal((d_model, d_model)) * scale).astype(np.float32)
        self.b_o = np.zeros(d_model, dtype=np.float32)

        self._cache: dict | None = None
        self.dW_qkv: np.ndarray | None = None
        self.db_qkv: np.ndarray | None = None
        self.dW_o: np.ndarray | None = None
        self.db_o: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (B, T, d_model) -> output: (B, T, d_model). Causal."""
        B, T, D = x.shape
        H, dk = self.n_heads, self.d_k

        # 1. Project to combined QKV — one matmul, then split.
        qkv = x @ self.W_qkv + self.b_qkv                              # (B, T, 3D)
        Q = qkv[..., :D]
        K = qkv[..., D : 2 * D]
        V = qkv[..., 2 * D :]

        # 2. Reshape into multi-head form: (B, T, D) -> (B, H, T, dk).
        Q = Q.reshape(B, T, H, dk).transpose(0, 2, 1, 3)
        K = K.reshape(B, T, H, dk).transpose(0, 2, 1, 3)
        V = V.reshape(B, T, H, dk).transpose(0, 2, 1, 3)

        # 3. Scaled dot-product scores — (B, H, T, T).
        scores = (Q @ K.transpose(0, 1, 3, 2)) / np.sqrt(dk)

        # 4. Causal mask — disallow attending to future positions.
        causal = np.triu(np.ones((T, T), dtype=bool), k=1)             # True above main diag
        scores = np.where(causal, np.float32(-1e9), scores)

        # 5. Softmax along the last axis (the keys axis), numerically stable.
        scores_shifted = scores - scores.max(axis=-1, keepdims=True)
        exp_scores = np.exp(scores_shifted)
        A = exp_scores / exp_scores.sum(axis=-1, keepdims=True)        # (B, H, T, T)

        # 6. Apply attention to values.
        av = A @ V                                                     # (B, H, T, dk)

        # 7. Concatenate heads: (B, H, T, dk) -> (B, T, D).
        concat = av.transpose(0, 2, 1, 3).reshape(B, T, D)

        # 8. Output projection.
        out = concat @ self.W_o + self.b_o                             # (B, T, D)

        self._cache = {
            "x": x, "Q": Q, "K": K, "V": V, "A": A, "concat": concat,
        }
        return out

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        """grad_out: (B, T, d_model). Returns dL/dx of the same shape.

        Each numbered step matches a section in notes/attention_backward.md.
        """
        c = self._cache
        assert c is not None, "MultiHeadSelfAttention.backward called before forward"
        x = c["x"]
        B, T, D = x.shape
        H, dk = self.n_heads, self.d_k

        # 2. Output projection: out = concat @ W_o + b_o
        concat_flat = c["concat"].reshape(-1, D)                       # (BT, D)
        grad_out_flat = grad_out.reshape(-1, D)                        # (BT, D)
        self.dW_o = concat_flat.T @ grad_out_flat                      # (D, D)
        self.db_o = grad_out_flat.sum(axis=0)                          # (D,)
        d_concat = (grad_out_flat @ self.W_o.T).reshape(B, T, D)       # (B, T, D)

        # Reshape (B, T, D) -> (B, H, T, dk) for the multi-head backward path.
        d_av = d_concat.reshape(B, T, H, dk).transpose(0, 2, 1, 3)     # (B, H, T, dk)

        # 3. Through av = A @ V.
        # dV: A^T @ d_av;  dA: d_av @ V^T  (last-two-axis transposes only)
        dV = c["A"].transpose(0, 1, 3, 2) @ d_av                       # (B, H, T, dk)
        dA = d_av @ c["V"].transpose(0, 1, 3, 2)                       # (B, H, T, T)

        # 4. Through softmax along the last axis.
        # dS = A * (dA - sum(dA * A, axis=-1, keepdims=True))
        # Causal-mask gradients are automatically zero at masked positions
        # because A=0 there -> dS=0 there. No re-masking needed.
        dS = c["A"] * (dA - (dA * c["A"]).sum(axis=-1, keepdims=True)) # (B, H, T, T)

        # 5. Through scaled scores: S = Q K^T / sqrt(dk).
        inv_sqrt_dk = 1.0 / np.sqrt(dk)
        dQ = (dS @ c["K"]) * inv_sqrt_dk                                # (B, H, T, dk)
        dK = (dS.transpose(0, 1, 3, 2) @ c["Q"]) * inv_sqrt_dk          # (B, H, T, dk)

        # 6. Reshape Q/K/V grads back to (B, T, D), concat into d_qkv.
        dQ_flat = dQ.transpose(0, 2, 1, 3).reshape(B, T, D)
        dK_flat = dK.transpose(0, 2, 1, 3).reshape(B, T, D)
        dV_flat = dV.transpose(0, 2, 1, 3).reshape(B, T, D)
        d_qkv = np.concatenate([dQ_flat, dK_flat, dV_flat], axis=-1)   # (B, T, 3D)

        # 7. Through QKV projection: qkv = x @ W_qkv + b_qkv.
        x_flat = x.reshape(-1, D)                                      # (BT, D)
        d_qkv_flat = d_qkv.reshape(-1, 3 * D)                          # (BT, 3D)
        self.dW_qkv = x_flat.T @ d_qkv_flat                            # (D, 3D)
        self.db_qkv = d_qkv_flat.sum(axis=0)                           # (3D,)
        return (d_qkv_flat @ self.W_qkv.T).reshape(B, T, D)            # dL/dx

    def params_and_grads(self):
        yield self.W_qkv, self.dW_qkv
        yield self.b_qkv, self.db_qkv
        yield self.W_o, self.dW_o
        yield self.b_o, self.db_o


if __name__ == "__main__":
    from src.gradcheck import numerical_gradient, relative_error, upcast_module

    rng = np.random.default_rng(0)
    attn = MultiHeadSelfAttention(d_model=16, n_heads=2, rng=rng)
    x = rng.standard_normal((2, 4, 16)).astype(np.float32)

    out = attn.forward(x)
    print(f"forward:  x{x.shape} -> out{out.shape}")
    A = attn._cache["A"]
    print(f"  attention weights sum to 1: {np.allclose(A.sum(axis=-1), 1.0, atol=1e-5)}")
    causal_violations = (A * np.triu(np.ones((4, 4), dtype=bool), k=1)).sum()
    print(f"  causal violations: {causal_violations:.6f} (must be 0)")

    # Gradient check.
    print("\nGradient check (upcasting params + input to float64)...")
    upcast_module(attn)
    x = x.astype(np.float64)

    grad_out = rng.standard_normal((2, 4, 16))                          # synthetic upstream grad
    out = attn.forward(x)
    loss = float((out * grad_out).sum())                                # synthetic loss
    dx_analytic = attn.backward(grad_out)

    def loss_fn():
        return float((attn.forward(x) * grad_out).sum())

    print(f"\n  {'param':<10}{'shape':<14}{'max rel err':>14}")
    for name, param, analytic in [
        ("W_qkv", attn.W_qkv, attn.dW_qkv),
        ("b_qkv", attn.b_qkv, attn.db_qkv),
        ("W_o",   attn.W_o,   attn.dW_o),
        ("b_o",   attn.b_o,   attn.db_o),
    ]:
        numerical = numerical_gradient(loss_fn, param, eps=1e-6)
        err = relative_error(analytic, numerical)
        flag = "PASS" if err < 1e-5 else "FAIL"
        print(f"  {name:<10}{str(param.shape):<14}{err:>14.2e}  {flag}")

    dx_num = numerical_gradient(loss_fn, x, eps=1e-6)
    err_x = relative_error(dx_analytic, dx_num)
    print(f"  {'dL/dx':<10}{str(x.shape):<14}{err_x:>14.2e}  {'PASS' if err_x < 1e-5 else 'FAIL'}")
