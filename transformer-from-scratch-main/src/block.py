"""A single Transformer decoder block (pre-LN architecture).

The 2017 paper used post-LN, but pre-LN has become the standard since
~2019 because it trains more stably without learning-rate warmup. The
flow is:

    x  ->  + ----- LN -> Attn ----- +
    |      |                        |
    |      +<------- residual ------+
    |
    +----  + ----- LN -> MLP  ----- +
           |                        |
           +<------- residual ------+
                                    |
                                   out

Backward: each residual additively splits the upstream gradient into
two paths — one straight through the residual, one through the
sublayer back to the input.
"""
from __future__ import annotations

import numpy as np

from src.attention import MultiHeadSelfAttention
from src.layer_norm import LayerNorm
from src.mlp import PositionWiseMLP


class TransformerBlock:
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int | None = None,
        *,
        rng: np.random.Generator,
    ) -> None:
        self.ln1 = LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, rng=rng)
        self.ln2 = LayerNorm(d_model)
        self.mlp = PositionWiseMLP(d_model, d_ff=d_ff, rng=rng)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = x + self.attn.forward(self.ln1.forward(x))
        x = x + self.mlp.forward(self.ln2.forward(x))
        return x

    def backward(self, grad_y: np.ndarray) -> np.ndarray:
        # Gradient through second residual (x + mlp(ln2(x))):
        #   the residual passes grad_y through unchanged, the sublayer branch
        #   receives grad_y and propagates back through mlp -> ln2 -> x.
        grad_x = grad_y + self.ln2.backward(self.mlp.backward(grad_y))

        # Gradient through first residual (x + attn(ln1(x))):
        grad_x = grad_x + self.ln1.backward(self.attn.backward(grad_x))

        return grad_x

    def params_and_grads(self):
        yield from self.ln1.params_and_grads()
        yield from self.attn.params_and_grads()
        yield from self.ln2.params_and_grads()
        yield from self.mlp.params_and_grads()


if __name__ == "__main__":
    from src.gradcheck import numerical_gradient, relative_error, upcast_module

    rng = np.random.default_rng(0)
    block = TransformerBlock(d_model=16, n_heads=2, d_ff=32, rng=rng)

    # Upcast all sub-modules.
    for sub in (block.ln1, block.attn, block.ln2, block.mlp.fc1, block.mlp.fc2):
        upcast_module(sub)

    x = rng.standard_normal((2, 4, 16))
    grad_out = rng.standard_normal((2, 4, 16))

    out = block.forward(x)
    dx = block.backward(grad_out)

    def loss_fn():
        return float((block.forward(x) * grad_out).sum())

    print(f"TransformerBlock gradient check:")
    print(f"  {'param':<14}{'shape':<14}{'max rel err':>14}")
    for name, (param, analytic) in zip(
        ["ln1.gamma", "ln1.beta",
         "attn.W_qkv", "attn.b_qkv", "attn.W_o", "attn.b_o",
         "ln2.gamma", "ln2.beta",
         "mlp.fc1.W", "mlp.fc1.b", "mlp.fc2.W", "mlp.fc2.b"],
        list(block.params_and_grads()),
    ):
        numerical = numerical_gradient(loss_fn, param, eps=1e-6)
        err = relative_error(analytic, numerical)
        flag = "PASS" if err < 1e-5 else "FAIL"
        print(f"  {name:<14}{str(param.shape):<14}{err:>14.2e}  {flag}")

    dx_num = numerical_gradient(loss_fn, x, eps=1e-6)
    err_x = relative_error(dx, dx_num)
    print(f"  {'dL/dx':<14}{str(x.shape):<14}{err_x:>14.2e}  {'PASS' if err_x < 1e-5 else 'FAIL'}")
