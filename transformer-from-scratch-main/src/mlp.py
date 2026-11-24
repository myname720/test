"""Position-wise feed-forward block.

Two Linear layers with a ReLU in between, applied independently at every
position in the sequence:

    y = Linear_2(ReLU(Linear_1(x)))

The original Transformer used d_ff = 4 * d_model. The "position-wise"
qualifier just emphasises that the same MLP weights are applied to every
position — there's no cross-position mixing here, which is what attention
is for.
"""
from __future__ import annotations

import numpy as np

from src.layers import Linear, ReLU


class PositionWiseMLP:
    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        *,
        rng: np.random.Generator,
    ) -> None:
        if d_ff is None:
            d_ff = 4 * d_model
        self.fc1 = Linear(d_model, d_ff, rng=rng)
        self.relu = ReLU()
        self.fc2 = Linear(d_ff, d_model, rng=rng)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.fc2.forward(self.relu.forward(self.fc1.forward(x)))

    def backward(self, grad_y: np.ndarray) -> np.ndarray:
        return self.fc1.backward(self.relu.backward(self.fc2.backward(grad_y)))

    def params_and_grads(self):
        yield self.fc1.W, self.fc1.dW
        yield self.fc1.b, self.fc1.db
        yield self.fc2.W, self.fc2.dW
        yield self.fc2.b, self.fc2.db


if __name__ == "__main__":
    from src.gradcheck import numerical_gradient, relative_error, upcast_module

    rng = np.random.default_rng(0)
    mlp = PositionWiseMLP(d_model=16, d_ff=32, rng=rng)

    # Cast everything to float64 for the gradcheck.
    upcast_module(mlp.fc1)
    upcast_module(mlp.fc2)
    x = rng.standard_normal((2, 4, 16))

    grad_out = rng.standard_normal((2, 4, 16))
    out = mlp.forward(x)
    dx = mlp.backward(grad_out)

    def loss_fn():
        return float((mlp.forward(x) * grad_out).sum())

    print(f"PositionWiseMLP gradient check:")
    print(f"  {'param':<12}{'shape':<14}{'max rel err':>14}")
    for name, param, analytic in [
        ("fc1.W", mlp.fc1.W, mlp.fc1.dW),
        ("fc1.b", mlp.fc1.b, mlp.fc1.db),
        ("fc2.W", mlp.fc2.W, mlp.fc2.dW),
        ("fc2.b", mlp.fc2.b, mlp.fc2.db),
    ]:
        numerical = numerical_gradient(loss_fn, param, eps=1e-6)
        err = relative_error(analytic, numerical)
        flag = "PASS" if err < 1e-5 else "FAIL"
        print(f"  {name:<12}{str(param.shape):<14}{err:>14.2e}  {flag}")

    dx_num = numerical_gradient(loss_fn, x, eps=1e-6)
    err_x = relative_error(dx, dx_num)
    print(f"  {'dL/dx':<12}{str(x.shape):<14}{err_x:>14.2e}  {'PASS' if err_x < 1e-5 else 'FAIL'}")
