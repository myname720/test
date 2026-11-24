# Layer Normalisation — derivation

These notes derive the forward and backward passes of LayerNorm
(Ba et al., 2016) as implemented in `src/layer_norm.py`.

## 1. Forward

LayerNorm normalises along the **last** axis (the feature axis) for each
example independently. For an input vector $x \in \mathbb{R}^{d}$, the
operation is:

$$
\mu = \frac{1}{d} \sum_{i} x_i
\qquad
\sigma^2 = \frac{1}{d} \sum_{i} (x_i - \mu)^2
\qquad
\hat x_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}
$$

then a learnable affine transform per feature:

$$
y_i = \gamma_i \hat x_i + \beta_i
$$

with learnable scale $\gamma \in \mathbb{R}^d$ and shift $\beta \in \mathbb{R}^d$.

The point of normalising along the feature axis (not the batch axis as in
batch norm) is that it works at any batch size and at inference time
without needing running statistics — every operation is local to a single
example.

## 2. Backward — the affine part is trivial

$y = \gamma \hat x + \beta$, both elementwise, so:

$$
\frac{\partial L}{\partial \beta_i} = \frac{\partial L}{\partial y_i}
\qquad
\frac{\partial L}{\partial \gamma_i} = \frac{\partial L}{\partial y_i} \, \hat x_i
\qquad
\frac{\partial L}{\partial \hat x_i} = \frac{\partial L}{\partial y_i} \, \gamma_i
$$

Sum $\partial L / \partial \beta$ and $\partial L / \partial \gamma$
across the batch and time axes (since these parameters are shared
across positions); the per-example $\partial L / \partial \hat x$ is
what feeds the next step.

## 3. Backward — the normalisation step is fiddly

$\hat x_i = (x_i - \mu) / \sigma$ where I write $\sigma := \sqrt{\sigma^2 + \epsilon}$
for brevity. The wrinkle is that $\mu$ and $\sigma$ are functions of every
$x_i$, so a perturbation in any single $x_i$ ripples through all $\hat x_j$.

Differentiate $\hat x_i$ w.r.t. $x_j$:

$$
\frac{\partial \hat x_i}{\partial x_j}
= \frac{1}{\sigma} \left[ \delta_{ij} - \frac{1}{d} - \frac{(x_i - \mu)(x_j - \mu)}{d \sigma^2} \right]
$$

Plugging into the chain rule and collecting terms yields the standard
"three-term" expression:

$$
\boxed{\;\frac{\partial L}{\partial x_i} = \frac{1}{\sigma} \left( \frac{\partial L}{\partial \hat x_i} - \frac{1}{d} \sum_j \frac{\partial L}{\partial \hat x_j} - \frac{\hat x_i}{d} \sum_j \frac{\partial L}{\partial \hat x_j} \, \hat x_j \right)\;}
$$

Three terms:

1. $\partial L / \partial \hat x_i$ — the direct gradient.
2. $-\frac{1}{d} \sum_j \partial L / \partial \hat x_j$ — correction
   because $\mu$ depends on every $x_j$ (mean-removal contribution).
3. $-\frac{\hat x_i}{d} \sum_j (\partial L / \partial \hat x_j) \hat x_j$ —
   correction because $\sigma$ depends on every $x_j$ (variance-removal
   contribution).

## 4. Vectorisation

In code, working over the last axis only (broadcasting batch / time):

```python
x_hat = (x - mu) / sigma                     # forward output
dxhat = grad_y * gamma                       # affine backward
mean_dxhat        = dxhat.mean(axis=-1, keepdims=True)
mean_dxhat_xhat   = (dxhat * x_hat).mean(axis=-1, keepdims=True)
dx = (dxhat - mean_dxhat - x_hat * mean_dxhat_xhat) / sigma
```

Three reductions along the last axis, three broadcast subtractions —
no Python loops, all vectorised. The implementation matches this
expression line for line.

## 5. Verification

A finite-difference gradient check on a `(B=2, T=3, d=8)` input with random
γ and β passes to ~1e-7 relative error on every parameter and on
$\partial L / \partial x$ — see `src/layer_norm.py::__main__`.
