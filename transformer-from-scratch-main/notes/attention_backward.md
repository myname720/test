# Multi-head attention — backward pass

These notes derive the gradient flowing through `MultiHeadSelfAttention`
in `src/attention.py`. Each step here corresponds to a line in the
`backward()` method. The forward pass is in `notes/attention.md`.

We use the convention that `backward()` receives `dL/d(output)` and
returns `dL/d(input)`, with parameter gradients accumulated on the layer.

## 1. Variables and shapes

Recap from forward, for a single layer with batch size $B$, sequence
length $T$, model dimension $d$, $H$ heads, $d_k = d/H$:

| Symbol | Shape | Meaning |
|---|---|---|
| $x$ | $(B, T, d)$ | layer input |
| $W_{qkv}$ | $(d, 3d)$ | combined QKV projection |
| $b_{qkv}$ | $(3d,)$ | combined QKV bias |
| $\text{qkv}$ | $(B, T, 3d)$ | $x W_{qkv} + b_{qkv}$ |
| $Q, K, V$ | $(B, H, T, d_k)$ | reshaped per-head queries/keys/values |
| $S$ | $(B, H, T, T)$ | scaled scores $Q K^\top / \sqrt{d_k}$ (post-mask) |
| $A$ | $(B, H, T, T)$ | softmax of $S$ along last axis |
| $\text{av}$ | $(B, H, T, d_k)$ | $A V$ |
| $\text{concat}$ | $(B, T, d)$ | reshaped $\text{av}$ |
| $W_o, b_o$ | $(d, d), (d,)$ | output projection |
| $\text{out}$ | $(B, T, d)$ | $\text{concat} \, W_o + b_o$ |

## 2. Output projection

$\text{out} = \text{concat} \, W_o + b_o$, so by the standard linear-layer rule:

$$
\frac{\partial L}{\partial W_o} = \text{concat}^\top \frac{\partial L}{\partial \text{out}}
\qquad
\frac{\partial L}{\partial b_o} = \sum_{B, T} \frac{\partial L}{\partial \text{out}}
\qquad
\frac{\partial L}{\partial \text{concat}} = \frac{\partial L}{\partial \text{out}} \, W_o^\top
$$

Reshape $\partial L / \partial \text{concat}$ from $(B, T, d)$ back into
$(B, H, T, d_k)$ to get $\partial L / \partial \text{av}$.

## 3. Through the attention application

$\text{av} = A V$. From the standard matmul rule:

$$
\frac{\partial L}{\partial V} = A^\top \frac{\partial L}{\partial \text{av}}
\qquad
\frac{\partial L}{\partial A} = \frac{\partial L}{\partial \text{av}} \, V^\top
$$

(The transposes act on the last two axes; the head and batch axes are
broadcast.)

## 4. Through the softmax — the trickiest step

For each row $i$ of $A$ along the last axis, $A_{ij} = e^{S_{ij}} / \sum_l e^{S_{il}}$.
The Jacobian of softmax has a non-trivial off-diagonal structure:

$$
\frac{\partial A_{ij}}{\partial S_{ik}} = A_{ij} (\delta_{jk} - A_{ik})
$$

Applying the chain rule, we get for the gradient through one row:

$$
\frac{\partial L}{\partial S_{ik}}
= \sum_j \frac{\partial L}{\partial A_{ij}} \cdot A_{ij} (\delta_{jk} - A_{ik})
= A_{ik} \left( \frac{\partial L}{\partial A_{ik}} - \sum_j \frac{\partial L}{\partial A_{ij}} A_{ij} \right)
$$

In vectorised form (broadcasting over batch and head axes):

$$
\frac{\partial L}{\partial S} = A \odot \left( \frac{\partial L}{\partial A} - \left( \frac{\partial L}{\partial A} \odot A \right)\!\!.\text{sum}(\text{axis}{=}{-}1, \text{keepdims})\right)
$$

The expression in parentheses is the row-wise centred gradient — it
subtracts each row's softmax-weighted mean before scaling by $A$.

### Causal mask: gradients at masked positions are automatically zero

At masked positions $j > i$, $S_{ij} = -\infty$ (in practice, $-10^9$),
so $A_{ij} = 0$. Then the formula above gives
$\partial L / \partial S_{ij} = 0 \cdot (\ldots) = 0$ at those positions.
**No re-masking is needed in the backward pass.**

## 5. Through the scaled dot-product

$S = Q K^\top / \sqrt{d_k}$. Taking the differential:

$$
\frac{\partial L}{\partial Q} = \frac{1}{\sqrt{d_k}} \, \frac{\partial L}{\partial S} \, K
\qquad
\frac{\partial L}{\partial K} = \frac{1}{\sqrt{d_k}} \, \left(\frac{\partial L}{\partial S}\right)^\top Q
$$

Again the transposes act on the last two axes only.

## 6. Reshape back to per-position vectors

We had $Q = q_{\text{flat}}.\text{reshape}(B, T, H, d_k).\text{transpose}(0, 2, 1, 3)$
in forward. Reversing this:

$$
q_{\text{flat}} = \text{reshape}(\text{transpose}(\partial L / \partial Q, (0, 2, 1, 3)))
$$

with shape $(B, T, d)$ each for $\partial L / \partial Q$,
$\partial L / \partial K$, $\partial L / \partial V$.

Then the gradient w.r.t. the combined QKV projection output is:

$$
\frac{\partial L}{\partial \text{qkv}} = \text{concat}\!\left(\frac{\partial L}{\partial Q},\ \frac{\partial L}{\partial K},\ \frac{\partial L}{\partial V}\right)
$$

along the last axis, shape $(B, T, 3d)$.

## 7. Through the QKV projection

$\text{qkv} = x W_{qkv} + b_{qkv}$, standard linear:

$$
\frac{\partial L}{\partial W_{qkv}} = x^\top \frac{\partial L}{\partial \text{qkv}}
\qquad
\frac{\partial L}{\partial b_{qkv}} = \sum_{B, T} \frac{\partial L}{\partial \text{qkv}}
\qquad
\frac{\partial L}{\partial x} = \frac{\partial L}{\partial \text{qkv}} \, W_{qkv}^\top
$$

Returned by `backward()` and propagated to whatever fed this layer.

## 8. Verification

The implementation is verified against a finite-difference numerical
gradient on a small model (`d_model=16`, `n_heads=2`, `B=2`, `T=4`):

  - max relative error on every parameter < 1e-5 (with float64 upcast)
  - max relative error on $\partial L / \partial x$ < 1e-5

This establishes that every step above is implemented correctly and
the implementation matches the math.
