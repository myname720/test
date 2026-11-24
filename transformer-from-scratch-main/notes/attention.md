# Scaled dot-product attention — derivation

These notes derive the attention operation as it appears in
[Vaswani et al. (2017)](https://arxiv.org/abs/1706.03762), starting from the
underlying problem and ending at the equation implemented in
`src/attention.py`. Each section corresponds to one piece of the eventual
implementation.

## 1. The problem attention is solving

A sequence model processing $T$ tokens needs each position to access
information from every other position. RNNs do this by passing a hidden
state through time — information from token 1 reaches token $T$ only after
$T-1$ recurrence steps, which causes the vanishing gradient problem and
prevents parallel computation.

Attention solves both problems by allowing every position to read directly
from every other position, in parallel, in a single layer.

## 2. Notation

For a single sequence of length $T$ with embedding dimension $d$:

- $X \in \mathbb{R}^{T \times d}$ — input token embeddings (one row per token).
- $Q = X W_Q$, $K = X W_K$, $V = X W_V$ — queries, keys, values, where each
  $W_\bullet \in \mathbb{R}^{d \times d_k}$ is a learned linear projection.
- $d_k$ is the per-head dimension ($d_k = d / H$ where $H$ is the number of heads).

Reading the metaphor: each token *queries* the sequence for relevant
information; each token also exposes a *key* (its address) and a *value*
(its contents). The attention weights compare a query to every key and use
the resulting similarities to mix the values.

## 3. Single-head scaled dot-product attention

The forward operation:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{Q K^\top}{\sqrt{d_k}} \right) V
$$

Step by step for one query row $q_i \in \mathbb{R}^{d_k}$:

1. **Compatibility scores** between $q_i$ and every key:
   $s_i = q_i K^\top \in \mathbb{R}^{T}$ — a scalar similarity per token.
2. **Scale** by $\sqrt{d_k}$ (Section 4 derives why):
   $\tilde{s}_i = s_i / \sqrt{d_k}$.
3. **Normalise** to a probability distribution:
   $\alpha_i = \text{softmax}(\tilde{s}_i) \in \mathbb{R}^{T}$, with $\sum_t \alpha_{i,t} = 1$.
4. **Mix the values**: $\text{out}_i = \sum_t \alpha_{i,t} v_t = \alpha_i V$.

Stacking over all $T$ queries gives the matrix form above.

## 4. Why scale by $\sqrt{d_k}$?

This is the most-asked attention interview question. The argument is about
the variance of the dot product.

Assume each component of $q_i$ and $k_j$ is an independent zero-mean
unit-variance random variable. Then for a single dot product

$$
q_i \cdot k_j = \sum_{l=1}^{d_k} q_{i,l} \, k_{j,l}
$$

each term $q_{i,l} k_{j,l}$ has mean zero and variance one (variance of a
product of independent zero-mean unit-variance variables). Summing $d_k$
such independent terms:

$$
\mathbb{E}[q_i \cdot k_j] = 0
\qquad
\text{Var}(q_i \cdot k_j) = d_k
$$

So the **standard deviation of the dot product grows as $\sqrt{d_k}$**.
For $d_k = 64$ that's $\sigma \approx 8$ — large enough that softmax saturates
into a near-one-hot distribution where almost all gradient flows through one
position. Backprop through near-saturated softmax has vanishing gradients
on the unselected positions.

Dividing by $\sqrt{d_k}$ rescales the variance back to $\text{Var} = 1$, which
keeps the softmax in its non-saturated regime regardless of $d_k$.

## 5. Causal masking

For a decoder language model, position $i$ must not see positions $j > i$
(future tokens). The mask is applied to the scaled scores **before** softmax:

$$
\tilde{s}_{i,j} \leftarrow
\begin{cases}
\tilde{s}_{i,j} & \text{if } j \le i \\\\
-\infty & \text{otherwise}
\end{cases}
$$

After softmax, the $-\infty$ entries map to $0$, so $\alpha_{i,j} = 0$ for
$j > i$. Importantly the mask is applied on the scores, not the weights —
zeroing weights post-softmax doesn't preserve the probability simplex.

In practice $-\infty$ is approximated by a very large negative number
(e.g. $-10^9$) for numerical reasons.

## 6. Multi-head attention

A single attention "head" can only emphasise one type of relationship at a
time (e.g. syntactic agreement). Multi-head attention runs $H$ attention
operations in parallel, each with its own learned projections, and
concatenates their outputs:

$$
\text{MultiHead}(X) = \text{Concat}(\text{head}_1, \ldots, \text{head}_H) W_O
$$

where each

$$
\text{head}_h = \text{Attention}(X W_Q^{(h)}, X W_K^{(h)}, X W_V^{(h)})
$$

with $W_Q^{(h)}, W_K^{(h)}, W_V^{(h)} \in \mathbb{R}^{d \times d_k}$ and
the output projection $W_O \in \mathbb{R}^{H d_k \times d}$. We choose
$d_k = d / H$ so the total parameter count is comparable to a
single-head version with full $d \times d$ projections.

**Implementation note.** In the code, the $H$ heads are computed jointly by
reshaping a single $(T, d)$ projection into $(T, H, d_k)$, transposing to
$(H, T, d_k)$, doing batched attention across the head axis, then
reshaping back. This gives one batched matmul per QKV projection rather
than $H$ small matmuls — much faster on real hardware.

## 7. Position encoding

Attention is permutation-invariant: shuffling the rows of $X$ shuffles the
rows of the output identically. We have to inject position information
explicitly, before the first attention layer.

The original paper uses **sinusoidal** position encodings. For position
$pos \in \{0, \ldots, T-1\}$ and dimension $i \in \{0, \ldots, d-1\}$:

$$
\text{PE}(pos, 2i)   = \sin\!\left(\frac{pos}{10000^{2i/d}}\right)
\qquad
\text{PE}(pos, 2i+1) = \cos\!\left(\frac{pos}{10000^{2i/d}}\right)
$$

The position encoding is **added** to the token embedding:
$X \leftarrow X + \text{PE}$.

Two properties this design buys:

1. **Bounded values.** Each component lies in $[-1, 1]$, so adding it
   doesn't overwhelm the token embedding.
2. **Linear shift relation.** For any fixed offset $k$, $\text{PE}(pos+k)$
   can be written as a linear function of $\text{PE}(pos)$ (using the
   sin/cos angle-addition identities). This lets the model learn relative
   position relationships through linear projections.

A learned position embedding (a separate lookup table indexed by position)
is also viable and is what GPT-2 uses. The trade-off is that learned
embeddings can't extrapolate to sequences longer than seen during training.
Sinusoidal encodings extrapolate by construction, at the cost of some
expressivity.

This implementation uses sinusoidal encodings; learned encodings are
trivially swappable.

## 8. Complexity

For sequence length $T$ and dimension $d$:

- **Attention:** $\mathcal{O}(T^2 d)$ — the $QK^\top$ product is the $T^2$
  bottleneck. Memory is $\mathcal{O}(T^2)$ for the attention matrix.
- **Position-wise MLP:** $\mathcal{O}(T \cdot d^2)$ — independent of sequence
  length per position.

For small $T$ (our case: $T = 128$) the MLP dominates. For $T \gg d$
attention dominates and motivates the entire field of efficient-attention
research (FlashAttention, linear attention, sliding-window, etc.) — all
out of scope for this implementation.

## 9. Forward implementation summary

The eventual `attention.forward(X)` will:

1. Compute $Q, K, V$ via three linear projections of $X$.
2. Reshape into $H$ heads.
3. Compute scaled scores $S = QK^\top / \sqrt{d_k}$.
4. Apply causal mask (set upper triangle to $-\infty$).
5. Apply softmax along the last axis to get $\alpha$.
6. Compute output $O = \alpha V$.
7. Concatenate heads and apply output projection $W_O$.

The backward pass is straightforward chain rule through these steps; it is
derived alongside the implementation in a follow-up commit
(`notes/attention_backward.md`).

## References

- Vaswani et al., *Attention Is All You Need* (2017),
  https://arxiv.org/abs/1706.03762
- Phuong & Hutter, *Formal Algorithms for Transformers* (2022),
  https://arxiv.org/abs/2207.09238 — concise pseudocode reference.
