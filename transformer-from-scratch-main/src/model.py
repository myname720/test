"""Decoder-only Transformer language model.

Composes the components built in the rest of `src/`:

    token_emb + position_emb -> [TransformerBlock × N] -> LayerNorm -> Linear -> logits

The final Linear projects from `d_model` back to `vocab_size` (the
"language-model head"). It does not share weights with the token
embedding — for clarity and to keep gradient flow obvious.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.block import TransformerBlock
from src.embedding import TokenEmbedding, sinusoidal_position_encoding
from src.layer_norm import LayerNorm
from src.layers import Linear


@dataclass
class GPTConfig:
    vocab_size: int
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int | None = None     # defaults to 4 * d_model
    block_size: int = 64        # max sequence length the model can handle


class GPT:
    def __init__(self, config: GPTConfig, *, rng: np.random.Generator) -> None:
        self.config = config
        self.token_emb = TokenEmbedding(config.vocab_size, config.d_model, rng=rng)
        self.pos_enc = sinusoidal_position_encoding(config.block_size, config.d_model)
        self.blocks = [
            TransformerBlock(config.d_model, config.n_heads, d_ff=config.d_ff, rng=rng)
            for _ in range(config.n_layers)
        ]
        self.ln_f = LayerNorm(config.d_model)
        self.lm_head = Linear(config.d_model, config.vocab_size, rng=rng)

    def forward(self, idx: np.ndarray) -> np.ndarray:
        """idx: (B, T) integer tokens -> logits: (B, T, vocab_size)."""
        B, T = idx.shape
        assert T <= self.config.block_size, f"sequence length {T} > block_size {self.config.block_size}"

        x = self.token_emb.forward(idx) + self.pos_enc[:T]              # (B, T, d_model)
        for block in self.blocks:
            x = block.forward(x)
        x = self.ln_f.forward(x)
        return self.lm_head.forward(x)                                  # (B, T, vocab_size)

    def backward(self, grad_logits: np.ndarray) -> None:
        """Backprop. Gradients accumulate on every parameter; nothing returned."""
        grad = self.lm_head.backward(grad_logits)
        grad = self.ln_f.backward(grad)
        for block in reversed(self.blocks):
            grad = block.backward(grad)
        # Position encoding is constant — no backward through it.
        # Token embedding consumes the gradient and updates self.token_emb.dW.
        self.token_emb.backward(grad)

    def params_and_grads(self):
        yield self.token_emb.W, self.token_emb.dW
        for block in self.blocks:
            yield from block.params_and_grads()
        yield from self.ln_f.params_and_grads()
        yield self.lm_head.W, self.lm_head.dW
        yield self.lm_head.b, self.lm_head.db


def parameter_count(model: GPT) -> int:
    return sum(p.size for p, _ in model.params_and_grads())


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    config = GPTConfig(vocab_size=65, d_model=64, n_heads=4, n_layers=4, block_size=64)
    model = GPT(config, rng=rng)

    print(f"Model: d_model={config.d_model}, n_heads={config.n_heads}, "
          f"n_layers={config.n_layers}, vocab={config.vocab_size}")
    print(f"Parameter count: {parameter_count(model):,}")

    idx = rng.integers(0, config.vocab_size, size=(2, 16), dtype=np.int64)
    logits = model.forward(idx)
    print(f"\nforward: idx{idx.shape} -> logits{logits.shape}")
    print(f"  logits std: {logits.std():.4f}, range: [{logits.min():+.2f}, {logits.max():+.2f}]")

    # Smoke test: gradient backward runs without shape errors.
    grad = rng.standard_normal(logits.shape).astype(np.float32)
    model.backward(grad)
    print(f"\nbackward: ran without error.")
    grad_norms = [(name, np.linalg.norm(g)) for name, (p, g) in zip(
        ["tok_emb"] + [f"block{i}" for i in range(config.n_layers)] + ["ln_f", "head"],
        [(model.token_emb.W, model.token_emb.dW)]
        + [(model.blocks[i].attn.W_qkv, model.blocks[i].attn.dW_qkv) for i in range(config.n_layers)]
        + [(model.ln_f.gamma, model.ln_f.dgamma), (model.lm_head.W, model.lm_head.dW)],
    )]
    for name, norm in grad_norms:
        print(f"  ||grad {name}|| = {norm:.4f}")
