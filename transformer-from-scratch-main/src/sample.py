"""Autoregressive sampling — generate text from a trained GPT.

The basic loop: for each step, run the model on the current sequence,
take the logits at the last position, divide by `temperature`, softmax,
and sample a new token from the resulting categorical distribution.
Append it to the sequence and repeat.

No KV-cache — each step is a full forward pass. For block_size=64 and a
small model this is plenty fast (~milliseconds per token).
"""
from __future__ import annotations

import numpy as np

from src.losses import softmax
from src.model import GPT


def sample(
    model: GPT,
    prompt_ids: np.ndarray,
    *,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate up to `max_new_tokens` continuation tokens.

    Args:
        prompt_ids: 1-D array of int token IDs (length >= 1).
        temperature: divides logits before softmax. Higher = more random.
        top_k: if set, restrict sampling to the top-k most likely tokens.

    Returns:
        Full token sequence including the prompt and the continuation.
    """
    block_size = model.config.block_size
    ids = prompt_ids.astype(np.int64).copy()

    for _ in range(max_new_tokens):
        # Take the last `block_size` tokens — model can't see further back.
        ctx = ids[-block_size:]
        logits = model.forward(ctx[None, :])                   # (1, T, V)
        next_logits = logits[0, -1, :] / temperature           # (V,)

        if top_k is not None:
            # Keep top-k by setting all others to -inf so softmax zeros them.
            cutoff = np.partition(next_logits, -top_k)[-top_k]
            next_logits = np.where(next_logits >= cutoff, next_logits, -np.inf)

        probs = softmax(next_logits)
        next_id = int(rng.choice(len(probs), p=probs))
        ids = np.append(ids, next_id)

    return ids
