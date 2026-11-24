"""Tiny Shakespeare loader, character tokenizer, and minibatch iterator.

The dataset is the canonical Karpathy char-rnn corpus: ~1MB of Shakespeare's
plays concatenated, ~65 unique characters. Character-level tokenization
keeps the vocabulary tiny and avoids the complexity of subword units while
still producing recognisable text after training.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Iterator

import numpy as np

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_FILE = DATA_DIR / "tinyshakespeare.txt"


def download_if_missing(url: str = DATA_URL, target: Path = DATA_FILE) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        urllib.request.urlretrieve(url, target)
    return target


def build_vocab(text: str) -> tuple[dict[str, int], list[str]]:
    """Return (stoi, itos) — string-to-index dict and index-to-string list."""
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    return stoi, chars


def encode(text: str, stoi: dict[str, int]) -> np.ndarray:
    """Convert text to int32 token IDs."""
    return np.array([stoi[c] for c in text], dtype=np.int32)


def decode(ids: np.ndarray, itos: list[str]) -> str:
    """Convert token IDs back to text."""
    return "".join(itos[int(i)] for i in ids)


def split_train_val(ids: np.ndarray, ratio: float = 0.9) -> tuple[np.ndarray, np.ndarray]:
    n = int(ratio * len(ids))
    return ids[:n], ids[n:]


def iterate_batches(
    ids: np.ndarray,
    block_size: int,
    batch_size: int,
    *,
    rng: np.random.Generator,
    n_batches: int | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield random batches of (x, y) where y is x shifted by one position.

    Each row is a contiguous slice of `block_size` tokens; the target sequence
    is the same window shifted right by 1 (so position i predicts the token at
    position i+1). Sampling is uniform-random across the corpus, not sequential.
    """
    max_start = len(ids) - block_size - 1
    yielded = 0
    while True:
        starts = rng.integers(0, max_start, size=batch_size)
        x = np.stack([ids[s : s + block_size] for s in starts])
        y = np.stack([ids[s + 1 : s + 1 + block_size] for s in starts])
        yield x.astype(np.int64), y.astype(np.int64)
        yielded += 1
        if n_batches is not None and yielded >= n_batches:
            break


if __name__ == "__main__":
    download_if_missing()
    text = DATA_FILE.read_text()
    print(f"Corpus length: {len(text):,} characters")

    stoi, itos = build_vocab(text)
    print(f"Vocabulary size: {len(stoi)} ({''.join(itos[:30])!r}...)")

    ids = encode(text, stoi)
    train_ids, val_ids = split_train_val(ids)
    print(f"Train: {len(train_ids):,} tokens   Val: {len(val_ids):,} tokens")

    rng = np.random.default_rng(0)
    batches = iterate_batches(train_ids, block_size=64, batch_size=4, rng=rng, n_batches=2)
    for i, (x, y) in enumerate(batches):
        print(f"\nBatch {i}: x{x.shape} y{y.shape}")
        print(f"  x[0] decoded: {decode(x[0], itos)!r}")
        print(f"  y[0] decoded: {decode(y[0], itos)!r}")
