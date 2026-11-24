"""Mini-batch training loop for the GPT model.

Per step:
    1. sample a batch of (x, y) — random sequence windows from the corpus
    2. forward pass -> logits
    3. cross_entropy(logits, y) -> loss + grad_logits
    4. backward through the model
    5. AdamW step

Logs train and validation loss every `log_every` steps to runs/loss.csv.
"""
from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.data import iterate_batches
from src.losses import cross_entropy_loss
from src.model import GPT
from src.optim import AdamW

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"


@dataclass
class StepLog:
    step: int
    train_loss: float
    val_loss: float


def estimate_val_loss(model: GPT, val_ids: np.ndarray, *, block_size: int, batch_size: int,
                      n_batches: int, rng: np.random.Generator) -> float:
    losses = []
    for x, y in iterate_batches(val_ids, block_size, batch_size, rng=rng, n_batches=n_batches):
        logits = model.forward(x)
        loss, _ = cross_entropy_loss(logits, y)
        losses.append(loss)
    return float(np.mean(losses))


def train(
    model: GPT,
    train_ids: np.ndarray,
    val_ids: np.ndarray,
    *,
    n_steps: int,
    block_size: int,
    batch_size: int,
    lr: float,
    weight_decay: float = 0.1,
    log_every: int = 50,
    val_batches: int = 16,
    rng: np.random.Generator,
    log_path: Path | None = None,
) -> list[StepLog]:
    optimizer = AdamW(model.params_and_grads, lr=lr, weight_decay=weight_decay)
    history: list[StepLog] = []

    val_rng = np.random.default_rng(rng.integers(0, 2**32))
    train_iter = iterate_batches(train_ids, block_size, batch_size, rng=rng, n_batches=n_steps)

    t0 = time.time()
    running_loss = 0.0
    running_count = 0

    for step, (x, y) in enumerate(train_iter, start=1):
        logits = model.forward(x)
        loss, grad_logits = cross_entropy_loss(logits, y)
        model.backward(grad_logits)
        optimizer.step()

        running_loss += loss
        running_count += 1

        if step % log_every == 0 or step == n_steps:
            train_loss = running_loss / running_count
            val_loss = estimate_val_loss(
                model, val_ids,
                block_size=block_size, batch_size=batch_size,
                n_batches=val_batches, rng=val_rng,
            )
            elapsed = time.time() - t0
            steps_per_sec = step / elapsed
            print(
                f"step {step:>5}/{n_steps}  train_loss={train_loss:.4f}  "
                f"val_loss={val_loss:.4f}  "
                f"{steps_per_sec:5.1f} steps/s"
            )
            history.append(StepLog(step, train_loss, val_loss))
            running_loss = 0.0
            running_count = 0

    if log_path is not None:
        log_path.parent.mkdir(exist_ok=True, parents=True)
        with log_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "train_loss", "val_loss"])
            for h in history:
                writer.writerow([h.step, f"{h.train_loss:.6f}", f"{h.val_loss:.6f}"])

    return history
