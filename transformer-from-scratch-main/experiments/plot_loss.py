"""Plot the training/validation loss curves saved to runs/loss.csv."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
LOSS_CSV = ROOT / "runs" / "loss.csv"
FIG_DIR = ROOT / "figures"


def main() -> None:
    if not LOSS_CSV.exists():
        raise FileNotFoundError(f"{LOSS_CSV} — run `python train.py` first.")

    steps, train_loss, val_loss = [], [], []
    with LOSS_CSV.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row["step"]))
            train_loss.append(float(row["train_loss"]))
            val_loss.append(float(row["val_loss"]))

    FIG_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(steps, train_loss, marker="o", markersize=4, color="tab:blue", label="train")
    ax.plot(steps, val_loss, marker="s", markersize=4, color="tab:orange", label="validation")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title(f"Tiny Shakespeare -- final val loss {val_loss[-1]:.3f}")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right")

    fig.tight_layout()
    out = FIG_DIR / "loss_curves.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
