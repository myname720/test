"""Top-level training entrypoint.

End-to-end: load Shakespeare, build vocab, instantiate the model, train for
`--steps` mini-batch steps, save loss curves, save final weights, and
generate sample text from a few prompts.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

from src.data import build_vocab, decode, download_if_missing, encode, split_train_val
from src.model import GPT, GPTConfig, parameter_count
from src.sample import sample
from src.train import train

ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "runs"
CKPT_DIR = ROOT / "checkpoints"
SAMPLES_DIR = ROOT / "samples"


def save_checkpoint(model: GPT, path: Path) -> None:
    """Pickle every parameter array (no torch state_dict, since we have no torch)."""
    path.parent.mkdir(exist_ok=True, parents=True)
    state = {
        "config": model.config,
        "params": [p.copy() for p, _ in model.params_and_grads()],
    }
    with path.open("wb") as f:
        pickle.dump(state, f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--prompt", type=str, default="ROMEO:")
    parser.add_argument("--sample-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    with open("chem.txt", "r", encoding="utf-8") as f:
        text = f.read()
    stoi, itos = build_vocab(text)
    ids = encode(text, stoi)
    train_ids, val_ids = split_train_val(ids)
    print(f"Corpus: {len(text):,} chars, vocab {len(stoi)}")
    print(f"Train: {len(train_ids):,}  Val: {len(val_ids):,}")

    config = GPTConfig(
        vocab_size=len(stoi),
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        block_size=args.block_size,
    )
    model = GPT(config, rng=rng)
    print(f"Model: d={config.d_model}, H={config.n_heads}, L={config.n_layers}, "
          f"params={parameter_count(model):,}")
    print(f"Training {args.steps} steps at lr={args.lr}, batch={args.batch_size}\n")

    history = train(
        model, train_ids, val_ids,
        n_steps=args.steps,
        block_size=config.block_size,
        batch_size=args.batch_size,
        lr=args.lr,
        log_every=args.log_every,
        rng=rng,
        log_path=RUNS_DIR / "loss.csv",
    )

    print(f"\nSaving checkpoint -> checkpoints/final.pkl")
    save_checkpoint(model, CKPT_DIR / "final.pkl")

    print(f"\nGenerating samples (temperature={args.temperature}, top_k=40)...")
    SAMPLES_DIR.mkdir(exist_ok=True)
    samples = []
    for prompt in [args.prompt, "HAMLET:", "To be or not to be,"]:
        prompt_ids = encode(prompt, stoi)
        out_ids = sample(
            model, prompt_ids,
            max_new_tokens=args.sample_tokens,
            temperature=args.temperature,
            top_k=40,
            rng=rng,
        )
        text_out = decode(out_ids, itos)
        samples.append((prompt, text_out))
        print(f"\n----- prompt: {prompt!r} -----")
        print(text_out)

    out_path = SAMPLES_DIR / "generations.txt"
    with out_path.open("w") as f:
        for prompt, text_out in samples:
            f.write(f"===== prompt: {prompt!r} =====\n")
            f.write(text_out)
            f.write("\n\n")
    print(f"\nSamples saved to {out_path}")
    print(f"Final val_loss: {history[-1].val_loss:.4f}")


if __name__ == "__main__":
    main()
