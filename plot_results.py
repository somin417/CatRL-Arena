"""Plot evaluation CSVs from evaluate.py."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

import settings as S


def load_csv(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def plot_results(csv_path: Path) -> None:
    rows = load_csv(csv_path)
    if not rows:
        raise ValueError(f"No rows in {csv_path}")

    episodes = [int(r["episode"]) for r in rows]
    returns = [float(r["return"]) for r in rows]
    broken = [int(r["broken_bricks"]) for r in rows]
    steps = [int(r["steps"]) for r in rows]

    out_dir = S.PLOTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = csv_path.stem

    specs = [
        ("return", returns, "Episode Return"),
        ("broken_bricks", broken, "Broken Bricks"),
        ("steps", steps, "Episode Steps"),
    ]
    for key, values, ylabel in specs:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(episodes, values, marker="o", linewidth=1.5)
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} — {stem}")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out_file = out_dir / f"{stem}_{key}.png"
        fig.savefig(out_file, dpi=120)
        plt.close(fig)
        print(f"Saved {out_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot CatBreak evaluation CSV.")
    parser.add_argument("--csv", type=str, required=True)
    args = parser.parse_args()
    plot_results(Path(args.csv))


if __name__ == "__main__":
    main()
