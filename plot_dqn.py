"""Plot DQN training and evaluation curves."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import settings as S

PLOT_SPECS = [
    ("train_episodes.csv", "return", "training_return.png", "Training Return"),
    ("train_episodes.csv", "broken_bricks", "training_broken_bricks.png", "Training Broken Bricks"),
    ("train_episodes.csv", "loss", "training_loss.png", "Training Loss"),
    ("eval_history.csv", "clear_rate", "eval_clear_rate.png", "Eval Clear Rate"),
    ("eval_history.csv", "avg_blocks_per_100_steps", "eval_blocks_per_100_steps.png", "Eval Blocks / 100 Steps"),
    ("eval_history.csv", "avg_return", "eval_avg_return.png", "Eval Avg Return"),
]

DEFAULT_RUNS = {
    "dqn": S.DQN_DIR,
    "dqn_per": S.DQN_PER_DIR,
    "dqn_cnn": S.DQN_CNN_DIR,
}

RUN_LABELS = {
    "dqn": "DQN (MLP)",
    "dqn_per": "DQN+PER",
    "dqn_cnn": "DQN-CNN",
}

RUN_COLORS = {
    "dqn": "#388bfd",
    "dqn_per": "#d29922",
    "dqn_cnn": "#63c174",
}


def load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def run_label(run_dir: Path) -> str:
    name = run_dir.name
    return RUN_LABELS.get(name, name)


def rolling_mean(values: list[float], window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr
    window = max(1, min(window, arr.size))
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(arr, kernel, mode="valid")


def _x_y(rows: list[dict], column: str) -> tuple[list[int], list[float]]:
    if not rows or column not in rows[0]:
        return [], []
    x_key = "episode" if "episode" in rows[0] else "eval_index"
    xs = [int(r[x_key]) for r in rows]
    ys = [float(r[column]) for r in rows]
    return xs, ys


def _plot_series(
    ax,
    xs,
    ys,
    title: str,
    column: str,
    xlabel: str,
    label: str | None = None,
    color: str | None = None,
) -> None:
    if not xs:
        return
    plot_kwargs = {"linewidth": 1.5}
    if label:
        plot_kwargs["label"] = label
    if color:
        plot_kwargs["color"] = color
    ax.plot(xs, ys, **plot_kwargs)
    ylab = "Clear Rate (%)" if column == "clear_rate" else title
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylab)
    ax.grid(True, alpha=0.3)


def save_single_plot(
    run_dir: Path,
    csv_name: str,
    column: str,
    filename: str,
    title: str,
) -> Path | None:
    rows = load_csv(run_dir / csv_name)
    xs, ys = _x_y(rows, column)
    if not xs:
        print(f"WARNING: skipping {run_dir.name}/{filename} — missing {csv_name}:{column}")
        return None

    if column == "clear_rate":
        ys = [y * 100 for y in ys]

    fig, ax = plt.subplots(figsize=(8, 4))
    label = run_label(run_dir)
    xlabel = "episode" if csv_name == "train_episodes.csv" else "eval episode"
    _plot_series(ax, xs, ys, title, column, xlabel, label=label)
    fig.suptitle(f"{label} — {title}", fontsize=12)
    fig.tight_layout()
    out_path = run_dir / "plots" / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path


def plot_dqn(run_dir: Path) -> Path:
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    label = run_label(run_dir)

    for csv_name, column, filename, title in PLOT_SPECS:
        save_single_plot(run_dir, csv_name, column, filename, title)

    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    fig.suptitle(f"{label} Training Dashboard", fontsize=14)
    axes_flat = axes.flatten()

    plot_idx = 0
    for csv_name, column, _filename, title in PLOT_SPECS:
        rows = load_csv(run_dir / csv_name)
        xs, ys = _x_y(rows, column)
        if not xs or plot_idx >= len(axes_flat):
            continue
        if column == "clear_rate":
            ys = [y * 100 for y in ys]
        ax = axes_flat[plot_idx]
        ax.plot(xs, ys, linewidth=1.5, color=RUN_COLORS.get(run_dir.name, "#388bfd"))
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("episode" if csv_name == "train_episodes.csv" else "eval episode")
        ax.set_ylabel("Clear Rate (%)" if column == "clear_rate" else title)
        ax.grid(True, alpha=0.3)
        plot_idx += 1

    for j in range(plot_idx, len(axes_flat)):
        axes_flat[j].axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = plots_dir / "dqn_dashboard.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path


def plot_dqn_comparison(
    run_dirs: dict[str, Path],
    out_dir: Path | None = None,
    smooth_window: int = 50,
) -> Path:
    out_dir = out_dir or (S.PLOTS_DIR / "dqn_compare")
    out_dir.mkdir(parents=True, exist_ok=True)

    comparison_specs = [
        ("train_episodes.csv", "return", "compare_training_return.png", "Training Return"),
        ("train_episodes.csv", "broken_bricks", "compare_training_broken_bricks.png", "Training Broken Bricks"),
        ("train_episodes.csv", "loss", "compare_training_loss.png", "Training Loss"),
        ("eval_history.csv", "avg_return", "compare_eval_return.png", "Eval Avg Return"),
        ("eval_history.csv", "clear_rate", "compare_eval_clear_rate.png", "Eval Clear Rate"),
        ("eval_history.csv", "avg_blocks_per_100_steps", "compare_eval_blocks_per_100.png", "Eval Blocks / 100 Steps"),
    ]

    for csv_name, column, filename, title in comparison_specs:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        has_data = False
        for key, run_dir in run_dirs.items():
            rows = load_csv(run_dir / csv_name)
            xs, ys = _x_y(rows, column)
            if not xs:
                print(f"WARNING: compare skip {key} — no {csv_name}:{column}")
                continue
            if column == "clear_rate":
                ys = [y * 100 for y in ys]
            label = RUN_LABELS.get(key, key)
            color = RUN_COLORS.get(key, None)
            ax.plot(xs, ys, alpha=0.25, linewidth=0.8, color=color)
            smooth = rolling_mean(ys, smooth_window)
            if smooth.size:
                smooth_x = xs[smooth_window - 1 :]
                ax.plot(smooth_x, smooth, linewidth=2.0, label=label, color=color)
                has_data = True
        if has_data:
            ax.set_title(f"{title} — DQN vs DQN+PER vs DQN-CNN", fontsize=12)
            ax.set_xlabel("episode" if "train" in csv_name else "eval episode")
            ax.set_ylabel("Clear Rate (%)" if column == "clear_rate" else title)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", fontsize=9)
            fig.tight_layout()
            out_path = out_dir / filename
            fig.savefig(out_path, dpi=120)
            plt.close(fig)
            print(f"Saved {out_path}")
        else:
            plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("DQN / DQN+PER / DQN-CNN Comparison", fontsize=14)
    for ax, (csv_name, column, _filename, title) in zip(axes.flatten(), comparison_specs):
        has_data = False
        for key, run_dir in run_dirs.items():
            rows = load_csv(run_dir / csv_name)
            xs, ys = _x_y(rows, column)
            if not xs:
                continue
            if column == "clear_rate":
                ys = [y * 100 for y in ys]
            label = RUN_LABELS.get(key, key)
            color = RUN_COLORS.get(key, None)
            smooth = rolling_mean(ys, smooth_window)
            if smooth.size:
                smooth_x = xs[smooth_window - 1 :]
                ax.plot(smooth_x, smooth, linewidth=1.8, label=label, color=color)
                has_data = True
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("episode" if "train" in csv_name else "eval episode")
        ax.grid(True, alpha=0.3)
        if has_data:
            ax.legend(fontsize=7, loc="best")
        else:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    dashboard_path = out_dir / "compare_dashboard.png"
    fig.savefig(dashboard_path, dpi=120)
    plt.close(fig)
    print(f"Saved {dashboard_path}")
    return dashboard_path


def plot_all_default_runs() -> None:
    for key, run_dir in DEFAULT_RUNS.items():
        if not run_dir.exists():
            print(f"WARNING: missing run dir {run_dir}")
            continue
        if not (run_dir / "train_episodes.csv").exists():
            print(f"WARNING: no train_episodes.csv in {run_dir}")
            continue
        print(f"Plotting {key} ...")
        plot_dqn(run_dir)

    available = {
        key: path
        for key, path in DEFAULT_RUNS.items()
        if (path / "train_episodes.csv").exists()
    }
    if len(available) >= 2:
        print("Plotting comparison ...")
        plot_dqn_comparison(available)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot DQN training runs and comparisons.")
    parser.add_argument("--run-dir", type=str, default=None, help="Plot a single run directory")
    parser.add_argument("--all", action="store_true", help="Plot dqn, dqn_per, dqn_cnn")
    parser.add_argument("--compare", action="store_true", help="Plot comparison across default runs")
    parser.add_argument(
        "--compare-out-dir",
        type=str,
        default=str(S.PLOTS_DIR / "dqn_compare"),
    )
    parser.add_argument("--smooth-window", type=int, default=50)
    args = parser.parse_args()

    if args.all:
        plot_all_default_runs()
        return

    if args.compare:
        available = {
            key: path
            for key, path in DEFAULT_RUNS.items()
            if (path / "train_episodes.csv").exists()
        }
        plot_dqn_comparison(available, Path(args.compare_out_dir), args.smooth_window)
        return

    run_dir = Path(args.run_dir or S.DQN_DIR)
    plot_dqn(run_dir)


if __name__ == "__main__":
    main()
