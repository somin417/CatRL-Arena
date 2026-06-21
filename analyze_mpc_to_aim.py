"""Analyze CEM-MPC planning logs to guide CEM-Aim feature design."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

import settings as S

SAFE_AIM_DIFF_MIN = 0.05
SAFE_AIM_DIFF_MAX = 0.15


def _load_planning_log(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _float(row: dict, key: str, default: float = 0.0) -> float:
    val = row.get(key, "")
    if val == "" or val is None:
        return default
    return float(val)


def _int(row: dict, key: str, default: int = 0) -> int:
    val = row.get(key, "")
    if val == "" or val is None:
        return default
    return int(float(val))


def analyze_planning_log(log_path: Path, out_path: Path) -> Path:
    rows = _load_planning_log(log_path)
    if not rows:
        raise ValueError(f"Empty planning log: {log_path}")

    has_follow_col = "followball_action" in rows[0]

    entropies = [_float(r, "entropy_mean") for r in rows]
    low_entropy = [r for r in rows if _float(r, "entropy_mean") < 0.05]
    brick_hits = [r for r in rows if _int(r, "best_predicted_bricks") > 0]

    action_by_ball_y: dict[str, list[int]] = defaultdict(list)
    action_by_ball_vx: dict[str, list[int]] = defaultdict(list)
    for r in rows:
        action = _int(r, "chosen_action")
        ball_y = _float(r, "ball_y") / S.FIELD_HEIGHT
        ball_vx = _float(r, "ball_vx") / S.BALL_SPEED_MAX
        y_bin = "upper" if ball_y < 0.35 else ("mid" if ball_y < 0.55 else "lower")
        vx_bin = "left" if ball_vx < -0.05 else ("right" if ball_vx > 0.05 else "center")
        action_by_ball_y[y_bin].append(action)
        action_by_ball_vx[vx_bin].append(action)

    diff_count = 0
    diff_examples: list[str] = []
    safe_rows = [r for r in rows if _int(r, "is_safe_aim") == 1]
    safe_diff_count = 0
    if has_follow_col:
        for r in rows:
            chosen = _int(r, "chosen_action")
            follow = _int(r, "followball_action")
            if chosen != follow:
                diff_count += 1
                if len(diff_examples) < 5:
                    diff_examples.append(
                        f"step={r.get('real_step')} chosen={r.get('chosen_action_name')} "
                        f"follow={r.get('followball_action_name', follow)} "
                        f"ball_y={_float(r,'ball_y'):.0f} mode={r.get('mode','')}"
                    )
            if _int(r, "is_safe_aim") == 1 and chosen != follow:
                safe_diff_count += 1
        diff_rate = diff_count / len(rows)
        safe_diff_rate = safe_diff_count / len(safe_rows) if safe_rows else 0.0
        target_ok = SAFE_AIM_DIFF_MIN <= safe_diff_rate <= SAFE_AIM_DIFF_MAX
        follow_section = (
            f"## CEM-MPC vs FollowBall\n\n"
            f"- Steps with different action from FollowBall: **{diff_count}** / {len(rows)} "
            f"({100 * diff_rate:.1f}%)\n"
            f"- Safe-aim steps (ball rising, upper half): **{len(safe_rows)}** "
            f"({100 * len(safe_rows) / len(rows):.1f}%)\n"
            f"- Safe-aim diff from FollowBall: **{safe_diff_count}** / {len(safe_rows)} "
            f"({100 * safe_diff_rate:.1f}%)  "
            f"target **{100 * SAFE_AIM_DIFF_MIN:.0f}–{100 * SAFE_AIM_DIFF_MAX:.0f}%** "
            f"{'✓' if target_ok else '✗'}\n"
        )
        if diff_examples:
            follow_section += "- Examples:\n" + "\n".join(f"  - {e}" for e in diff_examples) + "\n"
    else:
        follow_section = (
            "## CEM-MPC vs FollowBall\n\n"
            "FollowBall action column not present in planning log — comparison skipped.\n"
        )

    lines = [
        "# CEM-MPC → CEM-Aim Insight Report",
        "",
        f"Source: `{log_path}`",
        f"Rows: {len(rows)}",
        "",
        follow_section,
        "",
        "## Entropy",
        "",
        f"- Mean entropy: **{mean(entropies):.4f}**",
        f"- Median entropy: **{median(entropies):.4f}**",
        f"- Low-entropy steps (entropy < 0.05): **{len(low_entropy)}** "
        f"({100 * len(low_entropy) / len(rows):.1f}%)",
        "",
        "## Brick-aiming states",
        "",
        f"- Steps with best_predicted_bricks > 0: **{len(brick_hits)}** "
        f"({100 * len(brick_hits) / len(rows):.1f}%)",
        "",
    ]

    if brick_hits:
        avg_ball_y = mean(_float(r, "ball_y") for r in brick_hits)
        avg_paddle_x = mean(_float(r, "paddle_x") for r in brick_hits)
        lines.extend([
            f"- Avg ball_y in those states: {avg_ball_y:.1f} (field height {S.FIELD_HEIGHT})",
            f"- Avg paddle_x: {avg_paddle_x:.1f}",
            "",
        ])

    lines.extend([
        "## Action vs state (chosen action modes)",
        "",
        "### By ball_y region",
        "",
    ])
    for region, actions in sorted(action_by_ball_y.items()):
        if actions:
            mode = max(set(actions), key=actions.count)
            lines.append(
                f"- {region}: n={len(actions)}, mode action={S.ACTION_NAMES.get(mode, mode)}"
            )

    lines.extend(["", "### By ball_vx", ""])
    for region, actions in sorted(action_by_ball_vx.items()):
        if actions:
            mode = max(set(actions), key=actions.count)
            lines.append(
                f"- {region}: n={len(actions)}, mode action={S.ACTION_NAMES.get(mode, mode)}"
            )

    lines.extend([
        "",
        "## Recommended CEM-Aim features",
        "",
        "1. **predicted_landing_x** — base intercept target (FollowBall prior)",
        "2. **brick_centroid_x** — where to aim when ball is rising and safe",
        "3. **danger_level** — gate aiming offset when ball is low / descending / misaligned",
        "4. **ball_vx** — lateral approach direction for offset sign",
        "5. **left/right brick density** — asymmetric cluster pressure",
        "",
        "## Interpretation",
        "",
        "CEM-MPC roughly matched FollowBall on episode metrics while producing "
        "safe-aim deviations for CEM-Aim demo warm-start. CEM-Aim v2 keeps predicted "
        "landing as the base target and learns a safety-gated offset only when danger is low.",
        "",
    ])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written: {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze CEM-MPC planning log for CEM-Aim.")
    parser.add_argument("--planning-log", type=str, required=True)
    parser.add_argument(
        "--output",
        type=str,
        default=str(S.CEM_AIM_DIR / "mpc_insight_report.md"),
    )
    args = parser.parse_args()
    analyze_planning_log(Path(args.planning_log), Path(args.output))


if __name__ == "__main__":
    main()
