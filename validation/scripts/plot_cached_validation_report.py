"""Render a skimmable PNG summary from a cached validation JSON report."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

DEFAULT_REPORT_PATH = Path("validation/reports/cached_baseline_smoke.json")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", nargs="?", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    output_path = args.output or args.report.with_suffix(".png")
    render_report(args.report, output_path)
    print(output_path.as_posix())
    return 0


def render_report(report_path: Path, output_path: Path) -> None:
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    summary = cast(dict[str, Any], payload.get("summary", {}))
    grouped = cast(dict[str, list[dict[str, Any]]], summary.get("grouped_metrics", {}))
    worst = cast(dict[str, list[dict[str, Any]]], summary.get("worst_clips", {}))

    figure, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    figure.suptitle(_title(summary), fontsize=15, fontweight="bold")

    _plot_overall_metrics(axes[0][0], summary)
    _plot_group_metric(
        axes[0][1],
        grouped.get("environment_split", []),
        "environment_split",
        "By Environment",
    )
    _plot_group_metric(
        axes[1][0],
        grouped.get("activity_class", []),
        "activity_class",
        "By Activity",
    )
    _plot_worst_clips(
        axes[1][1],
        worst.get("mean_reference_error_m", []),
        "Worst Clips: Reference Error",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _title(summary: dict[str, Any]) -> str:
    frames = int(summary.get("evaluated_frames", 0))
    seconds = float(summary.get("total_seconds", 0.0))
    estimator = str(summary.get("estimator", "unknown"))
    return f"{estimator} cached validation: {frames:,} frames in {seconds:.1f}s"


def _plot_overall_metrics(axis, summary: dict[str, Any]) -> None:
    metrics = [
        ("reference", float(summary.get("mean_reference_error_m", 0.0))),
        ("pelvis", float(summary.get("mean_pelvis_error_m", 0.0))),
        ("pelvis xz", float(summary.get("mean_pelvis_horizontal_error_m", 0.0))),
        ("pelvis y", float(summary.get("mean_pelvis_vertical_error_m", 0.0))),
        ("torso", float(summary.get("mean_torso_error_m", 0.0))),
        ("spine", float(summary.get("mean_spine_error_m", 0.0))),
        ("chest", float(summary.get("mean_chest_error_m", 0.0))),
        ("left foot", float(summary.get("mean_left_foot_error_m", 0.0))),
        ("right foot", float(summary.get("mean_right_foot_error_m", 0.0))),
        ("contact foot", float(summary.get("mean_contact_foot_error_m", 0.0))),
        ("contact slip/s", float(summary.get("mean_contact_slip_m_per_s", 0.0))),
    ]
    labels = [label for label, _value in metrics]
    values = [value for _label, value in metrics]
    colors = [
        "#4c78a8",
        "#72b7b2",
        "#8cd17d",
        "#b6992d",
        "#54a24b",
        "#59a14f",
        "#9c755f",
        "#f58518",
        "#e45756",
        "#b279a2",
        "#af7aa1",
    ]
    bars = axis.bar(labels, values, color=colors[: len(values)])
    axis.set_title("Overall Metrics")
    axis.set_ylabel("meters / meters per second")
    axis.tick_params(axis="x", rotation=30)
    axis.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)


def _plot_group_metric(
    axis,
    groups: list[dict[str, Any]],
    label_key: str,
    title: str,
) -> None:
    if not groups:
        axis.set_title(title)
        axis.text(0.5, 0.5, "No grouped metrics", ha="center", va="center")
        axis.axis("off")
        return
    groups = sorted(
        groups,
        key=lambda group: float(group.get("mean_reference_error_m", 0.0)),
        reverse=True,
    )
    labels = [str(group.get(label_key, "unknown")) for group in groups]
    reference = [float(group.get("mean_reference_error_m", 0.0)) for group in groups]
    torso = [float(group.get("mean_torso_error_m", 0.0)) for group in groups]
    contact = [float(group.get("mean_contact_foot_error_m", 0.0)) for group in groups]
    positions = range(len(groups))
    axis.barh(
        [position + 0.24 for position in positions], reference, height=0.22, label="reference"
    )
    axis.barh([position for position in positions], torso, height=0.22, label="torso")
    axis.barh(
        [position - 0.24 for position in positions], contact, height=0.22, label="contact foot"
    )
    axis.set_yticks(list(positions), labels)
    axis.invert_yaxis()
    axis.set_title(title)
    axis.set_xlabel("mean error (m)")
    axis.legend(fontsize=8)


def _plot_worst_clips(axis, clips: list[dict[str, Any]], title: str) -> None:
    if not clips:
        axis.set_title(title)
        axis.text(0.5, 0.5, "No worst-clip entries", ha="center", va="center")
        axis.axis("off")
        return
    clips = list(reversed(clips[:10]))
    labels = [
        (
            f"{clip.get('clip', 'unknown')} | "
            f"{clip.get('environment_split', '?')} | "
            f"{clip.get('activity_class', '?')}"
        )
        for clip in clips
    ]
    values = [float(clip.get("mean_reference_error_m", 0.0)) for clip in clips]
    bars = axis.barh(labels, values, color="#e45756")
    axis.set_title(title)
    axis.set_xlabel("mean reference error (m)")
    axis.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)


if __name__ == "__main__":
    raise SystemExit(main())
