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
    clips = cast(list[dict[str, Any]], payload.get("clips", []))
    grouped = cast(dict[str, list[dict[str, Any]]], summary.get("grouped_metrics", {}))
    worst = cast(dict[str, list[dict[str, Any]]], summary.get("worst_clips", {}))

    figure, axes = plt.subplots(3, 3, figsize=(22, 15), constrained_layout=True)
    figure.suptitle(_title(summary), fontsize=15, fontweight="bold")

    _plot_overall_metrics(axes[0][0], summary)
    _plot_clip_distribution(axes[0][1], clips)
    _plot_plausibility_distribution(axes[0][2], clips)
    _plot_group_metric(
        axes[1][0],
        grouped.get("environment_split", []),
        "environment_split",
        "By Environment",
    )
    _plot_group_metric(
        axes[1][1],
        grouped.get("posture_subset", []) or grouped.get("activity_class", []),
        "posture_subset" if grouped.get("posture_subset") else "activity_class",
        "By Posture Slice" if grouped.get("posture_subset") else "By Activity",
    )
    _plot_clip_scatter(axes[1][2], clips)
    _plot_head_turn_scatter(axes[2][0], clips)
    _plot_worst_clips(
        axes[2][1],
        worst.get("mean_full_body_reference_error_m")
        or worst.get("reference_error_p95_m")
        or worst.get("mean_reference_error_m", []),
        "Worst Clips: Full-Body Error",
        "mean_full_body_reference_error_m",
    )
    _plot_worst_clips(
        axes[2][2],
        worst.get("avatar_implausibility_score_0_100", []),
        "Worst Clips: Avatar Implausibility",
        "avatar_implausibility_score_0_100",
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
        (
            "full body",
            _metric_value(
                summary,
                "mean_full_body_reference_error_m",
                "mean_reference_error_m",
            ),
        ),
        (
            "upper",
            _metric_value(
                summary,
                "mean_upper_body_reference_error_m",
                "mean_torso_error_m",
            ),
        ),
        (
            "lower",
            _metric_value(
                summary,
                "mean_lower_body_reference_error_m",
                "mean_reference_error_m",
            ),
        ),
        ("pelvis", float(summary.get("mean_pelvis_error_m", 0.0))),
        ("hand", float(summary.get("mean_hand_reference_error_m", 0.0))),
        ("shoulder", float(summary.get("mean_shoulder_error_m", 0.0))),
        ("elbow", float(summary.get("mean_elbow_error_m", 0.0))),
        ("knee", float(summary.get("mean_knee_error_m", 0.0))),
        ("ref p95", float(summary.get("reference_error_p95_m", 0.0))),
        ("contact foot", float(summary.get("mean_contact_foot_error_m", 0.0))),
        ("contact slip/s", float(summary.get("mean_contact_slip_m_per_s", 0.0))),
        ("low-foot slip/s", float(summary.get("mean_low_foot_slip_m_per_s", 0.0))),
        ("support dist", float(summary.get("mean_hips_support_distance_m", 0.0))),
    ]
    labels = [label for label, _value in metrics]
    values = [value for _label, value in metrics]
    colors = [
        "#4c78a8",
        "#72b7b2",
        "#54a24b",
        "#9c755f",
        "#f58518",
        "#e45756",
        "#ff9da6",
        "#b279a2",
        "#af7aa1",
        "#59a14f",
        "#edc948",
    ]
    positions = list(range(len(values)))
    bars = axis.barh(positions, values, color=colors[: len(values)])
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_title("Headline Means")
    axis.set_xlabel("meters / meters per second")
    axis.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)


def _plot_clip_distribution(axis, clips: list[dict[str, Any]]) -> None:
    metric_specs = [
        ("full body", "mean_full_body_reference_error_m"),
        ("upper", "mean_upper_body_reference_error_m"),
        ("lower", "mean_lower_body_reference_error_m"),
        ("pelvis", "mean_pelvis_error_m"),
        ("hand", "mean_hand_reference_error_m"),
        ("shoulder", "mean_shoulder_error_m"),
        ("elbow", "mean_elbow_error_m"),
        ("knee", "mean_knee_error_m"),
        ("ref p95", "reference_error_p95_m"),
        ("contact foot", "mean_contact_foot_error_m"),
        ("low-foot slip", "mean_low_foot_slip_m_per_s"),
        ("support dist", "mean_hips_support_distance_m"),
    ]
    series = [_metric_values(clips, key) for _label, key in metric_specs]
    labels = [label for label, _key in metric_specs]
    nonempty = [(label, values) for label, values in zip(labels, series, strict=True) if values]
    if not nonempty:
        axis.set_title("Per-Clip Distributions")
        axis.text(0.5, 0.5, "No clip metrics", ha="center", va="center")
        axis.axis("off")
        return

    labels = [label for label, _values in nonempty]
    series = [values for _label, values in nonempty]
    positions = list(range(1, len(series) + 1))
    parts = axis.violinplot(series, positions=positions, showmeans=True, showextrema=True)
    for body in parts["bodies"]:
        body.set_facecolor("#4c78a8")
        body.set_edgecolor("#2f4b68")
        body.set_alpha(0.32)
    for key in ("cmeans", "cmins", "cmaxes", "cbars"):
        parts[key].set_color("#2f4b68")

    for position, values in zip(positions, series, strict=True):
        sample = _sample_values(values, limit=350)
        offsets = _jitter_offsets(len(sample), width=0.12)
        axis.scatter(
            [position + offset for offset in offsets],
            sample,
            s=5,
            alpha=0.22,
            color="#111111",
            linewidths=0,
        )
    axis.set_xticks(positions, labels, rotation=25, ha="right")
    axis.set_title("Per-Clip Error Distributions")
    axis.set_ylabel("meters")
    axis.grid(axis="y", alpha=0.2)


def _plot_plausibility_distribution(axis, clips: list[dict[str, Any]]) -> None:
    metric_specs = [
        ("estimator", "avatar_implausibility_score_0_100"),
        ("GT/reference", "reference_avatar_implausibility_score_0_100"),
    ]
    series = [_metric_values(clips, key) for _label, key in metric_specs]
    labels = [label for label, _key in metric_specs]
    nonempty = [(label, values) for label, values in zip(labels, series, strict=True) if values]
    axis.set_title("Avatar Implausibility Distribution")
    if not nonempty:
        axis.text(0.5, 0.5, "No plausibility metrics", ha="center", va="center")
        axis.axis("off")
        return

    labels = [label for label, _values in nonempty]
    series = [values for _label, values in nonempty]
    parts = axis.violinplot(
        series,
        positions=list(range(1, len(series) + 1)),
        showmeans=True,
        showextrema=True,
    )
    colors = ("#e45756", "#4c78a8")
    for color, body in zip(colors, parts["bodies"], strict=False):
        body.set_facecolor(color)
        body.set_edgecolor("#333333")
        body.set_alpha(0.34)
    for key in ("cmeans", "cmins", "cmaxes", "cbars"):
        parts[key].set_color("#333333")

    for position, values in enumerate(series, start=1):
        quantiles = _quantiles(values, (0.5, 0.9, 0.95, 0.99))
        axis.text(
            position + 0.08,
            min(100.0, quantiles[0.99] + 2.0),
            (
                f"p50 {quantiles[0.5]:.1f}\n"
                f"p90 {quantiles[0.9]:.1f}\n"
                f"p95 {quantiles[0.95]:.1f}\n"
                f"p99 {quantiles[0.99]:.1f}"
            ),
            fontsize=8,
            va="bottom",
        )
    axis.set_xticks(list(range(1, len(labels) + 1)), labels)
    axis.set_ylabel("implausibility score (0-100)")
    axis.set_ylim(bottom=0.0)
    axis.grid(axis="y", alpha=0.2)


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
        key=lambda group: _metric_value(
            group,
            "mean_full_body_reference_error_m",
            "mean_reference_error_m",
        ),
        reverse=True,
    )
    labels = [str(group.get(label_key, "unknown")) for group in groups]
    full_body = [
        _metric_value(group, "mean_full_body_reference_error_m", "mean_reference_error_m")
        for group in groups
    ]
    upper = [
        _metric_value(group, "mean_upper_body_reference_error_m", "mean_torso_error_m")
        for group in groups
    ]
    lower = [
        _metric_value(group, "mean_lower_body_reference_error_m", "mean_reference_error_m")
        for group in groups
    ]
    positions = range(len(groups))
    axis.barh(
        [position + 0.24 for position in positions], full_body, height=0.22, label="full body"
    )
    axis.barh([position for position in positions], upper, height=0.22, label="upper")
    axis.barh([position - 0.24 for position in positions], lower, height=0.22, label="lower")
    axis.set_yticks(list(positions), labels)
    axis.invert_yaxis()
    axis.set_title(title)
    axis.set_xlabel("mean error (m)")
    axis.legend(fontsize=8)


def _plot_clip_scatter(axis, clips: list[dict[str, Any]]) -> None:
    if not clips:
        axis.set_title("Velocity Error vs Jerk")
        axis.text(0.5, 0.5, "No clip metrics", ha="center", va="center")
        axis.axis("off")
        return
    colors = {
        "flat": "#4c78a8",
        "nonflat": "#e45756",
        "unknown": "#9c755f",
    }
    plotted = False
    for split, color in colors.items():
        split_clips = [clip for clip in clips if str(clip.get("environment_split")) == split]
        x_values, y_values = _metric_pairs(
            split_clips,
            "mean_joint_velocity_error_m_per_s",
            "mean_joint_estimated_jerk_m_per_s3",
        )
        if not x_values or not y_values:
            continue
        axis.scatter(
            x_values,
            y_values,
            s=8,
            alpha=0.35,
            color=color,
            label=split,
            linewidths=0,
        )
        plotted = True
    axis.set_title("Velocity Error vs Jerk")
    if not plotted:
        axis.text(0.5, 0.5, "No temporal metrics", ha="center", va="center")
        axis.axis("off")
        return
    axis.set_xlabel("mean joint velocity error (m/s)")
    axis.set_ylabel("mean estimated jerk (m/s^3)")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)


def _plot_head_turn_scatter(axis, clips: list[dict[str, Any]]) -> None:
    x_values, y_values = _metric_pairs(
        clips,
        "max_reference_head_forward_angular_speed_deg_s",
        "max_head_forward_angular_speed_deg_s",
    )
    if not x_values or not y_values:
        axis.set_title("Head Angular Speed")
        axis.text(0.5, 0.5, "No head angular metrics", ha="center", va="center")
        axis.axis("off")
        return
    colors = [
        "#e45756"
        if float(clip.get("pct_estimated_fast_head_turns_without_reference_motion", 0.0)) >= 25.0
        else "#4c78a8"
        for clip in clips
        if isinstance(clip.get("max_reference_head_forward_angular_speed_deg_s"), (int, float))
        and isinstance(clip.get("max_head_forward_angular_speed_deg_s"), (int, float))
    ]
    axis.scatter(x_values, y_values, s=9, alpha=0.36, color=colors, linewidths=0)
    limit = max(max(x_values, default=0.0), max(y_values, default=0.0), 720.0)
    axis.plot([0.0, limit], [0.0, limit], color="#333333", linewidth=1.0, alpha=0.5)
    axis.axhline(720.0, color="#e45756", linewidth=0.8, alpha=0.35)
    axis.axvline(720.0, color="#4c78a8", linewidth=0.8, alpha=0.35)
    axis.set_xlim(left=0.0)
    axis.set_ylim(bottom=0.0)
    axis.set_title("Estimated vs GT Head Angular Speed")
    axis.set_xlabel("GT max head speed (deg/s)")
    axis.set_ylabel("estimated max head speed (deg/s)")
    axis.grid(alpha=0.2)


def _plot_worst_clips(
    axis,
    clips: list[dict[str, Any]],
    title: str,
    metric_key: str = "mean_reference_error_m",
) -> None:
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
            f"{clip.get('activity_class', '?')} | "
            f"{clip.get('posture_subset', '?')}"
        )
        for clip in clips
    ]
    values = [_clip_metric_value(clip, metric_key) for clip in clips]
    bars = axis.barh(labels, values, color="#e45756")
    axis.set_title(title)
    axis.set_xlabel("error (m)")
    axis.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)


def _metric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def _metric_value(row: dict[str, Any], key: str, *fallback_keys: str) -> float:
    for candidate_key in (key, *fallback_keys):
        value = row.get(candidate_key)
        if isinstance(value, (int, float)):
            return float(value)
    return 0.0


def _clip_metric_value(clip: dict[str, Any], metric_key: str) -> float:
    return _metric_value(clip, metric_key, "reference_error_p95_m", "mean_reference_error_m")


def _metric_pairs(
    rows: list[dict[str, Any]], x_key: str, y_key: str
) -> tuple[list[float], list[float]]:
    x_values = []
    y_values = []
    for row in rows:
        x_value = row.get(x_key)
        y_value = row.get(y_key)
        if isinstance(x_value, (int, float)) and isinstance(y_value, (int, float)):
            x_values.append(float(x_value))
            y_values.append(float(y_value))
    return x_values, y_values


def _sample_values(values: list[float], *, limit: int) -> list[float]:
    if len(values) <= limit:
        return values
    step = max(1, len(values) // limit)
    return values[::step][:limit]


def _quantiles(values: list[float], quantiles: Sequence[float]) -> dict[float, float]:
    sorted_values = sorted(values)
    if not sorted_values:
        return {quantile: 0.0 for quantile in quantiles}
    result = {}
    for quantile in quantiles:
        position = (len(sorted_values) - 1) * min(1.0, max(0.0, quantile))
        lower = int(position)
        upper = min(len(sorted_values) - 1, lower + 1)
        blend = position - lower
        result[quantile] = sorted_values[lower] * (1.0 - blend) + sorted_values[upper] * blend
    return result


def _jitter_offsets(count: int, *, width: float) -> list[float]:
    if count <= 1:
        return [0.0] * count
    return [((index * 37) % count) / (count - 1) * width * 2.0 - width for index in range(count)]


if __name__ == "__main__":
    raise SystemExit(main())
