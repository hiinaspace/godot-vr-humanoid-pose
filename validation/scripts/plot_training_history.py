"""Render a PNG summary from Torch training history JSONL files."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

DEFAULT_HISTORY_PATH = Path(".cache/ml/upper_body_models_dense_smoke/torch_mlp_history.jsonl")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("history", nargs="*", type=Path, default=[DEFAULT_HISTORY_PATH])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--title")
    args = parser.parse_args(argv)

    output_path = args.output or _default_output_path(args.history)
    render_history(args.history, output_path, title=args.title)
    print(output_path.as_posix())
    return 0


def render_history(
    history_paths: Sequence[Path], output_path: Path, *, title: str | None = None
) -> None:
    runs = [(path, _read_history(path)) for path in history_paths]
    if not runs:
        raise ValueError("At least one history path is required")
    for path, rows in runs:
        if not rows:
            raise ValueError(f"No history rows found in {path}")

    figure, axes = plt.subplots(2, 2, figsize=(17, 10), constrained_layout=True)
    figure.suptitle(title or _plot_title(runs), fontsize=14, fontweight="bold")

    _plot_metrics(
        axes[0][0],
        runs,
        (
            ("validation latent", "validation_mse"),
            ("train latent", "train_latent_mse"),
            ("train total", "train_loss"),
        ),
        "Latent And Training Loss",
        "MSE / weighted loss",
        log_scale=True,
    )
    _plot_metrics(
        axes[0][1],
        runs,
        (
            ("decoded position", "validation_decoded_position_mse"),
            ("torso", "validation_decoded_torso_mse"),
            ("shoulder", "validation_decoded_shoulder_mse"),
            ("elbow", "validation_decoded_elbow_mse"),
            ("stretch", "validation_decoded_stretch_mse"),
            ("reach hinge", "validation_decoded_reach_hinge_mse"),
        ),
        "Decoded Geometry",
        "MSE",
        log_scale=True,
    )
    _plot_metrics(
        axes[1][0],
        runs,
        (
            ("velocity", "validation_decoded_velocity_mse"),
            ("acceleration", "validation_decoded_acceleration_mse"),
            ("jerk", "validation_decoded_jerk_mse"),
            ("pole velocity", "validation_pole_velocity_mse"),
            ("pole excess jerk", "validation_pole_excess_jerk_mse"),
        ),
        "Temporal Metrics",
        "MSE",
        log_scale=True,
    )
    _plot_metrics(
        axes[1][1],
        runs,
        (
            ("epoch seconds", "epoch_seconds"),
            ("cuda allocated MiB", "cuda_memory_allocated_mib"),
            ("cuda max MiB", "cuda_max_memory_allocated_mib"),
        ),
        "Runtime And Memory",
        "seconds / MiB",
        log_scale=False,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _read_history(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _plot_metrics(
    axis,
    runs: Sequence[tuple[Path, list[dict[str, Any]]]],
    metrics: Sequence[tuple[str, str]],
    title: str,
    ylabel: str,
    *,
    log_scale: bool,
) -> None:
    plotted = False
    for run_index, (path, rows) in enumerate(runs):
        run_label = _run_label(path, run_index, len(runs))
        for label, key in metrics:
            points = _series(rows, key)
            if not points:
                continue
            epochs = [epoch for epoch, _value in points]
            values = [value for _epoch, value in points]
            axis.plot(
                epochs,
                values,
                marker="o",
                markersize=3,
                linewidth=1.5,
                label=f"{run_label}: {label}",
            )
            plotted = True
    axis.set_title(title)
    axis.set_xlabel("epoch")
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.25)
    if log_scale:
        axis.set_yscale("log")
    if plotted:
        axis.legend(fontsize=8)
    else:
        axis.text(
            0.5, 0.5, "No matching metrics", ha="center", va="center", transform=axis.transAxes
        )


def _series(rows: Sequence[Mapping[str, Any]], key: str) -> list[tuple[int, float]]:
    points: list[tuple[int, float]] = []
    for row in rows:
        value = row.get(key)
        epoch = row.get("epoch")
        if not isinstance(epoch, int) or not isinstance(value, int | float):
            continue
        numeric_value = float(value)
        if numeric_value >= 0.0:
            points.append((epoch, numeric_value))
    return points


def _default_output_path(history_paths: Sequence[Path]) -> Path:
    if len(history_paths) == 1:
        return history_paths[0].with_suffix(".png")
    return Path(".cache/ml/training_history_compare.png")


def _plot_title(runs: Sequence[tuple[Path, list[dict[str, Any]]]]) -> str:
    if len(runs) == 1:
        path, rows = runs[0]
        return f"{path.parent.name}/{path.name}: {len(rows)} rows"
    return f"Training history comparison: {len(runs)} runs"


def _run_label(path: Path, run_index: int, run_count: int) -> str:
    if run_count == 1:
        return path.stem.replace("_history", "")
    return f"{path.parent.name}:{path.stem.replace('_history', '') or run_index}"


if __name__ == "__main__":
    raise SystemExit(main())
