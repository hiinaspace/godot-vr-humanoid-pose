"""Analyze CMU gait statistics against sparse-VR procedural gait assumptions."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from math import atan2, degrees, sqrt
from pathlib import Path
from statistics import fmean
from typing import Any, cast

import matplotlib
import polars as pl

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from validation.vr_pose_validation.baseline import create_estimator
from validation.vr_pose_validation.basis_estimator import BasisPortEstimator
from validation.vr_pose_validation.cmu_metadata import ActivityClass, EnvironmentSplit
from validation.vr_pose_validation.geometry import (
    Vector3,
    cross,
    dot,
    horizontal_distance,
    length,
    normalize,
    rotate_vector,
    subtract,
)
from validation.vr_pose_validation.parquet_cache import (
    filter_clip_cache_files_by_environment,
    list_clip_cache_files,
)

DEFAULT_CACHE_DIR = Path(".cache/cmu/parquet/canonical_full")
DEFAULT_REPORT_PATH = Path("validation/reports/cmu_gait_flat_locomotion.json")
DEFAULT_STEPS_CSV_PATH = Path("validation/reports/cmu_gait_flat_locomotion_steps.csv")
DEFAULT_PLOT_PATH = Path("validation/reports/cmu_gait_flat_locomotion.png")
FOOT_NAMES = ("left_foot", "right_foot")
FEATURE_COLUMNS = (
    "mean_head_speed_m_s",
    "mean_head_height_m",
    "head_yaw_rate_deg_s",
    "mean_hand_speed_m_s",
    "hand_speed_asymmetry_m_s",
    "mean_hand_separation_m",
    "mean_hand_height_m",
    "hands_forward_offset_m",
)
ORACLE_FEATURE_COLUMNS = (
    *FEATURE_COLUMNS,
    "mean_hip_speed_m_s",
    "hip_turn_rate_deg_s",
)
PREDICTED_TARGETS = (
    "step_abs_forward_m",
    "step_width_m",
    "step_interval_s",
    "swing_duration_s",
)
WORLD_UP: Vector3 = (0.0, 1.0, 0.0)
WORLD_FORWARD: Vector3 = (0.0, 0.0, 1.0)


@dataclass(frozen=True, slots=True)
class ContactSegment:
    foot_name: str
    start_index: int
    end_index: int
    start_time_s: float
    end_time_s: float
    mean_position: Vector3

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_time_s - self.start_time_s)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--steps-csv", type=Path, default=DEFAULT_STEPS_CSV_PATH)
    parser.add_argument("--plot", type=Path, default=DEFAULT_PLOT_PATH)
    parser.add_argument("--max-clips", type=int)
    parser.add_argument("--subjects", nargs="*")
    parser.add_argument(
        "--environment-split",
        choices=("flat", "nonflat", "unknown"),
        nargs="*",
        default=("flat",),
    )
    parser.add_argument(
        "--activity-class",
        choices=(
            "idle_standing",
            "sitting",
            "stationary_upper_body",
            "locomotion",
            "dance_exercise",
            "swimming",
            "floor_pose",
            "sports",
            "unknown",
        ),
        nargs="*",
        default=("locomotion",),
    )
    parser.add_argument("--contact-height-threshold-m", type=float, default=0.08)
    parser.add_argument("--contact-speed-threshold-m-s", type=float, default=0.8)
    parser.add_argument("--min-contact-duration-s", type=float, default=0.04)
    parser.add_argument("--model-holdout-folds", type=int, default=5)
    args = parser.parse_args(argv)

    report, steps = analyze_cmu_gait(
        cache_dir=args.cache_dir,
        subjects=set(args.subjects or []),
        environment_splits=cast(set[EnvironmentSplit], set(args.environment_split or [])),
        activity_classes=cast(set[ActivityClass], set(args.activity_class or [])),
        max_clips=args.max_clips,
        contact_height_threshold_m=args.contact_height_threshold_m,
        contact_speed_threshold_m_s=args.contact_speed_threshold_m_s,
        min_contact_duration_s=args.min_contact_duration_s,
        model_holdout_folds=max(2, args.model_holdout_folds),
    )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if steps:
        args.steps_csv.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(steps).write_csv(args.steps_csv)

    render_gait_plot(report, steps, args.plot)
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0


def analyze_cmu_gait(
    *,
    cache_dir: Path,
    subjects: set[str] | None = None,
    environment_splits: set[EnvironmentSplit] | None = None,
    activity_classes: set[ActivityClass] | None = None,
    max_clips: int | None = None,
    contact_height_threshold_m: float = 0.08,
    contact_speed_threshold_m_s: float = 0.8,
    min_contact_duration_s: float = 0.04,
    model_holdout_folds: int = 5,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    paths = list_clip_cache_files(cache_dir, subjects or None)
    paths = filter_clip_cache_files_by_environment(
        paths,
        environment_splits=environment_splits,
        activity_classes=activity_classes,
    )
    if max_clips is not None:
        paths = paths[:max_clips]

    step_rows: list[dict[str, object]] = []
    clip_rows: list[dict[str, object]] = []
    total_frames = 0
    for path in paths:
        dataframe = pl.read_parquet(path)
        total_frames += dataframe.height
        clip_steps = extract_clip_gait_steps(
            dataframe,
            contact_height_threshold_m=contact_height_threshold_m,
            contact_speed_threshold_m_s=contact_speed_threshold_m_s,
            min_contact_duration_s=min_contact_duration_s,
        )
        step_rows.extend(clip_steps)
        clip_rows.append(_clip_summary(dataframe, clip_steps))

    summary = _gait_summary(
        cache_dir=cache_dir,
        paths_count=len(paths),
        total_frames=total_frames,
        environment_splits=environment_splits or set(),
        activity_classes=activity_classes or set(),
        step_rows=step_rows,
        model_holdout_folds=model_holdout_folds,
        contact_height_threshold_m=contact_height_threshold_m,
        contact_speed_threshold_m_s=contact_speed_threshold_m_s,
        min_contact_duration_s=min_contact_duration_s,
    )
    return {
        "summary": summary,
        "clip_summaries": clip_rows,
    }, step_rows


def extract_clip_gait_steps(
    dataframe: pl.DataFrame,
    *,
    contact_height_threshold_m: float = 0.08,
    contact_speed_threshold_m_s: float = 0.18,
    min_contact_duration_s: float = 0.04,
) -> list[dict[str, object]]:
    if dataframe.is_empty():
        return []

    columns = dataframe.to_dict(as_series=False)
    times = [float(value) for value in columns["timestamp_s"]]
    floor_height = _floor_height(columns)
    segments = [
        segment
        for foot_name in FOOT_NAMES
        for segment in _contact_segments(
            foot_name,
            times,
            _positions(columns, foot_name),
            floor_height_m=floor_height,
            contact_height_threshold_m=contact_height_threshold_m,
            contact_speed_threshold_m_s=contact_speed_threshold_m_s,
            min_contact_duration_s=min_contact_duration_s,
        )
    ]
    segments.sort(key=lambda segment: (segment.start_time_s, segment.foot_name))
    if not segments:
        return []

    subject = str(columns["subject"][0])
    clip = str(columns["clip"][0])
    metadata = {
        "subject": subject,
        "clip": clip,
        "environment_split": str(columns.get("environment_split", ["unknown"])[0]),
        "activity_class": str(columns.get("activity_class", ["unknown"])[0]),
        "motion_description": str(columns.get("motion_description", [""])[0]),
    }
    positions = {
        "hips": _positions(columns, "hips"),
        "head": _positions(columns, "head"),
        "left_hand": _positions(columns, "left_hand"),
        "right_hand": _positions(columns, "right_hand"),
    }

    previous_by_foot: dict[str, ContactSegment] = {}
    previous_step_time_s: float | None = None
    step_rows: list[dict[str, object]] = []
    for segment in segments:
        opposite_segment = previous_by_foot.get(_opposite_foot(segment.foot_name))
        same_foot_previous = previous_by_foot.get(segment.foot_name)
        if opposite_segment is None or opposite_segment.start_time_s >= segment.start_time_s:
            previous_by_foot[segment.foot_name] = segment
            continue

        frame_index = segment.start_index
        forward = _travel_forward(
            frame_index,
            times,
            positions["hips"],
            _rotations(columns, "head"),
        )
        right = normalize(cross(forward, WORLD_UP), fallback=(1.0, 0.0, 0.0))
        delta = subtract(segment.mean_position, opposite_segment.mean_position)
        signed_forward = dot(delta, forward)
        step_interval_s = (
            segment.start_time_s - previous_step_time_s
            if previous_step_time_s is not None
            else None
        )
        swing_duration_s = (
            segment.start_time_s - same_foot_previous.end_time_s
            if same_foot_previous is not None
            else None
        )
        step_rows.append(
            {
                **metadata,
                "foot": segment.foot_name,
                "frame_index": int(frame_index),
                "timestamp_s": segment.start_time_s,
                "contact_duration_s": segment.duration_s,
                "step_forward_m": signed_forward,
                "step_abs_forward_m": abs(signed_forward),
                "step_width_m": abs(dot(delta, right)),
                "step_distance_m": horizontal_distance(
                    segment.mean_position,
                    opposite_segment.mean_position,
                ),
                "step_interval_s": step_interval_s,
                "cadence_steps_per_min": 60.0 / step_interval_s
                if step_interval_s is not None and step_interval_s > 0.0
                else None,
                "swing_duration_s": swing_duration_s,
                **_sparse_features(frame_index, times, positions, forward),
            }
        )
        previous_by_foot[segment.foot_name] = segment
        previous_step_time_s = segment.start_time_s

    return step_rows


def render_gait_plot(
    report: dict[str, object],
    step_rows: list[dict[str, object]],
    output_path: Path,
) -> None:
    summary = cast(dict[str, Any], report.get("summary", {}))
    assumptions = cast(dict[str, float], summary.get("basis_footstep_assumptions", {}))
    figure, axes = plt.subplots(2, 3, figsize=(20, 10), constrained_layout=True)
    figure.suptitle(_plot_title(summary), fontsize=15, fontweight="bold")

    _plot_histogram(
        axes[0][0],
        step_rows,
        "step_abs_forward_m",
        "Step Forward Distance",
        "meters",
        markers={
            "forward lead": assumptions.get("footstep_forward_lead_m", 0.0),
            "max step": assumptions.get("footstep_max_step_length_m", 0.0),
        },
    )
    _plot_histogram(
        axes[0][1],
        step_rows,
        "step_width_m",
        "Step Width",
        "meters",
        markers={
            "1.8m stance": assumptions.get("stance_width_m_at_1p8", 0.0),
        },
    )
    _plot_histogram(
        axes[0][2],
        step_rows,
        "step_interval_s",
        "Step Interval",
        "seconds",
        markers={
            "swing duration": assumptions.get("step_duration_s", 0.0),
            "min interval": assumptions.get("footstep_min_interval_s", 0.0),
        },
    )
    _plot_scatter_with_bins(
        axes[1][0],
        step_rows,
        x_key="mean_head_speed_m_s",
        y_key="step_abs_forward_m",
        title="Head Speed vs Step Length",
        y_label="step forward distance (m)",
    )
    _plot_scatter_with_bins(
        axes[1][1],
        step_rows,
        x_key="mean_head_speed_m_s",
        y_key="step_interval_s",
        title="Head Speed vs Step Interval",
        y_label="seconds",
    )
    _plot_predictor_r2(axes[1][2], cast(dict[str, Any], summary.get("predictor_results", {})))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _contact_segments(
    foot_name: str,
    times: list[float],
    positions: list[Vector3],
    *,
    floor_height_m: float,
    contact_height_threshold_m: float,
    contact_speed_threshold_m_s: float,
    min_contact_duration_s: float,
) -> list[ContactSegment]:
    contact_flags = []
    for index, position in enumerate(positions):
        dt = max(times[index] - times[index - 1], 1.0 / 120.0) if index > 0 else 1.0 / 120.0
        previous = positions[index - 1] if index > 0 else position
        horizontal_speed = horizontal_distance(position, previous) / dt
        vertical_speed = abs(position[1] - previous[1]) / dt
        contact_flags.append(
            position[1] - floor_height_m <= contact_height_threshold_m
            and horizontal_speed <= contact_speed_threshold_m_s
            and vertical_speed <= contact_speed_threshold_m_s
        )

    segments: list[ContactSegment] = []
    start_index: int | None = None
    for index, contact in enumerate(contact_flags):
        if contact and start_index is None:
            start_index = index
        if start_index is not None and (not contact or index == len(contact_flags) - 1):
            end_index = index - 1 if not contact else index
            segment = _make_contact_segment(foot_name, start_index, end_index, times, positions)
            if segment.duration_s >= min_contact_duration_s:
                segments.append(segment)
            start_index = None
    return segments


def _make_contact_segment(
    foot_name: str,
    start_index: int,
    end_index: int,
    times: list[float],
    positions: list[Vector3],
) -> ContactSegment:
    segment_positions = positions[start_index : end_index + 1]
    return ContactSegment(
        foot_name=foot_name,
        start_index=start_index,
        end_index=end_index,
        start_time_s=times[start_index],
        end_time_s=times[end_index],
        mean_position=(
            fmean(position[0] for position in segment_positions),
            fmean(position[1] for position in segment_positions),
            fmean(position[2] for position in segment_positions),
        ),
    )


def _sparse_features(
    center_index: int,
    times: list[float],
    positions: dict[str, list[Vector3]],
    forward: Vector3,
) -> dict[str, float]:
    start_index, end_index = _window_indices(center_index, times, window_s=0.30)
    head_positions = positions["head"][start_index : end_index + 1]
    left_hand_positions = positions["left_hand"][start_index : end_index + 1]
    right_hand_positions = positions["right_hand"][start_index : end_index + 1]
    hip_positions = positions["hips"][start_index : end_index + 1]
    hand_speeds = [
        _mean_speed(times, left_hand_positions, start_index=start_index),
        _mean_speed(times, right_hand_positions, start_index=start_index),
    ]
    hand_center = _mean_position(
        [
            (
                (left[0] + right[0]) * 0.5,
                (left[1] + right[1]) * 0.5,
                (left[2] + right[2]) * 0.5,
            )
            for left, right in zip(left_hand_positions, right_hand_positions, strict=True)
        ]
    )
    head_center = _mean_position(head_positions)
    return {
        "mean_head_speed_m_s": _mean_speed(times, head_positions, start_index=start_index),
        "mean_head_height_m": fmean(position[1] for position in head_positions),
        "head_yaw_rate_deg_s": _yaw_rate(times, head_positions, start_index, end_index),
        "mean_hand_speed_m_s": fmean(hand_speeds),
        "hand_speed_asymmetry_m_s": abs(hand_speeds[0] - hand_speeds[1]),
        "mean_hand_separation_m": fmean(
            horizontal_distance(left, right)
            for left, right in zip(left_hand_positions, right_hand_positions, strict=True)
        ),
        "mean_hand_height_m": fmean(
            (left[1] + right[1]) * 0.5
            for left, right in zip(left_hand_positions, right_hand_positions, strict=True)
        ),
        "hands_forward_offset_m": dot(subtract(hand_center, head_center), forward),
        "mean_hip_speed_m_s": _mean_speed(times, hip_positions, start_index=start_index),
        "hip_turn_rate_deg_s": _yaw_rate(times, hip_positions, start_index, end_index),
    }


def _gait_summary(
    *,
    cache_dir: Path,
    paths_count: int,
    total_frames: int,
    environment_splits: set[EnvironmentSplit],
    activity_classes: set[ActivityClass],
    step_rows: list[dict[str, object]],
    model_holdout_folds: int,
    contact_height_threshold_m: float,
    contact_speed_threshold_m_s: float,
    min_contact_duration_s: float,
) -> dict[str, object]:
    step_frame = pl.DataFrame(step_rows) if step_rows else pl.DataFrame()
    settings = create_estimator("basis_footstep")
    assumptions = _basis_footstep_assumptions(settings)
    return {
        "cache_dir": cache_dir.as_posix(),
        "clips": paths_count,
        "frames": total_frames,
        "steps": len(step_rows),
        "environment_splits": sorted(environment_splits),
        "activity_classes": sorted(activity_classes),
        "contact_height_threshold_m": contact_height_threshold_m,
        "contact_speed_threshold_m_s": contact_speed_threshold_m_s,
        "min_contact_duration_s": min_contact_duration_s,
        "basis_footstep_assumptions": assumptions,
        "distribution": _distribution_summary(step_rows),
        "by_speed_bin": _speed_bin_summary(step_frame),
        "predictor_results": _predictor_results(step_rows, holdout_folds=model_holdout_folds),
    }


def _basis_footstep_assumptions(estimator: object) -> dict[str, float]:
    if not isinstance(estimator, BasisPortEstimator):
        return {}
    settings = estimator.settings
    return {
        "stance_width_ratio": settings.stance_width_ratio,
        "stance_width_m_at_1p8": settings.stance_width_ratio * 1.8,
        "footstep_release_stretch_m": settings.footstep_release_stretch_m,
        "footstep_speed_release_reduction_s": settings.footstep_speed_release_reduction_s,
        "footstep_forward_lead_m": settings.footstep_forward_lead_m,
        "footstep_speed_lead_s": settings.footstep_speed_lead_s,
        "footstep_max_forward_lead_m": settings.footstep_max_forward_lead_m,
        "footstep_max_step_length_m": settings.footstep_max_step_length_m,
        "footstep_min_interval_s": settings.footstep_min_interval_s,
        "footstep_trailing_release_stretch_m": settings.footstep_trailing_release_stretch_m,
        "footstep_catchup_release_stretch_m": settings.footstep_catchup_release_stretch_m,
        "step_duration_s": settings.step_duration_s,
        "footstep_speed_duration_reduction_s": settings.footstep_speed_duration_reduction_s,
    }


def _distribution_summary(step_rows: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    metrics = (
        "step_abs_forward_m",
        "step_forward_m",
        "step_width_m",
        "step_distance_m",
        "step_interval_s",
        "cadence_steps_per_min",
        "swing_duration_s",
        "contact_duration_s",
        "mean_head_speed_m_s",
        "mean_hand_speed_m_s",
        "mean_hip_speed_m_s",
    )
    return {metric: _numeric_summary(_row_values(step_rows, metric)) for metric in metrics}


def _speed_bin_summary(step_frame: pl.DataFrame) -> list[dict[str, object]]:
    if step_frame.is_empty():
        return []
    frame = step_frame.with_columns(
        pl.when(pl.col("mean_head_speed_m_s") < 0.25)
        .then(pl.lit("0 idle-ish"))
        .when(pl.col("mean_head_speed_m_s") < 0.75)
        .then(pl.lit("1 slow"))
        .when(pl.col("mean_head_speed_m_s") < 1.5)
        .then(pl.lit("2 walk"))
        .otherwise(pl.lit("3 fast"))
        .alias("head_speed_bin")
    )
    return cast(
        list[dict[str, object]],
        frame.group_by("head_speed_bin")
        .agg(
            pl.len().alias("steps"),
            pl.col("step_abs_forward_m").mean().alias("mean_step_abs_forward_m"),
            pl.col("step_abs_forward_m").median().alias("median_step_abs_forward_m"),
            pl.col("step_width_m").mean().alias("mean_step_width_m"),
            pl.col("step_interval_s").mean().alias("mean_step_interval_s"),
            pl.col("cadence_steps_per_min").mean().alias("mean_cadence_steps_per_min"),
        )
        .sort("head_speed_bin")
        .to_dicts(),
    )


def _predictor_results(
    step_rows: list[dict[str, object]],
    *,
    holdout_folds: int,
) -> dict[str, dict[str, dict[str, float]]]:
    feature_sets = {
        "head_speed_only": ("mean_head_speed_m_s",),
        "upper_body": FEATURE_COLUMNS,
        "upper_body_plus_oracle_hip": ORACLE_FEATURE_COLUMNS,
    }
    results: dict[str, dict[str, dict[str, float]]] = {}
    for target in PREDICTED_TARGETS:
        results[target] = {}
        for feature_set_name, features in feature_sets.items():
            results[target][feature_set_name] = _fit_linear_regression(
                step_rows,
                target=target,
                features=features,
                holdout_folds=holdout_folds,
            )
    return results


def _fit_linear_regression(
    rows: list[dict[str, object]],
    *,
    target: str,
    features: Sequence[str],
    holdout_folds: int,
    ridge: float = 1e-4,
) -> dict[str, float]:
    usable_rows = [
        row
        for row in rows
        if _float_or_none(row.get(target)) is not None
        and all(_float_or_none(row.get(feature)) is not None for feature in features)
    ]
    if len(usable_rows) < 20:
        return {"rows": float(len(usable_rows)), "test_r2": 0.0, "test_mae": 0.0}

    train_rows = [
        row for row in usable_rows if _subject_fold(str(row["subject"]), holdout_folds) != 0
    ]
    test_rows = [
        row for row in usable_rows if _subject_fold(str(row["subject"]), holdout_folds) == 0
    ]
    if not train_rows or not test_rows:
        split_index = max(1, int(len(usable_rows) * 0.8))
        train_rows = usable_rows[:split_index]
        test_rows = usable_rows[split_index:]

    feature_means = [
        fmean(cast(float, _float_or_none(row[feature])) for row in train_rows)
        for feature in features
    ]
    feature_scales = [
        _stddev(
            [cast(float, _float_or_none(row[feature])) for row in train_rows],
            mean=mean,
        )
        for feature, mean in zip(features, feature_means, strict=True)
    ]
    train_x = [
        _standardized_features(row, features, feature_means, feature_scales) for row in train_rows
    ]
    train_y = [cast(float, _float_or_none(row[target])) for row in train_rows]
    coefficients = _ridge_coefficients(train_x, train_y, ridge=ridge)

    predictions = [
        _predict(coefficients, _standardized_features(row, features, feature_means, feature_scales))
        for row in test_rows
    ]
    actual = [cast(float, _float_or_none(row[target])) for row in test_rows]
    train_mean = fmean(train_y)
    baseline_predictions = [train_mean] * len(actual)
    return {
        "rows": float(len(usable_rows)),
        "train_rows": float(len(train_rows)),
        "test_rows": float(len(test_rows)),
        "test_r2": _r2(actual, predictions),
        "test_mae": _mae(actual, predictions),
        "baseline_mae": _mae(actual, baseline_predictions),
        "test_rmse": _rmse(actual, predictions),
    }


def _ridge_coefficients(
    x_rows: list[list[float]], y_values: list[float], *, ridge: float
) -> list[float]:
    width = len(x_rows[0]) + 1
    matrix = [[0.0 for _ in range(width)] for _ in range(width)]
    rhs = [0.0 for _ in range(width)]
    for features, y_value in zip(x_rows, y_values, strict=True):
        row = [1.0, *features]
        for i, left_value in enumerate(row):
            rhs[i] += left_value * y_value
            for j, right_value in enumerate(row):
                matrix[i][j] += left_value * right_value
    for index in range(1, width):
        matrix[index][index] += ridge
    return _solve_linear_system(matrix, rhs)


def _solve_linear_system(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    size = len(rhs)
    augmented = [row[:] + [rhs_value] for row, rhs_value in zip(matrix, rhs, strict=True)]
    for pivot_index in range(size):
        best_index = max(
            range(pivot_index, size),
            key=lambda row_index: abs(augmented[row_index][pivot_index]),
        )
        augmented[pivot_index], augmented[best_index] = (
            augmented[best_index],
            augmented[pivot_index],
        )
        pivot = augmented[pivot_index][pivot_index]
        if abs(pivot) <= 1e-12:
            continue
        for column_index in range(pivot_index, size + 1):
            augmented[pivot_index][column_index] /= pivot
        for row_index in range(size):
            if row_index == pivot_index:
                continue
            factor = augmented[row_index][pivot_index]
            if abs(factor) <= 1e-12:
                continue
            for column_index in range(pivot_index, size + 1):
                augmented[row_index][column_index] -= factor * augmented[pivot_index][column_index]
    return [row[-1] for row in augmented]


def _clip_summary(dataframe: pl.DataFrame, step_rows: list[dict[str, object]]) -> dict[str, object]:
    if dataframe.is_empty():
        return {}
    first = dataframe.row(0, named=True)
    return {
        "subject": str(first["subject"]),
        "clip": str(first["clip"]),
        "frames": dataframe.height,
        "steps": len(step_rows),
        "environment_split": str(first.get("environment_split", "unknown")),
        "activity_class": str(first.get("activity_class", "unknown")),
        "motion_description": str(first.get("motion_description", "")),
        "mean_step_abs_forward_m": _mean(_row_values(step_rows, "step_abs_forward_m")),
        "mean_step_width_m": _mean(_row_values(step_rows, "step_width_m")),
        "mean_step_interval_s": _mean(_row_values(step_rows, "step_interval_s")),
        "mean_head_speed_m_s": _mean(_row_values(step_rows, "mean_head_speed_m_s")),
    }


def _floor_height(columns: dict[str, list[Any]]) -> float:
    heights = sorted(
        float(value) for foot_name in FOOT_NAMES for value in columns[f"{foot_name}_py"]
    )
    if not heights:
        return 0.0
    return heights[max(0, min(len(heights) - 1, int(len(heights) * 0.05)))]


def _positions(columns: dict[str, list[Any]], joint_name: str) -> list[Vector3]:
    return [
        (float(x), float(y), float(z))
        for x, y, z in zip(
            columns[f"{joint_name}_px"],
            columns[f"{joint_name}_py"],
            columns[f"{joint_name}_pz"],
            strict=True,
        )
    ]


def _rotations(
    columns: dict[str, list[Any]], joint_name: str
) -> list[tuple[float, float, float, float]]:
    return [
        (float(x), float(y), float(z), float(w))
        for x, y, z, w in zip(
            columns[f"{joint_name}_qx"],
            columns[f"{joint_name}_qy"],
            columns[f"{joint_name}_qz"],
            columns[f"{joint_name}_qw"],
            strict=True,
        )
    ]


def _travel_forward(
    center_index: int,
    times: list[float],
    hips: list[Vector3],
    head_rotations: list[tuple[float, float, float, float]],
) -> Vector3:
    start_index, end_index = _window_indices(center_index, times, window_s=0.40)
    hip_delta = subtract(hips[end_index], hips[start_index])
    hip_horizontal = (hip_delta[0], 0.0, hip_delta[2])
    if length(hip_horizontal) > 0.02:
        return normalize(hip_horizontal, fallback=WORLD_FORWARD)
    clip_delta = subtract(hips[-1], hips[0])
    clip_horizontal = (clip_delta[0], 0.0, clip_delta[2])
    if length(clip_horizontal) > 0.05:
        return normalize(clip_horizontal, fallback=WORLD_FORWARD)
    head_forward = rotate_vector(head_rotations[center_index], WORLD_FORWARD)
    return normalize((head_forward[0], 0.0, head_forward[2]), fallback=WORLD_FORWARD)


def _window_indices(center_index: int, times: list[float], *, window_s: float) -> tuple[int, int]:
    half_window = window_s * 0.5
    center_time = times[center_index]
    start_index = center_index
    while start_index > 0 and center_time - times[start_index - 1] <= half_window:
        start_index -= 1
    end_index = center_index
    while end_index + 1 < len(times) and times[end_index + 1] - center_time <= half_window:
        end_index += 1
    return start_index, end_index


def _mean_speed(
    all_times: list[float],
    positions: list[Vector3],
    *,
    start_index: int,
) -> float:
    if len(positions) < 2:
        return 0.0
    distances = []
    for local_index in range(1, len(positions)):
        global_index = start_index + local_index
        dt = max(all_times[global_index] - all_times[global_index - 1], 1.0 / 120.0)
        distances.append(
            horizontal_distance(positions[local_index], positions[local_index - 1]) / dt
        )
    return fmean(distances) if distances else 0.0


def _yaw_rate(
    times: list[float],
    positions: list[Vector3],
    start_index: int,
    end_index: int,
) -> float:
    if len(positions) < 2 or end_index <= start_index:
        return 0.0
    delta = subtract(positions[-1], positions[0])
    yaw = atan2(delta[0], delta[2])
    duration = max(times[end_index] - times[start_index], 1.0 / 120.0)
    return abs(degrees(yaw)) / duration


def _mean_position(positions: Iterable[Vector3]) -> Vector3:
    values = list(positions)
    if not values:
        return (0.0, 0.0, 0.0)
    return (
        fmean(position[0] for position in values),
        fmean(position[1] for position in values),
        fmean(position[2] for position in values),
    )


def _opposite_foot(foot_name: str) -> str:
    return "right_foot" if foot_name == "left_foot" else "left_foot"


def _row_values(rows: list[dict[str, object]], key: str) -> list[float]:
    values = []
    for row in rows:
        value = _float_or_none(row.get(key))
        if value is not None:
            values.append(value)
    return values


def _numeric_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "count": 0.0,
            "mean": 0.0,
            "p10": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "p95": 0.0,
        }
    sorted_values = sorted(values)
    return {
        "count": float(len(sorted_values)),
        "mean": fmean(sorted_values),
        "p10": _percentile(sorted_values, 0.10),
        "p25": _percentile(sorted_values, 0.25),
        "p50": _percentile(sorted_values, 0.50),
        "p75": _percentile(sorted_values, 0.75),
        "p90": _percentile(sorted_values, 0.90),
        "p95": _percentile(sorted_values, 0.95),
    }


def _percentile(sorted_values: list[float], quantile: float) -> float:
    if not sorted_values:
        return 0.0
    index = min(len(sorted_values) - 1, max(0, int((len(sorted_values) - 1) * quantile)))
    return sorted_values[index]


def _mean(values: list[float]) -> float:
    return fmean(values) if values else 0.0


def _subject_fold(subject: str, folds: int) -> int:
    return sum(ord(character) for character in subject) % folds


def _standardized_features(
    row: dict[str, object],
    features: Sequence[str],
    means: Sequence[float],
    scales: Sequence[float],
) -> list[float]:
    return [
        (cast(float, _float_or_none(row[feature])) - mean) / scale
        for feature, mean, scale in zip(features, means, scales, strict=True)
    ]


def _stddev(values: list[float], *, mean: float) -> float:
    if len(values) <= 1:
        return 1.0
    variance = fmean((value - mean) * (value - mean) for value in values)
    result = sqrt(variance)
    return result if result > 1e-9 else 1.0


def _predict(coefficients: list[float], features: list[float]) -> float:
    return coefficients[0] + sum(
        coefficient * feature
        for coefficient, feature in zip(coefficients[1:], features, strict=True)
    )


def _r2(actual: list[float], predicted: list[float]) -> float:
    if not actual:
        return 0.0
    mean_actual = fmean(actual)
    total = sum((value - mean_actual) * (value - mean_actual) for value in actual)
    if total <= 1e-12:
        return 0.0
    residual = sum(
        (actual_value - predicted_value) * (actual_value - predicted_value)
        for actual_value, predicted_value in zip(actual, predicted, strict=True)
    )
    return 1.0 - residual / total


def _mae(actual: list[float], predicted: list[float]) -> float:
    if not actual:
        return 0.0
    return fmean(
        abs(actual_value - predicted_value)
        for actual_value, predicted_value in zip(actual, predicted, strict=True)
    )


def _rmse(actual: list[float], predicted: list[float]) -> float:
    if not actual:
        return 0.0
    return sqrt(
        fmean(
            (actual_value - predicted_value) * (actual_value - predicted_value)
            for actual_value, predicted_value in zip(actual, predicted, strict=True)
        )
    )


def _float_or_none(value: object) -> float | None:
    if isinstance(value, (float, int)):
        return float(value)
    return None


def _plot_title(summary: dict[str, Any]) -> str:
    return (
        f"CMU gait analysis: {int(summary.get('steps', 0)):,} steps, "
        f"{int(summary.get('clips', 0)):,} clips"
    )


def _plot_histogram(
    axis,
    rows: list[dict[str, object]],
    key: str,
    title: str,
    xlabel: str,
    *,
    markers: dict[str, float] | None = None,
) -> None:
    values = _row_values(rows, key)
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    if not values:
        axis.text(0.5, 0.5, "No data", ha="center", va="center")
        axis.axis("off")
        return
    bins = 1 if max(values) - min(values) <= 1e-9 else 50
    axis.hist(values, bins=bins, color="#4c78a8", alpha=0.78)
    for label, value in (markers or {}).items():
        if value <= 0.0:
            continue
        axis.axvline(value, color="#e45756", linestyle="--", linewidth=1.4, label=label)
    axis.grid(axis="y", alpha=0.2)
    if markers:
        axis.legend(fontsize=8)


def _plot_scatter_with_bins(
    axis,
    rows: list[dict[str, object]],
    *,
    x_key: str,
    y_key: str,
    title: str,
    y_label: str,
) -> None:
    pairs = [
        (x_value, y_value)
        for row in rows
        if (x_value := _float_or_none(row.get(x_key))) is not None
        and (y_value := _float_or_none(row.get(y_key))) is not None
    ]
    axis.set_title(title)
    axis.set_xlabel("head speed (m/s)")
    axis.set_ylabel(y_label)
    if not pairs:
        axis.text(0.5, 0.5, "No data", ha="center", va="center")
        axis.axis("off")
        return
    axis.scatter(
        [pair[0] for pair in pairs],
        [pair[1] for pair in pairs],
        s=7,
        alpha=0.22,
        color="#111111",
        linewidths=0,
    )
    binned = _scatter_bins(pairs, bin_width=0.25)
    if binned:
        axis.plot(
            [pair[0] for pair in binned],
            [pair[1] for pair in binned],
            color="#e45756",
            linewidth=2.0,
            marker="o",
            markersize=3,
        )
    axis.grid(alpha=0.2)


def _scatter_bins(
    pairs: list[tuple[float, float]], *, bin_width: float
) -> list[tuple[float, float]]:
    bins: dict[int, list[float]] = {}
    for x_value, y_value in pairs:
        bins.setdefault(int(x_value / bin_width), []).append(y_value)
    return [
        ((bin_index + 0.5) * bin_width, fmean(values))
        for bin_index, values in sorted(bins.items())
        if len(values) >= 5
    ]


def _plot_predictor_r2(axis, predictor_results: dict[str, Any]) -> None:
    labels = []
    values = []
    for target in ("step_abs_forward_m", "step_width_m", "step_interval_s"):
        target_results = cast(dict[str, Any], predictor_results.get(target, {}))
        for feature_set in ("head_speed_only", "upper_body", "upper_body_plus_oracle_hip"):
            result = cast(dict[str, float], target_results.get(feature_set, {}))
            labels.append(f"{target.removeprefix('step_')}\n{feature_set}")
            values.append(float(result.get("test_r2", 0.0)))
    axis.set_title("Held-Out Linear Predictor R2")
    if not values:
        axis.text(0.5, 0.5, "No predictor data", ha="center", va="center")
        axis.axis("off")
        return
    colors = ["#4c78a8", "#72b7b2", "#f58518"] * 3
    axis.bar(range(len(values)), values, color=colors[: len(values)])
    axis.axhline(0.0, color="#333333", linewidth=0.8)
    axis.set_xticks(range(len(values)), labels, rotation=45, ha="right", fontsize=7)
    axis.set_ylabel("R2 on held-out subjects")
    axis.grid(axis="y", alpha=0.2)


if __name__ == "__main__":
    raise SystemExit(main())
