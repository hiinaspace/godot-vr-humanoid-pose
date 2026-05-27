"""Compare estimator error spectra over cached CMU clips."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from validation.scripts.run_cached_validation import _tracked_sample_from_cached_frame
from validation.scripts.run_learned_upper_body_validation import _filtered_paths
from validation.vr_pose_validation.baseline import EstimatorName, create_estimator
from validation.vr_pose_validation.calibration import calibration_from_reference_frames
from validation.vr_pose_validation.cmu_metadata import ActivityClass, EnvironmentSplit
from validation.vr_pose_validation.frequency import (
    DEFAULT_FREQUENCY_BANDS,
    FrequencyAnalysis,
    summarize_frequency_reference_sequence,
)
from validation.vr_pose_validation.metrics import infer_reference_floor_height
from validation.vr_pose_validation.parquet_cache import iter_cached_pose_frames
from validation.vr_pose_validation.types import GroundingContext, HumanoidPoseFrame, PoseEstimator
from validation.vr_pose_validation.upper_body_ml import (
    DEFAULT_CLEAN_ACTIVITY_CLASSES,
    EmaUpperBodyLatentProvider,
    LearnedUpperBodyEstimator,
    TorchUpperBodyLatentProvider,
)

DEFAULT_CACHE_DIR = Path(".cache/cmu/parquet/canonical_full")
DEFAULT_DATASET_DIR = Path(".cache/ml/upper_body_windows_dense")
DEFAULT_MODEL_DIR = Path(".cache/ml/upper_body_models")
DEFAULT_REPORT_PATH = Path("validation/reports/estimator_frequency.json")
FREQUENCY_SUMMARY_METRICS = (
    "upper_body_error_total_power_m2",
    "upper_body_error_band_0_2hz_m2",
    "upper_body_error_band_2_6hz_m2",
    "upper_body_error_band_6_12hz_m2",
    "upper_body_error_band_12hz_plus_m2",
    "upper_body_error_high_frequency_ratio",
    "upper_body_estimated_high_frequency_ratio",
    "upper_body_reference_high_frequency_ratio",
    "torso_error_high_frequency_ratio",
    "shoulder_error_high_frequency_ratio",
    "elbow_error_high_frequency_ratio",
    "hand_error_high_frequency_ratio",
)
type EstimatorFactory = Callable[[], PoseEstimator]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_MODEL_DIR / "torch_mlp.pt")
    parser.add_argument(
        "--normalization",
        type=Path,
        default=DEFAULT_MODEL_DIR / "normalization.json",
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--plot", type=Path)
    parser.add_argument("--window-size", type=int, default=40)
    parser.add_argument(
        "--latent-ema-alpha",
        type=float,
        default=1.0,
        help="Apply EMA smoothing to learned latents; 1.0 keeps raw model output.",
    )
    parser.add_argument("--max-clips", type=int)
    parser.add_argument("--subjects", nargs="*")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument(
        "--base-estimator",
        default="basis_research",
        choices=(
            "trivial",
            "planted",
            "basis",
            "basis_knee_hint",
            "basis_footstep",
            "basis_arm_lookup",
            "basis_arm_swivel",
            "basis_shrug",
            "basis_research",
        ),
    )
    parser.add_argument("--learned-label", default="learned_fk")
    parser.add_argument("--skip-learned", action="store_true")
    parser.add_argument("--include-spectra", action="store_true")
    parser.add_argument("--top-clips", type=int, default=8)
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
        default=DEFAULT_CLEAN_ACTIVITY_CLASSES,
    )
    parser.add_argument(
        "--validation-split-only",
        action="store_true",
        default=True,
        help="Restrict to subject/clip pairs marked validation in the sparse-window dataset.",
    )
    parser.add_argument(
        "--all-splits",
        action="store_false",
        dest="validation_split_only",
        help="Evaluate all filtered clips instead of only the sparse-window validation split.",
    )
    args = parser.parse_args(argv)

    result = analyze_estimator_frequency(
        cache_dir=args.cache_dir,
        dataset_dir=args.dataset_dir,
        artifact_path=args.artifact,
        normalization_path=args.normalization,
        report_path=args.report,
        plot_path=args.plot,
        window_size=max(2, args.window_size),
        latent_ema_alpha=args.latent_ema_alpha,
        max_clips=args.max_clips,
        subjects=set(args.subjects or []),
        allow_cpu=args.allow_cpu,
        base_estimator_name=cast(EstimatorName, args.base_estimator),
        learned_label=args.learned_label,
        include_learned=not args.skip_learned,
        include_spectra=args.include_spectra,
        top_clip_count=max(0, args.top_clips),
        environment_splits=cast(set[EnvironmentSplit], set(args.environment_split or [])),
        activity_classes=cast(set[ActivityClass], set(args.activity_class or [])),
        validation_split_only=args.validation_split_only,
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0


def analyze_estimator_frequency(
    *,
    cache_dir: Path,
    dataset_dir: Path,
    artifact_path: Path,
    normalization_path: Path,
    report_path: Path,
    plot_path: Path | None = None,
    window_size: int = 40,
    latent_ema_alpha: float = 1.0,
    max_clips: int | None = None,
    subjects: set[str] | None = None,
    allow_cpu: bool = False,
    base_estimator_name: EstimatorName = "basis_research",
    learned_label: str = "learned_fk",
    include_learned: bool = True,
    include_spectra: bool = False,
    top_clip_count: int = 8,
    environment_splits: set[EnvironmentSplit] | None = None,
    activity_classes: set[ActivityClass] | None = None,
    validation_split_only: bool = True,
) -> dict[str, object]:
    start = time.perf_counter()
    paths = _filtered_paths(
        cache_dir,
        dataset_dir=dataset_dir,
        subjects=subjects or set(),
        environment_splits=environment_splits,
        activity_classes=activity_classes,
        validation_split_only=validation_split_only,
    )
    if max_clips is not None:
        paths = paths[:max_clips]

    estimator_factories: dict[str, EstimatorFactory] = {
        base_estimator_name: lambda: create_estimator(base_estimator_name),
    }
    if include_learned:
        provider = TorchUpperBodyLatentProvider(
            artifact_path=artifact_path,
            normalization_path=normalization_path,
            window_size=window_size,
            allow_cpu=allow_cpu,
        )
        latent_provider = (
            EmaUpperBodyLatentProvider(provider=provider, alpha=latent_ema_alpha)
            if latent_ema_alpha < 1.0
            else provider
        )

        def learned_factory() -> PoseEstimator:
            latent_provider.reset()
            return LearnedUpperBodyEstimator(
                latent_provider=latent_provider,
                base_estimator=create_estimator(base_estimator_name),
            )

        estimator_factories[learned_label] = learned_factory

    clip_rows = [
        _analyze_clip_frequency(
            path,
            estimator_factories=estimator_factories,
            include_spectra=include_spectra,
        )
        for path in paths
    ]
    summary = _summary(
        clip_rows,
        estimator_labels=tuple(estimator_factories),
        cache_dir=cache_dir,
        dataset_dir=dataset_dir,
        artifact_path=artifact_path,
        normalization_path=normalization_path,
        window_size=window_size,
        latent_ema_alpha=latent_ema_alpha,
        base_estimator_name=base_estimator_name,
        learned_label=learned_label,
        include_learned=include_learned,
        environment_splits=environment_splits or set(),
        activity_classes=activity_classes or set(),
        validation_split_only=validation_split_only,
        top_clip_count=top_clip_count,
        total_seconds=time.perf_counter() - start,
    )
    result: dict[str, object] = {
        "summary": summary,
        "bands": [
            {"name": name, "low_hz": low_hz, "high_hz": high_hz}
            for name, low_hz, high_hz in DEFAULT_FREQUENCY_BANDS
        ],
        "clips": clip_rows,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if plot_path is not None:
        render_frequency_report(result, plot_path)
    return result


def render_frequency_report(report: Mapping[str, object], output_path: Path) -> None:
    summary = cast(dict[str, Any], report.get("summary", {}))
    estimator_metrics = cast(dict[str, dict[str, float]], summary.get("estimators", {}))
    if not estimator_metrics:
        return
    labels = list(estimator_metrics)
    band_names = [name for name, _low_hz, _high_hz in DEFAULT_FREQUENCY_BANDS]
    figure, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    figure.suptitle("Estimator Frequency Diagnostics", fontsize=14, fontweight="bold")

    _plot_band_metric(
        axes[0][0],
        estimator_metrics,
        labels,
        band_names,
        prefix="upper_body_error_band",
        title="Upper-Body Error Bandpower",
        ylabel="mean m^2 / component",
    )
    _plot_ratio_metric(
        axes[0][1],
        estimator_metrics,
        labels,
        (
            ("upper error", "upper_body_error_high_frequency_ratio"),
            ("upper est", "upper_body_estimated_high_frequency_ratio"),
            ("upper GT", "upper_body_reference_high_frequency_ratio"),
        ),
        title="High-Frequency Ratio",
    )
    _plot_ratio_metric(
        axes[1][0],
        estimator_metrics,
        labels,
        (
            ("torso", "torso_error_high_frequency_ratio"),
            ("shoulder", "shoulder_error_high_frequency_ratio"),
            ("elbow", "elbow_error_high_frequency_ratio"),
            ("hand", "hand_error_high_frequency_ratio"),
        ),
        title="Error High-Frequency Ratio By Region",
    )
    _plot_total_power(axes[1][1], estimator_metrics, labels)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _analyze_clip_frequency(
    path: Path,
    *,
    estimator_factories: Mapping[str, EstimatorFactory],
    include_spectra: bool,
) -> dict[str, object]:
    cached_frame_list = list(iter_cached_pose_frames(path))
    reference_frames = [frame.joints for frame in cached_frame_list]
    inferred_floor_height_m = infer_reference_floor_height(reference_frames)
    calibration = calibration_from_reference_frames(
        reference_frames,
        floor_height_m=inferred_floor_height_m,
    )
    grounding = GroundingContext(floor_height_m=inferred_floor_height_m)
    first_frame = cached_frame_list[0] if cached_frame_list else None
    row: dict[str, object] = {
        "subject": path.parent.name.split("=", 1)[-1],
        "clip": path.stem,
        "source_frames": len(cached_frame_list),
        "environment_split": first_frame.environment_split if first_frame else "unknown",
        "activity_class": first_frame.activity_class if first_frame else "unknown",
        "motion_description": first_frame.motion_description if first_frame else "",
        "estimators": {},
    }
    estimators = cast(dict[str, object], row["estimators"])
    for label, factory in estimator_factories.items():
        estimator = factory()
        poses: list[HumanoidPoseFrame] = []
        for cached_frame in cached_frame_list:
            poses.append(
                estimator.estimate(
                    _tracked_sample_from_cached_frame(cached_frame), calibration, grounding
                )
            )
        analysis = summarize_frequency_reference_sequence(
            poses,
            reference_frames,
            include_spectra=include_spectra,
        )
        estimators[label] = _analysis_payload(analysis)
    return row


def _analysis_payload(analysis: FrequencyAnalysis) -> dict[str, object]:
    payload: dict[str, object] = {"metrics": analysis.metrics}
    if analysis.spectra:
        payload["spectra"] = analysis.spectra
    return payload


def _summary(
    clips: Sequence[Mapping[str, object]],
    *,
    estimator_labels: Sequence[str],
    cache_dir: Path,
    dataset_dir: Path,
    artifact_path: Path,
    normalization_path: Path,
    window_size: int,
    latent_ema_alpha: float,
    base_estimator_name: str,
    learned_label: str,
    include_learned: bool,
    environment_splits: set[EnvironmentSplit],
    activity_classes: set[ActivityClass],
    validation_split_only: bool,
    top_clip_count: int,
    total_seconds: float,
) -> dict[str, object]:
    estimator_summaries = {
        label: _mean_estimator_metrics(clips, label) for label in estimator_labels
    }
    summary: dict[str, object] = {
        "cache_dir": cache_dir.as_posix(),
        "dataset_dir": dataset_dir.as_posix(),
        "artifact": artifact_path.as_posix(),
        "normalization": normalization_path.as_posix(),
        "window_size": window_size,
        "latent_ema_alpha": latent_ema_alpha,
        "base_estimator": base_estimator_name,
        "learned_label": learned_label,
        "include_learned": include_learned,
        "validation_split_only": validation_split_only,
        "environment_splits": sorted(environment_splits),
        "activity_classes": sorted(activity_classes),
        "clips": len(clips),
        "source_frames": sum(_int_value(clip.get("source_frames", 0)) for clip in clips),
        "total_seconds": total_seconds,
        "estimators": estimator_summaries,
        "worst_clips": {
            label: _worst_clips(clips, label, top_clip_count) for label in estimator_labels
        },
    }
    if include_learned and learned_label in estimator_summaries:
        baseline = estimator_summaries.get(base_estimator_name, {})
        learned = estimator_summaries[learned_label]
        summary[f"{learned_label}_minus_{base_estimator_name}"] = {
            metric: learned[metric] - baseline[metric] for metric in learned if metric in baseline
        }
    return summary


def _mean_estimator_metrics(clips: Sequence[Mapping[str, object]], label: str) -> dict[str, float]:
    rows: list[Mapping[str, float]] = []
    for clip in clips:
        estimators = cast(Mapping[str, Mapping[str, object]], clip.get("estimators", {}))
        payload = estimators.get(label)
        if payload is None:
            continue
        rows.append(cast(Mapping[str, float], payload.get("metrics", {})))
    metric_names = sorted({metric for row in rows for metric in row})
    return {metric: _mean(row[metric] for row in rows if metric in row) for metric in metric_names}


def _worst_clips(
    clips: Sequence[Mapping[str, object]],
    label: str,
    limit: int,
) -> dict[str, list[dict[str, object]]]:
    if limit <= 0:
        return {}
    return {
        metric: [
            {
                "subject": str(clip.get("subject", "")),
                "clip": str(clip.get("clip", "")),
                "motion_description": str(clip.get("motion_description", "")),
                metric: _estimator_metric(clip, label, metric),
            }
            for clip in sorted(
                clips,
                key=lambda item: _estimator_metric(item, label, metric),
                reverse=True,
            )[:limit]
        ]
        for metric in FREQUENCY_SUMMARY_METRICS
    }


def _estimator_metric(clip: Mapping[str, object], label: str, metric: str) -> float:
    estimators = cast(Mapping[str, Mapping[str, object]], clip.get("estimators", {}))
    payload = estimators.get(label)
    if payload is None:
        return 0.0
    metrics = cast(Mapping[str, float], payload.get("metrics", {}))
    return float(metrics.get(metric, 0.0))


def _plot_band_metric(
    axis: Any,
    estimator_metrics: Mapping[str, Mapping[str, float]],
    labels: Sequence[str],
    band_names: Sequence[str],
    *,
    prefix: str,
    title: str,
    ylabel: str,
) -> None:
    width = 0.8 / max(1, len(labels))
    positions = list(range(len(band_names)))
    for label_index, label in enumerate(labels):
        values = [
            estimator_metrics[label].get(f"{prefix}_{band_name}_m2", 0.0)
            for band_name in band_names
        ]
        offsets = [
            position + (label_index - (len(labels) - 1) / 2.0) * width for position in positions
        ]
        axis.bar(offsets, values, width=width, label=label)
    axis.set_xticks(positions, band_names)
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", alpha=0.2)
    axis.legend()


def _plot_ratio_metric(
    axis: Any,
    estimator_metrics: Mapping[str, Mapping[str, float]],
    labels: Sequence[str],
    metrics: Sequence[tuple[str, str]],
    *,
    title: str,
) -> None:
    width = 0.8 / max(1, len(labels))
    positions = list(range(len(metrics)))
    for label_index, label in enumerate(labels):
        values = [estimator_metrics[label].get(metric, 0.0) for _name, metric in metrics]
        offsets = [
            position + (label_index - (len(labels) - 1) / 2.0) * width for position in positions
        ]
        axis.bar(offsets, values, width=width, label=label)
    axis.set_xticks(positions, [name for name, _metric in metrics], rotation=20, ha="right")
    axis.set_ylim(0.0, 1.0)
    axis.set_title(title)
    axis.set_ylabel("ratio")
    axis.grid(axis="y", alpha=0.2)
    axis.legend()


def _plot_total_power(
    axis: Any,
    estimator_metrics: Mapping[str, Mapping[str, float]],
    labels: Sequence[str],
) -> None:
    metrics = (
        ("upper error", "upper_body_error_total_power_m2"),
        ("upper est", "upper_body_estimated_total_power_m2"),
        ("upper GT", "upper_body_reference_total_power_m2"),
    )
    _plot_ratio_metric(axis, estimator_metrics, labels, metrics, title="Total Position Variance")
    axis.set_ylim(bottom=0.0)
    axis.set_ylabel("mean m^2 / component")


def _mean(values: Iterable[float]) -> float:
    value_list = [float(value) for value in values]
    return sum(value_list) / len(value_list) if value_list else 0.0


def _int_value(value: object) -> int:
    return int(value) if isinstance(value, int | float | str) else 0


if __name__ == "__main__":
    raise SystemExit(main())
