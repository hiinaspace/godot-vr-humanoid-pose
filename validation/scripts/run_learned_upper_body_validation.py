"""Run downstream CMU metrics with a learned upper-body latent Torch model."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import cast

import polars as pl

from validation.scripts.run_baseline_validation import ClipResult, write_clip_csv
from validation.scripts.run_cached_validation import (
    TOP_CLIP_METRICS,
    evaluate_cached_clip_path,
)
from validation.vr_pose_validation.baseline import EstimatorName, create_estimator
from validation.vr_pose_validation.cmu_metadata import ActivityClass, EnvironmentSplit
from validation.vr_pose_validation.parquet_cache import (
    filter_clip_cache_files_by_environment,
    list_clip_cache_files,
)
from validation.vr_pose_validation.upper_body_ml import (
    DEFAULT_CLEAN_ACTIVITY_CLASSES,
    LearnedUpperBodyEstimator,
    TorchUpperBodyLatentProvider,
)

DEFAULT_CACHE_DIR = Path(".cache/cmu/parquet/canonical_full")
DEFAULT_DATASET_DIR = Path(".cache/ml/upper_body_windows")
DEFAULT_MODEL_DIR = Path(".cache/ml/upper_body_models")
DEFAULT_REPORT_PATH = Path("validation/reports/learned_upper_body_validation.json")
SUMMARY_METRICS = (
    "mean_reference_error_m",
    "mean_upper_body_reference_error_m",
    "mean_torso_error_m",
    "mean_shoulder_error_m",
    "mean_elbow_error_m",
    "mean_hand_reference_error_m",
    "mean_head_forward_error_deg",
    "mean_neck_axis_error_deg",
    "mean_spine_axis_error_deg",
    "mean_joint_velocity_error_m_per_s",
    "mean_joint_estimated_jerk_m_per_s3",
    "avatar_implausibility_score_0_100",
)


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
    parser.add_argument("--csv-report", type=Path)
    parser.add_argument("--window-size", type=int, default=40)
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
    parser.add_argument("--compare-baseline", action="store_true")
    parser.add_argument("--top-clips", type=int, default=10)
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

    result = run_learned_upper_body_validation(
        cache_dir=args.cache_dir,
        dataset_dir=args.dataset_dir,
        artifact_path=args.artifact,
        normalization_path=args.normalization,
        report_path=args.report,
        csv_report_path=args.csv_report,
        window_size=max(2, args.window_size),
        max_clips=args.max_clips,
        subjects=set(args.subjects or []),
        allow_cpu=args.allow_cpu,
        base_estimator_name=args.base_estimator,
        compare_baseline=args.compare_baseline,
        top_clip_count=max(0, args.top_clips),
        environment_splits=cast(set[EnvironmentSplit], set(args.environment_split or [])),
        activity_classes=cast(set[ActivityClass], set(args.activity_class or [])),
        validation_split_only=args.validation_split_only,
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0


def run_learned_upper_body_validation(
    *,
    cache_dir: Path,
    dataset_dir: Path,
    artifact_path: Path,
    normalization_path: Path,
    report_path: Path,
    csv_report_path: Path | None = None,
    window_size: int = 40,
    max_clips: int | None = None,
    subjects: set[str] | None = None,
    allow_cpu: bool = False,
    base_estimator_name: EstimatorName = "basis_research",
    compare_baseline: bool = False,
    top_clip_count: int = 10,
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

    provider = TorchUpperBodyLatentProvider(
        artifact_path=artifact_path,
        normalization_path=normalization_path,
        window_size=window_size,
        allow_cpu=allow_cpu,
    )
    learned_results: list[ClipResult] = []
    for path in paths:
        provider.reset()
        estimator = LearnedUpperBodyEstimator(
            latent_provider=provider,
            base_estimator=create_estimator(base_estimator_name),
        )
        learned_results.append(evaluate_cached_clip_path(path, estimator))

    baseline_results: list[ClipResult] | None = None
    if compare_baseline:
        baseline_results = [
            evaluate_cached_clip_path(path, create_estimator(base_estimator_name)) for path in paths
        ]

    summary = _summary(
        cache_dir=cache_dir,
        dataset_dir=dataset_dir,
        paths=paths,
        learned_results=learned_results,
        baseline_results=baseline_results,
        artifact_path=artifact_path,
        normalization_path=normalization_path,
        window_size=window_size,
        base_estimator_name=base_estimator_name,
        environment_splits=environment_splits or set(),
        activity_classes=activity_classes or set(),
        validation_split_only=validation_split_only,
        top_clip_count=top_clip_count,
        total_seconds=time.perf_counter() - start,
    )
    result: dict[str, object] = {
        "summary": summary,
        "clips": [asdict(result) for result in learned_results],
    }
    if baseline_results is not None:
        result["baseline_clips"] = [asdict(result) for result in baseline_results]

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if csv_report_path is not None:
        write_clip_csv(
            csv_report_path,
            cast(list[dict[str, object]], [asdict(result) for result in learned_results]),
        )
    return result


def _filtered_paths(
    cache_dir: Path,
    *,
    dataset_dir: Path,
    subjects: set[str],
    environment_splits: set[EnvironmentSplit] | None,
    activity_classes: set[ActivityClass] | None,
    validation_split_only: bool,
) -> list[Path]:
    paths = list_clip_cache_files(cache_dir, subjects or None)
    paths = filter_clip_cache_files_by_environment(
        paths,
        environment_splits=environment_splits,
        activity_classes=activity_classes,
    )
    if not validation_split_only:
        return paths
    validation_clips = _dataset_split_clips(dataset_dir, "validation")
    return [
        path
        for path in paths
        if (path.parent.name.split("=", 1)[-1], path.stem) in validation_clips
    ]


def _dataset_split_clips(dataset_dir: Path, split: str) -> set[tuple[str, str]]:
    frame = cast(pl.DataFrame, _scan_sparse_window_split_columns(dataset_dir).collect())
    return {
        (str(row["subject"]), str(row["clip"]))
        for row in frame.filter(pl.col("split") == split)
        .select("subject", "clip")
        .unique()
        .to_dicts()
    }


def _scan_sparse_window_split_columns(dataset_dir: Path) -> pl.LazyFrame:
    windows_path = dataset_dir / "windows.parquet"
    if windows_path.exists():
        return pl.scan_parquet(windows_path).select("subject", "clip", "split")
    shard_dir = dataset_dir / "windows_shards"
    if shard_dir.exists():
        return pl.scan_parquet(str(shard_dir / "*.parquet")).select("subject", "clip", "split")
    raise FileNotFoundError(
        f"Expected sparse-window dataset at {dataset_dir}; run build_sparse_window_dataset"
    )


def _summary(
    *,
    cache_dir: Path,
    dataset_dir: Path,
    paths: Sequence[Path],
    learned_results: Sequence[ClipResult],
    baseline_results: Sequence[ClipResult] | None,
    artifact_path: Path,
    normalization_path: Path,
    window_size: int,
    base_estimator_name: str,
    environment_splits: set[EnvironmentSplit],
    activity_classes: set[ActivityClass],
    validation_split_only: bool,
    top_clip_count: int,
    total_seconds: float,
) -> dict[str, object]:
    learned_summary = _metric_summary(learned_results)
    summary: dict[str, object] = {
        "cache_dir": cache_dir.as_posix(),
        "dataset_dir": dataset_dir.as_posix(),
        "artifact": artifact_path.as_posix(),
        "normalization": normalization_path.as_posix(),
        "window_size": window_size,
        "base_estimator": base_estimator_name,
        "validation_split_only": validation_split_only,
        "environment_splits": sorted(environment_splits),
        "activity_classes": sorted(activity_classes),
        "clips": len(learned_results),
        "subjects": len({result.subject for result in learned_results}),
        "source_frames": sum(result.source_frames for result in learned_results),
        "evaluated_frames": sum(result.evaluated_frames for result in learned_results),
        "total_seconds": total_seconds,
        "evaluated_frames_per_second": (
            sum(result.evaluated_frames for result in learned_results) / total_seconds
            if total_seconds > 0.0
            else 0.0
        ),
        "learned": learned_summary,
        "worst_clips": _worst_clips(learned_results, top_clip_count),
    }
    del paths
    if baseline_results is not None:
        baseline_summary = _metric_summary(baseline_results)
        summary["baseline"] = baseline_summary
        summary["learned_minus_baseline"] = {
            metric: learned_summary[metric] - baseline_summary[metric] for metric in SUMMARY_METRICS
        }
    return summary


def _metric_summary(results: Sequence[ClipResult]) -> dict[str, float]:
    return {
        metric: _mean(getattr(result, metric) for result in results) for metric in SUMMARY_METRICS
    }


def _worst_clips(results: Sequence[ClipResult], limit: int) -> dict[str, list[dict[str, object]]]:
    if limit <= 0:
        return {}
    return {
        metric: [
            {
                "subject": result.subject,
                "clip": result.clip,
                "motion_description": result.motion_description,
                metric: float(getattr(result, metric)),
            }
            for result in sorted(
                results,
                key=lambda clip_result: float(getattr(clip_result, metric)),
                reverse=True,
            )[:limit]
        ]
        for metric in TOP_CLIP_METRICS
        if all(hasattr(result, metric) for result in results)
    }


def _mean(values: Iterable[float]) -> float:
    value_list = [float(value) for value in values]
    return sum(value_list) / len(value_list) if value_list else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
