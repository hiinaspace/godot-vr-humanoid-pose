"""Run stateful baseline validation from the flat Parquet cache."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import polars as pl

from validation.scripts.run_baseline_validation import ClipResult, write_clip_csv
from validation.vr_pose_validation.baseline import EstimatorName, create_estimator
from validation.vr_pose_validation.cmu_metadata import ActivityClass, EnvironmentSplit
from validation.vr_pose_validation.geometry import Transform, add
from validation.vr_pose_validation.metrics import (
    ReferencePoseMetrics,
    evaluate_foot_contact_sequence,
    evaluate_frame,
    evaluate_reference_pose,
    infer_reference_floor_height,
    summarize_reference_errors,
    summarize_target_errors,
)
from validation.vr_pose_validation.parquet_cache import (
    CachedPoseFrame,
    cache_summary,
    filter_clip_cache_files_by_environment,
    iter_cached_pose_frames,
    list_clip_cache_files,
)
from validation.vr_pose_validation.synthetic_trackers import SyntheticTrackerConfig
from validation.vr_pose_validation.types import (
    AvatarCalibration,
    GroundingContext,
    TrackedPoseSample,
    TrackerRole,
    TrackerState,
)

DEFAULT_CACHE_DIR = Path(".cache/cmu/parquet/canonical")
DEFAULT_REPORT_PATH = Path("validation/reports/cached_baseline_smoke.json")
DEFAULT_SYNTHETIC_TRACKER_CONFIG = SyntheticTrackerConfig()
SUMMARY_METRICS = (
    "mean_target_error_m",
    "mean_pelvis_error_m",
    "mean_pelvis_horizontal_error_m",
    "mean_pelvis_vertical_error_m",
    "mean_chest_error_m",
    "mean_spine_error_m",
    "mean_torso_error_m",
    "mean_left_foot_error_m",
    "mean_right_foot_error_m",
    "mean_reference_error_m",
    "mean_contact_foot_error_m",
    "mean_contact_slip_m_per_s",
)
TOP_CLIP_METRICS = (
    "mean_reference_error_m",
    "mean_pelvis_error_m",
    "mean_pelvis_horizontal_error_m",
    "mean_pelvis_vertical_error_m",
    "mean_torso_error_m",
    "mean_left_foot_error_m",
    "mean_right_foot_error_m",
    "mean_contact_foot_error_m",
    "mean_contact_slip_m_per_s",
)


@dataclass(frozen=True, slots=True)
class CachedValidationTask:
    path: str
    estimator_name: EstimatorName


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--csv-report", type=Path)
    parser.add_argument("--subjects", nargs="*", help="Subject ids such as 07 08 09.")
    parser.add_argument("--max-clips", type=int)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument(
        "--top-clips",
        type=int,
        default=10,
        help="Include this many worst clips per summary metric.",
    )
    parser.add_argument(
        "--environment-split",
        choices=("flat", "nonflat", "unknown"),
        nargs="*",
        help="Only evaluate clips with these cached environment split labels.",
    )
    parser.add_argument(
        "--activity-class",
        choices=(
            "idle_standing",
            "sitting",
            "stationary_upper_body",
            "locomotion",
            "dance_exercise",
            "floor_pose",
            "sports",
            "unknown",
        ),
        nargs="*",
        help="Only evaluate clips with these cached activity class labels.",
    )
    parser.add_argument(
        "--estimator",
        choices=("trivial", "planted", "basis"),
        default="trivial",
        help="Estimator implementation to evaluate.",
    )
    args = parser.parse_args(argv)

    result = run_cached_validation(
        cache_dir=args.cache_dir,
        subjects=set(args.subjects or []),
        max_clips=args.max_clips,
        workers=max(1, args.workers),
        estimator_name=args.estimator,
        environment_splits=cast(set[EnvironmentSplit], set(args.environment_split or [])),
        activity_classes=cast(set[ActivityClass], set(args.activity_class or [])),
        top_clip_count=max(0, args.top_clips),
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.csv_report is not None:
        write_clip_csv(args.csv_report, cast(list[dict[str, object]], result["clips"]))

    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0


def run_cached_validation(
    *,
    cache_dir: Path,
    subjects: set[str],
    max_clips: int | None,
    workers: int,
    estimator_name: EstimatorName = "trivial",
    environment_splits: set[EnvironmentSplit] | None = None,
    activity_classes: set[ActivityClass] | None = None,
    top_clip_count: int = 10,
) -> dict[str, object]:
    start = time.perf_counter()
    paths = list_clip_cache_files(cache_dir, subjects or None)
    paths = filter_clip_cache_files_by_environment(
        paths,
        environment_splits=environment_splits,
        activity_classes=activity_classes,
    )
    if max_clips is not None:
        paths = paths[:max_clips]

    tasks = [
        CachedValidationTask(path=path.as_posix(), estimator_name=estimator_name) for path in paths
    ]
    if workers == 1:
        clip_results = [_evaluate_cached_clip(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            clip_results = list(executor.map(_evaluate_cached_clip, tasks))

    summary_frame = pl.DataFrame([_scalar_clip_result(result) for result in clip_results])
    cache_frame = cache_summary(paths)
    total_seconds = time.perf_counter() - start
    return {
        "summary": {
            "cache_dir": cache_dir.as_posix(),
            "clips": len(clip_results),
            "subjects": int(cache_frame.select(pl.col("subject").n_unique()).item())
            if len(cache_frame) > 0
            else 0,
            "source_frames": _sum_column(summary_frame, "source_frames"),
            "evaluated_frames": _sum_column(summary_frame, "evaluated_frames"),
            "workers": workers,
            "estimator": estimator_name,
            "environment_splits": sorted(environment_splits or []),
            "activity_classes": sorted(activity_classes or []),
            "total_seconds": total_seconds,
            "evaluated_frames_per_second": (
                _sum_column(summary_frame, "evaluated_frames") / total_seconds
                if total_seconds > 0.0
                else 0.0
            ),
            "mean_target_error_m": _mean_column(summary_frame, "mean_target_error_m"),
            "mean_pelvis_error_m": _mean_column(summary_frame, "mean_pelvis_error_m"),
            "mean_pelvis_horizontal_error_m": _mean_column(
                summary_frame,
                "mean_pelvis_horizontal_error_m",
            ),
            "mean_pelvis_vertical_error_m": _mean_column(
                summary_frame,
                "mean_pelvis_vertical_error_m",
            ),
            "mean_chest_error_m": _mean_column(summary_frame, "mean_chest_error_m"),
            "mean_spine_error_m": _mean_column(summary_frame, "mean_spine_error_m"),
            "mean_torso_error_m": _mean_column(summary_frame, "mean_torso_error_m"),
            "mean_left_foot_error_m": _mean_column(summary_frame, "mean_left_foot_error_m"),
            "mean_right_foot_error_m": _mean_column(summary_frame, "mean_right_foot_error_m"),
            "mean_reference_error_m": _mean_column(summary_frame, "mean_reference_error_m"),
            "contact_frames": _sum_column(summary_frame, "contact_frames"),
            "mean_contact_foot_error_m": _mean_column(
                summary_frame,
                "mean_contact_foot_error_m",
            ),
            "mean_contact_slip_m_per_s": _mean_column(
                summary_frame,
                "mean_contact_slip_m_per_s",
            ),
            "mean_joint_errors_m": _mean_joint_errors(clip_results),
            "grouped_metrics": _grouped_metrics(clip_results),
            "worst_clips": _worst_clips(clip_results, top_clip_count),
        },
        "clips": [asdict(result) for result in clip_results],
    }


def _evaluate_cached_clip(task: CachedValidationTask) -> ClipResult:
    start = time.perf_counter()
    estimator = create_estimator(task.estimator_name)
    calibration = AvatarCalibration()
    frame_metrics = []
    reference_metrics: list[ReferencePoseMetrics] = []
    estimated_poses = []
    cached_frame_list = list(iter_cached_pose_frames(Path(task.path)))
    reference_frames = [frame.joints for frame in cached_frame_list]
    inferred_floor_height_m = infer_reference_floor_height(reference_frames)
    grounding = GroundingContext(floor_height_m=inferred_floor_height_m)
    environment_split = "unknown"
    environment_tags: tuple[str, ...] = ()
    activity_class = "unknown"
    activity_tags: tuple[str, ...] = ()
    motion_description = ""
    for frame_count, cached_frame in enumerate(cached_frame_list):
        if frame_count == 0:
            environment_split = cached_frame.environment_split
            environment_tags = cached_frame.environment_tags
            activity_class = cached_frame.activity_class
            activity_tags = cached_frame.activity_tags
            motion_description = cached_frame.motion_description
        sample = _tracked_sample_from_cached_frame(cached_frame)
        pose = estimator.estimate(sample, calibration, grounding)
        estimated_poses.append(pose)
        frame_metrics.append(evaluate_frame(pose, sample, grounding.floor_height_m))
        reference_metrics.append(evaluate_reference_pose(pose, cached_frame.joints))
    target_summary = summarize_target_errors(frame_metrics)
    reference_summary = summarize_reference_errors(reference_metrics)
    contact_summary = evaluate_foot_contact_sequence(
        estimated_poses,
        reference_frames,
        floor_height_m=inferred_floor_height_m,
    )
    mean_joint_errors_m = cast(dict[str, float], reference_summary["mean_joint_errors_m"])
    subject, clip = _subject_and_clip_from_path(Path(task.path))
    frame_count = len(cached_frame_list)
    return ClipResult(
        clip=clip,
        subject=subject,
        source_frames=frame_count,
        evaluated_frames=frame_count,
        parse_seconds=0.0,
        fk_sample_estimate_metric_seconds=time.perf_counter() - start,
        mean_target_error_m=target_summary["mean_target_error_m"],
        mean_pelvis_error_m=cast(float, reference_summary["mean_pelvis_error_m"]),
        mean_pelvis_horizontal_error_m=cast(
            float,
            reference_summary["mean_pelvis_horizontal_error_m"],
        ),
        mean_pelvis_vertical_error_m=cast(
            float,
            reference_summary["mean_pelvis_vertical_error_m"],
        ),
        mean_chest_error_m=cast(float, reference_summary["mean_chest_error_m"]),
        mean_spine_error_m=cast(float, reference_summary["mean_spine_error_m"]),
        mean_torso_error_m=cast(float, reference_summary["mean_torso_error_m"]),
        mean_left_foot_error_m=cast(float, reference_summary["mean_left_foot_error_m"]),
        mean_right_foot_error_m=cast(float, reference_summary["mean_right_foot_error_m"]),
        mean_reference_error_m=cast(float, reference_summary["mean_reference_error_m"]),
        mean_joint_errors_m=mean_joint_errors_m,
        inferred_floor_height_m=inferred_floor_height_m,
        contact_frames=contact_summary.contact_frames,
        mean_contact_foot_error_m=contact_summary.mean_contact_foot_error_m,
        left_contact_slip_m_per_s=contact_summary.left_contact_slip_m_per_s,
        right_contact_slip_m_per_s=contact_summary.right_contact_slip_m_per_s,
        mean_contact_slip_m_per_s=contact_summary.mean_contact_slip_m_per_s,
        environment_split=environment_split,
        environment_tags=environment_tags,
        activity_class=activity_class,
        activity_tags=activity_tags,
        motion_description=motion_description,
    )


def _tracked_sample_from_cached_frame(frame: CachedPoseFrame) -> TrackedPoseSample:
    return TrackedPoseSample(
        timestamp_seconds=frame.timestamp_s,
        hmd=_tracker_state("head", frame),
        left_hand=_tracker_state("left_hand", frame),
        right_hand=_tracker_state("right_hand", frame),
    )


def _tracker_state(role: TrackerRole, frame: CachedPoseFrame) -> TrackerState:
    attachment = DEFAULT_SYNTHETIC_TRACKER_CONFIG.attachments[role]
    source = frame.joints[attachment.source_joint]
    transform = Transform(
        position=add(source.position, attachment.local_transform.position),
        rotation=source.rotation,
    )
    return TrackerState(
        role=role,
        transform=transform,
        confidence=attachment.confidence,
        attachment=attachment,
    )


def _subject_and_clip_from_path(path: Path) -> tuple[str, str]:
    parent = path.parent.name
    subject = parent.split("=", maxsplit=1)[1] if parent.startswith("subject=") else parent
    return subject, path.stem


def _sum_column(dataframe: pl.DataFrame, column: str) -> int:
    if len(dataframe) == 0:
        return 0
    value = dataframe.select(pl.col(column).sum()).item()
    return int(value or 0)


def _mean_column(dataframe: pl.DataFrame, column: str) -> float:
    if len(dataframe) == 0:
        return 0.0
    value = dataframe.select(pl.col(column).mean()).item()
    return float(value or 0.0)


def _mean(values: Iterable[float]) -> float:
    value_list = list(values)
    if not value_list:
        return 0.0
    return sum(value_list) / len(value_list)


def _scalar_clip_result(result: ClipResult) -> dict[str, Any]:
    data = asdict(result)
    data.pop("mean_joint_errors_m", None)
    data.pop("environment_tags", None)
    data.pop("activity_tags", None)
    data.pop("motion_description", None)
    return data


def _mean_joint_errors(results: Iterable[ClipResult]) -> dict[str, float]:
    result_list = list(results)
    joint_names = sorted({name for result in result_list for name in result.mean_joint_errors_m})
    return {
        name: _mean(
            result.mean_joint_errors_m[name]
            for result in result_list
            if name in result.mean_joint_errors_m
        )
        for name in joint_names
    }


def _grouped_metrics(results: list[ClipResult]) -> dict[str, list[dict[str, object]]]:
    return {
        "environment_split": _group_results(results, ("environment_split",)),
        "activity_class": _group_results(results, ("activity_class",)),
        "environment_split_activity_class": _group_results(
            results,
            ("environment_split", "activity_class"),
        ),
    }


def _group_results(
    results: list[ClipResult],
    group_fields: tuple[str, ...],
) -> list[dict[str, object]]:
    groups: dict[tuple[str, ...], list[ClipResult]] = {}
    for result in results:
        key = tuple(str(getattr(result, field)) for field in group_fields)
        groups.setdefault(key, []).append(result)
    return [
        _summarize_result_group(group_fields, key, group_results)
        for key, group_results in sorted(groups.items())
    ]


def _summarize_result_group(
    group_fields: tuple[str, ...],
    key: tuple[str, ...],
    results: list[ClipResult],
) -> dict[str, object]:
    summary: dict[str, object] = {
        field: value for field, value in zip(group_fields, key, strict=True)
    }
    summary.update(
        {
            "clips": len(results),
            "subjects": len({result.subject for result in results}),
            "source_frames": sum(result.source_frames for result in results),
            "evaluated_frames": sum(result.evaluated_frames for result in results),
            "mean_joint_errors_m": _mean_joint_errors(results),
        }
    )
    for metric in SUMMARY_METRICS:
        summary[metric] = _mean(float(getattr(result, metric)) for result in results)
    return summary


def _worst_clips(
    results: list[ClipResult],
    limit: int,
) -> dict[str, list[dict[str, object]]]:
    if limit <= 0:
        return {metric: [] for metric in TOP_CLIP_METRICS}
    return {
        metric: [
            _clip_metric_summary(result, metric)
            for result in sorted(
                results,
                key=lambda clip_result: float(getattr(clip_result, metric)),
                reverse=True,
            )[:limit]
        ]
        for metric in TOP_CLIP_METRICS
    }


def _clip_metric_summary(result: ClipResult, metric: str) -> dict[str, object]:
    return {
        "clip": result.clip,
        "subject": result.subject,
        "environment_split": result.environment_split,
        "activity_class": result.activity_class,
        "motion_description": result.motion_description,
        "evaluated_frames": result.evaluated_frames,
        metric: float(getattr(result, metric)),
    }


if __name__ == "__main__":
    raise SystemExit(main())
