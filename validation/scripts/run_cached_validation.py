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

from validation.scripts.run_baseline_validation import (
    AVATAR_PLAUSIBILITY_SUMMARY_FIELDS,
    REFERENCE_AVATAR_PLAUSIBILITY_SUMMARY_FIELDS,
    ClipResult,
    write_clip_csv,
)
from validation.vr_pose_validation.baseline import EstimatorName, create_estimator
from validation.vr_pose_validation.calibration import (
    calibration_from_reference_frames,
    humanoid_poses_from_reference_frames,
)
from validation.vr_pose_validation.cmu_metadata import ActivityClass, EnvironmentSplit
from validation.vr_pose_validation.geometry import Transform, add
from validation.vr_pose_validation.metrics import (
    FLEXION_SUMMARY_FIELDS,
    GAIT_PLAUSIBILITY_SUMMARY_FIELDS,
    ORIENTATION_SUMMARY_FIELDS,
    REFERENCE_ERROR_DISTRIBUTION_FIELDS,
    TEMPORAL_SUMMARY_FIELDS,
    ReferencePoseMetrics,
    evaluate_foot_contact_sequence,
    evaluate_frame,
    evaluate_gait_plausibility_sequence,
    evaluate_reference_pose,
    infer_reference_floor_height,
    summarize_joint_flexion_sequence,
    summarize_orientation_sequence,
    summarize_reference_error_distribution,
    summarize_reference_errors,
    summarize_target_errors,
    summarize_temporal_reference_sequence,
)
from validation.vr_pose_validation.parquet_cache import (
    CachedPoseFrame,
    cache_summary,
    filter_clip_cache_files_by_environment,
    iter_cached_pose_frames,
    list_clip_cache_files,
)
from validation.vr_pose_validation.plausibility import summarize_avatar_plausibility
from validation.vr_pose_validation.posture import summarize_reference_posture
from validation.vr_pose_validation.synthetic_trackers import (
    SyntheticTrackerConfig,
    head_tracker_rotation_from_pose_frame,
)
from validation.vr_pose_validation.types import (
    GroundingContext,
    PoseEstimator,
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
    "mean_head_reference_error_m",
    "mean_hand_reference_error_m",
    "mean_upper_body_reference_error_m",
    "mean_lower_body_reference_error_m",
    "mean_full_body_reference_error_m",
    "mean_shoulder_error_m",
    "mean_elbow_error_m",
    "mean_knee_error_m",
    "mean_left_foot_error_m",
    "mean_right_foot_error_m",
    "mean_reference_error_m",
    *REFERENCE_ERROR_DISTRIBUTION_FIELDS,
    *TEMPORAL_SUMMARY_FIELDS,
    *FLEXION_SUMMARY_FIELDS,
    *ORIENTATION_SUMMARY_FIELDS,
    *GAIT_PLAUSIBILITY_SUMMARY_FIELDS,
    *AVATAR_PLAUSIBILITY_SUMMARY_FIELDS,
    *REFERENCE_AVATAR_PLAUSIBILITY_SUMMARY_FIELDS,
    "mean_contact_foot_error_m",
    "mean_contact_slip_m_per_s",
)
TOP_CLIP_METRICS = (
    "mean_reference_error_m",
    "mean_pelvis_error_m",
    "mean_pelvis_horizontal_error_m",
    "mean_pelvis_vertical_error_m",
    "mean_torso_error_m",
    "mean_hand_reference_error_m",
    "mean_upper_body_reference_error_m",
    "mean_lower_body_reference_error_m",
    "mean_full_body_reference_error_m",
    "mean_shoulder_error_m",
    "mean_elbow_error_m",
    "mean_knee_error_m",
    "mean_left_foot_error_m",
    "mean_right_foot_error_m",
    "reference_error_p95_m",
    "pct_reference_error_over_50cm",
    "mean_joint_velocity_error_m_per_s",
    "mean_joint_acceleration_error_m_per_s2",
    "mean_joint_estimated_jerk_m_per_s3",
    "mean_elbow_flexion_error_deg",
    "mean_knee_flexion_error_deg",
    "mean_joint_flexion_error_deg",
    "pct_elbow_flexion_over_limit",
    "pct_knee_flexion_over_limit",
    "pct_joint_flexion_over_limit",
    "mean_head_forward_error_deg",
    "head_forward_error_p95_deg",
    "pct_head_forward_error_over_90deg",
    "mean_head_up_error_deg",
    "head_up_error_p95_deg",
    "pct_head_up_error_over_90deg",
    "mean_neck_axis_error_deg",
    "neck_axis_error_p95_deg",
    "pct_neck_axis_error_over_45deg",
    "mean_spine_axis_error_deg",
    "spine_axis_error_p95_deg",
    "pct_spine_axis_error_over_45deg",
    "p95_head_forward_angular_speed_deg_s",
    "max_head_forward_angular_speed_deg_s",
    "pct_head_forward_angular_speed_over_720deg_s",
    "p95_reference_head_forward_angular_speed_deg_s",
    "max_reference_head_forward_angular_speed_deg_s",
    "pct_reference_head_forward_angular_speed_over_720deg_s",
    "p95_reference_chest_forward_angular_speed_deg_s",
    "max_reference_chest_forward_angular_speed_deg_s",
    "p95_reference_spine_forward_angular_speed_deg_s",
    "max_reference_spine_forward_angular_speed_deg_s",
    "pct_reference_fast_head_turns_with_torso_follow",
    "pct_estimated_fast_head_turns_without_reference_motion",
    "mean_contact_foot_error_m",
    "mean_contact_slip_m_per_s",
    "mean_low_foot_slip_m_per_s",
    "p95_low_foot_slip_m_per_s",
    "pct_low_foot_slip_speed_over_20cm_s",
    "mean_hips_support_distance_m",
    "pct_hips_support_distance_over_60cm",
    "pct_hips_support_distance_over_90cm",
    "pct_estimated_stance_width_under_20cm",
    "pct_estimated_stance_width_over_90cm",
    "pct_estimated_leg_reach_over_110cm",
    "avatar_implausibility_score_0_100",
    "avatar_joint_limit_penalty",
    "avatar_sustained_awkwardness_penalty",
    "avatar_endpoint_stretch_penalty",
    "avatar_temporal_jerk_penalty",
    "avatar_pct_joint_limit_violation_frames",
    "avatar_pct_sustained_awkward_frames",
    "avatar_max_endpoint_stretch_ratio",
    "avatar_p95_endpoint_stretch_ratio",
    "avatar_p95_joint_jerk_m_per_s3",
    "avatar_max_joint_jerk_m_per_s3",
    "reference_avatar_implausibility_score_0_100",
    "reference_avatar_joint_limit_penalty",
    "reference_avatar_endpoint_stretch_penalty",
    "reference_avatar_temporal_jerk_penalty",
    "reference_avatar_max_endpoint_stretch_ratio",
    "reference_avatar_p95_endpoint_stretch_ratio",
    "reference_avatar_p95_joint_jerk_m_per_s3",
    "reference_avatar_max_joint_jerk_m_per_s3",
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
            "swimming",
            "floor_pose",
            "sports",
            "unknown",
        ),
        nargs="*",
        help="Only evaluate clips with these cached activity class labels.",
    )
    parser.add_argument(
        "--estimator",
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
            "mean_head_reference_error_m": _mean_column(
                summary_frame,
                "mean_head_reference_error_m",
            ),
            "mean_left_hand_reference_error_m": _mean_column(
                summary_frame,
                "mean_left_hand_reference_error_m",
            ),
            "mean_right_hand_reference_error_m": _mean_column(
                summary_frame,
                "mean_right_hand_reference_error_m",
            ),
            "mean_hand_reference_error_m": _mean_column(
                summary_frame,
                "mean_hand_reference_error_m",
            ),
            "mean_upper_body_reference_error_m": _mean_column(
                summary_frame,
                "mean_upper_body_reference_error_m",
            ),
            "mean_lower_body_reference_error_m": _mean_column(
                summary_frame,
                "mean_lower_body_reference_error_m",
            ),
            "mean_full_body_reference_error_m": _mean_column(
                summary_frame,
                "mean_full_body_reference_error_m",
            ),
            "mean_left_shoulder_error_m": _mean_column(
                summary_frame,
                "mean_left_shoulder_error_m",
            ),
            "mean_right_shoulder_error_m": _mean_column(
                summary_frame,
                "mean_right_shoulder_error_m",
            ),
            "mean_shoulder_error_m": _mean_column(summary_frame, "mean_shoulder_error_m"),
            "mean_left_elbow_error_m": _mean_column(
                summary_frame,
                "mean_left_elbow_error_m",
            ),
            "mean_right_elbow_error_m": _mean_column(
                summary_frame,
                "mean_right_elbow_error_m",
            ),
            "mean_elbow_error_m": _mean_column(summary_frame, "mean_elbow_error_m"),
            "mean_left_knee_error_m": _mean_column(summary_frame, "mean_left_knee_error_m"),
            "mean_right_knee_error_m": _mean_column(summary_frame, "mean_right_knee_error_m"),
            "mean_knee_error_m": _mean_column(summary_frame, "mean_knee_error_m"),
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
            "gait_low_foot_frames": _sum_column(summary_frame, "gait_low_foot_frames"),
            "avatar_plausibility_bucket_counts": _avatar_plausibility_bucket_counts(clip_results),
            "reference_avatar_plausibility_bucket_counts": (
                _reference_avatar_plausibility_bucket_counts(clip_results)
            ),
            **{
                metric: _mean_column(summary_frame, metric)
                for metric in (
                    *REFERENCE_ERROR_DISTRIBUTION_FIELDS,
                    *TEMPORAL_SUMMARY_FIELDS,
                    *FLEXION_SUMMARY_FIELDS,
                    *ORIENTATION_SUMMARY_FIELDS,
                    *GAIT_PLAUSIBILITY_SUMMARY_FIELDS,
                    *AVATAR_PLAUSIBILITY_SUMMARY_FIELDS,
                    *REFERENCE_AVATAR_PLAUSIBILITY_SUMMARY_FIELDS,
                )
            },
            "mean_joint_errors_m": _mean_joint_errors(clip_results),
            "posture_frame_counts": _posture_frame_counts(clip_results),
            "posture_subset_counts": _posture_subset_counts(clip_results),
            "posture_vr_candidate_clips": sum(
                1 for result in clip_results if result.posture_vr_candidate
            ),
            "posture_vr_candidate_frames": sum(
                result.evaluated_frames for result in clip_results if result.posture_vr_candidate
            ),
            "grouped_metrics": _grouped_metrics(clip_results),
            "worst_clips": _worst_clips(clip_results, top_clip_count),
        },
        "clips": [asdict(result) for result in clip_results],
    }


def _evaluate_cached_clip(task: CachedValidationTask) -> ClipResult:
    estimator = create_estimator(task.estimator_name)
    return evaluate_cached_clip_path(Path(task.path), estimator)


def evaluate_cached_clip_path(path: Path, estimator: PoseEstimator) -> ClipResult:
    """Evaluate one cached clip with a caller-provided stateful estimator instance."""

    start = time.perf_counter()
    frame_metrics = []
    reference_metrics: list[ReferencePoseMetrics] = []
    estimated_poses = []
    cached_frame_list = list(iter_cached_pose_frames(path))
    reference_frames = [frame.joints for frame in cached_frame_list]
    timestamps_seconds = [frame.timestamp_s for frame in cached_frame_list]
    inferred_floor_height_m = infer_reference_floor_height(reference_frames)
    calibration = calibration_from_reference_frames(
        reference_frames,
        floor_height_m=inferred_floor_height_m,
    )
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
    tail_summary = summarize_reference_error_distribution(reference_metrics)
    temporal_summary = summarize_temporal_reference_sequence(
        estimated_poses,
        reference_frames,
    )
    flexion_summary = summarize_joint_flexion_sequence(
        estimated_poses,
        reference_frames,
    )
    orientation_summary = summarize_orientation_sequence(
        estimated_poses,
        reference_frames,
    )
    contact_summary = evaluate_foot_contact_sequence(
        estimated_poses,
        reference_frames,
        floor_height_m=inferred_floor_height_m,
    )
    gait_plausibility_summary = evaluate_gait_plausibility_sequence(
        estimated_poses,
        floor_height_m=inferred_floor_height_m,
    )
    avatar_plausibility_summary = summarize_avatar_plausibility(
        estimated_poses,
        calibration=calibration,
        floor_height_m=inferred_floor_height_m,
    )
    reference_avatar_plausibility_summary = summarize_avatar_plausibility(
        humanoid_poses_from_reference_frames(
            reference_frames,
            timestamps_seconds=timestamps_seconds,
        ),
        calibration=calibration,
        floor_height_m=inferred_floor_height_m,
    )
    posture_summary = summarize_reference_posture(
        reference_frames,
        environment_split=environment_split,
        activity_class=activity_class,
        floor_height_m=inferred_floor_height_m,
        timestamps_seconds=timestamps_seconds,
    )
    mean_joint_errors_m = cast(dict[str, float], reference_summary["mean_joint_errors_m"])
    subject, clip = _subject_and_clip_from_path(path)
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
        mean_head_reference_error_m=cast(
            float,
            reference_summary["mean_head_reference_error_m"],
        ),
        mean_left_hand_reference_error_m=cast(
            float,
            reference_summary["mean_left_hand_reference_error_m"],
        ),
        mean_right_hand_reference_error_m=cast(
            float,
            reference_summary["mean_right_hand_reference_error_m"],
        ),
        mean_hand_reference_error_m=cast(
            float,
            reference_summary["mean_hand_reference_error_m"],
        ),
        mean_upper_body_reference_error_m=cast(
            float,
            reference_summary["mean_upper_body_reference_error_m"],
        ),
        mean_lower_body_reference_error_m=cast(
            float,
            reference_summary["mean_lower_body_reference_error_m"],
        ),
        mean_full_body_reference_error_m=cast(
            float,
            reference_summary["mean_full_body_reference_error_m"],
        ),
        mean_left_shoulder_error_m=cast(
            float,
            reference_summary["mean_left_shoulder_error_m"],
        ),
        mean_right_shoulder_error_m=cast(
            float,
            reference_summary["mean_right_shoulder_error_m"],
        ),
        mean_shoulder_error_m=cast(float, reference_summary["mean_shoulder_error_m"]),
        mean_left_elbow_error_m=cast(float, reference_summary["mean_left_elbow_error_m"]),
        mean_right_elbow_error_m=cast(float, reference_summary["mean_right_elbow_error_m"]),
        mean_elbow_error_m=cast(float, reference_summary["mean_elbow_error_m"]),
        mean_left_knee_error_m=cast(float, reference_summary["mean_left_knee_error_m"]),
        mean_right_knee_error_m=cast(float, reference_summary["mean_right_knee_error_m"]),
        mean_knee_error_m=cast(float, reference_summary["mean_knee_error_m"]),
        mean_left_foot_error_m=cast(float, reference_summary["mean_left_foot_error_m"]),
        mean_right_foot_error_m=cast(float, reference_summary["mean_right_foot_error_m"]),
        mean_reference_error_m=cast(float, reference_summary["mean_reference_error_m"]),
        reference_error_p50_m=tail_summary["reference_error_p50_m"],
        reference_error_p90_m=tail_summary["reference_error_p90_m"],
        reference_error_p95_m=tail_summary["reference_error_p95_m"],
        reference_error_p99_m=tail_summary["reference_error_p99_m"],
        pct_reference_error_over_25cm=tail_summary["pct_reference_error_over_25cm"],
        pct_reference_error_over_50cm=tail_summary["pct_reference_error_over_50cm"],
        pct_reference_error_over_100cm=tail_summary["pct_reference_error_over_100cm"],
        pelvis_error_p95_m=tail_summary["pelvis_error_p95_m"],
        torso_error_p95_m=tail_summary["torso_error_p95_m"],
        shoulder_error_p95_m=tail_summary["shoulder_error_p95_m"],
        elbow_error_p95_m=tail_summary["elbow_error_p95_m"],
        knee_error_p95_m=tail_summary["knee_error_p95_m"],
        feet_error_p95_m=tail_summary["feet_error_p95_m"],
        mean_joint_velocity_error_m_per_s=temporal_summary["mean_joint_velocity_error_m_per_s"],
        mean_joint_acceleration_error_m_per_s2=temporal_summary[
            "mean_joint_acceleration_error_m_per_s2"
        ],
        mean_joint_estimated_jerk_m_per_s3=temporal_summary["mean_joint_estimated_jerk_m_per_s3"],
        mean_pelvis_velocity_error_m_per_s=temporal_summary["mean_pelvis_velocity_error_m_per_s"],
        mean_pelvis_acceleration_error_m_per_s2=temporal_summary[
            "mean_pelvis_acceleration_error_m_per_s2"
        ],
        mean_pelvis_estimated_jerk_m_per_s3=temporal_summary["mean_pelvis_estimated_jerk_m_per_s3"],
        mean_torso_velocity_error_m_per_s=temporal_summary["mean_torso_velocity_error_m_per_s"],
        mean_torso_acceleration_error_m_per_s2=temporal_summary[
            "mean_torso_acceleration_error_m_per_s2"
        ],
        mean_torso_estimated_jerk_m_per_s3=temporal_summary["mean_torso_estimated_jerk_m_per_s3"],
        mean_shoulder_velocity_error_m_per_s=temporal_summary[
            "mean_shoulder_velocity_error_m_per_s"
        ],
        mean_shoulder_acceleration_error_m_per_s2=temporal_summary[
            "mean_shoulder_acceleration_error_m_per_s2"
        ],
        mean_shoulder_estimated_jerk_m_per_s3=temporal_summary[
            "mean_shoulder_estimated_jerk_m_per_s3"
        ],
        mean_elbow_velocity_error_m_per_s=temporal_summary["mean_elbow_velocity_error_m_per_s"],
        mean_elbow_acceleration_error_m_per_s2=temporal_summary[
            "mean_elbow_acceleration_error_m_per_s2"
        ],
        mean_elbow_estimated_jerk_m_per_s3=temporal_summary["mean_elbow_estimated_jerk_m_per_s3"],
        mean_knee_velocity_error_m_per_s=temporal_summary["mean_knee_velocity_error_m_per_s"],
        mean_knee_acceleration_error_m_per_s2=temporal_summary[
            "mean_knee_acceleration_error_m_per_s2"
        ],
        mean_knee_estimated_jerk_m_per_s3=temporal_summary["mean_knee_estimated_jerk_m_per_s3"],
        mean_feet_velocity_error_m_per_s=temporal_summary["mean_feet_velocity_error_m_per_s"],
        mean_feet_acceleration_error_m_per_s2=temporal_summary[
            "mean_feet_acceleration_error_m_per_s2"
        ],
        mean_feet_estimated_jerk_m_per_s3=temporal_summary["mean_feet_estimated_jerk_m_per_s3"],
        mean_joint_errors_m=mean_joint_errors_m,
        inferred_floor_height_m=inferred_floor_height_m,
        contact_frames=contact_summary.contact_frames,
        mean_contact_foot_error_m=contact_summary.mean_contact_foot_error_m,
        left_contact_slip_m_per_s=contact_summary.left_contact_slip_m_per_s,
        right_contact_slip_m_per_s=contact_summary.right_contact_slip_m_per_s,
        mean_contact_slip_m_per_s=contact_summary.mean_contact_slip_m_per_s,
        gait_low_foot_frames=gait_plausibility_summary.low_foot_frames,
        mean_low_foot_slip_m_per_s=gait_plausibility_summary.mean_low_foot_slip_m_per_s,
        p95_low_foot_slip_m_per_s=gait_plausibility_summary.p95_low_foot_slip_m_per_s,
        pct_low_foot_slip_speed_over_20cm_s=(
            gait_plausibility_summary.pct_low_foot_slip_speed_over_20cm_s
        ),
        mean_hips_support_distance_m=gait_plausibility_summary.mean_hips_support_distance_m,
        pct_hips_support_distance_over_60cm=(
            gait_plausibility_summary.pct_hips_support_distance_over_60cm
        ),
        pct_hips_support_distance_over_90cm=(
            gait_plausibility_summary.pct_hips_support_distance_over_90cm
        ),
        mean_estimated_stance_width_m=gait_plausibility_summary.mean_estimated_stance_width_m,
        pct_estimated_stance_width_under_20cm=(
            gait_plausibility_summary.pct_estimated_stance_width_under_20cm
        ),
        pct_estimated_stance_width_over_90cm=(
            gait_plausibility_summary.pct_estimated_stance_width_over_90cm
        ),
        mean_estimated_leg_reach_m=gait_plausibility_summary.mean_estimated_leg_reach_m,
        pct_estimated_leg_reach_over_110cm=(
            gait_plausibility_summary.pct_estimated_leg_reach_over_110cm
        ),
        environment_split=environment_split,
        environment_tags=environment_tags,
        activity_class=activity_class,
        activity_tags=activity_tags,
        motion_description=motion_description,
        mean_left_elbow_flexion_error_deg=flexion_summary["mean_left_elbow_flexion_error_deg"],
        mean_right_elbow_flexion_error_deg=flexion_summary["mean_right_elbow_flexion_error_deg"],
        mean_elbow_flexion_error_deg=flexion_summary["mean_elbow_flexion_error_deg"],
        mean_left_knee_flexion_error_deg=flexion_summary["mean_left_knee_flexion_error_deg"],
        mean_right_knee_flexion_error_deg=flexion_summary["mean_right_knee_flexion_error_deg"],
        mean_knee_flexion_error_deg=flexion_summary["mean_knee_flexion_error_deg"],
        mean_joint_flexion_error_deg=flexion_summary["mean_joint_flexion_error_deg"],
        pct_left_elbow_flexion_over_limit=flexion_summary["pct_left_elbow_flexion_over_limit"],
        pct_right_elbow_flexion_over_limit=flexion_summary["pct_right_elbow_flexion_over_limit"],
        pct_elbow_flexion_over_limit=flexion_summary["pct_elbow_flexion_over_limit"],
        pct_left_knee_flexion_over_limit=flexion_summary["pct_left_knee_flexion_over_limit"],
        pct_right_knee_flexion_over_limit=flexion_summary["pct_right_knee_flexion_over_limit"],
        pct_knee_flexion_over_limit=flexion_summary["pct_knee_flexion_over_limit"],
        pct_joint_flexion_over_limit=flexion_summary["pct_joint_flexion_over_limit"],
        mean_head_forward_error_deg=orientation_summary["mean_head_forward_error_deg"],
        head_forward_error_p95_deg=orientation_summary["head_forward_error_p95_deg"],
        pct_head_forward_error_over_90deg=orientation_summary["pct_head_forward_error_over_90deg"],
        mean_head_up_error_deg=orientation_summary["mean_head_up_error_deg"],
        head_up_error_p95_deg=orientation_summary["head_up_error_p95_deg"],
        pct_head_up_error_over_90deg=orientation_summary["pct_head_up_error_over_90deg"],
        mean_neck_axis_error_deg=orientation_summary["mean_neck_axis_error_deg"],
        neck_axis_error_p95_deg=orientation_summary["neck_axis_error_p95_deg"],
        pct_neck_axis_error_over_45deg=orientation_summary["pct_neck_axis_error_over_45deg"],
        mean_spine_axis_error_deg=orientation_summary["mean_spine_axis_error_deg"],
        spine_axis_error_p95_deg=orientation_summary["spine_axis_error_p95_deg"],
        pct_spine_axis_error_over_45deg=orientation_summary["pct_spine_axis_error_over_45deg"],
        p95_head_forward_angular_speed_deg_s=orientation_summary[
            "p95_head_forward_angular_speed_deg_s"
        ],
        max_head_forward_angular_speed_deg_s=orientation_summary[
            "max_head_forward_angular_speed_deg_s"
        ],
        pct_head_forward_angular_speed_over_720deg_s=orientation_summary[
            "pct_head_forward_angular_speed_over_720deg_s"
        ],
        p95_reference_head_forward_angular_speed_deg_s=orientation_summary[
            "p95_reference_head_forward_angular_speed_deg_s"
        ],
        max_reference_head_forward_angular_speed_deg_s=orientation_summary[
            "max_reference_head_forward_angular_speed_deg_s"
        ],
        pct_reference_head_forward_angular_speed_over_720deg_s=orientation_summary[
            "pct_reference_head_forward_angular_speed_over_720deg_s"
        ],
        p95_reference_chest_forward_angular_speed_deg_s=orientation_summary[
            "p95_reference_chest_forward_angular_speed_deg_s"
        ],
        max_reference_chest_forward_angular_speed_deg_s=orientation_summary[
            "max_reference_chest_forward_angular_speed_deg_s"
        ],
        p95_reference_spine_forward_angular_speed_deg_s=orientation_summary[
            "p95_reference_spine_forward_angular_speed_deg_s"
        ],
        max_reference_spine_forward_angular_speed_deg_s=orientation_summary[
            "max_reference_spine_forward_angular_speed_deg_s"
        ],
        pct_reference_fast_head_turns_with_torso_follow=orientation_summary[
            "pct_reference_fast_head_turns_with_torso_follow"
        ],
        pct_estimated_fast_head_turns_without_reference_motion=orientation_summary[
            "pct_estimated_fast_head_turns_without_reference_motion"
        ],
        avatar_plausibility_bucket=avatar_plausibility_summary.bucket,
        avatar_plausibility_tags=avatar_plausibility_summary.tags,
        avatar_plausibility_score_0_100=avatar_plausibility_summary.plausibility_score_0_100,
        avatar_implausibility_score_0_100=(
            100.0 - avatar_plausibility_summary.plausibility_score_0_100
        ),
        avatar_joint_limit_penalty=avatar_plausibility_summary.joint_limit_penalty,
        avatar_sustained_awkwardness_penalty=(
            avatar_plausibility_summary.sustained_awkwardness_penalty
        ),
        avatar_endpoint_stretch_penalty=avatar_plausibility_summary.endpoint_stretch_penalty,
        avatar_temporal_jerk_penalty=avatar_plausibility_summary.temporal_jerk_penalty,
        avatar_min_joint_limit_margin_deg=avatar_plausibility_summary.min_joint_limit_margin_deg,
        avatar_mean_joint_limit_excess_deg=(
            avatar_plausibility_summary.mean_joint_limit_excess_deg
        ),
        avatar_pct_joint_limit_violation_frames=(
            avatar_plausibility_summary.pct_joint_limit_violation_frames
        ),
        avatar_pct_sustained_awkward_frames=(
            avatar_plausibility_summary.pct_sustained_awkward_frames
        ),
        avatar_max_endpoint_stretch_ratio=avatar_plausibility_summary.max_endpoint_stretch_ratio,
        avatar_p95_endpoint_stretch_ratio=avatar_plausibility_summary.p95_endpoint_stretch_ratio,
        avatar_pct_endpoint_stretch_over_105pct=(
            avatar_plausibility_summary.pct_endpoint_stretch_over_105pct
        ),
        avatar_p95_joint_jerk_m_per_s3=avatar_plausibility_summary.p95_joint_jerk_m_per_s3,
        avatar_max_joint_jerk_m_per_s3=avatar_plausibility_summary.max_joint_jerk_m_per_s3,
        avatar_pct_joint_jerk_over_threshold=(
            avatar_plausibility_summary.pct_joint_jerk_over_threshold
        ),
        reference_avatar_plausibility_bucket=reference_avatar_plausibility_summary.bucket,
        reference_avatar_plausibility_tags=reference_avatar_plausibility_summary.tags,
        reference_avatar_plausibility_score_0_100=(
            reference_avatar_plausibility_summary.plausibility_score_0_100
        ),
        reference_avatar_implausibility_score_0_100=(
            100.0 - reference_avatar_plausibility_summary.plausibility_score_0_100
        ),
        reference_avatar_joint_limit_penalty=(
            reference_avatar_plausibility_summary.joint_limit_penalty
        ),
        reference_avatar_sustained_awkwardness_penalty=(
            reference_avatar_plausibility_summary.sustained_awkwardness_penalty
        ),
        reference_avatar_endpoint_stretch_penalty=(
            reference_avatar_plausibility_summary.endpoint_stretch_penalty
        ),
        reference_avatar_temporal_jerk_penalty=(
            reference_avatar_plausibility_summary.temporal_jerk_penalty
        ),
        reference_avatar_min_joint_limit_margin_deg=(
            reference_avatar_plausibility_summary.min_joint_limit_margin_deg
        ),
        reference_avatar_mean_joint_limit_excess_deg=(
            reference_avatar_plausibility_summary.mean_joint_limit_excess_deg
        ),
        reference_avatar_pct_joint_limit_violation_frames=(
            reference_avatar_plausibility_summary.pct_joint_limit_violation_frames
        ),
        reference_avatar_pct_sustained_awkward_frames=(
            reference_avatar_plausibility_summary.pct_sustained_awkward_frames
        ),
        reference_avatar_max_endpoint_stretch_ratio=(
            reference_avatar_plausibility_summary.max_endpoint_stretch_ratio
        ),
        reference_avatar_p95_endpoint_stretch_ratio=(
            reference_avatar_plausibility_summary.p95_endpoint_stretch_ratio
        ),
        reference_avatar_pct_endpoint_stretch_over_105pct=(
            reference_avatar_plausibility_summary.pct_endpoint_stretch_over_105pct
        ),
        reference_avatar_p95_joint_jerk_m_per_s3=(
            reference_avatar_plausibility_summary.p95_joint_jerk_m_per_s3
        ),
        reference_avatar_max_joint_jerk_m_per_s3=(
            reference_avatar_plausibility_summary.max_joint_jerk_m_per_s3
        ),
        reference_avatar_pct_joint_jerk_over_threshold=(
            reference_avatar_plausibility_summary.pct_joint_jerk_over_threshold
        ),
        posture_class=posture_summary.posture_class,
        posture_subset=posture_summary.posture_subset,
        posture_vr_candidate=posture_summary.vr_candidate,
        posture_frame_count=posture_summary.frame_count,
        posture_upright_frames=posture_summary.upright_frames,
        posture_crouched_frames=posture_summary.crouched_frames,
        posture_sitting_frames=posture_summary.sitting_frames,
        posture_reclined_frames=posture_summary.reclined_frames,
        posture_floor_reclined_frames=posture_summary.floor_reclined_frames,
        posture_inverted_frames=posture_summary.inverted_frames,
        posture_unknown_frames=posture_summary.unknown_frames,
        posture_upright_pct=posture_summary.upright_pct,
        posture_crouched_pct=posture_summary.crouched_pct,
        posture_sitting_pct=posture_summary.sitting_pct,
        posture_reclined_pct=posture_summary.reclined_pct,
        posture_floor_reclined_pct=posture_summary.floor_reclined_pct,
        posture_inverted_pct=posture_summary.inverted_pct,
        posture_unknown_pct=posture_summary.unknown_pct,
        mean_head_height_m=posture_summary.mean_head_height_m,
        mean_hip_height_m=posture_summary.mean_hip_height_m,
        mean_torso_verticality=posture_summary.mean_torso_verticality,
        hip_horizontal_displacement_m=posture_summary.hip_horizontal_displacement_m,
        hip_horizontal_path_m=posture_summary.hip_horizontal_path_m,
        mean_hip_horizontal_speed_m_per_s=posture_summary.mean_hip_horizontal_speed_m_per_s,
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
    rotation = (
        head_tracker_rotation_from_pose_frame(frame.joints) if role == "head" else source.rotation
    )
    transform = Transform(
        position=add(source.position, attachment.local_transform.position),
        rotation=rotation,
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
    data.pop("avatar_plausibility_tags", None)
    data.pop("reference_avatar_plausibility_tags", None)
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


def _avatar_plausibility_bucket_counts(results: Iterable[ClipResult]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        counts[result.avatar_plausibility_bucket] = (
            counts.get(result.avatar_plausibility_bucket, 0) + 1
        )
    return dict(sorted(counts.items()))


def _reference_avatar_plausibility_bucket_counts(
    results: Iterable[ClipResult],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        counts[result.reference_avatar_plausibility_bucket] = (
            counts.get(result.reference_avatar_plausibility_bucket, 0) + 1
        )
    return dict(sorted(counts.items()))


def _grouped_metrics(results: list[ClipResult]) -> dict[str, list[dict[str, object]]]:
    return {
        "environment_split": _group_results(results, ("environment_split",)),
        "activity_class": _group_results(results, ("activity_class",)),
        "posture_class": _group_results(results, ("posture_class",)),
        "posture_subset": _group_results(results, ("posture_subset",)),
        "environment_split_activity_class": _group_results(
            results,
            ("environment_split", "activity_class"),
        ),
        "activity_class_posture_subset": _group_results(
            results,
            ("activity_class", "posture_subset"),
        ),
        "environment_split_posture_subset": _group_results(
            results,
            ("environment_split", "posture_subset"),
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
            "posture_frame_counts": _posture_frame_counts(results),
            "posture_subset_counts": _posture_subset_counts(results),
            "posture_vr_candidate_clips": sum(
                1 for result in results if result.posture_vr_candidate
            ),
            "posture_vr_candidate_frames": sum(
                result.evaluated_frames for result in results if result.posture_vr_candidate
            ),
            "avatar_plausibility_bucket_counts": _avatar_plausibility_bucket_counts(results),
            "reference_avatar_plausibility_bucket_counts": (
                _reference_avatar_plausibility_bucket_counts(results)
            ),
            "posture_upright_pct": _mean(result.posture_upright_pct for result in results),
            "posture_crouched_pct": _mean(result.posture_crouched_pct for result in results),
            "posture_sitting_pct": _mean(result.posture_sitting_pct for result in results),
            "posture_reclined_pct": _mean(result.posture_reclined_pct for result in results),
            "posture_floor_reclined_pct": _mean(
                result.posture_floor_reclined_pct for result in results
            ),
            "posture_inverted_pct": _mean(result.posture_inverted_pct for result in results),
            "mean_head_height_m": _mean(result.mean_head_height_m for result in results),
            "mean_hip_height_m": _mean(result.mean_hip_height_m for result in results),
            "mean_torso_verticality": _mean(result.mean_torso_verticality for result in results),
            "mean_hip_horizontal_speed_m_per_s": _mean(
                result.mean_hip_horizontal_speed_m_per_s for result in results
            ),
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
        "posture_class": result.posture_class,
        "posture_subset": result.posture_subset,
        "motion_description": result.motion_description,
        "evaluated_frames": result.evaluated_frames,
        metric: float(getattr(result, metric)),
    }


def _posture_frame_counts(results: Iterable[ClipResult]) -> dict[str, int]:
    result_list = list(results)
    return {
        "upright": sum(result.posture_upright_frames for result in result_list),
        "crouched": sum(result.posture_crouched_frames for result in result_list),
        "sitting": sum(result.posture_sitting_frames for result in result_list),
        "reclined": sum(result.posture_reclined_frames for result in result_list),
        "floor_reclined": sum(result.posture_floor_reclined_frames for result in result_list),
        "inverted": sum(result.posture_inverted_frames for result in result_list),
        "unknown": sum(result.posture_unknown_frames for result in result_list),
    }


def _posture_subset_counts(results: Iterable[ClipResult]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        counts[result.posture_subset] = counts.get(result.posture_subset, 0) + 1
    return dict(sorted(counts.items()))


if __name__ == "__main__":
    raise SystemExit(main())
