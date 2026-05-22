"""Run trivial baseline validation over CMU ASF/AMC clips."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast
from zipfile import ZipFile

from validation.vr_pose_validation.baseline import EstimatorName, create_estimator
from validation.vr_pose_validation.cmu import (
    CmuSkeleton,
    parse_amc_text_sampled,
    parse_asf_text,
)
from validation.vr_pose_validation.cmu_fk import (
    canonical_pose_from_cmu,
    iter_cmu_pose_frames,
)
from validation.vr_pose_validation.metrics import (
    FLEXION_SUMMARY_FIELDS,
    REFERENCE_ERROR_DISTRIBUTION_FIELDS,
    TEMPORAL_SUMMARY_FIELDS,
    ReferencePoseMetrics,
    evaluate_frame,
    evaluate_reference_pose,
    summarize_joint_flexion_sequence,
    summarize_reference_error_distribution,
    summarize_reference_errors,
    summarize_target_errors,
    summarize_temporal_reference_sequence,
)
from validation.vr_pose_validation.synthetic_trackers import iter_three_point_samples
from validation.vr_pose_validation.types import AvatarCalibration, GroundingContext

DEFAULT_ARCHIVE_PATH = Path(".cache/cmu/allasfamc.zip")
DEFAULT_REPORT_PATH = Path("validation/reports/baseline_smoke.json")


@dataclass(frozen=True, slots=True)
class ClipResult:
    clip: str
    subject: str
    source_frames: int
    evaluated_frames: int
    parse_seconds: float
    fk_sample_estimate_metric_seconds: float
    mean_target_error_m: float
    mean_pelvis_error_m: float
    mean_pelvis_horizontal_error_m: float
    mean_pelvis_vertical_error_m: float
    mean_chest_error_m: float
    mean_spine_error_m: float
    mean_torso_error_m: float
    mean_left_shoulder_error_m: float
    mean_right_shoulder_error_m: float
    mean_shoulder_error_m: float
    mean_left_elbow_error_m: float
    mean_right_elbow_error_m: float
    mean_elbow_error_m: float
    mean_left_knee_error_m: float
    mean_right_knee_error_m: float
    mean_knee_error_m: float
    mean_left_foot_error_m: float
    mean_right_foot_error_m: float
    mean_reference_error_m: float
    reference_error_p50_m: float
    reference_error_p90_m: float
    reference_error_p95_m: float
    reference_error_p99_m: float
    pct_reference_error_over_25cm: float
    pct_reference_error_over_50cm: float
    pct_reference_error_over_100cm: float
    pelvis_error_p95_m: float
    torso_error_p95_m: float
    shoulder_error_p95_m: float
    elbow_error_p95_m: float
    knee_error_p95_m: float
    feet_error_p95_m: float
    mean_joint_velocity_error_m_per_s: float
    mean_joint_acceleration_error_m_per_s2: float
    mean_joint_estimated_jerk_m_per_s3: float
    mean_pelvis_velocity_error_m_per_s: float
    mean_pelvis_acceleration_error_m_per_s2: float
    mean_pelvis_estimated_jerk_m_per_s3: float
    mean_torso_velocity_error_m_per_s: float
    mean_torso_acceleration_error_m_per_s2: float
    mean_torso_estimated_jerk_m_per_s3: float
    mean_shoulder_velocity_error_m_per_s: float
    mean_shoulder_acceleration_error_m_per_s2: float
    mean_shoulder_estimated_jerk_m_per_s3: float
    mean_elbow_velocity_error_m_per_s: float
    mean_elbow_acceleration_error_m_per_s2: float
    mean_elbow_estimated_jerk_m_per_s3: float
    mean_knee_velocity_error_m_per_s: float
    mean_knee_acceleration_error_m_per_s2: float
    mean_knee_estimated_jerk_m_per_s3: float
    mean_feet_velocity_error_m_per_s: float
    mean_feet_acceleration_error_m_per_s2: float
    mean_feet_estimated_jerk_m_per_s3: float
    mean_joint_errors_m: dict[str, float]
    inferred_floor_height_m: float = 0.0
    contact_frames: int = 0
    mean_contact_foot_error_m: float = 0.0
    left_contact_slip_m_per_s: float = 0.0
    right_contact_slip_m_per_s: float = 0.0
    mean_contact_slip_m_per_s: float = 0.0
    environment_split: str = "unknown"
    environment_tags: tuple[str, ...] = ()
    activity_class: str = "unknown"
    activity_tags: tuple[str, ...] = ()
    motion_description: str = ""
    mean_left_elbow_flexion_error_deg: float = 0.0
    mean_right_elbow_flexion_error_deg: float = 0.0
    mean_elbow_flexion_error_deg: float = 0.0
    mean_left_knee_flexion_error_deg: float = 0.0
    mean_right_knee_flexion_error_deg: float = 0.0
    mean_knee_flexion_error_deg: float = 0.0
    mean_joint_flexion_error_deg: float = 0.0
    pct_left_elbow_flexion_over_limit: float = 0.0
    pct_right_elbow_flexion_over_limit: float = 0.0
    pct_elbow_flexion_over_limit: float = 0.0
    pct_left_knee_flexion_over_limit: float = 0.0
    pct_right_knee_flexion_over_limit: float = 0.0
    pct_knee_flexion_over_limit: float = 0.0
    pct_joint_flexion_over_limit: float = 0.0
    posture_class: str = "unknown"
    posture_subset: str = "unknown_ood"
    posture_vr_candidate: bool = False
    posture_frame_count: int = 0
    posture_upright_frames: int = 0
    posture_crouched_frames: int = 0
    posture_sitting_frames: int = 0
    posture_reclined_frames: int = 0
    posture_floor_reclined_frames: int = 0
    posture_unknown_frames: int = 0
    posture_upright_pct: float = 0.0
    posture_crouched_pct: float = 0.0
    posture_sitting_pct: float = 0.0
    posture_reclined_pct: float = 0.0
    posture_floor_reclined_pct: float = 0.0
    posture_unknown_pct: float = 0.0
    mean_head_height_m: float = 0.0
    mean_hip_height_m: float = 0.0
    mean_torso_verticality: float = 0.0
    hip_horizontal_displacement_m: float = 0.0
    hip_horizontal_path_m: float = 0.0
    mean_hip_horizontal_speed_m_per_s: float = 0.0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--csv-report", type=Path)
    parser.add_argument("--subjects", nargs="*", help="Subject ids such as 07 08 09.")
    parser.add_argument("--max-clips", type=int)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames-per-clip", type=int)
    parser.add_argument(
        "--estimator",
        choices=("trivial", "planted", "basis", "basis_knee_hint"),
        default="trivial",
        help="Estimator implementation to evaluate.",
    )
    args = parser.parse_args(argv)

    result = run_validation(
        archive_path=args.archive,
        subjects=set(args.subjects or []),
        max_clips=args.max_clips,
        frame_stride=args.frame_stride,
        max_frames_per_clip=args.max_frames_per_clip,
        estimator_name=args.estimator,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.csv_report is not None:
        write_clip_csv(args.csv_report, cast(list[dict[str, object]], result["clips"]))

    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0


def run_validation(
    *,
    archive_path: Path,
    subjects: set[str],
    max_clips: int | None,
    frame_stride: int,
    max_frames_per_clip: int | None,
    estimator_name: EstimatorName = "trivial",
) -> dict[str, object]:
    start = time.perf_counter()
    clip_results: list[ClipResult] = []
    calibration = AvatarCalibration()
    grounding = GroundingContext()

    with ZipFile(archive_path) as zf:
        names = zf.namelist()
        asf_by_subject = {
            _subject_from_asf(name): name for name in names if name.lower().endswith(".asf")
        }
        amc_names = [
            name
            for name in sorted(names)
            if name.lower().endswith(".amc")
            and (not subjects or _subject_from_amc(name) in subjects)
        ]
        if max_clips is not None:
            amc_names = amc_names[:max_clips]

        skeleton_cache: dict[str, CmuSkeleton] = {}
        for amc_name in amc_names:
            subject = _subject_from_amc(amc_name)
            asf_name = asf_by_subject[subject]
            if subject not in skeleton_cache:
                skeleton_cache[subject] = parse_asf_text(
                    zf.read(asf_name).decode("utf-8", errors="replace")
                )

            parse_start = time.perf_counter()
            frame_selection = parse_amc_text_sampled(
                zf.read(amc_name).decode("utf-8", errors="replace"),
                frame_stride=frame_stride,
                max_frames=max_frames_per_clip,
            )
            parse_seconds = time.perf_counter() - parse_start
            work_start = time.perf_counter()
            estimator = create_estimator(estimator_name)
            frame_metrics = []
            reference_metrics: list[ReferencePoseMetrics] = []
            pose_frames = list(
                iter_cmu_pose_frames(
                    skeleton_cache[subject],
                    frame_selection.frames,
                )
            )
            canonical_frames = [canonical_pose_from_cmu(frame) for frame in pose_frames]
            samples = iter_three_point_samples(canonical_frames)
            estimated_poses = []
            for reference_frame, sample in zip(canonical_frames, samples, strict=True):
                pose = estimator.estimate(sample, calibration, grounding)
                estimated_poses.append(pose)
                frame_metrics.append(evaluate_frame(pose, sample, grounding.floor_height_m))
                reference_metrics.append(evaluate_reference_pose(pose, reference_frame))
            target_summary = summarize_target_errors(frame_metrics)
            reference_summary = summarize_reference_errors(reference_metrics)
            tail_summary = summarize_reference_error_distribution(reference_metrics)
            temporal_summary = summarize_temporal_reference_sequence(
                estimated_poses,
                canonical_frames,
            )
            flexion_summary = summarize_joint_flexion_sequence(
                estimated_poses,
                canonical_frames,
            )
            mean_joint_errors_m = cast(
                dict[str, float],
                reference_summary["mean_joint_errors_m"],
            )
            clip_results.append(
                ClipResult(
                    clip=amc_name,
                    subject=subject,
                    source_frames=frame_selection.source_frame_count,
                    evaluated_frames=len(canonical_frames),
                    parse_seconds=parse_seconds,
                    fk_sample_estimate_metric_seconds=time.perf_counter() - work_start,
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
                    mean_left_shoulder_error_m=cast(
                        float,
                        reference_summary["mean_left_shoulder_error_m"],
                    ),
                    mean_right_shoulder_error_m=cast(
                        float,
                        reference_summary["mean_right_shoulder_error_m"],
                    ),
                    mean_shoulder_error_m=cast(
                        float,
                        reference_summary["mean_shoulder_error_m"],
                    ),
                    mean_left_elbow_error_m=cast(
                        float,
                        reference_summary["mean_left_elbow_error_m"],
                    ),
                    mean_right_elbow_error_m=cast(
                        float,
                        reference_summary["mean_right_elbow_error_m"],
                    ),
                    mean_elbow_error_m=cast(float, reference_summary["mean_elbow_error_m"]),
                    mean_left_knee_error_m=cast(
                        float,
                        reference_summary["mean_left_knee_error_m"],
                    ),
                    mean_right_knee_error_m=cast(
                        float,
                        reference_summary["mean_right_knee_error_m"],
                    ),
                    mean_knee_error_m=cast(float, reference_summary["mean_knee_error_m"]),
                    mean_left_foot_error_m=cast(
                        float,
                        reference_summary["mean_left_foot_error_m"],
                    ),
                    mean_right_foot_error_m=cast(
                        float,
                        reference_summary["mean_right_foot_error_m"],
                    ),
                    mean_reference_error_m=cast(
                        float,
                        reference_summary["mean_reference_error_m"],
                    ),
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
                    mean_joint_velocity_error_m_per_s=temporal_summary[
                        "mean_joint_velocity_error_m_per_s"
                    ],
                    mean_joint_acceleration_error_m_per_s2=temporal_summary[
                        "mean_joint_acceleration_error_m_per_s2"
                    ],
                    mean_joint_estimated_jerk_m_per_s3=temporal_summary[
                        "mean_joint_estimated_jerk_m_per_s3"
                    ],
                    mean_pelvis_velocity_error_m_per_s=temporal_summary[
                        "mean_pelvis_velocity_error_m_per_s"
                    ],
                    mean_pelvis_acceleration_error_m_per_s2=temporal_summary[
                        "mean_pelvis_acceleration_error_m_per_s2"
                    ],
                    mean_pelvis_estimated_jerk_m_per_s3=temporal_summary[
                        "mean_pelvis_estimated_jerk_m_per_s3"
                    ],
                    mean_torso_velocity_error_m_per_s=temporal_summary[
                        "mean_torso_velocity_error_m_per_s"
                    ],
                    mean_torso_acceleration_error_m_per_s2=temporal_summary[
                        "mean_torso_acceleration_error_m_per_s2"
                    ],
                    mean_torso_estimated_jerk_m_per_s3=temporal_summary[
                        "mean_torso_estimated_jerk_m_per_s3"
                    ],
                    mean_shoulder_velocity_error_m_per_s=temporal_summary[
                        "mean_shoulder_velocity_error_m_per_s"
                    ],
                    mean_shoulder_acceleration_error_m_per_s2=temporal_summary[
                        "mean_shoulder_acceleration_error_m_per_s2"
                    ],
                    mean_shoulder_estimated_jerk_m_per_s3=temporal_summary[
                        "mean_shoulder_estimated_jerk_m_per_s3"
                    ],
                    mean_elbow_velocity_error_m_per_s=temporal_summary[
                        "mean_elbow_velocity_error_m_per_s"
                    ],
                    mean_elbow_acceleration_error_m_per_s2=temporal_summary[
                        "mean_elbow_acceleration_error_m_per_s2"
                    ],
                    mean_elbow_estimated_jerk_m_per_s3=temporal_summary[
                        "mean_elbow_estimated_jerk_m_per_s3"
                    ],
                    mean_knee_velocity_error_m_per_s=temporal_summary[
                        "mean_knee_velocity_error_m_per_s"
                    ],
                    mean_knee_acceleration_error_m_per_s2=temporal_summary[
                        "mean_knee_acceleration_error_m_per_s2"
                    ],
                    mean_knee_estimated_jerk_m_per_s3=temporal_summary[
                        "mean_knee_estimated_jerk_m_per_s3"
                    ],
                    mean_feet_velocity_error_m_per_s=temporal_summary[
                        "mean_feet_velocity_error_m_per_s"
                    ],
                    mean_feet_acceleration_error_m_per_s2=temporal_summary[
                        "mean_feet_acceleration_error_m_per_s2"
                    ],
                    mean_feet_estimated_jerk_m_per_s3=temporal_summary[
                        "mean_feet_estimated_jerk_m_per_s3"
                    ],
                    mean_joint_errors_m=mean_joint_errors_m,
                    mean_left_elbow_flexion_error_deg=flexion_summary[
                        "mean_left_elbow_flexion_error_deg"
                    ],
                    mean_right_elbow_flexion_error_deg=flexion_summary[
                        "mean_right_elbow_flexion_error_deg"
                    ],
                    mean_elbow_flexion_error_deg=flexion_summary["mean_elbow_flexion_error_deg"],
                    mean_left_knee_flexion_error_deg=flexion_summary[
                        "mean_left_knee_flexion_error_deg"
                    ],
                    mean_right_knee_flexion_error_deg=flexion_summary[
                        "mean_right_knee_flexion_error_deg"
                    ],
                    mean_knee_flexion_error_deg=flexion_summary["mean_knee_flexion_error_deg"],
                    mean_joint_flexion_error_deg=flexion_summary["mean_joint_flexion_error_deg"],
                    pct_left_elbow_flexion_over_limit=flexion_summary[
                        "pct_left_elbow_flexion_over_limit"
                    ],
                    pct_right_elbow_flexion_over_limit=flexion_summary[
                        "pct_right_elbow_flexion_over_limit"
                    ],
                    pct_elbow_flexion_over_limit=flexion_summary["pct_elbow_flexion_over_limit"],
                    pct_left_knee_flexion_over_limit=flexion_summary[
                        "pct_left_knee_flexion_over_limit"
                    ],
                    pct_right_knee_flexion_over_limit=flexion_summary[
                        "pct_right_knee_flexion_over_limit"
                    ],
                    pct_knee_flexion_over_limit=flexion_summary["pct_knee_flexion_over_limit"],
                    pct_joint_flexion_over_limit=flexion_summary["pct_joint_flexion_over_limit"],
                )
            )

    total_seconds = time.perf_counter() - start
    total_source_frames = sum(result.source_frames for result in clip_results)
    total_evaluated_frames = sum(result.evaluated_frames for result in clip_results)
    summary = {
        "archive": archive_path.as_posix(),
        "clips": len(clip_results),
        "source_frames": total_source_frames,
        "evaluated_frames": total_evaluated_frames,
        "frame_stride": frame_stride,
        "max_frames_per_clip": max_frames_per_clip,
        "estimator": estimator_name,
        "total_seconds": total_seconds,
        "evaluated_frames_per_second": (
            total_evaluated_frames / total_seconds if total_seconds > 0.0 else 0.0
        ),
        "mean_target_error_m": _mean(result.mean_target_error_m for result in clip_results),
        "mean_pelvis_error_m": _mean(result.mean_pelvis_error_m for result in clip_results),
        "mean_pelvis_horizontal_error_m": _mean(
            result.mean_pelvis_horizontal_error_m for result in clip_results
        ),
        "mean_pelvis_vertical_error_m": _mean(
            result.mean_pelvis_vertical_error_m for result in clip_results
        ),
        "mean_chest_error_m": _mean(result.mean_chest_error_m for result in clip_results),
        "mean_spine_error_m": _mean(result.mean_spine_error_m for result in clip_results),
        "mean_torso_error_m": _mean(result.mean_torso_error_m for result in clip_results),
        "mean_left_shoulder_error_m": _mean(
            result.mean_left_shoulder_error_m for result in clip_results
        ),
        "mean_right_shoulder_error_m": _mean(
            result.mean_right_shoulder_error_m for result in clip_results
        ),
        "mean_shoulder_error_m": _mean(result.mean_shoulder_error_m for result in clip_results),
        "mean_left_elbow_error_m": _mean(result.mean_left_elbow_error_m for result in clip_results),
        "mean_right_elbow_error_m": _mean(
            result.mean_right_elbow_error_m for result in clip_results
        ),
        "mean_elbow_error_m": _mean(result.mean_elbow_error_m for result in clip_results),
        "mean_left_knee_error_m": _mean(result.mean_left_knee_error_m for result in clip_results),
        "mean_right_knee_error_m": _mean(result.mean_right_knee_error_m for result in clip_results),
        "mean_knee_error_m": _mean(result.mean_knee_error_m for result in clip_results),
        "mean_left_foot_error_m": _mean(result.mean_left_foot_error_m for result in clip_results),
        "mean_right_foot_error_m": _mean(result.mean_right_foot_error_m for result in clip_results),
        "mean_reference_error_m": _mean(result.mean_reference_error_m for result in clip_results),
        "mean_joint_errors_m": _mean_joint_errors(clip_results),
    }
    for metric in (
        *REFERENCE_ERROR_DISTRIBUTION_FIELDS,
        *TEMPORAL_SUMMARY_FIELDS,
        *FLEXION_SUMMARY_FIELDS,
    ):
        summary[metric] = _mean(float(getattr(result, metric)) for result in clip_results)

    return {
        "summary": summary,
        "clips": [asdict(result) for result in clip_results],
    }


def write_clip_csv(path: Path, results: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ClipResult.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(results)


def _subject_from_asf(name: str) -> str:
    return Path(name).stem


def _subject_from_amc(name: str) -> str:
    return Path(name).stem.split("_", maxsplit=1)[0]


def _mean(values: Iterable[float]) -> float:
    value_list = list(values)
    if not value_list:
        return 0.0
    return sum(value_list) / len(value_list)


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


if __name__ == "__main__":
    raise SystemExit(main())
