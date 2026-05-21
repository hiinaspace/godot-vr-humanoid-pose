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
    ReferencePoseMetrics,
    evaluate_frame,
    evaluate_reference_pose,
    summarize_reference_errors,
    summarize_target_errors,
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
    mean_left_foot_error_m: float
    mean_right_foot_error_m: float
    mean_reference_error_m: float
    mean_joint_errors_m: dict[str, float]


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
        choices=("trivial", "planted", "basis"),
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
            for reference_frame, sample in zip(canonical_frames, samples, strict=True):
                pose = estimator.estimate(sample, calibration, grounding)
                frame_metrics.append(evaluate_frame(pose, sample, grounding.floor_height_m))
                reference_metrics.append(evaluate_reference_pose(pose, reference_frame))
            target_summary = summarize_target_errors(frame_metrics)
            reference_summary = summarize_reference_errors(reference_metrics)
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
                    mean_joint_errors_m=mean_joint_errors_m,
                )
            )

    total_seconds = time.perf_counter() - start
    total_source_frames = sum(result.source_frames for result in clip_results)
    total_evaluated_frames = sum(result.evaluated_frames for result in clip_results)
    return {
        "summary": {
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
            "mean_left_foot_error_m": _mean(
                result.mean_left_foot_error_m for result in clip_results
            ),
            "mean_right_foot_error_m": _mean(
                result.mean_right_foot_error_m for result in clip_results
            ),
            "mean_reference_error_m": _mean(
                result.mean_reference_error_m for result in clip_results
            ),
            "mean_joint_errors_m": _mean_joint_errors(clip_results),
        },
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
