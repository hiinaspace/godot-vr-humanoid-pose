"""Compare an external pose cache against a reference canonical pose cache."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import polars as pl

from validation.scripts.run_baseline_validation import ClipResult
from validation.scripts.run_cached_validation import (
    SUMMARY_METRICS,
    TOP_CLIP_METRICS,
    _avatar_plausibility_bucket_counts,
    _clip_metric_summary,
    _grouped_metrics,
    _mean_column,
    _mean_joint_errors,
    _posture_frame_counts,
    _posture_subset_counts,
    _reference_avatar_plausibility_bucket_counts,
    _scalar_clip_result,
    _sum_column,
    evaluate_cached_clip_path,
)
from validation.vr_pose_validation.calibration import humanoid_poses_from_reference_frames
from validation.vr_pose_validation.parquet_cache import (
    CachedPoseFrame,
    iter_cached_pose_frames,
    list_clip_cache_files,
)
from validation.vr_pose_validation.types import (
    AvatarCalibration,
    GroundingContext,
    HumanoidPoseFrame,
    TrackedPoseSample,
)

DEFAULT_REFERENCE_CACHE_DIR = Path(".cache/sparseposer/parquet/canonical")
DEFAULT_PREDICTION_CACHE_DIR = Path(".cache/sparseposer/parquet/model_xsens_eval")
DEFAULT_REPORT_PATH = Path("validation/reports/sparseposer_model_xsens_eval.json")


@dataclass(frozen=True, slots=True)
class PoseCacheComparisonTask:
    reference_path: Path
    prediction_path: Path
    max_frames: int
    skip_initial_frames: int


@dataclass(slots=True)
class SequencePoseEstimator:
    poses: list[HumanoidPoseFrame]
    index: int = 0

    def estimate(
        self,
        sample: TrackedPoseSample,
        calibration: AvatarCalibration,
        grounding: GroundingContext,
    ) -> HumanoidPoseFrame:
        del calibration, grounding
        if self.index >= len(self.poses):
            raise IndexError("prediction pose sequence is shorter than reference sequence")
        pose = self.poses[self.index]
        self.index += 1
        return HumanoidPoseFrame(
            timestamp_seconds=sample.timestamp_seconds,
            root=pose.root,
            joints=pose.joints,
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-cache-dir", type=Path, default=DEFAULT_REFERENCE_CACHE_DIR)
    parser.add_argument("--prediction-cache-dir", type=Path, default=DEFAULT_PREDICTION_CACHE_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--csv-report", type=Path)
    parser.add_argument("--subjects", nargs="*")
    parser.add_argument("--dataset-split", choices=("train", "eval"))
    parser.add_argument("--max-clips", type=int)
    parser.add_argument("--top-clips", type=int, default=10)
    parser.add_argument(
        "--skip-initial-frames",
        type=int,
        default=0,
        help="Drop this many leading frames from both reference and prediction.",
    )
    args = parser.parse_args(argv)

    result = run_pose_cache_comparison(
        reference_cache_dir=args.reference_cache_dir,
        prediction_cache_dir=args.prediction_cache_dir,
        subjects=set(args.subjects or []),
        dataset_split=args.dataset_split,
        max_clips=args.max_clips,
        top_clip_count=max(0, args.top_clips),
        skip_initial_frames=max(0, args.skip_initial_frames),
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.csv_report is not None:
        _write_clip_csv(args.csv_report, cast(list[dict[str, object]], result["clips"]))
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0


def run_pose_cache_comparison(
    *,
    reference_cache_dir: Path,
    prediction_cache_dir: Path,
    subjects: set[str],
    dataset_split: str | None,
    max_clips: int | None,
    top_clip_count: int,
    skip_initial_frames: int,
) -> dict[str, object]:
    start = time.perf_counter()
    reference_paths = list_clip_cache_files(reference_cache_dir, subjects or None)
    if dataset_split is not None:
        reference_paths = [
            path for path in reference_paths if _first_frame_dataset_split(path) == dataset_split
        ]
    if max_clips is not None:
        reference_paths = reference_paths[:max_clips]

    tasks = [
        task
        for path in reference_paths
        if (task := _comparison_task(path, prediction_cache_dir, skip_initial_frames)) is not None
    ]
    clip_results = [_evaluate_pose_cache_pair(task) for task in tasks]
    total_seconds = time.perf_counter() - start
    summary_frame = pl.DataFrame([_scalar_clip_result(result) for result in clip_results])

    return {
        "summary": {
            "reference_cache_dir": reference_cache_dir.as_posix(),
            "prediction_cache_dir": prediction_cache_dir.as_posix(),
            "clips": len(clip_results),
            "missing_prediction_clips": len(reference_paths) - len(tasks),
            "subjects": len({result.subject for result in clip_results}),
            "source_frames": _sum_column(summary_frame, "source_frames"),
            "evaluated_frames": _sum_column(summary_frame, "evaluated_frames"),
            "dataset_split": dataset_split,
            "skip_initial_frames": skip_initial_frames,
            "total_seconds": total_seconds,
            "evaluated_frames_per_second": (
                _sum_column(summary_frame, "evaluated_frames") / total_seconds
                if total_seconds > 0.0
                else 0.0
            ),
            "mean_joint_errors_m": _mean_joint_errors(clip_results),
            "grouped_metrics": _grouped_metrics(clip_results),
            "worst_clips": _worst_clips(clip_results, top_clip_count),
            "avatar_plausibility_bucket_counts": _avatar_plausibility_bucket_counts(clip_results),
            "reference_avatar_plausibility_bucket_counts": (
                _reference_avatar_plausibility_bucket_counts(clip_results)
            ),
            "posture_frame_counts": _posture_frame_counts(clip_results),
            "posture_subset_counts": _posture_subset_counts(clip_results),
            **{
                metric: _mean_column(summary_frame, metric)
                for metric in (
                    "mean_target_error_m",
                    "mean_pelvis_error_m",
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
                    *SUMMARY_METRICS,
                )
            },
        },
        "clips": [asdict(result) for result in clip_results],
    }


def _comparison_task(
    reference_path: Path,
    prediction_cache_dir: Path,
    skip_initial_frames: int,
) -> PoseCacheComparisonTask | None:
    prediction_path = prediction_cache_dir / reference_path.parent.name / reference_path.name
    if not prediction_path.exists():
        return None
    reference_frame_count = _parquet_height(reference_path)
    prediction_frame_count = _parquet_height(prediction_path)
    max_frames = min(reference_frame_count, prediction_frame_count) - skip_initial_frames
    if max_frames <= 0:
        return None
    return PoseCacheComparisonTask(
        reference_path=reference_path,
        prediction_path=prediction_path,
        max_frames=max_frames,
        skip_initial_frames=skip_initial_frames,
    )


def _evaluate_pose_cache_pair(task: PoseCacheComparisonTask) -> ClipResult:
    prediction_frames = _load_prediction_frames(
        task.prediction_path,
        skip_initial_frames=task.skip_initial_frames,
        max_frames=task.max_frames,
    )
    estimator = SequencePoseEstimator(
        poses=humanoid_poses_from_reference_frames(
            [frame.joints for frame in prediction_frames],
            timestamps_seconds=[frame.timestamp_s for frame in prediction_frames],
        )
    )
    return evaluate_cached_clip_path(
        task.reference_path,
        estimator,
        max_frames=task.max_frames,
        skip_initial_frames=task.skip_initial_frames,
    )


def _load_prediction_frames(
    path: Path,
    *,
    skip_initial_frames: int,
    max_frames: int,
) -> list[CachedPoseFrame]:
    frames = list(iter_cached_pose_frames(path))
    return frames[skip_initial_frames : skip_initial_frames + max_frames]


def _first_frame_dataset_split(path: Path) -> str:
    frame = pl.read_parquet(path, columns=["dataset_split"], n_rows=1)
    if frame.height == 0 or "dataset_split" not in frame.columns:
        return ""
    return str(frame["dataset_split"][0])


def _parquet_height(path: Path) -> int:
    return pl.read_parquet(path, columns=["frame_index"]).height


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


def _write_clip_csv(path: Path, clips: Iterable[dict[str, object]]) -> None:
    rows = list(clips)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
