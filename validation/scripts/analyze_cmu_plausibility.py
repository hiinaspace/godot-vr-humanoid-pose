"""Analyze GT-only biomechanical plausibility for cached CMU clips."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

from validation.vr_pose_validation.cmu_metadata import ActivityClass, EnvironmentSplit
from validation.vr_pose_validation.metrics import infer_reference_floor_height
from validation.vr_pose_validation.parquet_cache import (
    filter_clip_cache_files_by_environment,
    iter_cached_pose_frames,
    list_clip_cache_files,
)
from validation.vr_pose_validation.plausibility import summarize_reference_plausibility
from validation.vr_pose_validation.posture import summarize_reference_posture

DEFAULT_CACHE_DIR = Path(".cache/cmu/parquet/canonical_full")
DEFAULT_OUTPUT = Path("validation/reports/cmu_reference_plausibility.json")
DEFAULT_CSV_OUTPUT = Path("validation/reports/cmu_reference_plausibility.csv")
CSV_FIELDS = (
    "subject",
    "clip",
    "motion_description",
    "environment_split",
    "environment_tags",
    "activity_class",
    "activity_tags",
    "posture_subset",
    "posture_class",
    "posture_inverted_pct",
    "posture_floor_reclined_pct",
    "posture_reclined_pct",
    "reference_plausibility_bucket",
    "reference_plausibility_tags",
    "reference_frame_count",
    "reference_min_head_height_m",
    "reference_mean_head_height_m",
    "reference_min_hip_height_m",
    "reference_p05_torso_verticality",
    "reference_mean_torso_verticality",
    "reference_p95_joint_speed_m_per_s",
    "reference_max_joint_speed_m_per_s",
    "reference_pct_frames_over_20m_s",
    "reference_p95_joint_acceleration_m_per_s2",
    "reference_max_joint_acceleration_m_per_s2",
    "reference_max_bone_length_cv",
    "reference_p95_bone_length_cv",
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--csv-output", type=Path, default=DEFAULT_CSV_OUTPUT)
    parser.add_argument("--subjects", nargs="*", help="Subject ids such as 07 08 09.")
    parser.add_argument("--max-clips", type=int)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument(
        "--environment-split",
        action="append",
        choices=("flat", "nonflat", "unknown"),
        dest="environment_splits",
    )
    parser.add_argument(
        "--activity-class",
        action="append",
        choices=(
            "idle_standing",
            "stationary_upper_body",
            "locomotion",
            "sitting",
            "dance_exercise",
            "sports",
            "floor_pose",
            "swimming",
            "unknown",
        ),
        dest="activity_classes",
    )
    args = parser.parse_args(argv)

    rows = analyze_cache(
        cache_dir=args.cache_dir,
        subjects=set(args.subjects or ()),
        environment_splits=set(args.environment_splits or ()),
        activity_classes=set(args.activity_classes or ()),
        max_clips=args.max_clips,
        workers=max(1, args.workers),
    )
    payload = {
        "summary": _summary(rows),
        "clips": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(args.csv_output, rows)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    return 0


def analyze_cache(
    *,
    cache_dir: Path,
    subjects: set[str] | None = None,
    environment_splits: set[EnvironmentSplit] | None = None,
    activity_classes: set[ActivityClass] | None = None,
    max_clips: int | None = None,
    workers: int = 1,
) -> list[dict[str, Any]]:
    paths = list_clip_cache_files(cache_dir, subjects or None)
    paths = filter_clip_cache_files_by_environment(
        paths,
        environment_splits=environment_splits or None,
        activity_classes=activity_classes or None,
    )
    if max_clips is not None:
        paths = paths[: max(0, max_clips)]
    if workers == 1:
        rows = [_analyze_clip(path.as_posix()) for path in paths]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(_analyze_clip, (path.as_posix() for path in paths)))
    return sorted(rows, key=lambda row: (str(row["subject"]), str(row["clip"])))


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field, "")) for field in CSV_FIELDS})


def _analyze_clip(path_text: str) -> dict[str, Any]:
    frames = list(iter_cached_pose_frames(Path(path_text)))
    reference_frames = [frame.joints for frame in frames]
    timestamps = [frame.timestamp_s for frame in frames]
    floor_height_m = infer_reference_floor_height(reference_frames)
    first_frame = frames[0]
    posture_summary = summarize_reference_posture(
        reference_frames,
        environment_split=first_frame.environment_split,
        activity_class=first_frame.activity_class,
        floor_height_m=floor_height_m,
        timestamps_seconds=timestamps,
    )
    plausibility = summarize_reference_plausibility(
        reference_frames,
        timestamps_seconds=timestamps,
        floor_height_m=floor_height_m,
        posture_summary=posture_summary,
    )
    plausibility_payload = asdict(plausibility)
    return {
        "subject": first_frame.subject,
        "clip": first_frame.clip,
        "motion_description": first_frame.motion_description,
        "environment_split": first_frame.environment_split,
        "environment_tags": first_frame.environment_tags,
        "activity_class": first_frame.activity_class,
        "activity_tags": first_frame.activity_tags,
        "posture_subset": posture_summary.posture_subset,
        "posture_class": posture_summary.posture_class,
        "posture_inverted_pct": posture_summary.inverted_pct,
        "posture_floor_reclined_pct": posture_summary.floor_reclined_pct,
        "posture_reclined_pct": posture_summary.reclined_pct,
        "reference_plausibility_bucket": plausibility_payload.pop("bucket"),
        "reference_plausibility_tags": tuple(plausibility_payload.pop("tags")),
        **{f"reference_{key}": value for key, value in plausibility_payload.items()},
    }


def _summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    buckets = Counter(str(row["reference_plausibility_bucket"]) for row in rows)
    tags: Counter[str] = Counter()
    for row in rows:
        tags.update(str(tag) for tag in row["reference_plausibility_tags"])
    return {
        "clips": len(rows),
        "buckets": dict(sorted(buckets.items())),
        "tags": dict(sorted(tags.items())),
    }


def _csv_value(value: object) -> object:
    if isinstance(value, (list, tuple)):
        return ";".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return value


if __name__ == "__main__":
    raise SystemExit(main())
