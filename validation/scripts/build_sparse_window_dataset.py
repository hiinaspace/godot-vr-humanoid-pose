"""Build sparse 3-point tracker windows for upper-body latent ML experiments."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import cast

from validation.vr_pose_validation.cmu_metadata import ActivityClass, EnvironmentSplit
from validation.vr_pose_validation.upper_body_ml import (
    DEFAULT_CLEAN_ACTIVITY_CLASSES,
    SparseWindowConfig,
    TrackingSpace,
    build_sparse_window_dataset,
)

DEFAULT_CACHE_DIR = Path(".cache/cmu/parquet/canonical_full")
DEFAULT_OUTPUT_DIR = Path(".cache/ml/upper_body_windows")


def _default_workers(stride: int) -> int:
    cpu_count = os.cpu_count() or 1
    if stride <= 1:
        return max(1, min(cpu_count, 2))
    return max(1, min(cpu_count, 16))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--window-size", type=int, default=40)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument(
        "--workers",
        type=int,
        help="Worker processes. Defaults to 2 for stride-1 dense exports, else up to 16.",
    )
    parser.add_argument(
        "--shard-rows",
        type=int,
        default=50_000,
        help="Write window rows in Parquet shards instead of one in-memory file.",
    )
    parser.add_argument("--max-clips", type=int)
    parser.add_argument("--subjects", nargs="*")
    parser.add_argument("--holdout-folds", type=int, default=5)
    parser.add_argument(
        "--tracking-space",
        choices=("head_yaw", "calibration_yaw"),
        default="calibration_yaw",
        help=(
            "Coordinate basis for sparse windows and upper-body targets. "
            "calibration_yaw keeps clip-initial yaw axes stable while keeping "
            "positions relative to the current HMD/head origin."
        ),
    )
    parser.add_argument(
        "--estimator",
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
        help="Heuristic estimator used for baseline latent columns.",
    )
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
        default=tuple(sorted(DEFAULT_CLEAN_ACTIVITY_CLASSES)),
    )
    args = parser.parse_args(argv)

    config = SparseWindowConfig(
        window_size=max(2, args.window_size),
        stride=max(1, args.stride),
        estimator_name=args.estimator,
        holdout_folds=max(2, args.holdout_folds),
        environment_splits=cast(tuple[EnvironmentSplit, ...], tuple(args.environment_split)),
        activity_classes=cast(tuple[ActivityClass, ...], tuple(args.activity_class)),
        tracking_space=cast(TrackingSpace, args.tracking_space),
        workers=max(1, args.workers if args.workers is not None else _default_workers(args.stride)),
        shard_rows=max(0, args.shard_rows),
    )
    summary = build_sparse_window_dataset(
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        config=config,
        subjects=set(args.subjects or []),
        max_clips=args.max_clips,
    )
    print(json.dumps(asdict(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
