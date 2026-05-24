"""Run a lightweight black-box search over upper-body Basis estimator settings."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, cast

from validation.scripts.run_cached_validation import _tracked_sample_from_cached_frame
from validation.vr_pose_validation.baseline import create_estimator
from validation.vr_pose_validation.basis_estimator import (
    ArmBendStrategy,
    BasisEstimatorSettings,
    BasisPortEstimator,
)
from validation.vr_pose_validation.metrics import (
    ReferencePoseMetrics,
    evaluate_reference_pose,
    infer_reference_floor_height,
    summarize_joint_flexion_sequence,
    summarize_reference_error_distribution,
    summarize_reference_errors,
)
from validation.vr_pose_validation.parquet_cache import (
    clip_cache_path,
    iter_cached_pose_frames,
)
from validation.vr_pose_validation.types import AvatarCalibration, GroundingContext

DEFAULT_CACHE_DIR = Path(".cache/cmu/parquet/canonical_full")
DEFAULT_SEED_REPORT = Path("validation/reports/experiment_basis_flat_upper.json")
DEFAULT_OUTPUT = Path("validation/reports/upper_body_blackbox_tuning.jsonl")
DEFAULT_CSV_OUTPUT = Path("validation/reports/upper_body_blackbox_tuning.csv")
DEFAULT_ALLOWED_POSTURES = ("vr_stationary", "vr_locomotion", "sitting")
EXCLUDED_DESCRIPTION_TERMS = ("motorcycle",)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--seed-report", type=Path, default=DEFAULT_SEED_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--csv-output", type=Path, default=DEFAULT_CSV_OUTPUT)
    parser.add_argument("--trials", type=int, default=64)
    parser.add_argument("--clips", type=int, default=96)
    parser.add_argument("--workers", type=int, default=max(1, min(os.cpu_count() or 1, 8)))
    parser.add_argument("--seed", type=int, default=20260523)
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    clip_rows = _select_clip_rows(args.seed_report, limit=max(1, args.clips), rng=rng)
    clip_paths = [
        clip_cache_path(args.cache_dir, str(row["subject"]), str(row["clip"])).as_posix()
        for row in clip_rows
    ]
    trials = _trial_settings(max(1, args.trials), rng=rng)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("", encoding="utf-8")
    if args.csv_output.exists():
        args.csv_output.unlink()

    started_at = time.perf_counter()
    best: dict[str, Any] | None = None
    for trial_index, settings in enumerate(trials):
        trial_started_at = time.perf_counter()
        metrics = _evaluate_trial(
            clip_paths,
            settings,
            workers=max(1, args.workers),
        )
        row = {
            "trial": trial_index,
            "elapsed_seconds": time.perf_counter() - started_at,
            "trial_seconds": time.perf_counter() - trial_started_at,
            "clip_count": len(clip_paths),
            "objective": _objective(metrics),
            **metrics,
            **_settings_summary(settings),
        }
        if best is None or float(row["objective"]) < float(best["objective"]):
            best = row
            row["is_best"] = True
        else:
            row["is_best"] = False
        with args.output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        _write_csv(args.csv_output, row)
        print(
            json.dumps(
                {
                    "trial": trial_index,
                    "objective": row["objective"],
                    "best_objective": best["objective"],
                    "mean_elbow_error_m": row["mean_elbow_error_m"],
                    "elbow_error_p95_m": row["elbow_error_p95_m"],
                    "mean_shoulder_error_m": row["mean_shoulder_error_m"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    if best is not None:
        print(json.dumps({"best": best}, indent=2, sort_keys=True))
    return 0


def _select_clip_rows(
    report_path: Path,
    *,
    limit: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    rows = [
        cast(dict[str, Any], row)
        for row in payload.get("clips", [])
        if isinstance(row, dict) and _clip_is_upper_body_candidate(row)
    ]
    ranked = sorted(rows, key=_clip_selection_score, reverse=True)
    selected = ranked[: max(1, limit * 2 // 3)]
    remaining = [row for row in rows if row not in selected]
    rng.shuffle(remaining)
    selected.extend(remaining[: max(0, limit - len(selected))])
    return selected[:limit]


def _clip_is_upper_body_candidate(row: dict[str, Any]) -> bool:
    if str(row.get("environment_split", "")) != "flat":
        return False
    if str(row.get("posture_subset", "")) not in DEFAULT_ALLOWED_POSTURES:
        return False
    description = str(row.get("motion_description", "")).lower()
    return not any(term in description for term in EXCLUDED_DESCRIPTION_TERMS)


def _clip_selection_score(row: dict[str, Any]) -> float:
    return (
        _float(row.get("mean_elbow_error_m")) * 2.2
        + _float(row.get("mean_shoulder_error_m")) * 1.2
        + _float(row.get("mean_upper_body_reference_error_m"))
        + _float(row.get("mean_elbow_flexion_error_deg")) * 0.003
        + _float(row.get("pct_elbow_flexion_over_limit")) * 0.003
    )


def _trial_settings(
    count: int,
    *,
    rng: random.Random,
) -> list[BasisEstimatorSettings]:
    trials = [
        cast(BasisPortEstimator, create_estimator("basis")).settings,
        cast(BasisPortEstimator, create_estimator("basis_arm_lookup")).settings,
        cast(BasisPortEstimator, create_estimator("basis_arm_swivel")).settings,
    ]
    while len(trials) < count:
        strategy: ArmBendStrategy = "manikin_swivel" if rng.random() < 0.68 else "default"
        base = BasisEstimatorSettings(arm_bend_strategy=strategy)
        trials.append(
            replace(
                base,
                shoulder_width_ratio=rng.uniform(0.175, 0.230),
                upper_arm_ratio=rng.uniform(0.160, 0.190),
                lower_arm_ratio=rng.uniform(0.135, 0.175),
                arm_bend_lookup_blend=0.0
                if strategy == "manikin_swivel"
                else rng.uniform(0.0, 1.0),
                arm_swivel_down_weight=rng.uniform(0.20, 0.90),
                arm_swivel_outward_weight=rng.uniform(0.05, 0.70),
                arm_swivel_backward_weight=rng.uniform(0.05, 0.85),
                arm_swivel_cross_body_inward_weight=rng.uniform(0.00, 0.65),
                arm_swivel_overhead_relief=rng.uniform(0.10, 0.90),
                shoulder_reach_engage_ratio=rng.uniform(0.50, 0.88),
                shoulder_elevation_gain_m=rng.uniform(0.00, 0.12),
                shoulder_protraction_gain_m=rng.uniform(0.00, 0.10),
                shoulder_cross_body_gain_m=rng.uniform(0.00, 0.08),
                shoulder_reach_slack_m=rng.uniform(0.00, 0.06),
                max_shoulder_reach_shift_m=rng.uniform(0.06, 0.28),
            )
        )
    return trials[:count]


def _evaluate_trial(
    clip_paths: Sequence[str],
    settings: BasisEstimatorSettings,
    *,
    workers: int,
) -> dict[str, float]:
    settings_payload = asdict(settings)
    if workers == 1:
        clip_metrics = [_evaluate_clip((path, settings_payload)) for path in clip_paths]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            clip_metrics = list(
                executor.map(
                    _evaluate_clip,
                    ((path, settings_payload) for path in clip_paths),
                )
            )
    return {
        key: _mean(metric[key] for metric in clip_metrics)
        for key in (
            "mean_elbow_error_m",
            "elbow_error_p95_m",
            "mean_shoulder_error_m",
            "shoulder_error_p95_m",
            "mean_upper_body_reference_error_m",
            "mean_elbow_flexion_error_deg",
            "pct_elbow_flexion_over_limit",
        )
    }


def _evaluate_clip(task: tuple[str, dict[str, Any]]) -> dict[str, float]:
    path, settings_payload = task
    settings = BasisEstimatorSettings(**settings_payload)
    estimator = BasisPortEstimator(settings)
    cached_frame_list = list(iter_cached_pose_frames(Path(path)))
    reference_frames = [frame.joints for frame in cached_frame_list]
    floor_height_m = infer_reference_floor_height(reference_frames)
    grounding = GroundingContext(floor_height_m=floor_height_m)
    calibration = AvatarCalibration()
    reference_metrics: list[ReferencePoseMetrics] = []
    estimated_frames = []
    for cached_frame in cached_frame_list:
        pose = estimator.estimate(
            _tracked_sample_from_cached_frame(cached_frame),
            calibration,
            grounding,
        )
        estimated_frames.append(pose)
        reference_metrics.append(evaluate_reference_pose(pose, cached_frame.joints))
    reference_summary = summarize_reference_errors(reference_metrics)
    tail_summary = summarize_reference_error_distribution(reference_metrics)
    flexion_summary = summarize_joint_flexion_sequence(estimated_frames, reference_frames)
    return {
        "mean_elbow_error_m": cast(float, reference_summary["mean_elbow_error_m"]),
        "elbow_error_p95_m": tail_summary["elbow_error_p95_m"],
        "mean_shoulder_error_m": cast(float, reference_summary["mean_shoulder_error_m"]),
        "shoulder_error_p95_m": tail_summary["shoulder_error_p95_m"],
        "mean_upper_body_reference_error_m": cast(
            float,
            reference_summary["mean_upper_body_reference_error_m"],
        ),
        "mean_elbow_flexion_error_deg": flexion_summary["mean_elbow_flexion_error_deg"],
        "pct_elbow_flexion_over_limit": flexion_summary["pct_elbow_flexion_over_limit"],
    }


def _objective(metrics: dict[str, float]) -> float:
    return (
        metrics["mean_elbow_error_m"] * 2.0
        + metrics["elbow_error_p95_m"] * 0.9
        + metrics["mean_shoulder_error_m"] * 1.2
        + metrics["mean_upper_body_reference_error_m"] * 0.9
        + metrics["mean_elbow_flexion_error_deg"] * 0.003
        + metrics["pct_elbow_flexion_over_limit"] * 0.004
    )


def _settings_summary(settings: BasisEstimatorSettings) -> dict[str, float | str]:
    return {
        "arm_bend_strategy": settings.arm_bend_strategy,
        "arm_bend_lookup_blend": settings.arm_bend_lookup_blend,
        "shoulder_width_ratio": settings.shoulder_width_ratio,
        "upper_arm_ratio": settings.upper_arm_ratio,
        "lower_arm_ratio": settings.lower_arm_ratio,
        "arm_swivel_down_weight": settings.arm_swivel_down_weight,
        "arm_swivel_outward_weight": settings.arm_swivel_outward_weight,
        "arm_swivel_backward_weight": settings.arm_swivel_backward_weight,
        "arm_swivel_cross_body_inward_weight": settings.arm_swivel_cross_body_inward_weight,
        "arm_swivel_overhead_relief": settings.arm_swivel_overhead_relief,
        "shoulder_reach_engage_ratio": settings.shoulder_reach_engage_ratio,
        "shoulder_elevation_gain_m": settings.shoulder_elevation_gain_m,
        "shoulder_protraction_gain_m": settings.shoulder_protraction_gain_m,
        "shoulder_cross_body_gain_m": settings.shoulder_cross_body_gain_m,
        "shoulder_reach_slack_m": settings.shoulder_reach_slack_m,
        "max_shoulder_reach_shift_m": settings.max_shoulder_reach_shift_m,
    }


def _write_csv(path: Path, row: Mapping[str, object]) -> None:
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _float(value: object) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _mean(values: Iterable[float]) -> float:
    values_tuple = tuple(values)
    if not values_tuple:
        return 0.0
    return sum(values_tuple) / len(values_tuple)


if __name__ == "__main__":
    raise SystemExit(main())
