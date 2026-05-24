"""Summarize live capture sessions and calibrated tracker-vs-estimate errors."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import polars as pl

DEFAULT_CAPTURE_DIR = Path(".cache/live_capture")
TRACKED_JOINT_ROLES = (
    "left_elbow",
    "right_elbow",
    "left_knee",
    "right_knee",
    "left_foot",
    "right_foot",
    "pelvis",
)
DEFAULT_EXPECTED_CALIBRATED_ROLES = TRACKED_JOINT_ROLES


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "session",
        nargs="?",
        type=Path,
        default=DEFAULT_CAPTURE_DIR,
        help="Live capture session directory, or a parent directory containing session=* folders.",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    parser.add_argument(
        "--expected-role",
        action="append",
        dest="expected_roles",
        help=(
            "Expected calibrated tracker role. Repeat to override the default "
            "feet/knees/elbows/pelvis role set."
        ),
    )
    args = parser.parse_args(argv)

    session_dir = resolve_live_capture_session(args.session)
    summary = analyze_live_capture(
        session_dir,
        expected_roles=tuple(args.expected_roles)
        if args.expected_roles is not None
        else DEFAULT_EXPECTED_CALIBRATED_ROLES,
    )
    payload = json.dumps(summary, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
    return 0


def resolve_live_capture_session(path: Path) -> Path:
    if path.name.startswith("session="):
        return path
    candidates = [
        candidate for candidate in path.glob("session=*") if (candidate / "frames.parquet").exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"No live capture sessions with frames found under {path}")
    return max(candidates, key=lambda candidate: (candidate / "frames.parquet").stat().st_mtime)


def analyze_live_capture(
    session_dir: Path,
    *,
    expected_roles: Iterable[str] | None = DEFAULT_EXPECTED_CALIBRATED_ROLES,
) -> dict[str, Any]:
    frames = _read_table(session_dir / "frames.parquet")
    devices = _read_table(session_dir / "devices.parquet")
    joints = _read_table(session_dir / "joints.parquet")

    return {
        "session_dir": str(session_dir),
        "frames": frames.height,
        "devices": devices.height,
        "joints": joints.height,
        "source_sessions": _source_session_count(frames),
        "clips": _clip_summaries(frames),
        "calibrated_tracker_coverage": _calibrated_tracker_coverage(devices, expected_roles),
        "calibrated_tracker_errors_m": _calibrated_tracker_errors(devices, joints),
    }


def _read_table(path: Path) -> pl.DataFrame:
    if not path.exists():
        return pl.DataFrame()
    table = pl.read_parquet(path)
    if "clip_index" not in table.columns:
        table = _with_legacy_clip_index(table)
    return table


def _with_legacy_clip_index(table: pl.DataFrame) -> pl.DataFrame:
    if "frame_index" not in table.columns:
        return table.with_columns(pl.lit(0).alias("clip_index"))
    if "session_id" not in table.columns:
        table = table.with_columns(pl.lit("").alias("session_id"))
    return (
        table.with_row_index("_capture_row_order")
        .with_columns(
            (pl.col("frame_index") < pl.col("frame_index").shift(1).over("session_id").fill_null(0))
            .cast(pl.Int64)
            .alias("_clip_reset")
        )
        .with_columns(pl.col("_clip_reset").cum_sum().over("session_id").alias("clip_index"))
        .sort("_capture_row_order")
        .drop("_capture_row_order", "_clip_reset")
    )


def _source_session_count(frames: pl.DataFrame) -> int:
    if frames.is_empty() or "session_id" not in frames.columns:
        return 0
    return frames.select(pl.col("session_id").n_unique()).item()


def _clip_summaries(frames: pl.DataFrame) -> list[dict[str, Any]]:
    required = {"session_id", "clip_index", "timestamp_seconds"}
    if frames.is_empty() or not required.issubset(frames.columns):
        return []
    clips = (
        frames.group_by("session_id", "clip_index")
        .agg(
            pl.len().alias("frames"),
            pl.col("timestamp_seconds").min().alias("start_seconds"),
            pl.col("timestamp_seconds").max().alias("end_seconds"),
        )
        .with_columns((pl.col("end_seconds") - pl.col("start_seconds")).alias("duration_seconds"))
        .sort("session_id", "clip_index")
    )
    return _json_rows(clips)


def _calibrated_tracker_errors(devices: pl.DataFrame, joints: pl.DataFrame) -> dict[str, Any]:
    device_required = {
        "session_id",
        "clip_index",
        "frame_index",
        "role",
        "device_class",
        "device_px",
        "device_py",
        "device_pz",
    }
    joint_required = {
        "session_id",
        "clip_index",
        "frame_index",
        "joint",
        "joint_px",
        "joint_py",
        "joint_pz",
    }
    if (
        devices.is_empty()
        or joints.is_empty()
        or not device_required.issubset(devices.columns)
        or not joint_required.issubset(joints.columns)
    ):
        return {"available": False, "reason": "missing capture tables or columns"}

    trackers = devices.filter(
        (pl.col("device_class") == "CalibratedTracker") & pl.col("role").is_in(TRACKED_JOINT_ROLES)
    ).select(
        "session_id",
        "clip_index",
        "frame_index",
        "role",
        "device_px",
        "device_py",
        "device_pz",
    )
    if trackers.is_empty():
        return {"available": False, "reason": "no calibrated optional tracker rows"}

    estimates = joints.filter(pl.col("joint").is_in(TRACKED_JOINT_ROLES)).select(
        "session_id",
        "clip_index",
        "frame_index",
        pl.col("joint").alias("role"),
        "joint_px",
        "joint_py",
        "joint_pz",
    )
    joined = trackers.join(
        estimates,
        on=["session_id", "clip_index", "frame_index", "role"],
        how="inner",
    )
    if joined.is_empty():
        return {"available": False, "reason": "no matching tracker and estimated joint rows"}

    errors = joined.with_columns(
        (
            (pl.col("joint_px") - pl.col("device_px")).pow(2)
            + (pl.col("joint_py") - pl.col("device_py")).pow(2)
            + (pl.col("joint_pz") - pl.col("device_pz")).pow(2)
        )
        .sqrt()
        .alias("error_m")
    )
    by_role = (
        errors.group_by("role")
        .agg(
            pl.len().alias("samples"),
            pl.col("error_m").mean().alias("mean_m"),
            pl.col("error_m").median().alias("median_m"),
            pl.col("error_m").quantile(0.95).alias("p95_m"),
            pl.col("error_m").max().alias("max_m"),
        )
        .sort("role")
    )
    by_clip_role = (
        errors.group_by("session_id", "clip_index", "role")
        .agg(
            pl.len().alias("samples"),
            pl.col("error_m").mean().alias("mean_m"),
            pl.col("error_m").quantile(0.95).alias("p95_m"),
        )
        .sort("mean_m", descending=True)
        .head(24)
    )
    return {
        "available": True,
        "samples": errors.height,
        "by_role": _json_rows(by_role),
        "worst_clip_roles": _json_rows(by_clip_role),
    }


def _calibrated_tracker_coverage(
    devices: pl.DataFrame,
    expected_roles: Iterable[str] | None,
) -> dict[str, Any]:
    required = {"session_id", "clip_index", "role", "device_class"}
    if devices.is_empty() or not required.issubset(devices.columns):
        return {"available": False, "reason": "missing device table or columns"}

    trackers = devices.filter(pl.col("device_class") == "CalibratedTracker")
    if trackers.is_empty():
        return {"available": False, "reason": "no calibrated optional tracker rows"}

    by_clip = (
        trackers.group_by("session_id", "clip_index")
        .agg(
            pl.col("role").unique().sort().alias("roles"),
            pl.len().alias("samples"),
            pl.col("frame_index").n_unique().alias("frames")
            if "frame_index" in trackers.columns
            else pl.len().alias("frames"),
        )
        .sort("session_id", "clip_index")
    )
    expected = sorted(set(expected_roles or ()))
    rows = []
    for row in by_clip.to_dicts():
        roles = sorted(str(role) for role in row["roles"])
        row["roles"] = roles
        if expected:
            row["missing_roles"] = [role for role in expected if role not in roles]
            row["unexpected_roles"] = [role for role in roles if role not in expected]
            row["calibration_ok"] = not row["missing_roles"] and not row["unexpected_roles"]
        else:
            row["missing_roles"] = []
            row["unexpected_roles"] = []
            row["calibration_ok"] = True
        rows.append({key: _json_value(value) for key, value in row.items()})

    role_counts = (
        trackers.group_by("role")
        .agg(
            pl.len().alias("samples"),
            pl.col("session_id").n_unique().alias("source_sessions"),
            pl.struct("session_id", "clip_index").n_unique().alias("clips"),
        )
        .sort("role")
    )
    return {
        "available": True,
        "expected_roles": expected,
        "by_role": _json_rows(role_counts),
        "by_clip": rows,
        "bad_clips": [row for row in rows if not row["calibration_ok"]],
    }


def _json_rows(dataframe: pl.DataFrame) -> list[dict[str, Any]]:
    rows = dataframe.to_dicts()
    return [{key: _json_value(value) for key, value in row.items()} for row in rows]


def _json_value(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    return value


if __name__ == "__main__":
    raise SystemExit(main())
