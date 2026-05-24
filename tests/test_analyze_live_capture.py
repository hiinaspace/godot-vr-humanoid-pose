from __future__ import annotations

from typing import cast

import polars as pl
from pytest import approx

from validation.scripts.analyze_live_capture import analyze_live_capture
from validation.vr_pose_validation.live_capture import LiveCaptureParquetWriter


def test_analyze_live_capture_reports_clip_and_calibrated_elbow_error(tmp_path) -> None:
    writer = LiveCaptureParquetWriter(tmp_path, session_id="recorder")
    message = _frame_message()
    writer.append_message(message)
    summary = writer.flush()

    analysis = analyze_live_capture(summary.session_dir)

    assert analysis["frames"] == 1
    assert analysis["source_sessions"] == 1
    assert analysis["clips"][0]["clip_index"] == 3
    errors = analysis["calibrated_tracker_errors_m"]
    assert errors["available"]
    assert errors["samples"] == 1
    assert errors["by_role"][0]["role"] == "left_elbow"
    assert errors["by_role"][0]["mean_m"] == approx(0.1)
    coverage = analysis["calibrated_tracker_coverage"]
    assert coverage["available"]
    assert coverage["bad_clips"][0]["missing_roles"] == [
        "left_foot",
        "left_knee",
        "pelvis",
        "right_elbow",
        "right_foot",
        "right_knee",
    ]


def test_analyze_live_capture_flags_unexpected_calibrated_chest_role(tmp_path) -> None:
    writer = LiveCaptureParquetWriter(tmp_path, session_id="recorder")
    message = _frame_message()
    sample = cast(dict[str, object], message["sample"])
    sample["optional_trackers"] = {
        "chest": _tracker("chest", [0.0, 1.35, 0.05], [0.0, 0.0, 0.0, 1.0])
    }
    writer.append_message(message)
    summary = writer.flush()

    analysis = analyze_live_capture(summary.session_dir)

    bad_clip = analysis["calibrated_tracker_coverage"]["bad_clips"][0]
    assert bad_clip["unexpected_roles"] == ["chest"]


def test_analyze_live_capture_splits_legacy_clips_by_frame_reset(tmp_path) -> None:
    session_dir = tmp_path / "session=legacy"
    session_dir.mkdir()
    pl.DataFrame(
        {
            "session_id": ["godot-live", "godot-live", "godot-live", "godot-live"],
            "frame_index": [0, 1, 0, 1],
            "timestamp_seconds": [1.0, 2.0, 10.0, 11.0],
        }
    ).write_parquet(session_dir / "frames.parquet")

    analysis = analyze_live_capture(session_dir)

    assert [clip["clip_index"] for clip in analysis["clips"]] == [0, 1]
    assert [clip["frames"] for clip in analysis["clips"]] == [2, 2]


def _frame_message() -> dict[str, object]:
    identity = [0.0, 0.0, 0.0, 1.0]
    return {
        "format": "godot_vr_humanoid_pose.live_capture.v1",
        "type": "frame",
        "session_id": "godot-live",
        "frame_index": 0,
        "clip_index": 3,
        "timestamp_seconds": 10.0,
        "sample": {
            "timestamp_seconds": 10.0,
            "hmd": _tracker("head", [0.0, 1.6, 0.0], identity),
            "left_hand": _tracker("left_hand", [-0.3, 1.1, 0.3], identity),
            "right_hand": _tracker("right_hand", [0.3, 1.1, 0.3], identity),
            "optional_trackers": {
                "left_elbow": _tracker("left_elbow", [-0.45, 1.2, 0.15], identity)
            },
        },
        "devices": [
            _device("head", "HMD", [0.0, 1.6, 0.0], identity),
            _device("left_hand", "Controller", [-0.3, 1.1, 0.3], identity),
            _device("right_hand", "Controller", [0.3, 1.1, 0.3], identity),
        ],
        "estimated_frame": {
            "timestamp_seconds": 10.0,
            "root": {"position": [0.0, 1.0, 0.0], "rotation": identity},
            "joints": {
                "left_elbow": {
                    "transform": {"position": [-0.35, 1.2, 0.15], "rotation": identity},
                    "estimated": True,
                    "confidence": 1.0,
                },
            },
        },
    }


def _tracker(role: str, position: list[float], rotation: list[float]) -> dict[str, object]:
    return {
        "role": role,
        "valid": True,
        "confidence": 1.0,
        "transform": {"position": position, "rotation": rotation},
    }


def _device(
    role: str,
    device_class: str,
    position: list[float],
    rotation: list[float],
) -> dict[str, object]:
    value = _tracker(role, position, rotation)
    value["device_class"] = device_class
    value["device_id"] = role
    return value
