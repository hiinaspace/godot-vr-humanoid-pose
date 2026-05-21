"""Export a small Python-vs-Godot estimator parity fixture."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from validation.vr_pose_validation.baseline import BasisPortEstimator
from validation.vr_pose_validation.geometry import Transform, axis_name_quaternion
from validation.vr_pose_validation.synthetic_trackers import (
    SyntheticTrackerConfig,
    generate_three_point_samples,
)
from validation.vr_pose_validation.types import (
    AvatarCalibration,
    GroundingContext,
    HumanoidPoseFrame,
    JointPose,
    TrackedPoseSample,
    TrackerAttachment,
    TrackerState,
)

DEFAULT_OUTPUT = Path("godot/tests/fixtures/basis_parity_fixture.json")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Fixture path to write. Defaults to {DEFAULT_OUTPUT}.",
    )
    args = parser.parse_args(argv)

    fixture = build_fixture()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


def build_fixture() -> dict[str, Any]:
    calibration = AvatarCalibration(
        human_height_m=1.72,
        foot_height_m=0.075,
        bone_lengths_m={
            "head_to_hips": 0.59,
            "head_to_neck": 0.075,
            "neck_to_chest": 0.15,
            "chest_to_spine": 0.16,
            "spine_to_hips": 0.205,
            "shoulder_width": 0.40,
            "hip_width": 0.20,
            "stance_width": 0.32,
            "upper_arm": 0.30,
            "lower_arm": 0.26,
            "upper_leg": 0.42,
            "lower_leg": 0.43,
        },
    )
    grounding = GroundingContext(floor_height_m=0.0)
    frames = _source_frames()
    samples = generate_three_point_samples(
        frames,
        SyntheticTrackerConfig(frame_rate_hz=60.0),
    )
    estimator = BasisPortEstimator()
    expected = [estimator.estimate(sample, calibration, grounding) for sample in samples]

    return {
        "schema_version": 1,
        "description": "Stateful BasisPortEstimator parity fixture exported from Python.",
        "calibration": _calibration_to_json(calibration),
        "grounding": _grounding_to_json(grounding),
        "samples": [_sample_to_json(sample) for sample in samples],
        "expected_frames": [_pose_frame_to_json(frame) for frame in expected],
    }


def _source_frames() -> list[dict[str, Transform]]:
    return [
        _source_frame(root_x=0.00, root_z=0.00, head_y=1.62, yaw_degrees=0.0),
        _source_frame(root_x=0.02, root_z=0.04, head_y=1.61, yaw_degrees=5.0),
        _source_frame(root_x=0.05, root_z=0.12, head_y=1.54, yaw_degrees=9.0),
        _source_frame(root_x=0.08, root_z=0.28, head_y=1.49, yaw_degrees=14.0),
        _source_frame(root_x=0.11, root_z=0.43, head_y=1.58, yaw_degrees=18.0),
    ]


def _source_frame(
    *,
    root_x: float,
    root_z: float,
    head_y: float,
    yaw_degrees: float,
) -> dict[str, Transform]:
    rotation = axis_name_quaternion("y", yaw_degrees)
    return {
        "head": Transform(position=(root_x, head_y, root_z), rotation=rotation),
        "left_hand": Transform(
            position=(root_x - 0.32, head_y - 0.36, root_z + 0.18),
            rotation=rotation,
        ),
        "right_hand": Transform(
            position=(root_x + 0.30, head_y - 0.34, root_z + 0.16),
            rotation=rotation,
        ),
    }


def _calibration_to_json(calibration: AvatarCalibration) -> dict[str, Any]:
    return {
        "avatar_scale": calibration.avatar_scale,
        "bone_lengths_m": dict(calibration.bone_lengths_m),
        "eye_to_head": _transform_to_json(calibration.eye_to_head),
        "foot_height_m": calibration.foot_height_m,
        "human_height_m": calibration.human_height_m,
        "left_controller_to_wrist": _transform_to_json(calibration.left_controller_to_wrist),
        "right_controller_to_wrist": _transform_to_json(calibration.right_controller_to_wrist),
    }


def _grounding_to_json(grounding: GroundingContext) -> dict[str, Any]:
    return {
        "character_velocity": list(grounding.character_velocity),
        "floor_height_m": grounding.floor_height_m,
        "floor_normal": list(grounding.floor_normal),
        "locomotion_state": grounding.locomotion_state,
    }


def _sample_to_json(sample: TrackedPoseSample) -> dict[str, Any]:
    return {
        "character_velocity": list(sample.character_velocity),
        "hmd": _tracker_state_to_json(sample.hmd),
        "left_hand": _tracker_state_to_json(sample.left_hand),
        "optional_trackers": {
            role: _tracker_state_to_json(tracker)
            for role, tracker in sample.optional_trackers.items()
        },
        "right_hand": _tracker_state_to_json(sample.right_hand),
        "timestamp_seconds": sample.timestamp_seconds,
    }


def _tracker_state_to_json(tracker: TrackerState) -> dict[str, Any]:
    return {
        "attachment": _attachment_to_json(tracker.attachment) if tracker.attachment else None,
        "confidence": tracker.confidence,
        "role": tracker.role,
        "transform": _transform_to_json(tracker.transform),
        "valid": tracker.valid,
    }


def _attachment_to_json(attachment: TrackerAttachment) -> dict[str, Any]:
    return {
        "confidence": attachment.confidence,
        "latency_seconds": attachment.latency_seconds,
        "local_transform": _transform_to_json(attachment.local_transform),
        "noise_profile_id": attachment.noise_profile_id,
        "role": attachment.role,
        "source_joint": attachment.source_joint,
    }


def _pose_frame_to_json(frame: HumanoidPoseFrame) -> dict[str, Any]:
    return {
        "joints": {name: _joint_pose_to_json(frame.joints[name]) for name in sorted(frame.joints)},
        "root": _transform_to_json(frame.root),
        "timestamp_seconds": frame.timestamp_seconds,
    }


def _joint_pose_to_json(joint: JointPose) -> dict[str, Any]:
    return {
        "confidence": joint.confidence,
        "estimated": joint.estimated,
        "transform": _transform_to_json(joint.transform),
    }


def _transform_to_json(transform: Transform) -> dict[str, list[float]]:
    return {
        "position": list(transform.position),
        "rotation": list(transform.rotation),
    }


if __name__ == "__main__":
    raise SystemExit(main())
