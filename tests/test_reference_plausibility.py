from __future__ import annotations

from validation.vr_pose_validation.geometry import Transform
from validation.vr_pose_validation.parquet_cache import CACHE_JOINTS
from validation.vr_pose_validation.plausibility import (
    summarize_avatar_plausibility,
    summarize_reference_plausibility,
)
from validation.vr_pose_validation.types import AvatarCalibration, HumanoidPoseFrame, JointPose


def test_reference_plausibility_accepts_smooth_upright_motion() -> None:
    frames = [
        _reference_frame(head=(index * 0.005, 1.6, 0.0), hips=(index * 0.005, 1.0, 0.0))
        for index in range(40)
    ]

    summary = summarize_reference_plausibility(
        frames,
        timestamps_seconds=[index / 120.0 for index in range(len(frames))],
        floor_height_m=0.0,
    )

    assert summary.bucket == "plausible"
    assert summary.tags == ()


def test_reference_plausibility_rejects_head_below_floor() -> None:
    frames = [_reference_frame(head=(0.0, -0.2, 0.0), hips=(0.0, 0.5, 0.0)) for _index in range(40)]

    summary = summarize_reference_plausibility(
        frames,
        timestamps_seconds=[index / 120.0 for index in range(len(frames))],
        floor_height_m=0.0,
    )

    assert summary.bucket == "implausible_reference"
    assert "head_below_floor" in summary.tags


def test_reference_plausibility_rejects_motion_discontinuity() -> None:
    frames = [_reference_frame(head=(0.0, 1.6, 0.0), hips=(0.0, 1.0, 0.0)) for _index in range(40)]
    frames[20] = _reference_frame(head=(8.0, 1.6, 0.0), hips=(8.0, 1.0, 0.0))

    summary = summarize_reference_plausibility(
        frames,
        timestamps_seconds=[index / 120.0 for index in range(len(frames))],
        floor_height_m=0.0,
    )

    assert summary.bucket == "implausible_reference"
    assert "motion_discontinuity" in summary.tags
    assert summary.max_joint_speed_m_per_s > 25.0


def test_avatar_plausibility_accepts_clean_output_sequence() -> None:
    poses = [
        _avatar_pose(timestamp_seconds=index / 120.0, offset_x=index * 0.002) for index in range(40)
    ]

    summary = summarize_avatar_plausibility(
        poses,
        calibration=AvatarCalibration(human_height_m=2.0),
        floor_height_m=0.0,
    )

    assert summary.bucket == "plausible"
    assert summary.tags == ()
    assert summary.plausibility_score_0_100 > 99.0
    assert summary.endpoint_stretch_penalty == 0.0


def test_avatar_plausibility_flags_endpoint_stretch() -> None:
    poses = [
        _avatar_pose(
            timestamp_seconds=index / 120.0,
            left_hand=(-0.22, 0.15, 0.0),
        )
        for index in range(40)
    ]

    summary = summarize_avatar_plausibility(
        poses,
        calibration=AvatarCalibration(human_height_m=2.0),
        floor_height_m=0.0,
    )

    assert summary.bucket == "implausible_output"
    assert "endpoint_stretch" in summary.tags
    assert summary.max_endpoint_stretch_ratio > 1.25
    assert summary.endpoint_stretch_penalty > 0.0


def test_avatar_plausibility_flags_sustained_joint_limit_violation() -> None:
    poses = [
        _avatar_pose(
            timestamp_seconds=index / 120.0,
            left_elbow=(-0.22, 1.18, 0.0),
            left_hand=(-0.22, 1.50, 0.0),
        )
        for index in range(40)
    ]

    summary = summarize_avatar_plausibility(
        poses,
        calibration=AvatarCalibration(human_height_m=2.0),
        floor_height_m=0.0,
    )

    assert summary.bucket == "implausible_output"
    assert "joint_limit_violation" in summary.tags
    assert "sustained_awkwardness" in summary.tags
    assert summary.min_joint_limit_margin_deg < 0.0
    assert summary.joint_limit_penalty > 0.0


def test_avatar_plausibility_flags_temporal_jerk_without_gt() -> None:
    offsets = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    poses = [
        _avatar_pose(timestamp_seconds=index / 120.0, offset_x=offset)
        for index, offset in enumerate(offsets)
    ]

    summary = summarize_avatar_plausibility(
        poses,
        calibration=AvatarCalibration(human_height_m=2.0),
        floor_height_m=0.0,
    )

    assert summary.bucket == "implausible_output"
    assert "temporal_jerk_spike" in summary.tags
    assert summary.max_joint_jerk_m_per_s3 > 2500.0
    assert summary.temporal_jerk_penalty > 0.0


def _reference_frame(
    *,
    head: tuple[float, float, float],
    hips: tuple[float, float, float],
) -> dict[str, Transform]:
    joints = {
        joint: Transform(position=hips, rotation=(0.0, 0.0, 0.0, 1.0)) for joint in CACHE_JOINTS
    }
    joints.update(
        {
            "head": Transform(position=head),
            "hips": Transform(position=hips),
            "chest": Transform(position=(hips[0], hips[1] + 0.42, hips[2])),
            "spine": Transform(position=(hips[0], hips[1] + 0.25, hips[2])),
            "left_shoulder": Transform(position=(hips[0] - 0.22, hips[1] + 0.48, hips[2])),
            "right_shoulder": Transform(position=(hips[0] + 0.22, hips[1] + 0.48, hips[2])),
            "left_elbow": Transform(position=(hips[0] - 0.36, hips[1] + 0.34, hips[2])),
            "right_elbow": Transform(position=(hips[0] + 0.36, hips[1] + 0.34, hips[2])),
            "left_hand": Transform(position=(hips[0] - 0.42, hips[1] + 0.12, hips[2])),
            "right_hand": Transform(position=(hips[0] + 0.42, hips[1] + 0.12, hips[2])),
            "left_foot": Transform(position=(hips[0] - 0.12, 0.0, hips[2])),
            "right_foot": Transform(position=(hips[0] + 0.12, 0.0, hips[2])),
            "left_knee": Transform(position=(hips[0] - 0.12, hips[1] * 0.5, hips[2])),
            "right_knee": Transform(position=(hips[0] + 0.12, hips[1] * 0.5, hips[2])),
        }
    )
    return joints


def _avatar_pose(
    *,
    timestamp_seconds: float,
    offset_x: float = 0.0,
    left_elbow: tuple[float, float, float] | None = None,
    left_hand: tuple[float, float, float] | None = None,
) -> HumanoidPoseFrame:
    joints = {
        "head": (0.0, 1.82, 0.0),
        "chest": (0.0, 1.50, 0.0),
        "spine": (0.0, 1.25, 0.0),
        "hips": (0.0, 1.00, 0.0),
        "left_shoulder": (-0.22, 1.50, 0.0),
        "right_shoulder": (0.22, 1.50, 0.0),
        "left_elbow": left_elbow or (-0.22, 1.152, 0.0),
        "right_elbow": (0.22, 1.152, 0.0),
        "left_hand": left_hand or (-0.22, 0.850, 0.0),
        "right_hand": (0.22, 0.850, 0.0),
        "left_knee": (-0.11, 0.510, 0.0),
        "right_knee": (0.11, 0.510, 0.0),
        "left_foot": (-0.11, 0.018, 0.0),
        "right_foot": (0.11, 0.018, 0.0),
    }
    offset_joints = {
        name: JointPose(
            transform=Transform(
                position=(position[0] + offset_x, position[1], position[2]),
            )
        )
        for name, position in joints.items()
    }
    return HumanoidPoseFrame(
        timestamp_seconds=timestamp_seconds,
        root=Transform(position=offset_joints["hips"].transform.position),
        joints=offset_joints,
    )
