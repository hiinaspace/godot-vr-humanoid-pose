from __future__ import annotations

from pytest import approx

from validation.vr_pose_validation.baseline import BasisEstimatorSettings, BasisPortEstimator
from validation.vr_pose_validation.geometry import Transform
from validation.vr_pose_validation.synthetic_trackers import (
    generate_three_point_samples,
    pose_frame_from_positions,
)
from validation.vr_pose_validation.types import (
    AvatarCalibration,
    GroundingContext,
    TrackedPoseSample,
    TrackerState,
)


def _sample_with_head(head_position: tuple[float, float, float]) -> TrackedPoseSample:
    return generate_three_point_samples(
        [
            pose_frame_from_positions(
                {
                    "head": head_position,
                    "left_hand": (-0.3, 1.2, -0.2),
                    "right_hand": (0.3, 1.2, -0.2),
                }
            )
        ]
    )[0]


def test_basis_virtual_spine_uses_source_head_and_forward_bias() -> None:
    estimator = BasisPortEstimator(
        BasisEstimatorSettings(
            head_to_hips_ratio=0.34,
            hips_forward_bias_m=0.02,
            stance_width_ratio=0.2,
        )
    )
    sample = _sample_with_head((0.0, 1.6, 0.0))

    pose = estimator.estimate(
        sample,
        AvatarCalibration(human_height_m=2.0, foot_height_m=0.05),
        GroundingContext(),
    )

    assert pose.joints["head"].transform.position == sample.hmd.transform.position
    assert pose.joints["hips"].transform.position == approx((0.0, 0.92, 0.02))
    assert pose.joints["chest"].transform.position == approx((0.0, 1.3416, 0.0076))
    assert pose.joints["spine"].transform.position == approx((0.0, 1.1376, 0.0136))


def test_basis_virtual_spine_blends_optional_pelvis_tracker_xz_only() -> None:
    estimator = BasisPortEstimator(
        BasisEstimatorSettings(
            head_to_hips_ratio=0.3,
            hips_forward_bias_m=0.0,
            hips_xz_follow_blend=0.5,
        )
    )
    sample = _sample_with_head((0.0, 1.6, 0.0))
    sample = TrackedPoseSample(
        timestamp_seconds=sample.timestamp_seconds,
        hmd=sample.hmd,
        left_hand=sample.left_hand,
        right_hand=sample.right_hand,
        optional_trackers={
            "pelvis": TrackerState(
                role="pelvis",
                transform=Transform(position=(1.0, 0.2, 2.0)),
            )
        },
    )

    pose = estimator.estimate(sample, AvatarCalibration(human_height_m=2.0), GroundingContext())

    assert pose.joints["hips"].transform.position == approx((0.5, 1.0, 1.0))


def test_basis_sticky_feet_hold_until_anchor_stretch_exceeds_limit() -> None:
    estimator = BasisPortEstimator(
        BasisEstimatorSettings(
            head_to_hips_ratio=0.3,
            hips_forward_bias_m=0.0,
            stance_width_ratio=0.2,
            max_anchor_stretch_m=0.5,
            plant_velocity_threshold_m_s=100.0,
        )
    )
    calibration = AvatarCalibration(human_height_m=2.0, foot_height_m=0.05)
    grounding = GroundingContext()

    first = estimator.estimate(_sample_with_head((0.0, 1.6, 0.0)), calibration, grounding)
    second = estimator.estimate(_sample_with_head((0.2, 1.6, 0.0)), calibration, grounding)
    third = estimator.estimate(_sample_with_head((1.0, 1.6, 0.0)), calibration, grounding)

    assert (
        second.joints["left_foot"].transform.position
        == first.joints["left_foot"].transform.position
    )
    assert (
        third.joints["left_foot"].transform.position != first.joints["left_foot"].transform.position
    )


def test_basis_sticky_feet_release_when_projected_target_moves_fast() -> None:
    estimator = BasisPortEstimator(
        BasisEstimatorSettings(
            head_to_hips_ratio=0.3,
            hips_forward_bias_m=0.0,
            stance_width_ratio=0.2,
            max_anchor_stretch_m=10.0,
            plant_velocity_threshold_m_s=0.01,
        )
    )
    calibration = AvatarCalibration(human_height_m=2.0, foot_height_m=0.05)
    grounding = GroundingContext()

    first = estimator.estimate(_sample_with_head((0.0, 1.6, 0.0)), calibration, grounding)
    second = estimator.estimate(_sample_with_head((0.2, 1.6, 0.0)), calibration, grounding)

    assert (
        second.joints["left_foot"].transform.position
        != first.joints["left_foot"].transform.position
    )


def test_basis_outputs_two_bone_limb_joints() -> None:
    pose = BasisPortEstimator().estimate(
        _sample_with_head((0.0, 1.6, 0.0)),
        AvatarCalibration(human_height_m=1.8),
        GroundingContext(),
    )

    assert set(pose.joints) >= {
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_knee",
        "right_knee",
    }
    assert (
        pose.joints["left_foot"].transform.position[1]
        < pose.joints["left_knee"].transform.position[1]
        < pose.joints["hips"].transform.position[1]
    )
