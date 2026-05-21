from __future__ import annotations

from pytest import approx

from validation.vr_pose_validation.baseline import (
    BasisPortEstimator,
    PlantedFeetBaselineEstimator,
    TrivialBaselineEstimator,
)
from validation.vr_pose_validation.metrics import (
    evaluate_frame,
    evaluate_reference_pose,
    summarize_reference_errors,
    summarize_target_errors,
)
from validation.vr_pose_validation.synthetic_trackers import (
    generate_three_point_samples,
    pose_frame_from_positions,
)
from validation.vr_pose_validation.types import AvatarCalibration, GroundingContext


def _sample_frame():
    return generate_three_point_samples(
        [
            pose_frame_from_positions(
                {
                    "head": (0, 1.6, 0),
                    "left_hand": (-0.3, 1.2, -0.2),
                    "right_hand": (0.3, 1.2, -0.2),
                }
            )
        ]
    )[0]


def test_trivial_baseline_tracks_head_and_hands_exactly() -> None:
    sample = _sample_frame()
    estimator = TrivialBaselineEstimator()
    calibration = AvatarCalibration(human_height_m=1.7, foot_height_m=0.08)
    grounding = GroundingContext(floor_height_m=0.0)

    pose = estimator.estimate(sample, calibration, grounding)
    metrics = evaluate_frame(pose, sample)

    assert pose.joints["head"].estimated is False
    assert metrics.target_errors.mean_error_m == 0.0
    assert metrics.foot_grounding.left_hover_m == approx(0.08)
    assert summarize_target_errors([metrics])["mean_target_error_m"] == 0.0


def test_planted_feet_baseline_keeps_anchors_until_step_threshold() -> None:
    estimator = PlantedFeetBaselineEstimator(step_threshold_m=10.0)
    calibration = AvatarCalibration()
    grounding = GroundingContext()
    first = estimator.estimate(_sample_frame(), calibration, grounding)
    second = estimator.estimate(_sample_frame(), calibration, grounding)

    assert (
        first.joints["left_foot"].transform.position
        == second.joints["left_foot"].transform.position
    )


def test_basis_port_scaffold_uses_pose_estimator_interface() -> None:
    sample = _sample_frame()
    pose = BasisPortEstimator().estimate(sample, AvatarCalibration(), GroundingContext())

    assert set(pose.joints) >= {
        "head",
        "left_hand",
        "right_hand",
        "hips",
        "left_foot",
        "right_foot",
    }


def test_reference_pose_summary_reports_optional_joint_errors() -> None:
    sample = _sample_frame()
    pose = BasisPortEstimator().estimate(sample, AvatarCalibration(), GroundingContext())
    reference = {
        name: joint.transform
        for name, joint in pose.joints.items()
        if name in {"hips", "left_foot", "right_foot", "left_knee"}
    }

    metrics = evaluate_reference_pose(pose, reference)
    summary = summarize_reference_errors([metrics])

    assert metrics.joint_errors_m["left_knee"] == 0.0
    assert summary["mean_joint_errors_m"] == {
        "hips": 0.0,
        "left_foot": 0.0,
        "left_knee": 0.0,
        "right_foot": 0.0,
    }
