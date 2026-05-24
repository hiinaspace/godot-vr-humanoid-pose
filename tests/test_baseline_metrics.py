from __future__ import annotations

from pytest import approx

from validation.vr_pose_validation.baseline import (
    BasisPortEstimator,
    PlantedFeetBaselineEstimator,
    TrivialBaselineEstimator,
)
from validation.vr_pose_validation.geometry import Transform, axis_name_quaternion
from validation.vr_pose_validation.metrics import (
    ReferencePoseMetrics,
    evaluate_foot_contact_sequence,
    evaluate_frame,
    evaluate_gait_plausibility_sequence,
    evaluate_reference_pose,
    infer_reference_floor_height,
    summarize_joint_flexion_sequence,
    summarize_orientation_sequence,
    summarize_reference_error_distribution,
    summarize_reference_errors,
    summarize_target_errors,
    summarize_temporal_reference_sequence,
)
from validation.vr_pose_validation.synthetic_trackers import (
    generate_three_point_samples,
    pose_frame_from_positions,
)
from validation.vr_pose_validation.types import (
    AvatarCalibration,
    GroundingContext,
    HumanoidPoseFrame,
    JointPose,
)


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
    assert metrics.pelvis_horizontal_error_m == 0.0
    assert metrics.pelvis_vertical_error_m == 0.0
    assert metrics.mean_torso_error_m == 0.0
    assert summary["mean_torso_error_m"] == 0.0
    assert summary["mean_left_knee_error_m"] == 0.0
    assert summary["mean_knee_error_m"] == 0.0
    assert summary["mean_joint_errors_m"] == {
        "hips": 0.0,
        "left_foot": 0.0,
        "left_knee": 0.0,
        "right_foot": 0.0,
    }


def test_reference_pose_summary_reports_limb_pair_errors() -> None:
    pose = HumanoidPoseFrame(
        timestamp_seconds=0.0,
        root=Transform(),
        joints={
            "head": JointPose(Transform(position=(0.0, 1.7, 0.0))),
            "hips": JointPose(Transform(position=(0.0, 0.0, 0.0))),
            "left_foot": JointPose(Transform(position=(0.0, -1.0, 0.0))),
            "right_foot": JointPose(Transform(position=(1.0, -1.0, 0.0))),
            "left_hand": JointPose(Transform(position=(-0.6, 1.0, 0.0))),
            "right_hand": JointPose(Transform(position=(0.6, 1.0, 0.0))),
            "left_shoulder": JointPose(Transform(position=(-0.2, 1.4, 0.0))),
            "right_shoulder": JointPose(Transform(position=(0.2, 1.4, 0.0))),
            "left_elbow": JointPose(Transform(position=(-0.4, 1.0, 0.0))),
            "right_elbow": JointPose(Transform(position=(0.4, 1.0, 0.0))),
            "left_knee": JointPose(Transform(position=(-0.2, 0.5, 0.0))),
            "right_knee": JointPose(Transform(position=(0.2, 0.5, 0.0))),
        },
    )
    reference = {
        "head": Transform(position=(0.0, 1.7, 0.0)),
        "hips": Transform(position=(0.0, 0.0, 0.0)),
        "left_foot": Transform(position=(0.0, -1.0, 0.0)),
        "right_foot": Transform(position=(1.0, -1.0, 0.0)),
        "left_hand": Transform(position=(-0.6, 1.0, 0.0)),
        "right_hand": Transform(position=(0.8, 1.0, 0.0)),
        "left_shoulder": Transform(position=(-0.1, 1.4, 0.0)),
        "right_shoulder": Transform(position=(0.4, 1.4, 0.0)),
        "left_elbow": Transform(position=(-0.1, 1.0, 0.0)),
        "right_elbow": Transform(position=(0.8, 1.0, 0.0)),
        "left_knee": Transform(position=(-0.7, 0.5, 0.0)),
        "right_knee": Transform(position=(0.8, 0.5, 0.0)),
    }

    summary = summarize_reference_errors([evaluate_reference_pose(pose, reference)])

    assert summary["mean_left_shoulder_error_m"] == approx(0.1)
    assert summary["mean_right_shoulder_error_m"] == approx(0.2)
    assert summary["mean_shoulder_error_m"] == approx(0.15)
    assert summary["mean_elbow_error_m"] == approx(0.35)
    assert summary["mean_knee_error_m"] == approx(0.55)
    assert summary["mean_hand_reference_error_m"] == approx(0.1)
    assert summary["mean_upper_body_reference_error_m"] == approx(1.2 / 7.0)
    assert summary["mean_lower_body_reference_error_m"] == approx(1.1 / 5.0)
    assert summary["mean_full_body_reference_error_m"] == approx(2.3 / 12.0)


def test_reference_error_distribution_reports_quantiles_and_thresholds() -> None:
    metrics = [
        ReferencePoseMetrics(
            pelvis_error_m=value,
            pelvis_horizontal_error_m=value,
            pelvis_vertical_error_m=0.0,
            chest_error_m=value,
            spine_error_m=value,
            left_foot_error_m=value,
            right_foot_error_m=value,
            joint_errors_m={
                "left_shoulder": value,
                "right_shoulder": value,
                "left_elbow": value,
                "right_elbow": value,
                "left_knee": value,
                "right_knee": value,
            },
        )
        for value in (0.0, 1.0, 2.0, 3.0)
    ]

    summary = summarize_reference_error_distribution(metrics)

    assert summary["reference_error_p50_m"] == approx(1.5)
    assert summary["reference_error_p90_m"] == approx(2.7)
    assert summary["reference_error_p95_m"] == approx(2.85)
    assert summary["reference_error_p99_m"] == approx(2.97)
    assert summary["pct_reference_error_over_25cm"] == approx(75.0)
    assert summary["pct_reference_error_over_50cm"] == approx(75.0)
    assert summary["pct_reference_error_over_100cm"] == approx(50.0)
    assert summary["shoulder_error_p95_m"] == approx(2.85)
    assert summary["feet_error_p95_m"] == approx(2.85)


def test_temporal_reference_sequence_reports_velocity_acceleration_and_jerk() -> None:
    reference_frames = [
        {"hips": Transform(position=(float(index), 0.0, 0.0))} for index in range(4)
    ]
    estimated_positions = (0.0, 2.0, 4.0, 8.0)
    poses = [
        HumanoidPoseFrame(
            timestamp_seconds=float(index),
            root=Transform(),
            joints={"hips": JointPose(Transform(position=(estimated_positions[index], 0.0, 0.0)))},
        )
        for index in range(4)
    ]

    summary = summarize_temporal_reference_sequence(poses, reference_frames)

    assert summary["mean_joint_velocity_error_m_per_s"] == approx(5.0 / 3.0)
    assert summary["mean_joint_acceleration_error_m_per_s2"] == approx(1.0)
    assert summary["mean_joint_estimated_jerk_m_per_s3"] == approx(2.0)
    assert summary["mean_pelvis_velocity_error_m_per_s"] == approx(5.0 / 3.0)
    assert summary["mean_torso_acceleration_error_m_per_s2"] == approx(1.0)
    assert summary["mean_feet_velocity_error_m_per_s"] == 0.0


def test_foot_contact_metrics_measure_estimated_slip_during_reference_contact() -> None:
    reference_frames = [
        {
            "left_foot": Transform(position=(0.0, 0.0, 0.0)),
            "right_foot": Transform(position=(0.3, 0.0, 0.0)),
        },
        {
            "left_foot": Transform(position=(0.0, 0.0, 0.0)),
            "right_foot": Transform(position=(0.3, 0.0, 0.0)),
        },
        {
            "left_foot": Transform(position=(0.0, 0.0, 0.0)),
            "right_foot": Transform(position=(0.3, 0.0, 0.0)),
        },
    ]
    poses = [
        _foot_pose(0.0, 0.0),
        _foot_pose(0.05, 1.0 / 120.0),
        _foot_pose(0.10, 2.0 / 120.0),
    ]

    metrics = evaluate_foot_contact_sequence(
        poses,
        reference_frames,
        floor_height_m=infer_reference_floor_height(reference_frames),
    )

    assert metrics.left_contact_frames == 3
    assert metrics.mean_left_contact_foot_error_m > 0.0
    assert metrics.left_contact_slip_m_per_s == approx(6.0)


def test_gait_plausibility_metrics_flag_output_side_foot_skate_and_support() -> None:
    poses = [
        _gait_pose(
            timestamp_seconds=0.0,
            hips=(0.0, 1.0, 0.0),
            left_foot=(0.0, 0.0, 0.0),
            right_foot=(0.3, 0.0, 0.0),
        ),
        _gait_pose(
            timestamp_seconds=1.0 / 120.0,
            hips=(0.8, 1.0, 0.0),
            left_foot=(0.05, 0.0, 0.0),
            right_foot=(0.35, 0.0, 0.0),
        ),
        _gait_pose(
            timestamp_seconds=2.0 / 120.0,
            hips=(1.2, 1.0, 0.0),
            left_foot=(0.10, 0.0, 0.0),
            right_foot=(0.40, 0.0, 0.0),
        ),
    ]

    metrics = evaluate_gait_plausibility_sequence(poses, floor_height_m=0.0)

    assert metrics.low_foot_frames == 6
    assert metrics.mean_low_foot_slip_m_per_s == approx(6.0)
    assert metrics.p95_low_foot_slip_m_per_s == approx(6.0)
    assert metrics.pct_low_foot_slip_speed_over_20cm_s == 100.0
    assert metrics.mean_hips_support_distance_m > 0.4
    assert metrics.pct_hips_support_distance_over_60cm == approx(100.0 / 3.0)
    assert metrics.mean_estimated_stance_width_m == approx(0.3)
    assert metrics.pct_estimated_stance_width_under_20cm == 0.0
    assert metrics.pct_estimated_leg_reach_over_110cm > 0.0


def test_joint_flexion_metrics_report_angle_error_and_limit_violations() -> None:
    reference_frames = [
        {
            "left_shoulder": Transform(position=(-1.0, 0.0, 0.0)),
            "left_elbow": Transform(position=(0.0, 0.0, 0.0)),
            "left_hand": Transform(position=(1.0, 0.0, 0.0)),
            "right_shoulder": Transform(position=(1.0, 0.0, 0.0)),
            "right_elbow": Transform(position=(0.0, 0.0, 0.0)),
            "right_hand": Transform(position=(-1.0, 0.0, 0.0)),
            "hips": Transform(position=(0.0, 1.0, 0.0)),
            "left_knee": Transform(position=(0.0, 0.0, 0.0)),
            "left_foot": Transform(position=(0.0, -1.0, 0.0)),
            "right_knee": Transform(position=(1.0, 0.0, 0.0)),
            "right_foot": Transform(position=(1.0, -1.0, 0.0)),
        }
    ]
    poses = [
        HumanoidPoseFrame(
            timestamp_seconds=0.0,
            root=Transform(),
            joints={
                "left_shoulder": JointPose(Transform(position=(-1.0, 0.0, 0.0))),
                "left_elbow": JointPose(Transform(position=(0.0, 0.0, 0.0))),
                "left_hand": JointPose(Transform(position=(0.0, -1.0, 0.0))),
                "right_shoulder": JointPose(Transform(position=(1.0, 0.0, 0.0))),
                "right_elbow": JointPose(Transform(position=(0.0, 0.0, 0.0))),
                "right_hand": JointPose(Transform(position=(0.1, 0.0, 0.0))),
                "hips": JointPose(Transform(position=(0.0, 1.0, 0.0))),
                "left_knee": JointPose(Transform(position=(0.0, 0.0, 0.0))),
                "left_foot": JointPose(Transform(position=(0.0, -1.0, 0.0))),
                "right_knee": JointPose(Transform(position=(1.0, 0.0, 0.0))),
                "right_foot": JointPose(Transform(position=(1.0, -1.0, 0.0))),
            },
        )
    ]

    summary = summarize_joint_flexion_sequence(poses, reference_frames)

    assert summary["mean_left_elbow_flexion_error_deg"] == approx(90.0)
    assert summary["mean_right_elbow_flexion_error_deg"] == approx(180.0)
    assert summary["mean_knee_flexion_error_deg"] == approx(0.0)
    assert summary["pct_left_elbow_flexion_over_limit"] == 0.0
    assert summary["pct_right_elbow_flexion_over_limit"] == 100.0
    assert summary["pct_elbow_flexion_over_limit"] == approx(50.0)
    assert summary["pct_joint_flexion_over_limit"] == approx(25.0)


def test_orientation_metrics_report_head_flip_and_torso_axis_error() -> None:
    reference_frames = [
        _orientation_reference(),
        _orientation_reference(),
    ]
    poses = [
        _orientation_pose(
            timestamp_seconds=0.0,
            head_rotation=Transform().rotation,
            hips=(0.0, 0.0, 0.0),
            chest=(0.0, 1.0, 0.0),
            head=(0.0, 2.0, 0.0),
        ),
        _orientation_pose(
            timestamp_seconds=0.1,
            head_rotation=axis_name_quaternion("y", 180.0),
            hips=(0.0, 0.0, 0.0),
            chest=(2.0, 1.0, 0.0),
            head=(0.0, 2.0, 0.0),
        ),
    ]

    summary = summarize_orientation_sequence(poses, reference_frames)

    assert summary["mean_head_forward_error_deg"] == approx(90.0)
    assert summary["pct_head_forward_error_over_90deg"] == approx(50.0)
    assert summary["mean_head_up_error_deg"] == approx(0.0)
    assert summary["mean_neck_axis_error_deg"] == approx(31.717474, abs=1e-6)
    assert summary["mean_spine_axis_error_deg"] == approx(31.717474, abs=1e-6)
    assert summary["pct_neck_axis_error_over_45deg"] == approx(50.0)
    assert summary["pct_spine_axis_error_over_45deg"] == approx(50.0)
    assert summary["max_head_forward_angular_speed_deg_s"] == approx(1800.0)
    assert summary["pct_head_forward_angular_speed_over_720deg_s"] == approx(100.0)


def _foot_pose(left_x: float, timestamp_seconds: float) -> HumanoidPoseFrame:
    joints = {
        "left_foot": JointPose(Transform(position=(left_x, 0.0, 0.0))),
        "right_foot": JointPose(Transform(position=(0.3, 0.0, 0.0))),
    }
    return HumanoidPoseFrame(
        timestamp_seconds=timestamp_seconds,
        root=Transform(),
        joints=joints,
    )


def _gait_pose(
    *,
    timestamp_seconds: float,
    hips: tuple[float, float, float],
    left_foot: tuple[float, float, float],
    right_foot: tuple[float, float, float],
) -> HumanoidPoseFrame:
    joints = {
        "hips": JointPose(Transform(position=hips)),
        "left_foot": JointPose(Transform(position=left_foot)),
        "right_foot": JointPose(Transform(position=right_foot)),
    }
    return HumanoidPoseFrame(
        timestamp_seconds=timestamp_seconds,
        root=Transform(position=hips),
        joints=joints,
    )


def _orientation_reference() -> dict[str, Transform]:
    return {
        "hips": Transform(position=(0.0, 0.0, 0.0)),
        "chest": Transform(position=(0.0, 1.0, 0.0)),
        "head": Transform(position=(0.0, 2.0, 0.0)),
    }


def _orientation_pose(
    *,
    timestamp_seconds: float,
    head_rotation: tuple[float, float, float, float],
    hips: tuple[float, float, float],
    chest: tuple[float, float, float],
    head: tuple[float, float, float],
) -> HumanoidPoseFrame:
    return HumanoidPoseFrame(
        timestamp_seconds=timestamp_seconds,
        root=Transform(position=hips),
        joints={
            "hips": JointPose(Transform(position=hips)),
            "chest": JointPose(Transform(position=chest)),
            "head": JointPose(Transform(position=head, rotation=head_rotation)),
        },
    )
