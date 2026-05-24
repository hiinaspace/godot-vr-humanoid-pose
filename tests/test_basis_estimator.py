from __future__ import annotations

from pytest import approx

from validation.vr_pose_validation.baseline import (
    BasisEstimatorSettings,
    BasisPortEstimator,
    create_estimator,
)
from validation.vr_pose_validation.geometry import (
    Transform,
    cross,
    distance,
    dot,
    horizontal_distance,
    look_rotation,
    normalize,
    subtract,
)
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


def _manual_sample(
    head_position: tuple[float, float, float],
    *,
    timestamp_seconds: float,
    forward: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> TrackedPoseSample:
    x, _, z = head_position
    return TrackedPoseSample(
        timestamp_seconds=timestamp_seconds,
        hmd=TrackerState(
            role="head",
            transform=Transform(position=head_position, rotation=look_rotation(forward)),
        ),
        left_hand=TrackerState(
            role="left_hand",
            transform=Transform(position=(x + 0.3, 1.2, z + 0.2)),
        ),
        right_hand=TrackerState(
            role="right_hand",
            transform=Transform(position=(x - 0.3, 1.2, z + 0.2)),
        ),
    )


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
    assert pose.joints["neck"].transform.position == approx((0.0, 1.52, 0.0))
    assert pose.joints["hips"].transform.position == approx((0.0, 0.92, 0.02))
    assert pose.joints["chest"].transform.position == approx((0.0, 1.34, 0.006))
    assert pose.joints["spine"].transform.position == approx((0.0, 1.14, 0.0126666667))


def test_basis_virtual_spine_preserves_calibrated_head_to_hips_length() -> None:
    estimator = BasisPortEstimator(
        BasisEstimatorSettings(
            head_to_hips_ratio=0.34,
            hips_forward_bias_m=0.0,
        )
    )
    sample = _sample_with_head((0.0, 1.6, 0.0))

    pose = estimator.estimate(
        sample,
        AvatarCalibration(
            human_height_m=2.0,
            bone_lengths_m={"head_to_hips": 0.8, "head_to_neck": 0.1},
        ),
        GroundingContext(),
    )

    assert pose.joints["hips"].transform.position == approx((0.0, 0.8, 0.0))
    assert pose.joints["chest"].transform.position[1] > pose.joints["spine"].transform.position[1]


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
                transform=Transform(position=(0.1, 0.2, 0.2)),
            )
        },
    )

    pose = estimator.estimate(sample, AvatarCalibration(human_height_m=2.0), GroundingContext())

    assert pose.joints["hips"].transform.position == approx((0.05, 1.0, 0.1))


def test_basis_virtual_spine_can_lean_against_planted_support() -> None:
    estimator = BasisPortEstimator(
        BasisEstimatorSettings(
            head_to_hips_ratio=0.3,
            hips_forward_bias_m=0.0,
            stance_width_ratio=0.2,
            support_hips_xz_follow_blend=1.0,
            max_torso_lean_degrees=45.0,
        )
    )
    calibration = AvatarCalibration(human_height_m=2.0, foot_height_m=0.05)
    grounding = GroundingContext()

    estimator.estimate(_sample_with_head((0.0, 1.6, 0.0)), calibration, grounding)
    leaned = estimator.estimate(_sample_with_head((0.0, 1.6, 0.2)), calibration, grounding)

    assert leaned.joints["hips"].transform.position[2] < 0.1
    assert (
        leaned.joints["chest"].transform.position[2] > leaned.joints["hips"].transform.position[2]
    )
    assert (
        leaned.joints["chest"].transform.position[2] < leaned.joints["neck"].transform.position[2]
    )


def test_basis_virtual_spine_places_neck_from_pitched_head_rotation() -> None:
    head_rotation = look_rotation(normalize((0.0, -0.4, 1.0)))
    sample = _sample_with_head((0.0, 1.6, 0.0))
    sample = TrackedPoseSample(
        timestamp_seconds=sample.timestamp_seconds,
        hmd=TrackerState(
            role="head",
            transform=Transform(position=(0.0, 1.6, 0.0), rotation=head_rotation),
        ),
        left_hand=sample.left_hand,
        right_hand=sample.right_hand,
    )

    pose = BasisPortEstimator().estimate(
        sample,
        AvatarCalibration(human_height_m=2.0),
        GroundingContext(),
    )

    assert pose.joints["neck"].transform.position[1] < pose.joints["head"].transform.position[1]
    assert pose.joints["neck"].transform.position[2] < pose.joints["head"].transform.position[2]


def test_basis_body_right_matches_canonical_forward_convention() -> None:
    pose = BasisPortEstimator().estimate(
        _sample_with_head((0.0, 1.6, 0.0)),
        AvatarCalibration(human_height_m=1.8),
        GroundingContext(),
    )

    assert (
        pose.joints["left_shoulder"].transform.position[0]
        > pose.joints["right_shoulder"].transform.position[0]
    )


def test_basis_body_forward_flips_when_controllers_would_cross_arms() -> None:
    sample = TrackedPoseSample(
        timestamp_seconds=0.0,
        hmd=TrackerState(
            role="head",
            transform=Transform(
                position=(0.0, 1.6, 0.0),
                rotation=look_rotation(normalize((0.0, -0.4, -1.0))),
            ),
        ),
        left_hand=TrackerState(
            role="left_hand",
            transform=Transform(position=(0.55, 1.25, 0.0)),
        ),
        right_hand=TrackerState(
            role="right_hand",
            transform=Transform(position=(-0.55, 1.25, 0.0)),
        ),
    )

    pose = BasisPortEstimator().estimate(
        sample,
        AvatarCalibration(human_height_m=1.8),
        GroundingContext(),
    )
    left_shoulder = pose.joints["left_shoulder"].transform.position
    right_shoulder = pose.joints["right_shoulder"].transform.position

    assert left_shoulder[0] > right_shoulder[0]
    assert distance(left_shoulder, sample.left_hand.transform.position) < distance(
        left_shoulder,
        sample.right_hand.transform.position,
    )
    assert distance(right_shoulder, sample.right_hand.transform.position) < distance(
        right_shoulder,
        sample.left_hand.transform.position,
    )


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
        "neck",
    }
    assert (
        pose.joints["left_foot"].transform.position[1]
        < pose.joints["left_knee"].transform.position[1]
        < pose.joints["hips"].transform.position[1]
    )


def test_basis_shoulder_follows_reaching_controller_to_preserve_arm_lengths() -> None:
    estimator = BasisPortEstimator()
    sample = TrackedPoseSample(
        timestamp_seconds=0.0,
        hmd=TrackerState(
            role="head",
            transform=Transform(position=(0.0, 1.6, 0.0), rotation=look_rotation((0.0, 0.0, 1.0))),
        ),
        left_hand=TrackerState(
            role="left_hand",
            transform=Transform(position=(0.85, 1.22, 0.18)),
        ),
        right_hand=TrackerState(
            role="right_hand",
            transform=Transform(position=(-0.3, 1.22, 0.18)),
        ),
    )
    calibration = AvatarCalibration(human_height_m=1.8)

    pose = estimator.estimate(sample, calibration, GroundingContext())

    shoulder = pose.joints["left_shoulder"].transform.position
    elbow = pose.joints["left_elbow"].transform.position
    hand = pose.joints["left_hand"].transform.position
    upper_arm = calibration.human_height_m * estimator.settings.upper_arm_ratio
    lower_arm = calibration.human_height_m * estimator.settings.lower_arm_ratio

    assert shoulder[0] > 0.22
    assert distance(shoulder, hand) <= (
        upper_arm + lower_arm + estimator.settings.shoulder_reach_slack_m + 1e-9
    )
    assert distance(shoulder, elbow) == approx(upper_arm)
    assert distance(elbow, hand) <= lower_arm + estimator.settings.shoulder_reach_slack_m + 0.0002


def test_basis_elbow_bend_hint_points_behind_controller_reach() -> None:
    sample = TrackedPoseSample(
        timestamp_seconds=0.0,
        hmd=TrackerState(
            role="head",
            transform=Transform(position=(0.0, 1.6, 0.0), rotation=look_rotation((0.0, 0.0, 1.0))),
        ),
        left_hand=TrackerState(
            role="left_hand",
            transform=Transform(position=(0.4, 1.2, 0.1)),
        ),
        right_hand=TrackerState(
            role="right_hand",
            transform=Transform(position=(-0.3, 1.22, 0.18)),
        ),
    )

    pose = BasisPortEstimator().estimate(
        sample,
        AvatarCalibration(human_height_m=1.8),
        GroundingContext(),
    )

    shoulder = pose.joints["left_shoulder"].transform.position
    elbow = pose.joints["left_elbow"].transform.position
    hand = pose.joints["left_hand"].transform.position
    assert elbow[2] < shoulder[2]
    assert elbow[2] < hand[2]


def test_manikin_swivel_arm_hint_prefers_lower_elbow_orbit() -> None:
    sample = TrackedPoseSample(
        timestamp_seconds=0.0,
        hmd=TrackerState(
            role="head",
            transform=Transform(position=(0.0, 1.6, 0.0), rotation=look_rotation((0.0, 0.0, 1.0))),
        ),
        left_hand=TrackerState(
            role="left_hand",
            transform=Transform(position=(0.38, 1.25, 0.18)),
        ),
        right_hand=TrackerState(
            role="right_hand",
            transform=Transform(position=(-0.38, 1.25, 0.18)),
        ),
    )

    pose = BasisPortEstimator(BasisEstimatorSettings(arm_bend_strategy="manikin_swivel")).estimate(
        sample,
        AvatarCalibration(human_height_m=1.8),
        GroundingContext(),
    )

    shoulder = pose.joints["left_shoulder"].transform.position
    elbow = pose.joints["left_elbow"].transform.position
    hand = pose.joints["left_hand"].transform.position
    assert elbow[1] < (shoulder[1] + hand[1]) * 0.5


def test_basis_knee_hint_factory_isolated_from_default_basis() -> None:
    default = create_estimator("basis")
    experiment = create_estimator("basis_knee_hint")

    assert isinstance(default, BasisPortEstimator)
    assert isinstance(experiment, BasisPortEstimator)
    assert not default.settings.knee_hint_enabled
    assert experiment.settings.knee_hint_enabled


def test_basis_footstep_factory_isolated_from_default_basis() -> None:
    default = create_estimator("basis")
    experiment = create_estimator("basis_footstep")

    assert isinstance(default, BasisPortEstimator)
    assert isinstance(experiment, BasisPortEstimator)
    assert not default.settings.footstep_planner_enabled
    assert experiment.settings.footstep_planner_enabled
    assert experiment.settings.knee_hint_enabled
    assert experiment.settings.stance_width_ratio > default.settings.stance_width_ratio
    assert experiment.settings.step_duration_s > default.settings.step_duration_s


def test_basis_research_factory_combines_footstep_and_swivel_arm_settings() -> None:
    experiment = create_estimator("basis_research")

    assert isinstance(experiment, BasisPortEstimator)
    assert experiment.settings.footstep_planner_enabled
    assert experiment.settings.knee_hint_enabled
    assert experiment.settings.arm_bend_strategy == "manikin_swivel"


def test_basis_knee_hint_smooths_across_yaw_changes() -> None:
    estimator = BasisPortEstimator(
        BasisEstimatorSettings(
            knee_hint_enabled=True,
            knee_hint_lerp_speed=1.0,
            support_hips_xz_follow_blend=0.0,
        )
    )
    calibration = AvatarCalibration(human_height_m=2.0, foot_height_m=0.05)
    grounding = GroundingContext()

    estimator.estimate(
        _manual_sample((0.0, 1.6, 0.0), timestamp_seconds=0.0),
        calibration,
        grounding,
    )
    first_hint = estimator._foot_states["left_foot"].filtered_knee_hint
    estimator.estimate(
        _manual_sample(
            (0.0, 1.6, 0.0),
            timestamp_seconds=1.0 / 120.0,
            forward=(1.0, 0.0, 0.0),
        ),
        calibration,
        grounding,
    )
    second_hint = estimator._foot_states["left_foot"].filtered_knee_hint

    assert first_hint is not None
    assert second_hint is not None
    assert dot(first_hint, second_hint) > 0.9


def test_basis_footstep_planner_holds_planted_anchor_below_release_stretch() -> None:
    estimator = BasisPortEstimator(
        BasisEstimatorSettings(
            footstep_planner_enabled=True,
            head_to_hips_ratio=0.3,
            hips_forward_bias_m=0.0,
            support_hips_xz_follow_blend=0.0,
            stance_width_ratio=0.2,
            footstep_release_stretch_m=0.5,
        )
    )
    calibration = AvatarCalibration(human_height_m=2.0, foot_height_m=0.05)
    grounding = GroundingContext()

    first = estimator.estimate(
        _manual_sample((0.0, 1.6, 0.0), timestamp_seconds=0.0),
        calibration,
        grounding,
    )
    second = estimator.estimate(
        _manual_sample((0.1, 1.6, 0.0), timestamp_seconds=0.1),
        calibration,
        grounding,
    )

    assert (
        second.joints["left_foot"].transform.position
        == first.joints["left_foot"].transform.position
    )
    assert estimator._foot_states["left_foot"].phase == "planted"
    assert estimator._active_swing_foot is None


def test_basis_footstep_planner_releases_one_foot_to_forward_step_target() -> None:
    estimator = BasisPortEstimator(
        BasisEstimatorSettings(
            footstep_planner_enabled=True,
            head_to_hips_ratio=0.3,
            hips_forward_bias_m=0.0,
            support_hips_xz_follow_blend=0.0,
            stance_width_ratio=0.2,
            footstep_release_stretch_m=0.1,
            footstep_forward_lead_m=0.2,
            footstep_speed_lead_s=0.0,
            footstep_max_neutral_stretch_m=2.0,
            footstep_min_interval_s=0.0,
            step_duration_s=0.4,
            step_lift_height_m=0.08,
        )
    )
    calibration = AvatarCalibration(human_height_m=2.0, foot_height_m=0.05)
    grounding = GroundingContext()

    first = estimator.estimate(
        _manual_sample((0.0, 1.6, 0.0), timestamp_seconds=0.0),
        calibration,
        grounding,
    )
    estimator.estimate(
        _manual_sample((0.0, 1.6, 0.35), timestamp_seconds=0.1),
        calibration,
        grounding,
    )
    third = estimator.estimate(
        _manual_sample((0.0, 1.6, 0.45), timestamp_seconds=0.2),
        calibration,
        grounding,
    )

    assert estimator._active_swing_foot == "left_foot"
    assert estimator._foot_states["left_foot"].phase == "swing"
    assert estimator._foot_states["right_foot"].phase == "planted"
    assert third.joints["left_foot"].transform.position[1] > calibration.foot_height_m
    assert (
        third.joints["left_foot"].transform.position[2]
        > first.joints["left_foot"].transform.position[2]
    )
    assert (
        third.joints["right_foot"].transform.position
        == first.joints["right_foot"].transform.position
    )


def test_basis_footstep_planner_alternates_and_prevents_crossover() -> None:
    estimator = BasisPortEstimator(
        BasisEstimatorSettings(
            footstep_planner_enabled=True,
            head_to_hips_ratio=0.3,
            hips_forward_bias_m=0.0,
            support_hips_xz_follow_blend=0.0,
            stance_width_ratio=0.2,
            footstep_release_stretch_m=0.1,
            footstep_forward_lead_m=0.2,
            footstep_speed_lead_s=0.0,
            footstep_max_neutral_stretch_m=2.0,
            footstep_min_interval_s=0.0,
            step_duration_s=0.4,
            step_lift_height_m=0.08,
        )
    )
    calibration = AvatarCalibration(human_height_m=2.0, foot_height_m=0.05)
    grounding = GroundingContext()
    forward = (0.0, 0.0, 1.0)
    right = normalize(cross(forward, (0.0, 1.0, 0.0)))

    estimator.estimate(
        _manual_sample((0.0, 1.6, 0.0), timestamp_seconds=0.0),
        calibration,
        grounding,
    )
    estimator.estimate(
        _manual_sample((0.0, 1.6, 0.35), timestamp_seconds=0.1),
        calibration,
        grounding,
    )
    pose = estimator.estimate(
        _manual_sample((0.0, 1.6, 0.65), timestamp_seconds=0.5),
        calibration,
        grounding,
    )

    swinging_feet = [
        foot_name
        for foot_name in ("left_foot", "right_foot")
        if estimator._foot_states[foot_name].phase == "swing"
    ]
    assert swinging_feet == ["right_foot"]

    hips = pose.joints["hips"].transform.position
    left_lateral = dot(subtract(pose.joints["left_foot"].transform.position, hips), right)
    right_lateral = dot(subtract(pose.joints["right_foot"].transform.position, hips), right)

    assert left_lateral < 0.0
    assert right_lateral > 0.0


def test_basis_footstep_planner_bounds_fast_motion_leg_stretch() -> None:
    estimator = BasisPortEstimator(
        BasisEstimatorSettings(
            footstep_planner_enabled=True,
            head_to_hips_ratio=0.3,
            hips_forward_bias_m=0.0,
            support_hips_xz_follow_blend=0.0,
            stance_width_ratio=0.2,
            footstep_release_stretch_m=0.1,
            footstep_forward_lead_m=0.24,
            footstep_speed_lead_s=0.0,
            footstep_max_forward_lead_m=0.24,
            footstep_max_neutral_stretch_m=0.45,
            footstep_min_interval_s=0.0,
            step_duration_s=0.25,
        )
    )
    calibration = AvatarCalibration(human_height_m=2.0, foot_height_m=0.05)
    grounding = GroundingContext()
    max_horizontal_leg_stretch = 0.0

    for frame_index in range(30):
        pose = estimator.estimate(
            _manual_sample(
                (0.0, 1.6, frame_index * 0.12),
                timestamp_seconds=frame_index / 30.0,
            ),
            calibration,
            grounding,
        )
        hips = pose.joints["hips"].transform.position
        max_horizontal_leg_stretch = max(
            max_horizontal_leg_stretch,
            horizontal_distance(hips, pose.joints["left_foot"].transform.position),
            horizontal_distance(hips, pose.joints["right_foot"].transform.position),
        )

    assert max_horizontal_leg_stretch <= 0.52


def test_basis_footstep_planner_aligns_startup_to_current_body() -> None:
    estimator = BasisPortEstimator(
        BasisEstimatorSettings(
            footstep_planner_enabled=True,
            head_to_hips_ratio=0.3,
            hips_forward_bias_m=0.0,
            support_hips_xz_follow_blend=0.0,
            stance_width_ratio=0.2,
            footstep_startup_align_s=0.05,
        )
    )
    calibration = AvatarCalibration(human_height_m=2.0, foot_height_m=0.05)
    grounding = GroundingContext()

    estimator.estimate(
        _manual_sample((0.0, 1.6, 0.0), timestamp_seconds=0.0),
        calibration,
        grounding,
    )
    pose = estimator.estimate(
        _manual_sample((0.0, 1.6, 0.35), timestamp_seconds=0.02),
        calibration,
        grounding,
    )

    hips = pose.joints["hips"].transform.position

    assert estimator._foot_states["left_foot"].phase == "planted"
    assert estimator._foot_states["right_foot"].phase == "planted"
    assert horizontal_distance(hips, pose.joints["left_foot"].transform.position) < 0.25
    assert horizontal_distance(hips, pose.joints["right_foot"].transform.position) < 0.25


def test_basis_footstep_planner_steps_trailing_foot_after_stop() -> None:
    estimator = BasisPortEstimator(
        BasisEstimatorSettings(
            footstep_planner_enabled=True,
            head_to_hips_ratio=0.3,
            hips_forward_bias_m=0.0,
            support_hips_xz_follow_blend=0.0,
            stance_width_ratio=0.2,
            footstep_release_stretch_m=10.0,
            footstep_forward_lead_m=0.0,
            footstep_speed_lead_s=0.0,
            footstep_catchup_release_stretch_m=10.0,
            footstep_trailing_release_stretch_m=0.2,
            footstep_stop_speed_m_s=0.05,
            footstep_max_neutral_stretch_m=2.0,
            footstep_min_interval_s=10.0,
            step_duration_s=0.2,
        )
    )
    calibration = AvatarCalibration(human_height_m=2.0, foot_height_m=0.05)
    grounding = GroundingContext()

    estimator.estimate(
        _manual_sample((0.0, 1.6, 0.0), timestamp_seconds=0.0),
        calibration,
        grounding,
    )
    before_recovery = estimator.estimate(
        _manual_sample((0.0, 1.6, 0.35), timestamp_seconds=0.1),
        calibration,
        grounding,
    )
    before_hips = before_recovery.joints["hips"].transform.position
    before_left_distance = horizontal_distance(
        before_hips,
        before_recovery.joints["left_foot"].transform.position,
    )

    estimator.estimate(
        _manual_sample((0.0, 1.6, 0.35), timestamp_seconds=0.3),
        calibration,
        grounding,
    )
    after_recovery = estimator.estimate(
        _manual_sample((0.0, 1.6, 0.35), timestamp_seconds=0.55),
        calibration,
        grounding,
    )
    after_hips = after_recovery.joints["hips"].transform.position
    after_left_distance = horizontal_distance(
        after_hips,
        after_recovery.joints["left_foot"].transform.position,
    )

    assert after_left_distance < before_left_distance
