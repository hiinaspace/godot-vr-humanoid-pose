"""Bootstrap pose-estimator baselines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from validation.vr_pose_validation.basis_estimator import (
    ArmBendStrategy,
    BasisEstimatorSettings,
    BasisPortEstimator,
)
from validation.vr_pose_validation.geometry import Transform, Vector3, distance, with_y
from validation.vr_pose_validation.types import (
    AvatarCalibration,
    GroundingContext,
    HumanoidPoseFrame,
    JointPose,
    PoseEstimator,
    TrackedPoseSample,
)

type EstimatorName = Literal[
    "trivial",
    "planted",
    "basis",
    "basis_knee_hint",
    "basis_footstep",
    "basis_arm_lookup",
    "basis_arm_swivel",
    "basis_shrug",
    "basis_research",
]

__all__ = [
    "ArmBendStrategy",
    "BasisEstimatorSettings",
    "BasisPortEstimator",
    "EstimatorName",
    "PlantedFeetBaselineEstimator",
    "TrivialBaselineEstimator",
    "create_estimator",
]


@dataclass(slots=True)
class TrivialBaselineEstimator:
    """Head-driven torso, tracked hands, and feet projected under the hips."""

    hip_drop_ratio: float = 0.55
    stance_width_m: float = 0.28

    def estimate(
        self,
        sample: TrackedPoseSample,
        calibration: AvatarCalibration,
        grounding: GroundingContext,
    ) -> HumanoidPoseFrame:
        head_position = sample.hmd.transform.position
        hip_y = max(
            grounding.floor_height_m + calibration.foot_height_m,
            head_position[1] - calibration.human_height_m * self.hip_drop_ratio,
        )
        hip_position = (head_position[0], hip_y, head_position[2])
        left_foot = (
            hip_position[0] - self.stance_width_m * 0.5,
            grounding.floor_height_m + calibration.foot_height_m,
            hip_position[2],
        )
        right_foot = (
            hip_position[0] + self.stance_width_m * 0.5,
            grounding.floor_height_m + calibration.foot_height_m,
            hip_position[2],
        )
        joints = {
            "head": JointPose(sample.hmd.transform, sample.hmd.confidence, estimated=False),
            "left_hand": JointPose(
                sample.left_hand.transform,
                sample.left_hand.confidence,
                estimated=False,
            ),
            "right_hand": JointPose(
                sample.right_hand.transform,
                sample.right_hand.confidence,
                estimated=False,
            ),
            "hips": JointPose(Transform(position=hip_position), estimated=True),
            "left_foot": JointPose(Transform(position=left_foot), estimated=True),
            "right_foot": JointPose(Transform(position=right_foot), estimated=True),
        }
        return HumanoidPoseFrame(
            timestamp_seconds=sample.timestamp_seconds,
            root=Transform(position=hip_position),
            joints=joints,
        )


@dataclass(slots=True)
class PlantedFeetBaselineEstimator(TrivialBaselineEstimator):
    """Simple sticky-feet variant for exercising foot-slip metrics."""

    step_threshold_m: float = 0.35
    _anchors: dict[str, Vector3] = field(default_factory=dict, init=False)

    def estimate(
        self,
        sample: TrackedPoseSample,
        calibration: AvatarCalibration,
        grounding: GroundingContext,
    ) -> HumanoidPoseFrame:
        frame = TrivialBaselineEstimator.estimate(self, sample, calibration, grounding)
        hips = frame.joints["hips"].transform.position
        updated_joints = dict(frame.joints)
        for foot_name in ("left_foot", "right_foot"):
            default_position = updated_joints[foot_name].transform.position
            anchor = self._anchors.get(foot_name, default_position)
            if distance(with_y(hips, anchor[1]), anchor) > self.step_threshold_m:
                anchor = default_position
            self._anchors[foot_name] = anchor
            updated_joints[foot_name] = JointPose(
                transform=Transform(position=anchor),
                confidence=0.75,
                estimated=True,
            )
        return HumanoidPoseFrame(
            timestamp_seconds=frame.timestamp_seconds,
            root=frame.root,
            joints=updated_joints,
        )


def create_estimator(name: EstimatorName) -> PoseEstimator:
    if name == "trivial":
        return TrivialBaselineEstimator()
    if name == "planted":
        return PlantedFeetBaselineEstimator()
    if name == "basis":
        return BasisPortEstimator()
    if name == "basis_knee_hint":
        return BasisPortEstimator(BasisEstimatorSettings(knee_hint_enabled=True))
    if name == "basis_arm_lookup":
        return BasisPortEstimator(BasisEstimatorSettings(arm_bend_strategy="basis_lookup"))
    if name == "basis_arm_swivel":
        return BasisPortEstimator(BasisEstimatorSettings(arm_bend_strategy="manikin_swivel"))
    if name == "basis_shrug":
        return BasisPortEstimator(_shoulder_shrug_settings())
    if name == "basis_footstep":
        return BasisPortEstimator(_footstep_settings())
    if name == "basis_research":
        return BasisPortEstimator(
            _footstep_settings(
                arm_bend_strategy="manikin_swivel",
            )
        )
    raise ValueError(f"Unknown estimator: {name}")


def _shoulder_shrug_settings() -> BasisEstimatorSettings:
    return BasisEstimatorSettings(
        shoulder_shrug_enabled=True,
        shoulder_shrug_start_ratio=0.26,
        shoulder_shrug_full_ratio=0.78,
        shoulder_shrug_min_reach_ratio=0.25,
        shoulder_shrug_full_reach_ratio=0.86,
        shoulder_shrug_abduction_start_ratio=0.24,
        shoulder_shrug_abduction_full_ratio=0.78,
        shoulder_shrug_max_elevation_m=0.08,
        shoulder_shrug_forward_gain_m=0.05,
    )


def _footstep_settings(
    *,
    arm_bend_strategy: ArmBendStrategy = "default",
) -> BasisEstimatorSettings:
    return BasisEstimatorSettings(
        footstep_planner_enabled=True,
        knee_hint_enabled=True,
        arm_bend_strategy=arm_bend_strategy,
        stance_width_ratio=0.22,
        support_hips_xz_follow_blend=0.0,
        max_anchor_stretch_m=0.14,
        footstep_release_stretch_m=0.24,
        footstep_speed_release_reduction_s=0.04,
        footstep_forward_lead_m=0.18,
        footstep_speed_lead_s=0.08,
        footstep_max_forward_lead_m=0.40,
        footstep_max_step_length_m=0.85,
        footstep_swing_retarget_stretch_m=0.20,
        footstep_max_neutral_stretch_m=0.60,
        footstep_min_lateral_ratio=0.45,
        footstep_min_interval_s=0.14,
        footstep_startup_align_s=0.0,
        footstep_catchup_release_stretch_m=0.58,
        footstep_trailing_release_stretch_m=0.34,
        footstep_stop_speed_m_s=0.12,
        footstep_catchup_duration_s=0.20,
        footstep_speed_duration_reduction_s=0.05,
        plant_velocity_threshold_m_s=100.0,
        step_duration_s=0.28,
        step_lift_height_m=0.07,
    )
