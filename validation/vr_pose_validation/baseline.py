"""Bootstrap pose-estimator baselines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from validation.vr_pose_validation.basis_estimator import (
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

type EstimatorName = Literal["trivial", "planted", "basis"]

__all__ = [
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
    raise ValueError(f"Unknown estimator: {name}")
