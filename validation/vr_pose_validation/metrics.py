"""Validation metrics for sparse-tracker baselines."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean

from validation.vr_pose_validation.geometry import Transform, distance
from validation.vr_pose_validation.types import (
    HumanoidPoseFrame,
    TrackedPoseSample,
    TrackerState,
)


@dataclass(frozen=True, slots=True)
class TargetErrorMetrics:
    head_error_m: float
    left_hand_error_m: float
    right_hand_error_m: float

    @property
    def mean_error_m(self) -> float:
        return fmean((self.head_error_m, self.left_hand_error_m, self.right_hand_error_m))


@dataclass(frozen=True, slots=True)
class FootGroundingMetrics:
    left_penetration_m: float
    right_penetration_m: float
    left_hover_m: float
    right_hover_m: float


@dataclass(frozen=True, slots=True)
class FrameMetrics:
    timestamp_seconds: float
    target_errors: TargetErrorMetrics
    foot_grounding: FootGroundingMetrics


@dataclass(frozen=True, slots=True)
class ReferencePoseMetrics:
    pelvis_error_m: float
    left_foot_error_m: float
    right_foot_error_m: float

    @property
    def mean_error_m(self) -> float:
        return fmean((self.pelvis_error_m, self.left_foot_error_m, self.right_foot_error_m))


def evaluate_frame(
    pose: HumanoidPoseFrame,
    sample: TrackedPoseSample,
    floor_height_m: float = 0.0,
) -> FrameMetrics:
    return FrameMetrics(
        timestamp_seconds=pose.timestamp_seconds,
        target_errors=TargetErrorMetrics(
            head_error_m=_joint_to_tracker_error(pose, "head", sample.hmd),
            left_hand_error_m=_joint_to_tracker_error(pose, "left_hand", sample.left_hand),
            right_hand_error_m=_joint_to_tracker_error(pose, "right_hand", sample.right_hand),
        ),
        foot_grounding=FootGroundingMetrics(
            left_penetration_m=_penetration(pose, "left_foot", floor_height_m),
            right_penetration_m=_penetration(pose, "right_foot", floor_height_m),
            left_hover_m=_hover(pose, "left_foot", floor_height_m),
            right_hover_m=_hover(pose, "right_foot", floor_height_m),
        ),
    )


def summarize_target_errors(metrics: list[FrameMetrics]) -> dict[str, float]:
    if not metrics:
        return {
            "mean_head_error_m": 0.0,
            "mean_left_hand_error_m": 0.0,
            "mean_right_hand_error_m": 0.0,
            "mean_target_error_m": 0.0,
        }
    return {
        "mean_head_error_m": fmean(metric.target_errors.head_error_m for metric in metrics),
        "mean_left_hand_error_m": fmean(
            metric.target_errors.left_hand_error_m for metric in metrics
        ),
        "mean_right_hand_error_m": fmean(
            metric.target_errors.right_hand_error_m for metric in metrics
        ),
        "mean_target_error_m": fmean(metric.target_errors.mean_error_m for metric in metrics),
    }


def evaluate_reference_pose(
    pose: HumanoidPoseFrame,
    reference_joints: dict[str, Transform],
) -> ReferencePoseMetrics:
    return ReferencePoseMetrics(
        pelvis_error_m=_pose_to_reference_error(pose, reference_joints, "hips"),
        left_foot_error_m=_pose_to_reference_error(pose, reference_joints, "left_foot"),
        right_foot_error_m=_pose_to_reference_error(pose, reference_joints, "right_foot"),
    )


def summarize_reference_errors(metrics: list[ReferencePoseMetrics]) -> dict[str, float]:
    if not metrics:
        return {
            "mean_pelvis_error_m": 0.0,
            "mean_left_foot_error_m": 0.0,
            "mean_right_foot_error_m": 0.0,
            "mean_reference_error_m": 0.0,
        }
    return {
        "mean_pelvis_error_m": fmean(metric.pelvis_error_m for metric in metrics),
        "mean_left_foot_error_m": fmean(metric.left_foot_error_m for metric in metrics),
        "mean_right_foot_error_m": fmean(metric.right_foot_error_m for metric in metrics),
        "mean_reference_error_m": fmean(metric.mean_error_m for metric in metrics),
    }


def _joint_to_tracker_error(
    pose: HumanoidPoseFrame,
    joint_name: str,
    tracker: TrackerState,
) -> float:
    return distance(pose.joints[joint_name].transform.position, tracker.transform.position)


def _penetration(pose: HumanoidPoseFrame, joint_name: str, floor_height_m: float) -> float:
    y = pose.joints[joint_name].transform.position[1]
    return max(0.0, floor_height_m - y)


def _hover(pose: HumanoidPoseFrame, joint_name: str, floor_height_m: float) -> float:
    y = pose.joints[joint_name].transform.position[1]
    return max(0.0, y - floor_height_m)


def _pose_to_reference_error(
    pose: HumanoidPoseFrame,
    reference_joints: dict[str, Transform],
    joint_name: str,
) -> float:
    reference = reference_joints[joint_name]
    return distance(pose.joints[joint_name].transform.position, reference.position)
