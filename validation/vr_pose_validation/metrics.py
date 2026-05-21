"""Validation metrics for sparse-tracker baselines."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from statistics import fmean

from validation.vr_pose_validation.geometry import Transform, Vector3, distance, horizontal_distance
from validation.vr_pose_validation.types import (
    HumanoidPoseFrame,
    TrackedPoseSample,
    TrackerState,
)

FOOT_JOINTS = ("left_foot", "right_foot")


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
    pelvis_horizontal_error_m: float
    pelvis_vertical_error_m: float
    chest_error_m: float | None
    spine_error_m: float | None
    left_foot_error_m: float
    right_foot_error_m: float
    joint_errors_m: Mapping[str, float] = field(default_factory=dict)

    @property
    def mean_error_m(self) -> float:
        return fmean((self.pelvis_error_m, self.left_foot_error_m, self.right_foot_error_m))

    @property
    def mean_torso_error_m(self) -> float:
        values = [
            error
            for error in (self.pelvis_error_m, self.chest_error_m, self.spine_error_m)
            if error is not None
        ]
        return fmean(values) if values else 0.0


@dataclass(frozen=True, slots=True)
class FootContactSequenceMetrics:
    inferred_floor_height_m: float
    left_contact_frames: int
    right_contact_frames: int
    left_contact_duration_s: float
    right_contact_duration_s: float
    left_contact_slip_m: float
    right_contact_slip_m: float
    mean_left_contact_foot_error_m: float
    mean_right_contact_foot_error_m: float

    @property
    def contact_frames(self) -> int:
        return self.left_contact_frames + self.right_contact_frames

    @property
    def left_contact_slip_m_per_s(self) -> float:
        return _rate(self.left_contact_slip_m, self.left_contact_duration_s)

    @property
    def right_contact_slip_m_per_s(self) -> float:
        return _rate(self.right_contact_slip_m, self.right_contact_duration_s)

    @property
    def mean_contact_slip_m_per_s(self) -> float:
        total_slip = self.left_contact_slip_m + self.right_contact_slip_m
        total_duration = self.left_contact_duration_s + self.right_contact_duration_s
        return _rate(total_slip, total_duration)

    @property
    def mean_contact_foot_error_m(self) -> float:
        weighted_error = (
            self.mean_left_contact_foot_error_m * self.left_contact_frames
            + self.mean_right_contact_foot_error_m * self.right_contact_frames
        )
        return _rate(weighted_error, float(self.contact_frames))


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
    joint_errors = {
        joint_name: _pose_to_reference_error(pose, reference_joints, joint_name)
        for joint_name in sorted(set(pose.joints).intersection(reference_joints))
    }
    pelvis_error = _pose_to_reference_error(pose, reference_joints, "hips")
    return ReferencePoseMetrics(
        pelvis_error_m=pelvis_error,
        pelvis_horizontal_error_m=_pose_to_reference_horizontal_error(
            pose,
            reference_joints,
            "hips",
        ),
        pelvis_vertical_error_m=_pose_to_reference_vertical_error(
            pose,
            reference_joints,
            "hips",
        ),
        chest_error_m=_optional_pose_to_reference_error(pose, reference_joints, "chest"),
        spine_error_m=_optional_pose_to_reference_error(pose, reference_joints, "spine"),
        left_foot_error_m=_pose_to_reference_error(pose, reference_joints, "left_foot"),
        right_foot_error_m=_pose_to_reference_error(pose, reference_joints, "right_foot"),
        joint_errors_m=joint_errors,
    )


def summarize_reference_errors(metrics: list[ReferencePoseMetrics]) -> dict[str, object]:
    if not metrics:
        return {
            "mean_pelvis_error_m": 0.0,
            "mean_pelvis_horizontal_error_m": 0.0,
            "mean_pelvis_vertical_error_m": 0.0,
            "mean_chest_error_m": 0.0,
            "mean_spine_error_m": 0.0,
            "mean_torso_error_m": 0.0,
            "mean_left_foot_error_m": 0.0,
            "mean_right_foot_error_m": 0.0,
            "mean_reference_error_m": 0.0,
            "mean_joint_errors_m": {},
        }
    joint_names = sorted({name for metric in metrics for name in metric.joint_errors_m})
    return {
        "mean_pelvis_error_m": fmean(metric.pelvis_error_m for metric in metrics),
        "mean_pelvis_horizontal_error_m": fmean(
            metric.pelvis_horizontal_error_m for metric in metrics
        ),
        "mean_pelvis_vertical_error_m": fmean(metric.pelvis_vertical_error_m for metric in metrics),
        "mean_chest_error_m": _mean_optional(metric.chest_error_m for metric in metrics),
        "mean_spine_error_m": _mean_optional(metric.spine_error_m for metric in metrics),
        "mean_torso_error_m": fmean(metric.mean_torso_error_m for metric in metrics),
        "mean_left_foot_error_m": fmean(metric.left_foot_error_m for metric in metrics),
        "mean_right_foot_error_m": fmean(metric.right_foot_error_m for metric in metrics),
        "mean_reference_error_m": fmean(metric.mean_error_m for metric in metrics),
        "mean_joint_errors_m": {
            name: fmean(
                metric.joint_errors_m[name] for metric in metrics if name in metric.joint_errors_m
            )
            for name in joint_names
        },
    }


def infer_reference_floor_height(
    reference_frames: Iterable[dict[str, Transform]],
    *,
    quantile: float = 0.05,
) -> float:
    heights = sorted(
        transform.position[1]
        for frame in reference_frames
        for joint_name, transform in frame.items()
        if joint_name in FOOT_JOINTS
    )
    if not heights:
        return 0.0
    index = min(len(heights) - 1, max(0, int(len(heights) * quantile)))
    return heights[index]


def evaluate_foot_contact_sequence(
    poses: Iterable[HumanoidPoseFrame],
    reference_frames: Iterable[dict[str, Transform]],
    *,
    floor_height_m: float | None = None,
    contact_height_threshold_m: float = 0.08,
    contact_speed_threshold_m_s: float = 0.18,
) -> FootContactSequenceMetrics:
    pose_list = list(poses)
    reference_list = list(reference_frames)
    inferred_floor = (
        infer_reference_floor_height(reference_list) if floor_height_m is None else floor_height_m
    )
    accumulators = {
        "left_foot": _FootContactAccumulator(),
        "right_foot": _FootContactAccumulator(),
    }
    for pose, reference in zip(pose_list, reference_list, strict=False):
        for foot_name, accumulator in accumulators.items():
            if foot_name not in pose.joints or foot_name not in reference:
                continue
            accumulator.add_frame(
                timestamp_seconds=pose.timestamp_seconds,
                estimated_position=pose.joints[foot_name].transform.position,
                reference_position=reference[foot_name].position,
                floor_height_m=inferred_floor,
                contact_height_threshold_m=contact_height_threshold_m,
                contact_speed_threshold_m_s=contact_speed_threshold_m_s,
            )

    left = accumulators["left_foot"].summary()
    right = accumulators["right_foot"].summary()
    return FootContactSequenceMetrics(
        inferred_floor_height_m=inferred_floor,
        left_contact_frames=left.contact_frames,
        right_contact_frames=right.contact_frames,
        left_contact_duration_s=left.contact_duration_s,
        right_contact_duration_s=right.contact_duration_s,
        left_contact_slip_m=left.contact_slip_m,
        right_contact_slip_m=right.contact_slip_m,
        mean_left_contact_foot_error_m=left.mean_contact_foot_error_m,
        mean_right_contact_foot_error_m=right.mean_contact_foot_error_m,
    )


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
    if joint_name not in pose.joints or joint_name not in reference_joints:
        return 0.0
    reference = reference_joints[joint_name]
    return distance(pose.joints[joint_name].transform.position, reference.position)


def _optional_pose_to_reference_error(
    pose: HumanoidPoseFrame,
    reference_joints: dict[str, Transform],
    joint_name: str,
) -> float | None:
    if joint_name not in pose.joints or joint_name not in reference_joints:
        return None
    return _pose_to_reference_error(pose, reference_joints, joint_name)


def _pose_to_reference_horizontal_error(
    pose: HumanoidPoseFrame,
    reference_joints: dict[str, Transform],
    joint_name: str,
) -> float:
    if joint_name not in pose.joints or joint_name not in reference_joints:
        return 0.0
    return horizontal_distance(
        pose.joints[joint_name].transform.position,
        reference_joints[joint_name].position,
    )


def _pose_to_reference_vertical_error(
    pose: HumanoidPoseFrame,
    reference_joints: dict[str, Transform],
    joint_name: str,
) -> float:
    if joint_name not in pose.joints or joint_name not in reference_joints:
        return 0.0
    return abs(
        pose.joints[joint_name].transform.position[1] - reference_joints[joint_name].position[1]
    )


@dataclass(frozen=True, slots=True)
class _FootContactAccumulatorSummary:
    contact_frames: int
    contact_duration_s: float
    contact_slip_m: float
    mean_contact_foot_error_m: float


@dataclass(slots=True)
class _FootContactAccumulator:
    previous_timestamp_seconds: float | None = None
    previous_reference_position: Vector3 | None = None
    previous_estimated_position: Vector3 | None = None
    previous_contact: bool = False
    contact_frames: int = 0
    contact_duration_s: float = 0.0
    contact_slip_m: float = 0.0
    contact_foot_error_sum_m: float = 0.0

    def add_frame(
        self,
        *,
        timestamp_seconds: float,
        estimated_position: Vector3,
        reference_position: Vector3,
        floor_height_m: float,
        contact_height_threshold_m: float,
        contact_speed_threshold_m_s: float,
    ) -> None:
        dt = (
            max(timestamp_seconds - self.previous_timestamp_seconds, 1.0 / 120.0)
            if self.previous_timestamp_seconds is not None
            else 1.0 / 120.0
        )
        reference_horizontal_speed = (
            horizontal_distance(reference_position, self.previous_reference_position) / dt
            if self.previous_reference_position is not None
            else 0.0
        )
        reference_vertical_speed = (
            abs(reference_position[1] - self.previous_reference_position[1]) / dt
            if self.previous_reference_position is not None
            else 0.0
        )
        contact = (
            reference_position[1] - floor_height_m <= contact_height_threshold_m
            and reference_horizontal_speed <= contact_speed_threshold_m_s
            and reference_vertical_speed <= contact_speed_threshold_m_s
        )
        if contact:
            self.contact_frames += 1
            self.contact_foot_error_sum_m += distance(estimated_position, reference_position)
            if self.previous_contact and self.previous_estimated_position is not None:
                self.contact_slip_m += horizontal_distance(
                    estimated_position,
                    self.previous_estimated_position,
                )
                self.contact_duration_s += dt

        self.previous_timestamp_seconds = timestamp_seconds
        self.previous_reference_position = reference_position
        self.previous_estimated_position = estimated_position
        self.previous_contact = contact

    def summary(self) -> _FootContactAccumulatorSummary:
        return _FootContactAccumulatorSummary(
            contact_frames=self.contact_frames,
            contact_duration_s=self.contact_duration_s,
            contact_slip_m=self.contact_slip_m,
            mean_contact_foot_error_m=_rate(
                self.contact_foot_error_sum_m,
                float(self.contact_frames),
            ),
        )


def _rate(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0.0 else 0.0


def _mean_optional(values: Iterable[float | None]) -> float:
    value_list = [value for value in values if value is not None]
    return fmean(value_list) if value_list else 0.0
