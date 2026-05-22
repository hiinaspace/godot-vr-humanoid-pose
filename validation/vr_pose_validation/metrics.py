"""Validation metrics for sparse-tracker baselines."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from math import acos, degrees, floor
from statistics import fmean

from validation.vr_pose_validation.geometry import (
    Transform,
    Vector3,
    distance,
    dot,
    horizontal_distance,
    length,
    scale,
    subtract,
)
from validation.vr_pose_validation.types import (
    HumanoidPoseFrame,
    TrackedPoseSample,
    TrackerState,
)

FOOT_JOINTS = ("left_foot", "right_foot")
JOINT_PAIR_METRICS = {
    "shoulder": ("left_shoulder", "right_shoulder"),
    "elbow": ("left_elbow", "right_elbow"),
    "knee": ("left_knee", "right_knee"),
}
REFERENCE_ERROR_DISTRIBUTION_FIELDS = (
    "reference_error_p50_m",
    "reference_error_p90_m",
    "reference_error_p95_m",
    "reference_error_p99_m",
    "pct_reference_error_over_25cm",
    "pct_reference_error_over_50cm",
    "pct_reference_error_over_100cm",
    "pelvis_error_p95_m",
    "torso_error_p95_m",
    "shoulder_error_p95_m",
    "elbow_error_p95_m",
    "knee_error_p95_m",
    "feet_error_p95_m",
)
TEMPORAL_REGION_JOINTS = {
    "joint": (),
    "pelvis": ("hips",),
    "torso": ("hips", "chest", "spine"),
    "shoulder": ("left_shoulder", "right_shoulder"),
    "elbow": ("left_elbow", "right_elbow"),
    "knee": ("left_knee", "right_knee"),
    "feet": FOOT_JOINTS,
}
TEMPORAL_SUMMARY_FIELDS = tuple(
    field_name
    for region in TEMPORAL_REGION_JOINTS
    for field_name in (
        f"mean_{region}_velocity_error_m_per_s",
        f"mean_{region}_acceleration_error_m_per_s2",
        f"mean_{region}_estimated_jerk_m_per_s3",
    )
)
FLEXION_JOINTS = {
    "left_elbow": ("left_shoulder", "left_elbow", "left_hand"),
    "right_elbow": ("right_shoulder", "right_elbow", "right_hand"),
    "left_knee": ("hips", "left_knee", "left_foot"),
    "right_knee": ("hips", "right_knee", "right_foot"),
}
FLEXION_GROUPS = {
    "elbow": ("left_elbow", "right_elbow"),
    "knee": ("left_knee", "right_knee"),
    "joint": tuple(FLEXION_JOINTS),
}
DEFAULT_FLEXION_LIMITS_DEG = {
    "left_elbow": 150.0,
    "right_elbow": 150.0,
    "left_knee": 135.0,
    "right_knee": 135.0,
}
FLEXION_SUMMARY_FIELDS = (
    "mean_left_elbow_flexion_error_deg",
    "mean_right_elbow_flexion_error_deg",
    "mean_elbow_flexion_error_deg",
    "mean_left_knee_flexion_error_deg",
    "mean_right_knee_flexion_error_deg",
    "mean_knee_flexion_error_deg",
    "mean_joint_flexion_error_deg",
    "pct_left_elbow_flexion_over_limit",
    "pct_right_elbow_flexion_over_limit",
    "pct_elbow_flexion_over_limit",
    "pct_left_knee_flexion_over_limit",
    "pct_right_knee_flexion_over_limit",
    "pct_knee_flexion_over_limit",
    "pct_joint_flexion_over_limit",
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
            "mean_left_shoulder_error_m": 0.0,
            "mean_right_shoulder_error_m": 0.0,
            "mean_shoulder_error_m": 0.0,
            "mean_left_elbow_error_m": 0.0,
            "mean_right_elbow_error_m": 0.0,
            "mean_elbow_error_m": 0.0,
            "mean_left_knee_error_m": 0.0,
            "mean_right_knee_error_m": 0.0,
            "mean_knee_error_m": 0.0,
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
        "mean_left_shoulder_error_m": _mean_joint_error(metrics, "left_shoulder"),
        "mean_right_shoulder_error_m": _mean_joint_error(metrics, "right_shoulder"),
        "mean_shoulder_error_m": _mean_joint_pair_error(metrics, "shoulder"),
        "mean_left_elbow_error_m": _mean_joint_error(metrics, "left_elbow"),
        "mean_right_elbow_error_m": _mean_joint_error(metrics, "right_elbow"),
        "mean_elbow_error_m": _mean_joint_pair_error(metrics, "elbow"),
        "mean_left_knee_error_m": _mean_joint_error(metrics, "left_knee"),
        "mean_right_knee_error_m": _mean_joint_error(metrics, "right_knee"),
        "mean_knee_error_m": _mean_joint_pair_error(metrics, "knee"),
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


def summarize_reference_error_distribution(
    metrics: list[ReferencePoseMetrics],
) -> dict[str, float]:
    if not metrics:
        return {field_name: 0.0 for field_name in REFERENCE_ERROR_DISTRIBUTION_FIELDS}

    reference_errors = [metric.mean_error_m for metric in metrics]
    shoulder_errors = _joint_pair_frame_errors(metrics, "shoulder")
    elbow_errors = _joint_pair_frame_errors(metrics, "elbow")
    knee_errors = _joint_pair_frame_errors(metrics, "knee")
    feet_errors = [
        fmean((metric.left_foot_error_m, metric.right_foot_error_m)) for metric in metrics
    ]
    return {
        "reference_error_p50_m": _quantile(reference_errors, 0.50),
        "reference_error_p90_m": _quantile(reference_errors, 0.90),
        "reference_error_p95_m": _quantile(reference_errors, 0.95),
        "reference_error_p99_m": _quantile(reference_errors, 0.99),
        "pct_reference_error_over_25cm": _percent_over(reference_errors, 0.25),
        "pct_reference_error_over_50cm": _percent_over(reference_errors, 0.50),
        "pct_reference_error_over_100cm": _percent_over(reference_errors, 1.00),
        "pelvis_error_p95_m": _quantile(
            [metric.pelvis_error_m for metric in metrics],
            0.95,
        ),
        "torso_error_p95_m": _quantile(
            [metric.mean_torso_error_m for metric in metrics],
            0.95,
        ),
        "shoulder_error_p95_m": _quantile(shoulder_errors, 0.95),
        "elbow_error_p95_m": _quantile(elbow_errors, 0.95),
        "knee_error_p95_m": _quantile(knee_errors, 0.95),
        "feet_error_p95_m": _quantile(feet_errors, 0.95),
    }


def summarize_temporal_reference_sequence(
    poses: Iterable[HumanoidPoseFrame],
    reference_frames: Iterable[dict[str, Transform]],
) -> dict[str, float]:
    pose_list = list(poses)
    reference_list = list(reference_frames)
    frame_count = min(len(pose_list), len(reference_list))
    velocity_errors = _empty_temporal_buckets()
    acceleration_errors = _empty_temporal_buckets()
    estimated_jerks = _empty_temporal_buckets()
    velocities_by_joint: dict[str, list[_TemporalDerivativeSample]] = {}
    accelerations_by_joint: dict[str, list[_TemporalDerivativeSample]] = {}

    for frame_index in range(1, frame_count):
        previous_pose = pose_list[frame_index - 1]
        pose = pose_list[frame_index]
        previous_reference = reference_list[frame_index - 1]
        reference = reference_list[frame_index]
        dt = _frame_dt(previous_pose.timestamp_seconds, pose.timestamp_seconds)
        joint_names = sorted(
            set(previous_pose.joints)
            .intersection(pose.joints)
            .intersection(previous_reference)
            .intersection(reference)
        )
        for joint_name in joint_names:
            estimated_velocity = _finite_difference(
                pose.joints[joint_name].transform.position,
                previous_pose.joints[joint_name].transform.position,
                dt,
            )
            reference_velocity = _finite_difference(
                reference[joint_name].position,
                previous_reference[joint_name].position,
                dt,
            )
            _add_temporal_value(
                velocity_errors,
                joint_name,
                distance(estimated_velocity, reference_velocity),
            )
            velocities_by_joint.setdefault(joint_name, []).append(
                _TemporalDerivativeSample(
                    timestamp_seconds=pose.timestamp_seconds,
                    estimated=estimated_velocity,
                    reference=reference_velocity,
                )
            )

    for joint_name, velocities in velocities_by_joint.items():
        for previous, current in zip(velocities, velocities[1:], strict=False):
            dt = _frame_dt(previous.timestamp_seconds, current.timestamp_seconds)
            estimated_acceleration = _finite_difference(current.estimated, previous.estimated, dt)
            reference_acceleration = _finite_difference(current.reference, previous.reference, dt)
            _add_temporal_value(
                acceleration_errors,
                joint_name,
                distance(estimated_acceleration, reference_acceleration),
            )
            accelerations_by_joint.setdefault(joint_name, []).append(
                _TemporalDerivativeSample(
                    timestamp_seconds=current.timestamp_seconds,
                    estimated=estimated_acceleration,
                    reference=reference_acceleration,
                )
            )

    for joint_name, accelerations in accelerations_by_joint.items():
        for previous, current in zip(accelerations, accelerations[1:], strict=False):
            dt = _frame_dt(previous.timestamp_seconds, current.timestamp_seconds)
            estimated_jerk = _finite_difference(current.estimated, previous.estimated, dt)
            _add_temporal_value(estimated_jerks, joint_name, length(estimated_jerk))

    return {
        field_name: value
        for region in TEMPORAL_REGION_JOINTS
        for field_name, value in (
            (
                f"mean_{region}_velocity_error_m_per_s",
                _mean_values(velocity_errors[region]),
            ),
            (
                f"mean_{region}_acceleration_error_m_per_s2",
                _mean_values(acceleration_errors[region]),
            ),
            (
                f"mean_{region}_estimated_jerk_m_per_s3",
                _mean_values(estimated_jerks[region]),
            ),
        )
    }


def summarize_joint_flexion_sequence(
    poses: Iterable[HumanoidPoseFrame],
    reference_frames: Iterable[dict[str, Transform]],
    *,
    flexion_limits_deg: Mapping[str, float] = DEFAULT_FLEXION_LIMITS_DEG,
) -> dict[str, float]:
    """Report position-derived elbow/knee flexion errors and coarse limit violations.

    The canonical validation stream does not yet guarantee biomechanical
    joint-local rotations. This derives a conservative flexion angle from three
    joint positions instead, where a straight limb is 0 degrees and larger
    values mean more bend.
    """

    angle_errors: dict[str, list[float]] = {joint_name: [] for joint_name in FLEXION_JOINTS}
    over_limit: dict[str, list[float]] = {joint_name: [] for joint_name in FLEXION_JOINTS}
    for pose, reference in zip(poses, reference_frames, strict=False):
        for joint_name, joint_chain in FLEXION_JOINTS.items():
            estimated_angle = _pose_flexion_angle_deg(pose, joint_chain)
            reference_angle = _reference_flexion_angle_deg(reference, joint_chain)
            if estimated_angle is None or reference_angle is None:
                continue
            angle_errors[joint_name].append(abs(estimated_angle - reference_angle))
            limit = flexion_limits_deg.get(joint_name)
            if limit is not None:
                over_limit[joint_name].append(1.0 if estimated_angle > limit else 0.0)

    return {
        "mean_left_elbow_flexion_error_deg": _mean_values(angle_errors["left_elbow"]),
        "mean_right_elbow_flexion_error_deg": _mean_values(angle_errors["right_elbow"]),
        "mean_elbow_flexion_error_deg": _mean_flexion_group(angle_errors, "elbow"),
        "mean_left_knee_flexion_error_deg": _mean_values(angle_errors["left_knee"]),
        "mean_right_knee_flexion_error_deg": _mean_values(angle_errors["right_knee"]),
        "mean_knee_flexion_error_deg": _mean_flexion_group(angle_errors, "knee"),
        "mean_joint_flexion_error_deg": _mean_flexion_group(angle_errors, "joint"),
        "pct_left_elbow_flexion_over_limit": _percent_flags(over_limit["left_elbow"]),
        "pct_right_elbow_flexion_over_limit": _percent_flags(over_limit["right_elbow"]),
        "pct_elbow_flexion_over_limit": _percent_flexion_group(over_limit, "elbow"),
        "pct_left_knee_flexion_over_limit": _percent_flags(over_limit["left_knee"]),
        "pct_right_knee_flexion_over_limit": _percent_flags(over_limit["right_knee"]),
        "pct_knee_flexion_over_limit": _percent_flexion_group(over_limit, "knee"),
        "pct_joint_flexion_over_limit": _percent_flexion_group(over_limit, "joint"),
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


@dataclass(frozen=True, slots=True)
class _TemporalDerivativeSample:
    timestamp_seconds: float
    estimated: Vector3
    reference: Vector3


def _rate(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0.0 else 0.0


def _mean_optional(values: Iterable[float | None]) -> float:
    value_list = [value for value in values if value is not None]
    return fmean(value_list) if value_list else 0.0


def _mean_joint_error(metrics: list[ReferencePoseMetrics], joint_name: str) -> float:
    return _mean_optional(metric.joint_errors_m.get(joint_name) for metric in metrics)


def _mean_joint_pair_error(metrics: list[ReferencePoseMetrics], pair_name: str) -> float:
    left_name, right_name = JOINT_PAIR_METRICS[pair_name]
    return _mean_optional(
        error
        for metric in metrics
        for error in (
            metric.joint_errors_m.get(left_name),
            metric.joint_errors_m.get(right_name),
        )
    )


def _joint_pair_frame_errors(metrics: list[ReferencePoseMetrics], pair_name: str) -> list[float]:
    left_name, right_name = JOINT_PAIR_METRICS[pair_name]
    values = []
    for metric in metrics:
        frame_values = [
            error
            for error in (
                metric.joint_errors_m.get(left_name),
                metric.joint_errors_m.get(right_name),
            )
            if error is not None
        ]
        if frame_values:
            values.append(fmean(frame_values))
    return values


def _quantile(values: Iterable[float], quantile: float) -> float:
    sorted_values = sorted(values)
    if not sorted_values:
        return 0.0
    clamped_quantile = min(1.0, max(0.0, quantile))
    position = (len(sorted_values) - 1) * clamped_quantile
    lower_index = floor(position)
    upper_index = min(len(sorted_values) - 1, lower_index + 1)
    upper_weight = position - lower_index
    lower_weight = 1.0 - upper_weight
    return sorted_values[lower_index] * lower_weight + sorted_values[upper_index] * upper_weight


def _percent_over(values: Iterable[float], threshold: float) -> float:
    value_list = list(values)
    if not value_list:
        return 0.0
    return sum(1 for value in value_list if value > threshold) / len(value_list) * 100.0


def _empty_temporal_buckets() -> dict[str, list[float]]:
    return {region: [] for region in TEMPORAL_REGION_JOINTS}


def _add_temporal_value(
    buckets: dict[str, list[float]],
    joint_name: str,
    value: float,
) -> None:
    for region in _temporal_regions_for_joint(joint_name):
        buckets[region].append(value)


def _temporal_regions_for_joint(joint_name: str) -> tuple[str, ...]:
    return tuple(
        region
        for region, joint_names in TEMPORAL_REGION_JOINTS.items()
        if region == "joint" or joint_name in joint_names
    )


def _finite_difference(current: Vector3, previous: Vector3, dt: float) -> Vector3:
    return scale(subtract(current, previous), 1.0 / dt)


def _pose_flexion_angle_deg(
    pose: HumanoidPoseFrame,
    joint_chain: tuple[str, str, str],
) -> float | None:
    if any(joint_name not in pose.joints for joint_name in joint_chain):
        return None
    proximal, middle, distal = (
        pose.joints[joint_name].transform.position for joint_name in joint_chain
    )
    return _flexion_angle_deg(proximal, middle, distal)


def _reference_flexion_angle_deg(
    reference: dict[str, Transform],
    joint_chain: tuple[str, str, str],
) -> float | None:
    if any(joint_name not in reference for joint_name in joint_chain):
        return None
    proximal, middle, distal = (reference[joint_name].position for joint_name in joint_chain)
    return _flexion_angle_deg(proximal, middle, distal)


def _flexion_angle_deg(proximal: Vector3, middle: Vector3, distal: Vector3) -> float | None:
    proximal_vector = subtract(proximal, middle)
    distal_vector = subtract(distal, middle)
    proximal_length = length(proximal_vector)
    distal_length = length(distal_vector)
    if proximal_length <= 1e-9 or distal_length <= 1e-9:
        return None
    cosine = dot(proximal_vector, distal_vector) / (proximal_length * distal_length)
    interior_angle = degrees(acos(min(1.0, max(-1.0, cosine))))
    return 180.0 - interior_angle


def _mean_flexion_group(values_by_joint: Mapping[str, list[float]], group_name: str) -> float:
    return _mean_values(
        value for joint_name in FLEXION_GROUPS[group_name] for value in values_by_joint[joint_name]
    )


def _percent_flags(values: Iterable[float]) -> float:
    value_list = list(values)
    if not value_list:
        return 0.0
    return fmean(value_list) * 100.0


def _percent_flexion_group(values_by_joint: Mapping[str, list[float]], group_name: str) -> float:
    return _percent_flags(
        value for joint_name in FLEXION_GROUPS[group_name] for value in values_by_joint[joint_name]
    )


def _frame_dt(previous_timestamp_seconds: float, timestamp_seconds: float) -> float:
    return max(timestamp_seconds - previous_timestamp_seconds, 1.0 / 120.0)


def _mean_values(values: Iterable[float]) -> float:
    value_list = list(values)
    return fmean(value_list) if value_list else 0.0
