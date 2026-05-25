"""Reference and output-side plausibility checks for validation clips."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from math import acos, degrees
from statistics import fmean
from typing import Literal

from validation.vr_pose_validation.geometry import (
    Transform,
    Vector3,
    distance,
    dot,
    length,
    scale,
    subtract,
)
from validation.vr_pose_validation.metrics import infer_reference_floor_height
from validation.vr_pose_validation.posture import PostureSummary, summarize_reference_posture
from validation.vr_pose_validation.types import AvatarCalibration, HumanoidPoseFrame

type ReferencePlausibilityBucket = Literal[
    "plausible",
    "needs_review",
    "implausible_reference",
]
type AvatarPlausibilityBucket = Literal[
    "plausible",
    "needs_review",
    "implausible_output",
]

REFERENCE_PLAUSIBILITY_JOINTS = (
    "head",
    "chest",
    "spine",
    "hips",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_hand",
    "right_hand",
    "left_knee",
    "right_knee",
    "left_foot",
    "right_foot",
)
REFERENCE_PLAUSIBILITY_BONES = (
    ("head", "chest"),
    ("chest", "spine"),
    ("spine", "hips"),
    ("chest", "left_shoulder"),
    ("chest", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_hand"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_hand"),
    ("hips", "left_knee"),
    ("left_knee", "left_foot"),
    ("hips", "right_knee"),
    ("right_knee", "right_foot"),
)
AVATAR_PLAUSIBILITY_JOINTS = REFERENCE_PLAUSIBILITY_JOINTS
WORLD_UP: Vector3 = (0.0, 1.0, 0.0)
DEFAULT_JERK_THRESHOLD_M_PER_S3 = 50000.0


@dataclass(frozen=True, slots=True)
class _FlexionLimitSpec:
    name: str
    chain: tuple[str, str, str]
    comfortable_max_deg: float
    hard_max_deg: float


@dataclass(frozen=True, slots=True)
class _AxisLimitSpec:
    name: str
    start_joint: str
    end_joint: str
    comfortable_max_deg: float
    hard_max_deg: float


@dataclass(frozen=True, slots=True)
class _LengthSpec:
    name: str
    start_joint: str
    end_joint: str
    calibration_names: tuple[str, ...]
    fallback_height_ratio: float


@dataclass(frozen=True, slots=True)
class _JointLimitObservation:
    name: str
    value_deg: float
    hard_margin_deg: float
    comfort_penalty: float


FLEXION_LIMIT_SPECS = (
    _FlexionLimitSpec(
        "left_elbow_flexion",
        ("left_shoulder", "left_elbow", "left_hand"),
        135.0,
        160.0,
    ),
    _FlexionLimitSpec(
        "right_elbow_flexion",
        ("right_shoulder", "right_elbow", "right_hand"),
        135.0,
        160.0,
    ),
    _FlexionLimitSpec("left_knee_flexion", ("hips", "left_knee", "left_foot"), 140.0, 165.0),
    _FlexionLimitSpec("right_knee_flexion", ("hips", "right_knee", "right_foot"), 140.0, 165.0),
)
AXIS_LIMIT_SPECS = (
    _AxisLimitSpec("neck_axis_from_vertical", "chest", "head", 55.0, 85.0),
    _AxisLimitSpec("spine_axis_from_vertical", "hips", "chest", 65.0, 95.0),
)
SEGMENT_LENGTH_SPECS = (
    _LengthSpec(
        "left_upper_arm",
        "left_shoulder",
        "left_elbow",
        ("left_upper_arm", "upper_arm", "humerus"),
        0.174,
    ),
    _LengthSpec(
        "right_upper_arm",
        "right_shoulder",
        "right_elbow",
        ("right_upper_arm", "upper_arm", "humerus"),
        0.174,
    ),
    _LengthSpec(
        "left_lower_arm",
        "left_elbow",
        "left_hand",
        ("left_lower_arm", "lower_arm", "forearm", "radius"),
        0.151,
    ),
    _LengthSpec(
        "right_lower_arm",
        "right_elbow",
        "right_hand",
        ("right_lower_arm", "lower_arm", "forearm", "radius"),
        0.151,
    ),
    _LengthSpec(
        "left_upper_leg",
        "hips",
        "left_knee",
        ("left_upper_leg", "upper_leg", "thigh", "femur"),
        0.245,
    ),
    _LengthSpec(
        "right_upper_leg",
        "hips",
        "right_knee",
        ("right_upper_leg", "upper_leg", "thigh", "femur"),
        0.245,
    ),
    _LengthSpec(
        "left_lower_leg",
        "left_knee",
        "left_foot",
        ("left_lower_leg", "lower_leg", "shin", "tibia"),
        0.246,
    ),
    _LengthSpec(
        "right_lower_leg",
        "right_knee",
        "right_foot",
        ("right_lower_leg", "lower_leg", "shin", "tibia"),
        0.246,
    ),
)
ENDPOINT_LENGTH_SPECS = (
    _LengthSpec("left_arm_reach", "left_shoulder", "left_hand", ("left_arm", "arm"), 0.325),
    _LengthSpec("right_arm_reach", "right_shoulder", "right_hand", ("right_arm", "arm"), 0.325),
    _LengthSpec("left_leg_reach", "hips", "left_foot", ("left_leg", "leg"), 0.491),
    _LengthSpec("right_leg_reach", "hips", "right_foot", ("right_leg", "leg"), 0.491),
)


@dataclass(frozen=True, slots=True)
class ReferencePlausibilitySummary:
    bucket: ReferencePlausibilityBucket
    tags: tuple[str, ...]
    frame_count: int
    inferred_floor_height_m: float
    min_head_height_m: float
    mean_head_height_m: float
    min_hip_height_m: float
    p05_torso_verticality: float
    mean_torso_verticality: float
    p95_joint_speed_m_per_s: float
    max_joint_speed_m_per_s: float
    pct_frames_over_20m_s: float
    p95_joint_acceleration_m_per_s2: float
    max_joint_acceleration_m_per_s2: float
    max_bone_length_cv: float
    p95_bone_length_cv: float


@dataclass(frozen=True, slots=True)
class AvatarPlausibilitySummary:
    """Cheap output-side kinematic plausibility score.

    This deliberately does not compare against mocap ground truth. It scores
    whether the emitted avatar pose sequence is biomechanically plausible enough
    to render, using only the estimated pose, floor, and calibration scale.
    """

    bucket: AvatarPlausibilityBucket
    tags: tuple[str, ...]
    frame_count: int
    inferred_floor_height_m: float
    human_height_m: float
    plausibility_score_0_100: float
    joint_limit_penalty: float
    sustained_awkwardness_penalty: float
    endpoint_stretch_penalty: float
    temporal_jerk_penalty: float
    min_joint_limit_margin_deg: float
    mean_joint_limit_excess_deg: float
    pct_joint_limit_violation_frames: float
    pct_sustained_awkward_frames: float
    max_endpoint_stretch_ratio: float
    p95_endpoint_stretch_ratio: float
    pct_endpoint_stretch_over_105pct: float
    p95_joint_jerk_m_per_s3: float
    max_joint_jerk_m_per_s3: float
    pct_joint_jerk_over_threshold: float


def summarize_reference_plausibility(
    reference_frames: Iterable[dict[str, Transform]],
    *,
    timestamps_seconds: Iterable[float] | None = None,
    floor_height_m: float | None = None,
    posture_summary: PostureSummary | None = None,
) -> ReferencePlausibilitySummary:
    frame_list = list(reference_frames)
    timestamp_list = list(timestamps_seconds) if timestamps_seconds is not None else None
    inferred_floor_height = (
        infer_reference_floor_height(frame_list) if floor_height_m is None else floor_height_m
    )
    posture = posture_summary or summarize_reference_posture(
        frame_list,
        floor_height_m=inferred_floor_height,
        timestamps_seconds=timestamp_list,
    )
    head_heights = _joint_heights(frame_list, "head", inferred_floor_height)
    hip_heights = _joint_heights(frame_list, "hips", inferred_floor_height)
    torso_verticalities = _torso_verticalities(frame_list)
    frame_max_speeds, joint_speeds = _joint_speeds(frame_list, timestamp_list)
    joint_accelerations = _joint_accelerations(frame_list, timestamp_list)
    bone_cvs = _bone_length_cvs(frame_list)

    tags = _plausibility_tags(
        frame_count=len(frame_list),
        min_head_height_m=min(head_heights, default=0.0),
        mean_head_height_m=_mean(head_heights),
        min_hip_height_m=min(hip_heights, default=0.0),
        p05_torso_verticality=_quantile(torso_verticalities, 0.05),
        p95_joint_speed_m_per_s=_quantile(joint_speeds, 0.95),
        max_joint_speed_m_per_s=max(joint_speeds, default=0.0),
        pct_frames_over_20m_s=_percent(
            sum(1 for speed in frame_max_speeds if speed >= 20.0),
            len(frame_max_speeds),
        ),
        p95_joint_acceleration_m_per_s2=_quantile(joint_accelerations, 0.95),
        max_bone_length_cv=max(bone_cvs, default=0.0),
        posture_summary=posture,
    )
    return ReferencePlausibilitySummary(
        bucket=_plausibility_bucket(tags),
        tags=tags,
        frame_count=len(frame_list),
        inferred_floor_height_m=inferred_floor_height,
        min_head_height_m=min(head_heights, default=0.0),
        mean_head_height_m=_mean(head_heights),
        min_hip_height_m=min(hip_heights, default=0.0),
        p05_torso_verticality=_quantile(torso_verticalities, 0.05),
        mean_torso_verticality=_mean(torso_verticalities),
        p95_joint_speed_m_per_s=_quantile(joint_speeds, 0.95),
        max_joint_speed_m_per_s=max(joint_speeds, default=0.0),
        pct_frames_over_20m_s=_percent(
            sum(1 for speed in frame_max_speeds if speed >= 20.0),
            len(frame_max_speeds),
        ),
        p95_joint_acceleration_m_per_s2=_quantile(joint_accelerations, 0.95),
        max_joint_acceleration_m_per_s2=max(joint_accelerations, default=0.0),
        max_bone_length_cv=max(bone_cvs, default=0.0),
        p95_bone_length_cv=_quantile(bone_cvs, 0.95),
    )


def summarize_avatar_plausibility(
    poses: Iterable[HumanoidPoseFrame],
    *,
    calibration: AvatarCalibration | None = None,
    human_height_m: float | None = None,
    floor_height_m: float | None = None,
    joint_jerk_threshold_m_per_s3: float = DEFAULT_JERK_THRESHOLD_M_PER_S3,
) -> AvatarPlausibilitySummary:
    """Summarize output-side plausibility independent of reference GT matching."""

    pose_list = list(poses)
    effective_calibration = calibration or AvatarCalibration()
    effective_height = (
        human_height_m
        if human_height_m is not None
        else effective_calibration.human_height_m * effective_calibration.avatar_scale
    )
    inferred_floor = (
        _infer_pose_floor_height(pose_list) if floor_height_m is None else floor_height_m
    )
    joint_limit_frames = [_joint_limit_frame_values(pose) for pose in pose_list]
    hard_margins = [
        observation.hard_margin_deg
        for frame_values in joint_limit_frames
        for observation in frame_values
    ]
    hard_excesses = [max(0.0, -margin) for margin in hard_margins]
    frame_hard_excesses = [
        max(
            (max(0.0, -observation.hard_margin_deg) for observation in frame_values),
            default=0.0,
        )
        for frame_values in joint_limit_frames
    ]
    frame_awkwardness = [
        max((observation.comfort_penalty for observation in frame_values), default=0.0)
        for frame_values in joint_limit_frames
    ]
    stretch_ratios = [
        ratio
        for pose in pose_list
        for ratio in _endpoint_stretch_ratios(
            pose,
            effective_calibration,
            human_height_m=effective_height,
        )
    ]
    jerk_values = _joint_jerks(pose_list)

    pct_joint_limit_violation_frames = _percent(
        sum(1 for excess in frame_hard_excesses if excess > 0.0),
        len(frame_hard_excesses),
    )
    pct_sustained_awkward_frames = _percent(
        sum(1 for awkwardness in frame_awkwardness if awkwardness >= 0.50),
        len(frame_awkwardness),
    )
    pct_endpoint_stretch_over_105pct = _percent(
        sum(1 for ratio in stretch_ratios if ratio >= 1.05),
        len(stretch_ratios),
    )
    pct_joint_jerk_over_threshold = _percent(
        sum(1 for jerk in jerk_values if jerk >= joint_jerk_threshold_m_per_s3),
        len(jerk_values),
    )

    joint_limit_penalty = _clamp01(
        _mean(hard_excesses) / 25.0 + pct_joint_limit_violation_frames / 100.0
    )
    sustained_awkwardness_penalty = _mean(frame_awkwardness)
    endpoint_stretch_penalty = _clamp01(
        _mean(max(0.0, ratio - 1.05) for ratio in stretch_ratios) / 0.20
        + pct_endpoint_stretch_over_105pct / 100.0
    )
    temporal_jerk_penalty = _clamp01(
        _quantile(jerk_values, 0.95) / joint_jerk_threshold_m_per_s3
        + pct_joint_jerk_over_threshold / 100.0
    )
    plausibility_score = max(
        0.0,
        100.0
        * (
            1.0
            - (
                joint_limit_penalty * 0.30
                + sustained_awkwardness_penalty * 0.25
                + endpoint_stretch_penalty * 0.25
                + temporal_jerk_penalty * 0.20
            )
        ),
    )
    tags = _avatar_plausibility_tags(
        pct_joint_limit_violation_frames=pct_joint_limit_violation_frames,
        pct_sustained_awkward_frames=pct_sustained_awkward_frames,
        max_endpoint_stretch_ratio=max(stretch_ratios, default=1.0),
        p95_endpoint_stretch_ratio=_quantile(stretch_ratios, 0.95),
        p95_joint_jerk_m_per_s3=_quantile(jerk_values, 0.95),
        max_joint_jerk_m_per_s3=max(jerk_values, default=0.0),
        pct_joint_jerk_over_threshold=pct_joint_jerk_over_threshold,
        joint_jerk_threshold_m_per_s3=joint_jerk_threshold_m_per_s3,
        floor_penetration_m=_max_floor_penetration(pose_list, inferred_floor),
    )
    return AvatarPlausibilitySummary(
        bucket=_avatar_plausibility_bucket(tags),
        tags=tags,
        frame_count=len(pose_list),
        inferred_floor_height_m=inferred_floor,
        human_height_m=effective_height,
        plausibility_score_0_100=plausibility_score,
        joint_limit_penalty=joint_limit_penalty,
        sustained_awkwardness_penalty=sustained_awkwardness_penalty,
        endpoint_stretch_penalty=endpoint_stretch_penalty,
        temporal_jerk_penalty=temporal_jerk_penalty,
        min_joint_limit_margin_deg=min(hard_margins, default=0.0),
        mean_joint_limit_excess_deg=_mean(hard_excesses),
        pct_joint_limit_violation_frames=pct_joint_limit_violation_frames,
        pct_sustained_awkward_frames=pct_sustained_awkward_frames,
        max_endpoint_stretch_ratio=max(stretch_ratios, default=1.0),
        p95_endpoint_stretch_ratio=_quantile(stretch_ratios, 0.95),
        pct_endpoint_stretch_over_105pct=pct_endpoint_stretch_over_105pct,
        p95_joint_jerk_m_per_s3=_quantile(jerk_values, 0.95),
        max_joint_jerk_m_per_s3=max(jerk_values, default=0.0),
        pct_joint_jerk_over_threshold=pct_joint_jerk_over_threshold,
    )


def _avatar_plausibility_tags(
    *,
    pct_joint_limit_violation_frames: float,
    pct_sustained_awkward_frames: float,
    max_endpoint_stretch_ratio: float,
    p95_endpoint_stretch_ratio: float,
    p95_joint_jerk_m_per_s3: float,
    max_joint_jerk_m_per_s3: float,
    pct_joint_jerk_over_threshold: float,
    joint_jerk_threshold_m_per_s3: float,
    floor_penetration_m: float,
) -> tuple[str, ...]:
    tags: list[str] = []
    if pct_joint_limit_violation_frames >= 5.0:
        tags.append("joint_limit_violation")
    if pct_sustained_awkward_frames >= 20.0:
        tags.append("sustained_awkwardness")
    if p95_endpoint_stretch_ratio >= 1.10 or max_endpoint_stretch_ratio >= 1.25:
        tags.append("endpoint_stretch")
    elif p95_endpoint_stretch_ratio >= 1.05:
        tags.append("minor_endpoint_stretch")
    if p95_joint_jerk_m_per_s3 >= joint_jerk_threshold_m_per_s3 or (
        max_joint_jerk_m_per_s3 >= joint_jerk_threshold_m_per_s3 * 20.0
        and pct_joint_jerk_over_threshold >= 0.5
    ):
        tags.append("temporal_jerk_spike")
    if floor_penetration_m > 0.05:
        tags.append("floor_penetration")
    return tuple(tags)


def _avatar_plausibility_bucket(tags: Sequence[str]) -> AvatarPlausibilityBucket:
    hard_tags = {
        "joint_limit_violation",
        "endpoint_stretch",
        "temporal_jerk_spike",
        "floor_penetration",
    }
    if hard_tags.intersection(tags):
        return "implausible_output"
    if tags:
        return "needs_review"
    return "plausible"


def _joint_limit_frame_values(pose: HumanoidPoseFrame) -> list[_JointLimitObservation]:
    observations: list[_JointLimitObservation] = []
    for spec in FLEXION_LIMIT_SPECS:
        flexion_angle = _pose_flexion_angle_deg(pose, spec.chain)
        if flexion_angle is None:
            continue
        observations.append(_joint_limit_observation(spec.name, flexion_angle, spec))
    for spec in AXIS_LIMIT_SPECS:
        axis_angle = _pose_axis_angle_from_vertical_deg(pose, spec.start_joint, spec.end_joint)
        if axis_angle is None:
            continue
        observations.append(_joint_limit_observation(spec.name, axis_angle, spec))
    return observations


def _joint_limit_observation(
    name: str,
    value_deg: float,
    spec: _FlexionLimitSpec | _AxisLimitSpec,
) -> _JointLimitObservation:
    comfort_range = max(spec.hard_max_deg - spec.comfortable_max_deg, 1e-9)
    return _JointLimitObservation(
        name=name,
        value_deg=value_deg,
        hard_margin_deg=spec.hard_max_deg - value_deg,
        comfort_penalty=_clamp01((value_deg - spec.comfortable_max_deg) / comfort_range),
    )


def _endpoint_stretch_ratios(
    pose: HumanoidPoseFrame,
    calibration: AvatarCalibration,
    *,
    human_height_m: float,
) -> list[float]:
    ratios: list[float] = []
    for spec in (*SEGMENT_LENGTH_SPECS, *ENDPOINT_LENGTH_SPECS):
        start_position = _pose_position(pose, spec.start_joint)
        end_position = _pose_position(pose, spec.end_joint)
        if start_position is None or end_position is None:
            continue
        expected_length = _expected_length(
            calibration,
            spec.calibration_names,
            fallback=human_height_m * spec.fallback_height_ratio,
        )
        if expected_length <= 1e-9:
            continue
        ratios.append(distance(start_position, end_position) / expected_length)
    return ratios


def _expected_length(
    calibration: AvatarCalibration,
    names: Sequence[str],
    *,
    fallback: float,
) -> float:
    for name in names:
        value = calibration.bone_lengths_m.get(name)
        if value is not None and value > 0.0:
            return value * calibration.avatar_scale
    return fallback


def _joint_jerks(poses: Sequence[HumanoidPoseFrame]) -> list[float]:
    jerks: list[float] = []
    for frame_index in range(len(poses) - 3):
        dt_0 = _pose_dt(poses, frame_index)
        dt_1 = _pose_dt(poses, frame_index + 1)
        dt_2 = _pose_dt(poses, frame_index + 2)
        if dt_0 <= 0.0 or dt_1 <= 0.0 or dt_2 <= 0.0:
            continue
        first_accel_dt = (dt_0 + dt_1) * 0.5
        second_accel_dt = (dt_1 + dt_2) * 0.5
        jerk_dt = (first_accel_dt + second_accel_dt) * 0.5
        if first_accel_dt <= 0.0 or second_accel_dt <= 0.0 or jerk_dt <= 0.0:
            continue

        window = poses[frame_index : frame_index + 4]
        for joint_name in AVATAR_PLAUSIBILITY_JOINTS:
            positions = [_pose_position(pose, joint_name) for pose in window]
            if any(position is None for position in positions):
                continue
            p0, p1, p2, p3 = positions
            if p0 is None or p1 is None or p2 is None or p3 is None:
                continue
            velocity_0 = scale(subtract(p1, p0), 1.0 / dt_0)
            velocity_1 = scale(subtract(p2, p1), 1.0 / dt_1)
            velocity_2 = scale(subtract(p3, p2), 1.0 / dt_2)
            acceleration_0 = scale(subtract(velocity_1, velocity_0), 1.0 / first_accel_dt)
            acceleration_1 = scale(subtract(velocity_2, velocity_1), 1.0 / second_accel_dt)
            jerks.append(length(subtract(acceleration_1, acceleration_0)) / jerk_dt)
    return jerks


def _pose_dt(poses: Sequence[HumanoidPoseFrame], index: int) -> float:
    if index + 1 >= len(poses):
        return 1.0 / 120.0
    return poses[index + 1].timestamp_seconds - poses[index].timestamp_seconds


def _pose_position(pose: HumanoidPoseFrame, joint_name: str) -> Vector3 | None:
    joint = pose.joints.get(joint_name)
    if joint is None:
        return None
    return joint.transform.position


def _pose_flexion_angle_deg(
    pose: HumanoidPoseFrame,
    joint_chain: tuple[str, str, str],
) -> float | None:
    positions = [_pose_position(pose, joint_name) for joint_name in joint_chain]
    if any(position is None for position in positions):
        return None
    proximal, middle, distal = positions
    if proximal is None or middle is None or distal is None:
        return None
    return _flexion_angle_deg(proximal, middle, distal)


def _pose_axis_angle_from_vertical_deg(
    pose: HumanoidPoseFrame,
    start_joint: str,
    end_joint: str,
) -> float | None:
    start_position = _pose_position(pose, start_joint)
    end_position = _pose_position(pose, end_joint)
    if start_position is None or end_position is None:
        return None
    axis = subtract(end_position, start_position)
    if length(axis) <= 1e-9:
        return None
    return _vector_angle_deg(axis, WORLD_UP)


def _flexion_angle_deg(proximal: Vector3, middle: Vector3, distal: Vector3) -> float | None:
    proximal_vector = subtract(proximal, middle)
    distal_vector = subtract(distal, middle)
    proximal_length = length(proximal_vector)
    distal_length = length(distal_vector)
    if proximal_length <= 1e-9 or distal_length <= 1e-9:
        return None
    interior_angle = _vector_angle_deg(proximal_vector, distal_vector)
    return 180.0 - interior_angle


def _vector_angle_deg(a: Vector3, b: Vector3) -> float:
    a_length = length(a)
    b_length = length(b)
    if a_length <= 1e-9 or b_length <= 1e-9:
        return 0.0
    cosine = dot(a, b) / (a_length * b_length)
    return degrees(acos(min(1.0, max(-1.0, cosine))))


def _infer_pose_floor_height(poses: Sequence[HumanoidPoseFrame]) -> float:
    heights = sorted(
        pose.joints[joint_name].transform.position[1]
        for pose in poses
        for joint_name in ("left_foot", "right_foot")
        if joint_name in pose.joints
    )
    if not heights:
        return 0.0
    index = min(len(heights) - 1, max(0, int(len(heights) * 0.05)))
    return heights[index]


def _max_floor_penetration(poses: Sequence[HumanoidPoseFrame], floor_height_m: float) -> float:
    return max(
        (
            max(0.0, floor_height_m - pose.joints[joint_name].transform.position[1])
            for pose in poses
            for joint_name in ("left_foot", "right_foot")
            if joint_name in pose.joints
        ),
        default=0.0,
    )


def _plausibility_tags(
    *,
    frame_count: int,
    min_head_height_m: float,
    mean_head_height_m: float,
    min_hip_height_m: float,
    p05_torso_verticality: float,
    p95_joint_speed_m_per_s: float,
    max_joint_speed_m_per_s: float,
    pct_frames_over_20m_s: float,
    p95_joint_acceleration_m_per_s2: float,
    max_bone_length_cv: float,
    posture_summary: PostureSummary,
) -> tuple[str, ...]:
    tags: list[str] = []
    if frame_count < 30:
        tags.append("too_short")
    if min_head_height_m < -0.10 or mean_head_height_m < 0.25:
        tags.append("head_below_floor")
    if min_hip_height_m < -0.10:
        tags.append("hips_below_floor")
    if posture_summary.inverted_pct >= 5.0:
        tags.append("inverted_frames")
    if posture_summary.floor_reclined_pct + posture_summary.reclined_pct >= 20.0:
        tags.append("reclined_frames")
    if p05_torso_verticality < 0.20:
        tags.append("near_horizontal_torso")
    if p95_joint_speed_m_per_s >= 30.0 or (
        max_joint_speed_m_per_s >= 200.0 and pct_frames_over_20m_s >= 0.50
    ):
        tags.append("motion_discontinuity")
    if p95_joint_acceleration_m_per_s2 >= 6000.0:
        tags.append("acceleration_spike")
    return tuple(tags)


def _plausibility_bucket(tags: Sequence[str]) -> ReferencePlausibilityBucket:
    hard_tags = {
        "head_below_floor",
        "hips_below_floor",
        "motion_discontinuity",
        "acceleration_spike",
    }
    if hard_tags.intersection(tags):
        return "implausible_reference"
    if tags:
        return "needs_review"
    return "plausible"


def _joint_heights(
    frames: Sequence[dict[str, Transform]],
    joint_name: str,
    floor_height_m: float,
) -> list[float]:
    return [
        frame[joint_name].position[1] - floor_height_m for frame in frames if joint_name in frame
    ]


def _torso_verticalities(frames: Sequence[dict[str, Transform]]) -> list[float]:
    values: list[float] = []
    for frame in frames:
        head = frame.get("head")
        hips = frame.get("hips")
        if head is None or hips is None:
            continue
        torso = subtract(head.position, hips.position)
        torso_length = length(torso)
        if torso_length > 1e-9:
            values.append(abs(torso[1]) / torso_length)
    return values


def _joint_speeds(
    frames: Sequence[dict[str, Transform]],
    timestamps_seconds: Sequence[float] | None,
) -> tuple[list[float], list[float]]:
    frame_max_speeds: list[float] = []
    joint_speeds: list[float] = []
    for previous_index, (previous_frame, current_frame) in enumerate(
        zip(frames, frames[1:], strict=False)
    ):
        dt = _frame_dt(previous_index, timestamps_seconds)
        if dt <= 0.0:
            continue
        speeds = [
            distance(previous_frame[joint_name].position, current_frame[joint_name].position) / dt
            for joint_name in REFERENCE_PLAUSIBILITY_JOINTS
            if joint_name in previous_frame and joint_name in current_frame
        ]
        if speeds:
            frame_max_speeds.append(max(speeds))
            joint_speeds.extend(speeds)
    return frame_max_speeds, joint_speeds


def _joint_accelerations(
    frames: Sequence[dict[str, Transform]],
    timestamps_seconds: Sequence[float] | None,
) -> list[float]:
    accelerations: list[float] = []
    for frame_index in range(1, len(frames) - 1):
        previous_frame = frames[frame_index - 1]
        current_frame = frames[frame_index]
        next_frame = frames[frame_index + 1]
        dt_previous = _frame_dt(frame_index - 1, timestamps_seconds)
        dt_next = _frame_dt(frame_index, timestamps_seconds)
        dt_center = (dt_previous + dt_next) * 0.5
        if dt_previous <= 0.0 or dt_next <= 0.0 or dt_center <= 0.0:
            continue
        for joint_name in REFERENCE_PLAUSIBILITY_JOINTS:
            if (
                joint_name not in previous_frame
                or joint_name not in current_frame
                or joint_name not in next_frame
            ):
                continue
            previous_velocity = scale(
                subtract(current_frame[joint_name].position, previous_frame[joint_name].position),
                1.0 / dt_previous,
            )
            next_velocity = scale(
                subtract(next_frame[joint_name].position, current_frame[joint_name].position),
                1.0 / dt_next,
            )
            accelerations.append(length(subtract(next_velocity, previous_velocity)) / dt_center)
    return accelerations


def _bone_length_cvs(frames: Sequence[dict[str, Transform]]) -> list[float]:
    cvs: list[float] = []
    for start_name, end_name in REFERENCE_PLAUSIBILITY_BONES:
        lengths = [
            distance(frame[start_name].position, frame[end_name].position)
            for frame in frames
            if start_name in frame and end_name in frame
        ]
        if not lengths:
            continue
        mean_length = _mean(lengths)
        if mean_length <= 1e-9:
            continue
        variance = _mean((value - mean_length) ** 2 for value in lengths)
        cvs.append(variance**0.5 / mean_length)
    return cvs


def _frame_dt(index: int, timestamps_seconds: Sequence[float] | None) -> float:
    if timestamps_seconds is None or index + 1 >= len(timestamps_seconds):
        return 1.0 / 120.0
    return timestamps_seconds[index + 1] - timestamps_seconds[index]


def _quantile(values: Iterable[float], q: float) -> float:
    value_list = sorted(values)
    if not value_list:
        return 0.0
    index = round((len(value_list) - 1) * min(1.0, max(0.0, q)))
    return value_list[index]


def _percent(count: int, total: int) -> float:
    return count / total * 100.0 if total > 0 else 0.0


def _mean(values: Iterable[float]) -> float:
    value_list = list(values)
    return fmean(value_list) if value_list else 0.0


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))
