"""Calibration helpers derived from canonical reference poses."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from statistics import fmean, median

from validation.vr_pose_validation.geometry import Transform, distance
from validation.vr_pose_validation.types import AvatarCalibration, HumanoidPoseFrame, JointPose

DEFAULT_HUMAN_HEIGHT_M = 1.7
DEFAULT_FOOT_HEIGHT_M = 0.08
FOOT_JOINTS = ("left_foot", "right_foot")

SEGMENT_JOINTS = {
    "head_to_neck": ("head", "neck"),
    "neck_to_chest": ("neck", "chest"),
    "chest_to_spine": ("chest", "spine"),
    "spine_to_hips": ("spine", "hips"),
    "left_upper_arm": ("left_shoulder", "left_elbow"),
    "right_upper_arm": ("right_shoulder", "right_elbow"),
    "left_lower_arm": ("left_elbow", "left_hand"),
    "right_lower_arm": ("right_elbow", "right_hand"),
    "left_upper_leg": ("hips", "left_knee"),
    "right_upper_leg": ("hips", "right_knee"),
    "left_lower_leg": ("left_knee", "left_foot"),
    "right_lower_leg": ("right_knee", "right_foot"),
    "shoulder_width": ("left_shoulder", "right_shoulder"),
}


def calibration_from_reference_frames(
    reference_frames: Iterable[Mapping[str, Transform]],
    *,
    floor_height_m: float | None = None,
) -> AvatarCalibration:
    """Infer clip-local avatar proportions from flattened CMU reference joints."""

    frame_list = list(reference_frames)
    lengths = _segment_lengths(frame_list)
    bone_lengths: dict[str, float] = dict(lengths)
    _add_pair_aliases(
        bone_lengths,
        aliases=("upper_arm", "humerus"),
        names=("left_upper_arm", "right_upper_arm"),
    )
    _add_pair_aliases(
        bone_lengths,
        aliases=("lower_arm", "forearm", "radius"),
        names=("left_lower_arm", "right_lower_arm"),
    )
    _add_pair_aliases(
        bone_lengths,
        aliases=("upper_leg", "thigh", "femur"),
        names=("left_upper_leg", "right_upper_leg"),
    )
    _add_pair_aliases(
        bone_lengths,
        aliases=("lower_leg", "shin", "tibia"),
        names=("left_lower_leg", "right_lower_leg"),
    )
    _add_sum_aliases(
        bone_lengths,
        aliases=("left_arm",),
        names=("left_upper_arm", "left_lower_arm"),
    )
    _add_sum_aliases(
        bone_lengths,
        aliases=("right_arm",),
        names=("right_upper_arm", "right_lower_arm"),
    )
    _add_pair_aliases(bone_lengths, aliases=("arm",), names=("left_arm", "right_arm"))
    _add_sum_aliases(
        bone_lengths,
        aliases=("left_leg",),
        names=("left_upper_leg", "left_lower_leg"),
    )
    _add_sum_aliases(
        bone_lengths,
        aliases=("right_leg",),
        names=("right_upper_leg", "right_lower_leg"),
    )
    _add_pair_aliases(bone_lengths, aliases=("leg",), names=("left_leg", "right_leg"))
    _copy_alias(bone_lengths, "shoulder_width", "left_shoulder_to_right_shoulder")

    torso_chain_length = _sum_existing(
        bone_lengths,
        ("head_to_neck", "neck_to_chest", "chest_to_spine", "spine_to_hips"),
    )
    if torso_chain_length is not None:
        bone_lengths["head_to_hips"] = torso_chain_length
        bone_lengths["hips_to_head"] = torso_chain_length

    human_height_m = _calibrated_height(
        bone_lengths,
        frame_list,
        floor_height_m=floor_height_m,
    )
    return AvatarCalibration(
        bone_lengths_m=bone_lengths,
        foot_height_m=DEFAULT_FOOT_HEIGHT_M,
        human_height_m=human_height_m,
    )


def humanoid_poses_from_reference_frames(
    reference_frames: Iterable[Mapping[str, Transform]],
    *,
    timestamps_seconds: Iterable[float] | None = None,
) -> list[HumanoidPoseFrame]:
    """Wrap reference transforms in the estimator output pose contract."""

    frame_list = list(reference_frames)
    timestamp_list = (
        list(timestamps_seconds)
        if timestamps_seconds is not None
        else [index / 120.0 for index in range(len(frame_list))]
    )
    poses: list[HumanoidPoseFrame] = []
    for index, frame in enumerate(frame_list):
        timestamp = timestamp_list[index] if index < len(timestamp_list) else index / 120.0
        root = frame.get("hips", Transform())
        poses.append(
            HumanoidPoseFrame(
                timestamp_seconds=timestamp,
                root=root,
                joints={
                    name: JointPose(transform=transform, estimated=False)
                    for name, transform in frame.items()
                },
            )
        )
    return poses


def _segment_lengths(
    frame_list: Sequence[Mapping[str, Transform]],
) -> dict[str, float]:
    lengths: dict[str, float] = {}
    for name, (start_joint, end_joint) in SEGMENT_JOINTS.items():
        values = [
            distance(frame[start_joint].position, frame[end_joint].position)
            for frame in frame_list
            if start_joint in frame and end_joint in frame
        ]
        non_zero_values = [value for value in values if value > 1e-6]
        if non_zero_values:
            lengths[name] = median(non_zero_values)
    return lengths


def _add_pair_aliases(
    bone_lengths: dict[str, float],
    *,
    aliases: tuple[str, ...],
    names: tuple[str, str],
) -> None:
    values = [bone_lengths[name] for name in names if name in bone_lengths]
    if not values:
        return
    value = fmean(values)
    for alias in aliases:
        bone_lengths[alias] = value


def _add_sum_aliases(
    bone_lengths: dict[str, float],
    *,
    aliases: tuple[str, ...],
    names: tuple[str, ...],
) -> None:
    values = [bone_lengths[name] for name in names if name in bone_lengths]
    if len(values) != len(names):
        return
    value = sum(values)
    for alias in aliases:
        bone_lengths[alias] = value


def _copy_alias(bone_lengths: dict[str, float], source: str, alias: str) -> None:
    value = bone_lengths.get(source)
    if value is not None:
        bone_lengths[alias] = value


def _calibrated_height(
    bone_lengths: Mapping[str, float],
    frame_list: Sequence[Mapping[str, Transform]],
    *,
    floor_height_m: float | None,
) -> float:
    torso = _sum_existing(
        bone_lengths,
        ("head_to_neck", "neck_to_chest", "chest_to_spine", "spine_to_hips"),
    )
    leg = bone_lengths.get("leg")
    chain_height = (
        torso + leg + DEFAULT_FOOT_HEIGHT_M if torso is not None and leg is not None else None
    )

    floor = _infer_floor_height(frame_list) if floor_height_m is None else floor_height_m
    head_heights = sorted(
        frame["head"].position[1] - floor for frame in frame_list if "head" in frame
    )
    upright_height = _quantile(head_heights, 0.95) if head_heights else None
    candidates = [
        value
        for value in (chain_height, upright_height)
        if value is not None
        and DEFAULT_HUMAN_HEIGHT_M * 0.5 <= value <= DEFAULT_HUMAN_HEIGHT_M * 1.6
    ]
    return max(candidates, default=DEFAULT_HUMAN_HEIGHT_M)


def _sum_existing(
    bone_lengths: Mapping[str, float],
    names: tuple[str, ...],
) -> float | None:
    values = [bone_lengths[name] for name in names if name in bone_lengths]
    if len(values) != len(names):
        return None
    return sum(values)


def _infer_floor_height(frame_list: Sequence[Mapping[str, Transform]]) -> float:
    heights = sorted(
        transform.position[1]
        for frame in frame_list
        for joint_name, transform in frame.items()
        if joint_name in FOOT_JOINTS
    )
    return _quantile(heights, 0.05) if heights else 0.0


def _quantile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    clamped_quantile = min(1.0, max(0.0, quantile))
    position = (len(values) - 1) * clamped_quantile
    lower = int(position)
    upper = min(len(values) - 1, lower + 1)
    blend = position - lower
    return values[lower] * (1.0 - blend) + values[upper] * blend
