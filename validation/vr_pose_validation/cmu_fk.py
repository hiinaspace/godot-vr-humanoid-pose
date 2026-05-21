"""Lightweight CMU ASF/AMC forward kinematics for validation smoke tests."""

from __future__ import annotations

from collections.abc import Iterator, Mapping

from validation.vr_pose_validation.cmu import (
    CmuBone,
    CmuFrame,
    CmuSkeleton,
    ordered_joint_rotation,
    scaled_root_translation,
)
from validation.vr_pose_validation.geometry import (
    IDENTITY_QUATERNION,
    Quaternion,
    Transform,
    add,
    euler_quaternion,
    multiply_quaternion,
    rotate_vector,
    scale,
)

type CmuPoseFrame = Mapping[str, Transform]

CANONICAL_CMU_JOINTS = {
    "head": "head",
    "left_hand": "lwrist",
    "right_hand": "rwrist",
    "hips": "root",
    "left_foot": "lfoot",
    "right_foot": "rfoot",
}


def iter_cmu_pose_frames(
    skeleton: CmuSkeleton,
    frames: list[CmuFrame],
    *,
    frame_stride: int = 1,
    max_frames: int | None = None,
) -> Iterator[CmuPoseFrame]:
    emitted = 0
    for index, frame in enumerate(frames):
        if index % frame_stride != 0:
            continue
        yield solve_cmu_pose_frame(skeleton, frame)
        emitted += 1
        if max_frames is not None and emitted >= max_frames:
            return


def solve_cmu_pose_frame(skeleton: CmuSkeleton, frame: CmuFrame) -> CmuPoseFrame:
    root_position = scaled_root_translation(frame, skeleton.root)
    root_rotation = _root_rotation(skeleton, frame)
    transforms: dict[str, Transform] = {
        "root": Transform(position=root_position, rotation=root_rotation)
    }
    rotations: dict[str, Quaternion] = {"root": root_rotation}

    for child_name in skeleton.hierarchy.get("root", ()):
        _solve_child(skeleton, frame, child_name, "root", transforms, rotations)
    return transforms


def canonical_pose_from_cmu(pose_frame: CmuPoseFrame) -> dict[str, Transform]:
    return {
        canonical: pose_frame[source]
        for canonical, source in CANONICAL_CMU_JOINTS.items()
        if source in pose_frame
    }


def _solve_child(
    skeleton: CmuSkeleton,
    frame: CmuFrame,
    bone_name: str,
    parent_name: str,
    transforms: dict[str, Transform],
    rotations: dict[str, Quaternion],
) -> None:
    bone = skeleton.bones[bone_name]
    parent_position = transforms[parent_name].position
    parent_rotation = rotations[parent_name]
    offset = scale(bone.direction, bone.length_m)
    position = add(parent_position, rotate_vector(parent_rotation, offset))
    rotation = multiply_quaternion(parent_rotation, _bone_local_rotation(skeleton, frame, bone))
    transforms[bone_name] = Transform(position=position, rotation=rotation)
    rotations[bone_name] = rotation

    for child_name in skeleton.hierarchy.get(bone_name, ()):
        _solve_child(skeleton, frame, child_name, bone_name, transforms, rotations)


def _root_rotation(skeleton: CmuSkeleton, frame: CmuFrame) -> Quaternion:
    root_values = frame.values.get("root", ())
    rotation_channels: list[str] = []
    rotation_values: list[float] = []
    for channel, value in zip(skeleton.root.order, root_values, strict=False):
        if channel.lower().startswith("r"):
            rotation_channels.append(channel)
            rotation_values.append(value)
    return euler_quaternion(tuple(rotation_channels), tuple(rotation_values))


def _bone_local_rotation(skeleton: CmuSkeleton, frame: CmuFrame, bone: CmuBone) -> Quaternion:
    joint_rotation = ordered_joint_rotation(frame, skeleton, bone.name)
    channels: list[str] = []
    values: list[float] = []
    for channel in bone.dof:
        if channel in joint_rotation:
            channels.append(channel)
            values.append(joint_rotation[channel])
    if not channels:
        return IDENTITY_QUATERNION
    axis_rotation = euler_quaternion(
        tuple(f"r{axis_name.lower()}" for axis_name in bone.axis_order),
        bone.axis,
    )
    dof_rotation = euler_quaternion(tuple(channels), tuple(values))
    return multiply_quaternion(axis_rotation, dof_rotation)
