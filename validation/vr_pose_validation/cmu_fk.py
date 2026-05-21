"""Lightweight CMU ASF/AMC forward kinematics for validation smoke tests."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass

from validation.vr_pose_validation.cmu import (
    CmuFrame,
    CmuSkeleton,
    scaled_root_translation,
)
from validation.vr_pose_validation.geometry import (
    IDENTITY_QUATERNION,
    Quaternion,
    Transform,
    Vector3,
    add,
    axis_name_quaternion,
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


@dataclass(frozen=True, slots=True)
class CompiledCmuFkBone:
    name: str
    parent_name: str
    offset_m: Vector3
    axis_rotation: Quaternion
    dof_axes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CompiledCmuFkPlan:
    root_rotation_axes: tuple[str, ...]
    root_rotation_value_indices: tuple[int, ...]
    bones: tuple[CompiledCmuFkBone, ...]


def compile_cmu_fk_plan(skeleton: CmuSkeleton) -> CompiledCmuFkPlan:
    root_rotation_axes: list[str] = []
    root_rotation_value_indices: list[int] = []
    for value_index, channel in enumerate(skeleton.root.order):
        axis = _axis_name_from_channel(channel)
        if channel.lower().startswith("r") and axis is not None:
            root_rotation_axes.append(axis)
            root_rotation_value_indices.append(value_index)

    bones: list[CompiledCmuFkBone] = []
    for child_name in skeleton.hierarchy.get("root", ()):
        _compile_child(skeleton, child_name, "root", bones)

    return CompiledCmuFkPlan(
        root_rotation_axes=tuple(root_rotation_axes),
        root_rotation_value_indices=tuple(root_rotation_value_indices),
        bones=tuple(bones),
    )


def iter_cmu_pose_frames(
    skeleton: CmuSkeleton,
    frames: list[CmuFrame],
    *,
    frame_stride: int = 1,
    max_frames: int | None = None,
) -> Iterator[CmuPoseFrame]:
    plan = compile_cmu_fk_plan(skeleton)
    emitted = 0
    for index, frame in enumerate(frames):
        if index % frame_stride != 0:
            continue
        yield solve_cmu_pose_frame(skeleton, frame, plan)
        emitted += 1
        if max_frames is not None and emitted >= max_frames:
            return


def solve_cmu_pose_frame(
    skeleton: CmuSkeleton,
    frame: CmuFrame,
    plan: CompiledCmuFkPlan | None = None,
) -> CmuPoseFrame:
    effective_plan = plan or compile_cmu_fk_plan(skeleton)
    root_position = scaled_root_translation(frame, skeleton.root)
    root_rotation = _root_rotation(frame, effective_plan)
    transforms: dict[str, Transform] = {
        "root": Transform(position=root_position, rotation=root_rotation)
    }
    rotations: dict[str, Quaternion] = {"root": root_rotation}

    for bone in effective_plan.bones:
        parent_position = transforms[bone.parent_name].position
        parent_rotation = rotations[bone.parent_name]
        position = add(parent_position, rotate_vector(parent_rotation, bone.offset_m))
        rotation = multiply_quaternion(
            parent_rotation,
            _bone_local_rotation(frame, bone),
        )
        transforms[bone.name] = Transform(position=position, rotation=rotation)
        rotations[bone.name] = rotation
    return transforms


def canonical_pose_from_cmu(pose_frame: CmuPoseFrame) -> dict[str, Transform]:
    return {
        canonical: pose_frame[source]
        for canonical, source in CANONICAL_CMU_JOINTS.items()
        if source in pose_frame
    }


def _compile_child(
    skeleton: CmuSkeleton,
    bone_name: str,
    parent_name: str,
    bones: list[CompiledCmuFkBone],
) -> None:
    bone = skeleton.bones[bone_name]
    bones.append(
        CompiledCmuFkBone(
            name=bone_name,
            parent_name=parent_name,
            offset_m=scale(bone.direction, bone.length_m),
            axis_rotation=euler_quaternion(
                tuple(f"r{axis_name.lower()}" for axis_name in bone.axis_order),
                bone.axis,
            ),
            dof_axes=tuple(
                axis for channel in bone.dof if (axis := _axis_name_from_channel(channel))
            ),
        )
    )

    for child_name in skeleton.hierarchy.get(bone_name, ()):
        _compile_child(skeleton, child_name, bone_name, bones)


def _root_rotation(frame: CmuFrame, plan: CompiledCmuFkPlan) -> Quaternion:
    root_values = frame.values.get("root", ())
    result = IDENTITY_QUATERNION
    for axis, value_index in zip(
        plan.root_rotation_axes,
        plan.root_rotation_value_indices,
        strict=True,
    ):
        if value_index < len(root_values):
            result = multiply_quaternion(
                result,
                axis_name_quaternion(axis, root_values[value_index]),
            )
    return result


def _bone_local_rotation(frame: CmuFrame, bone: CompiledCmuFkBone) -> Quaternion:
    if not bone.dof_axes:
        return bone.axis_rotation
    joint_values = frame.values.get(bone.name, ())
    dof_rotation = IDENTITY_QUATERNION
    for axis, value in zip(bone.dof_axes, joint_values, strict=False):
        dof_rotation = multiply_quaternion(dof_rotation, axis_name_quaternion(axis, value))
    return multiply_quaternion(bone.axis_rotation, dof_rotation)


def _axis_name_from_channel(channel: str) -> str | None:
    axis_name = channel.lower()[-1:]
    if axis_name in {"x", "y", "z"}:
        return axis_name
    return None
