"""Differentiable Torch decoder for upper-body runtime latents."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from validation.vr_pose_validation.upper_body_ml import (
    DECODED_REFERENCE_JOINTS,
    RUNTIME_DECODER_CONTEXT_JOINTS,
)

WORLD_UP = (0.0, 1.0, 0.0)
WORLD_DOWN = (0.0, -1.0, 0.0)
WORLD_FORWARD = (0.0, 0.0, 1.0)
WORLD_RIGHT = (1.0, 0.0, 0.0)


@dataclass(frozen=True, slots=True)
class RuntimeDecoderFeatureIndices:
    head: tuple[int, int, int]
    left_hand: tuple[int, int, int]
    right_hand: tuple[int, int, int]
    human_height: int
    shoulder_width: int
    upper_arm: int
    lower_arm: int
    arm_length: int


def runtime_decoder_feature_indices(
    feature_names: Sequence[str],
) -> RuntimeDecoderFeatureIndices | None:
    try:
        center_prefix = _center_prefix(feature_names)
    except ValueError:
        return None
    required = (
        f"{center_prefix}_head_position_x",
        f"{center_prefix}_head_position_y",
        f"{center_prefix}_head_position_z",
        f"{center_prefix}_left_hand_position_x",
        f"{center_prefix}_left_hand_position_y",
        f"{center_prefix}_left_hand_position_z",
        f"{center_prefix}_right_hand_position_x",
        f"{center_prefix}_right_hand_position_y",
        f"{center_prefix}_right_hand_position_z",
        "calibration_human_height_m",
        "calibration_shoulder_width_m",
        "calibration_upper_arm_m",
        "calibration_lower_arm_m",
        "calibration_arm_length_m",
    )
    if any(name not in feature_names for name in required):
        return None
    return RuntimeDecoderFeatureIndices(
        head=(
            feature_names.index(f"{center_prefix}_head_position_x"),
            feature_names.index(f"{center_prefix}_head_position_y"),
            feature_names.index(f"{center_prefix}_head_position_z"),
        ),
        left_hand=(
            feature_names.index(f"{center_prefix}_left_hand_position_x"),
            feature_names.index(f"{center_prefix}_left_hand_position_y"),
            feature_names.index(f"{center_prefix}_left_hand_position_z"),
        ),
        right_hand=(
            feature_names.index(f"{center_prefix}_right_hand_position_x"),
            feature_names.index(f"{center_prefix}_right_hand_position_y"),
            feature_names.index(f"{center_prefix}_right_hand_position_z"),
        ),
        human_height=feature_names.index("calibration_human_height_m"),
        shoulder_width=feature_names.index("calibration_shoulder_width_m"),
        upper_arm=feature_names.index("calibration_upper_arm_m"),
        lower_arm=feature_names.index("calibration_lower_arm_m"),
        arm_length=feature_names.index("calibration_arm_length_m"),
    )


def decode_upper_body_runtime_points(
    torch: Any,
    latents: Any,
    raw_features: Any,
    context_points: Any,
    context_masks: Any,
    *,
    feature_indices: RuntimeDecoderFeatureIndices,
    target_names: Sequence[str],
) -> tuple[Any, Any]:
    """Decode current upper-body latents with the same point logic as the runtime."""

    dtype = latents.dtype
    device = latents.device
    head = _context_point(
        torch,
        context_points,
        context_masks,
        "head",
        _head_from_features(torch, raw_features, feature_indices),
    )

    upper_arm = raw_features[:, feature_indices.upper_arm].clamp_min(1e-4)
    lower_arm = raw_features[:, feature_indices.lower_arm].clamp_min(1e-4)
    human_height = raw_features[:, feature_indices.human_height].clamp_min(0.1)
    left_hand = _context_point(
        torch,
        context_points,
        context_masks,
        "left_hand",
        _feature_point(raw_features, feature_indices.left_hand),
    )
    right_hand = _context_point(
        torch,
        context_points,
        context_masks,
        "right_hand",
        _feature_point(raw_features, feature_indices.right_hand),
    )
    chest = _decode_relative_point(latents, head, human_height, target_names, "chest")
    neck = _decode_relative_point(latents, head, human_height, target_names, "neck")
    left_shoulder = _decode_relative_point(
        latents, head, human_height, target_names, "left_shoulder"
    )
    right_shoulder = _decode_relative_point(
        latents,
        head,
        human_height,
        target_names,
        "right_shoulder",
    )
    left_elbow = _solve_elbow(
        torch,
        shoulder=left_shoulder,
        hand=left_hand,
        upper_arm=upper_arm,
        lower_arm=lower_arm,
        bend_hint=_decode_pole(latents, target_names, "left"),
    )
    right_elbow = _solve_elbow(
        torch,
        shoulder=right_shoulder,
        hand=right_hand,
        upper_arm=upper_arm,
        lower_arm=lower_arm,
        bend_hint=_decode_pole(latents, target_names, "right"),
    )
    points = torch.stack(
        (
            chest,
            neck,
            left_shoulder,
            right_shoulder,
            left_elbow,
            right_elbow,
        ),
        dim=1,
    )
    point_masks = _decoded_point_masks(torch, context_masks, dtype=dtype, device=device)
    return points, point_masks


def _decode_relative_point(
    latents: Any,
    head: Any,
    human_height: Any,
    target_names: Sequence[str],
    joint_name: str,
) -> Any:
    rel = torch_stack_columns(
        latents,
        (
            _target_index(target_names, f"{joint_name}_rel_x"),
            _target_index(target_names, f"{joint_name}_rel_y"),
            _target_index(target_names, f"{joint_name}_rel_z"),
        ),
    )
    return head + rel * human_height[:, None]


def _decode_pole(latents: Any, target_names: Sequence[str], side: str) -> Any:
    return torch_stack_columns(
        latents,
        (
            _target_index(target_names, f"{side}_elbow_pole_x"),
            _target_index(target_names, f"{side}_elbow_pole_y"),
            _target_index(target_names, f"{side}_elbow_pole_z"),
        ),
    )


def torch_stack_columns(values: Any, indices: tuple[int, int, int]) -> Any:
    return values[:, [indices[0], indices[1], indices[2]]]


def _solve_elbow(
    torch: Any,
    *,
    shoulder: Any,
    hand: Any,
    upper_arm: Any,
    lower_arm: Any,
    bend_hint: Any,
) -> Any:
    dtype = shoulder.dtype
    device = shoulder.device
    root_to_tip = hand - shoulder
    root_tip_distance = _norm(torch, root_to_tip).clamp_min(0.0001)
    max_reach = (upper_arm + lower_arm - 0.0001).clamp_min(0.0001)
    min_reach = (upper_arm - lower_arm).abs() + 0.0001
    clamped_distance = torch.minimum(torch.maximum(root_tip_distance, min_reach), max_reach)
    direction = _normalize(
        torch,
        root_to_tip,
        _constant(torch, WORLD_UP, dtype=dtype, device=device),
    )
    along = (
        upper_arm * upper_arm - lower_arm * lower_arm + clamped_distance * clamped_distance
    ) / (2.0 * clamped_distance)
    bend_height = torch.sqrt((upper_arm * upper_arm - along * along).clamp_min(0.0))
    bend_direction = _normalize(
        torch,
        _project_on_plane(torch, bend_hint, direction),
        _fallback_bend(torch, direction),
    )
    return shoulder + direction * along[:, None] + bend_direction * bend_height[:, None]


def _fallback_bend(torch: Any, direction: Any) -> Any:
    dtype = direction.dtype
    device = direction.device
    up = _constant(torch, WORLD_UP, dtype=dtype, device=device)
    horizontal_right = _normalize(
        torch,
        torch.cross(direction, up[None, :].expand_as(direction), dim=1),
        _constant(torch, WORLD_RIGHT, dtype=dtype, device=device),
    )
    return _normalize(
        torch,
        torch.cross(horizontal_right, direction, dim=1),
        _constant(torch, WORLD_DOWN, dtype=dtype, device=device),
    )


def _decoded_point_masks(torch: Any, context_masks: Any, *, dtype: Any, device: Any) -> Any:
    if context_masks.numel() == 0:
        return torch.zeros((0, len(DECODED_REFERENCE_JOINTS)), dtype=dtype, device=device)

    def context_mask(joint_name: str) -> Any:
        return context_masks[:, RUNTIME_DECODER_CONTEXT_JOINTS.index(joint_name)]

    head_mask = context_mask("head")
    left_hand_mask = context_mask("left_hand")
    right_hand_mask = context_mask("right_hand")
    return torch.stack(
        (
            head_mask,
            head_mask,
            head_mask,
            head_mask,
            head_mask * left_hand_mask,
            head_mask * right_hand_mask,
        ),
        dim=1,
    )


def _context_point(
    torch: Any,
    context_points: Any,
    context_masks: Any,
    joint_name: str,
    fallback: Any,
) -> Any:
    index = RUNTIME_DECODER_CONTEXT_JOINTS.index(joint_name)
    mask = context_masks[:, index].to(dtype=context_points.dtype)[:, None]
    return torch.where(mask > 0.5, context_points[:, index], fallback)


def _head_from_features(
    torch: Any, raw_features: Any, indices: RuntimeDecoderFeatureIndices
) -> Any:
    del torch
    return _feature_point(raw_features, indices.head)


def _feature_point(raw_features: Any, indices: tuple[int, int, int]) -> Any:
    return raw_features[:, [indices[0], indices[1], indices[2]]]


def _project_on_plane(torch: Any, vector: Any, normal: Any) -> Any:
    if vector.ndim == 1:
        vector = vector[None, :].expand_as(normal)
    normalized_normal = _normalize(
        torch, normal, _constant(torch, WORLD_UP, dtype=normal.dtype, device=normal.device)
    )
    return vector - normalized_normal * (vector * normalized_normal).sum(dim=1)[:, None]


def _normalize(torch: Any, vector: Any, fallback: Any) -> Any:
    if fallback.ndim == 1:
        fallback = fallback[None, :].expand_as(vector)
    magnitude = _norm(torch, vector)
    return torch.where(
        magnitude[:, None] > 1e-9, vector / magnitude.clamp_min(1e-9)[:, None], fallback
    )


def _norm(torch: Any, vector: Any) -> Any:
    return torch.linalg.vector_norm(vector, dim=1)


def _constant(torch: Any, values: tuple[float, float, float], *, dtype: Any, device: Any) -> Any:
    return torch.tensor(values, dtype=dtype, device=device)


def _target_index(target_names: Sequence[str], name: str) -> int:
    return target_names.index(name)


def _center_prefix(feature_names: Sequence[str]) -> str:
    for name in feature_names:
        if name.endswith("_head_position_y") and name.startswith("t+"):
            return name.removesuffix("_head_position_y")
    raise ValueError("Feature schema does not include a center head position")
