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
    head_y: int
    left_hand: tuple[int, int, int]
    right_hand: tuple[int, int, int]
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
        f"{center_prefix}_head_position_y",
        f"{center_prefix}_left_hand_position_x",
        f"{center_prefix}_left_hand_position_y",
        f"{center_prefix}_left_hand_position_z",
        f"{center_prefix}_right_hand_position_x",
        f"{center_prefix}_right_hand_position_y",
        f"{center_prefix}_right_hand_position_z",
        "calibration_shoulder_width_m",
        "calibration_upper_arm_m",
        "calibration_lower_arm_m",
        "calibration_arm_length_m",
    )
    if any(name not in feature_names for name in required):
        return None
    return RuntimeDecoderFeatureIndices(
        head_y=feature_names.index(f"{center_prefix}_head_position_y"),
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
    hips = _context_point(
        torch,
        context_points,
        context_masks,
        "hips",
        head - _constant(torch, WORLD_UP, dtype=dtype, device=device)[None, :],
    )
    spine = _context_point(torch, context_points, context_masks, "spine", (head + hips) * 0.6)
    chest = _context_point(torch, context_points, context_masks, "chest", (head + hips) * 0.35)
    neck = _context_point(torch, context_points, context_masks, "neck", chest * 0.3 + head * 0.7)

    torso_axis = hips - head
    torso_length = _norm(torch, torso_axis).clamp_min(1e-4)
    lean_x = latents[:, _target_index(target_names, "torso_lean_x")].clamp(-0.95, 0.95)
    lean_z = latents[:, _target_index(target_names, "torso_lean_z")].clamp(-0.95, 0.95)
    lean_y = torch.sqrt((1.0 - lean_x * lean_x - lean_z * lean_z).clamp_min(1e-4))
    hips_to_head = _normalize(
        torch,
        torch.stack((lean_x, lean_y, lean_z), dim=1),
        _constant(torch, WORLD_UP, dtype=dtype, device=device),
    )
    new_hips = head - hips_to_head * torso_length[:, None]
    moved = {
        "hips": new_hips,
        "spine": _lerp(head, new_hips, _torso_fraction(torch, spine, head, torso_axis)),
        "chest": _lerp(head, new_hips, _torso_fraction(torch, chest, head, torso_axis)),
        "neck": _lerp(head, new_hips, _torso_fraction(torch, neck, head, torso_axis)),
    }

    heading_sin = latents[:, _target_index(target_names, "torso_heading_sin")]
    heading_cos = latents[:, _target_index(target_names, "torso_heading_cos")]
    heading_norm = torch.sqrt(heading_sin * heading_sin + heading_cos * heading_cos).clamp_min(1e-6)
    heading_sin = heading_sin / heading_norm
    heading_cos = heading_cos / heading_norm
    forward = torch.stack((heading_sin, torch.zeros_like(heading_sin), heading_cos), dim=1)
    right = torch.stack((heading_cos, torch.zeros_like(heading_sin), -heading_sin), dim=1)
    up = _constant(torch, WORLD_UP, dtype=dtype, device=device)[None, :].expand_as(forward)

    shoulder_width = raw_features[:, feature_indices.shoulder_width]
    upper_arm = raw_features[:, feature_indices.upper_arm].clamp_min(1e-4)
    lower_arm = raw_features[:, feature_indices.lower_arm].clamp_min(1e-4)
    arm_length = raw_features[:, feature_indices.arm_length].clamp_min(0.1)
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
    left_shoulder = _decode_shoulder(
        latents,
        target_names=target_names,
        side="left",
        chest=moved["chest"],
        right=right,
        up=up,
        forward=forward,
        shoulder_width=shoulder_width,
        arm_length=arm_length,
    )
    right_shoulder = _decode_shoulder(
        latents,
        target_names=target_names,
        side="right",
        chest=moved["chest"],
        right=right,
        up=up,
        forward=forward,
        shoulder_width=shoulder_width,
        arm_length=arm_length,
    )
    left_elbow = _solve_elbow(
        torch,
        shoulder=left_shoulder,
        hand=left_hand,
        upper_arm=upper_arm,
        lower_arm=lower_arm,
        swivel_sin=latents[:, _target_index(target_names, "left_elbow_swivel_sin")],
        swivel_cos=latents[:, _target_index(target_names, "left_elbow_swivel_cos")],
    )
    right_elbow = _solve_elbow(
        torch,
        shoulder=right_shoulder,
        hand=right_hand,
        upper_arm=upper_arm,
        lower_arm=lower_arm,
        swivel_sin=latents[:, _target_index(target_names, "right_elbow_swivel_sin")],
        swivel_cos=latents[:, _target_index(target_names, "right_elbow_swivel_cos")],
    )
    points = torch.stack(
        (
            moved["chest"],
            moved["neck"],
            left_shoulder,
            right_shoulder,
            left_elbow,
            right_elbow,
        ),
        dim=1,
    )
    point_masks = _decoded_point_masks(torch, context_masks, dtype=dtype, device=device)
    return points, point_masks


def _decode_shoulder(
    latents: Any,
    *,
    target_names: Sequence[str],
    side: str,
    chest: Any,
    right: Any,
    up: Any,
    forward: Any,
    shoulder_width: Any,
    arm_length: Any,
) -> Any:
    sign = 1.0 if side == "left" else -1.0
    local_x = (
        sign * shoulder_width * 0.5
        + latents[:, _target_index(target_names, f"{side}_shoulder_lateral_delta")] * arm_length
    )
    local_y = latents[:, _target_index(target_names, f"{side}_shoulder_elevation")] * arm_length
    local_z = latents[:, _target_index(target_names, f"{side}_shoulder_protraction")] * arm_length
    return chest + right * local_x[:, None] + up * local_y[:, None] + forward * local_z[:, None]


def _solve_elbow(
    torch: Any,
    *,
    shoulder: Any,
    hand: Any,
    upper_arm: Any,
    lower_arm: Any,
    swivel_sin: Any,
    swivel_cos: Any,
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
    bend_hint = _bend_hint_from_swivel(
        torch,
        shoulder=shoulder,
        hand=hand,
        swivel_sin=swivel_sin,
        swivel_cos=swivel_cos,
    )
    bend_direction = _normalize(
        torch,
        _project_on_plane(torch, bend_hint, direction),
        _fallback_bend(torch, direction),
    )
    return shoulder + direction * along[:, None] + bend_direction * bend_height[:, None]


def _bend_hint_from_swivel(
    torch: Any,
    *,
    shoulder: Any,
    hand: Any,
    swivel_sin: Any,
    swivel_cos: Any,
) -> Any:
    dtype = shoulder.dtype
    device = shoulder.device
    arm_axis = _normalize(
        torch,
        hand - shoulder,
        _constant(torch, WORLD_UP, dtype=dtype, device=device),
    )
    base = _normalize(
        torch,
        _project_on_plane(
            torch, _constant(torch, WORLD_DOWN, dtype=dtype, device=device), arm_axis
        ),
        _fallback_bend(torch, arm_axis),
    )
    tangent = _normalize(
        torch,
        torch.cross(arm_axis, base, dim=1),
        _constant(torch, WORLD_RIGHT, dtype=dtype, device=device),
    )
    return _normalize(torch, tangent * swivel_sin[:, None] + base * swivel_cos[:, None], base)


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
    def context_mask(joint_name: str) -> Any:
        return context_masks[:, RUNTIME_DECODER_CONTEXT_JOINTS.index(joint_name)]

    chest_mask = context_mask("head") * context_mask("hips") * context_mask("chest")
    neck_mask = context_mask("head") * context_mask("hips") * context_mask("neck")
    left_shoulder_mask = chest_mask * context_mask("left_shoulder")
    right_shoulder_mask = chest_mask * context_mask("right_shoulder")
    left_elbow_mask = left_shoulder_mask * context_mask("left_hand")
    right_elbow_mask = right_shoulder_mask * context_mask("right_hand")
    if context_masks.numel() == 0:
        return torch.zeros((0, len(DECODED_REFERENCE_JOINTS)), dtype=dtype, device=device)
    return torch.stack(
        (
            chest_mask,
            neck_mask,
            left_shoulder_mask,
            right_shoulder_mask,
            left_elbow_mask,
            right_elbow_mask,
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
    zeros = torch.zeros_like(raw_features[:, indices.head_y])
    return torch.stack((zeros, raw_features[:, indices.head_y], zeros), dim=1)


def _feature_point(raw_features: Any, indices: tuple[int, int, int]) -> Any:
    return raw_features[:, [indices[0], indices[1], indices[2]]]


def _torso_fraction(torch: Any, position: Any, head: Any, head_to_hips: Any) -> Any:
    denominator = (head_to_hips * head_to_hips).sum(dim=1).clamp_min(1e-8)
    return (((position - head) * head_to_hips).sum(dim=1) / denominator).clamp(0.0, 1.0)


def _lerp(a: Any, b: Any, t: Any) -> Any:
    return a + (b - a) * t[:, None]


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
