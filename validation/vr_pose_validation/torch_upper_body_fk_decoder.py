"""Differentiable Torch FK decoder for upper-body rotation targets."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class UpperBodyFkFeatureIndices:
    head_y: int
    human_height: int
    head_to_neck: int
    neck_to_chest: int
    chest_to_spine: int
    spine_to_hips: int
    shoulder_width: int
    upper_arm: int


def upper_body_fk_feature_indices(
    feature_names: Sequence[str],
) -> UpperBodyFkFeatureIndices | None:
    center_prefix = "t+00" if "t+00_head_position_y" in feature_names else "t+000"
    required = (
        f"{center_prefix}_head_position_y",
        "calibration_human_height_m",
        "calibration_head_to_neck_m",
        "calibration_neck_to_chest_m",
        "calibration_chest_to_spine_m",
        "calibration_spine_to_hips_m",
        "calibration_shoulder_width_m",
        "calibration_upper_arm_m",
    )
    if any(name not in feature_names for name in required):
        return None
    return UpperBodyFkFeatureIndices(
        head_y=feature_names.index(f"{center_prefix}_head_position_y"),
        human_height=feature_names.index("calibration_human_height_m"),
        head_to_neck=feature_names.index("calibration_head_to_neck_m"),
        neck_to_chest=feature_names.index("calibration_neck_to_chest_m"),
        chest_to_spine=feature_names.index("calibration_chest_to_spine_m"),
        spine_to_hips=feature_names.index("calibration_spine_to_hips_m"),
        shoulder_width=feature_names.index("calibration_shoulder_width_m"),
        upper_arm=feature_names.index("calibration_upper_arm_m"),
    )


def decode_upper_body_fk_points(
    torch: Any,
    targets: Any,
    raw_features: Any,
    *,
    feature_indices: UpperBodyFkFeatureIndices,
    target_names: Sequence[str],
) -> Any:
    """Decode FK rotation targets to the standard local upper-body points."""

    dtype = targets.dtype
    device = targets.device
    zeros = torch.zeros_like(raw_features[:, feature_indices.head_y])
    head_origin = torch.stack((zeros, raw_features[:, feature_indices.head_y], zeros), dim=1)
    human_height = raw_features[:, feature_indices.human_height].clamp_min(0.1)
    hips = (
        head_origin + _vector_targets(targets, target_names, "fk_hips_rel") * human_height[:, None]
    )

    hips_rotation = _rotation6d_matrix(torch, targets, target_names, "fk_hips_rotation6d")
    spine_rotation = _compose_rotation(
        torch,
        hips_rotation,
        _rotation6d_matrix(torch, targets, target_names, "fk_spine_local_rotation6d"),
    )
    spine = hips + _rotate(
        torch,
        spine_rotation,
        _axis_offset(torch, raw_features[:, feature_indices.spine_to_hips], "y", dtype, device),
    )

    chest_rotation = _compose_rotation(
        torch,
        spine_rotation,
        _rotation6d_matrix(torch, targets, target_names, "fk_chest_local_rotation6d"),
    )
    chest = spine + _rotate(
        torch,
        chest_rotation,
        _axis_offset(torch, raw_features[:, feature_indices.chest_to_spine], "y", dtype, device),
    )

    head_rotation = _compose_rotation(
        torch,
        chest_rotation,
        _rotation6d_matrix(torch, targets, target_names, "fk_head_local_rotation6d"),
    )
    head = chest + _rotate(
        torch,
        head_rotation,
        _axis_offset(
            torch,
            raw_features[:, feature_indices.head_to_neck]
            + raw_features[:, feature_indices.neck_to_chest],
            "y",
            dtype,
            device,
        ),
    )
    neck = chest + 0.7 * (head - chest)

    left_shoulder_rotation = _compose_rotation(
        torch,
        chest_rotation,
        _rotation6d_matrix(torch, targets, target_names, "fk_left_shoulder_local_rotation6d"),
    )
    right_shoulder_rotation = _compose_rotation(
        torch,
        chest_rotation,
        _rotation6d_matrix(torch, targets, target_names, "fk_right_shoulder_local_rotation6d"),
    )
    half_shoulder_width = raw_features[:, feature_indices.shoulder_width] * 0.5
    left_shoulder = chest + _rotate(
        torch,
        left_shoulder_rotation,
        _axis_offset(torch, half_shoulder_width, "x", dtype, device),
    )
    right_shoulder = chest + _rotate(
        torch,
        right_shoulder_rotation,
        _axis_offset(torch, -half_shoulder_width, "x", dtype, device),
    )

    left_elbow_rotation = _compose_rotation(
        torch,
        left_shoulder_rotation,
        _rotation6d_matrix(torch, targets, target_names, "fk_left_elbow_local_rotation6d"),
    )
    right_elbow_rotation = _compose_rotation(
        torch,
        right_shoulder_rotation,
        _rotation6d_matrix(torch, targets, target_names, "fk_right_elbow_local_rotation6d"),
    )
    upper_arm = raw_features[:, feature_indices.upper_arm]
    left_elbow = left_shoulder + _rotate(
        torch,
        left_elbow_rotation,
        _axis_offset(torch, upper_arm, "x", dtype, device),
    )
    right_elbow = right_shoulder + _rotate(
        torch,
        right_elbow_rotation,
        _axis_offset(torch, -upper_arm, "x", dtype, device),
    )

    return torch.stack(
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


def fk_decoded_point_masks(torch: Any, target_mask: Any, target_names: Sequence[str]) -> Any:
    global_mask = _target_group_mask(
        target_mask,
        target_names,
        ("fk_hips_rel", "fk_hips_rotation6d"),
    )
    spine_mask = global_mask * _target_group_mask(
        target_mask,
        target_names,
        ("fk_spine_local_rotation6d",),
    )
    chest_mask = spine_mask * _target_group_mask(
        target_mask,
        target_names,
        ("fk_chest_local_rotation6d",),
    )
    head_mask = chest_mask * _target_group_mask(
        target_mask,
        target_names,
        ("fk_head_local_rotation6d",),
    )
    left_shoulder_mask = chest_mask * _target_group_mask(
        target_mask,
        target_names,
        ("fk_left_shoulder_local_rotation6d",),
    )
    right_shoulder_mask = chest_mask * _target_group_mask(
        target_mask,
        target_names,
        ("fk_right_shoulder_local_rotation6d",),
    )
    left_elbow_mask = left_shoulder_mask * _target_group_mask(
        target_mask,
        target_names,
        ("fk_left_elbow_local_rotation6d",),
    )
    right_elbow_mask = right_shoulder_mask * _target_group_mask(
        target_mask,
        target_names,
        ("fk_right_elbow_local_rotation6d",),
    )
    return torch.stack(
        (
            chest_mask,
            head_mask,
            left_shoulder_mask,
            right_shoulder_mask,
            left_elbow_mask,
            right_elbow_mask,
        ),
        dim=1,
    )


def _rotation6d_matrix(
    torch: Any,
    targets: Any,
    target_names: Sequence[str],
    prefix: str,
) -> Any:
    forward = _normalize(
        torch,
        _sequence_targets(targets, target_names, prefix, 0, 3),
        fallback=(0.0, 0.0, 1.0),
    )
    raw_up = _sequence_targets(targets, target_names, prefix, 3, 6)
    right = _normalize(torch, torch.cross(raw_up, forward, dim=1), fallback=(1.0, 0.0, 0.0))
    up = _normalize(torch, torch.cross(forward, right, dim=1), fallback=(0.0, 1.0, 0.0))
    return torch.stack((right, up, forward), dim=2)


def _vector_targets(targets: Any, target_names: Sequence[str], prefix: str) -> Any:
    return targets[
        :,
        [
            target_names.index(f"{prefix}_x"),
            target_names.index(f"{prefix}_y"),
            target_names.index(f"{prefix}_z"),
        ],
    ]


def _sequence_targets(
    targets: Any,
    target_names: Sequence[str],
    prefix: str,
    start: int,
    stop: int,
) -> Any:
    return targets[:, [target_names.index(f"{prefix}_{index}") for index in range(start, stop)]]


def _target_group_mask(
    target_mask: Any, target_names: Sequence[str], prefixes: Sequence[str]
) -> Any:
    mask = None
    for prefix in prefixes:
        width = 3 if prefix.endswith("_rel") else 6
        for index in range(width):
            column_name = f"{prefix}_{('xyz'[index] if width == 3 else index)}"
            column = target_mask[:, target_names.index(column_name)]
            mask = column if mask is None else mask * column
    if mask is None:
        raise ValueError("Expected at least one target mask prefix")
    return mask


def _compose_rotation(torch: Any, parent: Any, local: Any) -> Any:
    return torch.bmm(parent, local)


def _rotate(torch: Any, rotation: Any, vector: Any) -> Any:
    return torch.bmm(rotation, vector[:, :, None]).squeeze(2)


def _axis_offset(
    torch: Any,
    values: Any,
    axis: str,
    dtype: Any,
    device: Any,
) -> Any:
    zeros = torch.zeros_like(values, dtype=dtype, device=device)
    if axis == "x":
        return torch.stack((values, zeros, zeros), dim=1)
    if axis == "y":
        return torch.stack((zeros, values, zeros), dim=1)
    return torch.stack((zeros, zeros, values), dim=1)


def _normalize(torch: Any, vector: Any, *, fallback: tuple[float, float, float]) -> Any:
    magnitude = torch.linalg.vector_norm(vector, dim=1, keepdim=True)
    fallback_tensor = torch.tensor(fallback, dtype=vector.dtype, device=vector.device)
    normalized = vector / magnitude.clamp_min(1e-6)
    return torch.where(magnitude > 1e-6, normalized, fallback_tensor[None, :])
