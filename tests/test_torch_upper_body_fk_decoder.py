from __future__ import annotations

import pytest
from pytest import approx

from validation.vr_pose_validation.upper_body_ml import FK_TARGET_NAMES, feature_schema


def test_torch_fk_decoder_reconstructs_identity_upper_body_points() -> None:
    torch = pytest.importorskip("torch")

    from validation.vr_pose_validation.torch_upper_body_fk_decoder import (
        decode_upper_body_fk_points,
        fk_decoded_point_masks,
        upper_body_fk_feature_indices,
    )

    feature_names = feature_schema(1)
    feature_indices = upper_body_fk_feature_indices(feature_names)
    assert feature_indices is not None
    raw_features = torch.tensor([_feature_row(feature_names)], dtype=torch.float32)
    targets = torch.tensor([_identity_fk_targets()], dtype=torch.float32)
    masks = torch.ones_like(targets)

    points = decode_upper_body_fk_points(
        torch,
        targets,
        raw_features,
        feature_indices=feature_indices,
        target_names=FK_TARGET_NAMES,
    )
    point_masks = fk_decoded_point_masks(torch, masks, FK_TARGET_NAMES)

    assert point_masks.tolist() == [[1.0] * 6]
    assert points[0, 0].tolist() == approx((0.0, 1.32, 0.0), abs=1e-6)
    assert points[0, 1].tolist() == approx((0.0, 1.488, 0.0), abs=1e-6)
    assert points[0, 2].tolist() == approx((0.20, 1.32, 0.0), abs=1e-6)
    assert points[0, 3].tolist() == approx((-0.20, 1.32, 0.0), abs=1e-6)
    assert points[0, 4].tolist() == approx((0.50, 1.32, 0.0), abs=1e-6)
    assert points[0, 5].tolist() == approx((-0.50, 1.32, 0.0), abs=1e-6)


def test_torch_fk_decoder_has_finite_gradients() -> None:
    torch = pytest.importorskip("torch")

    from validation.vr_pose_validation.torch_upper_body_fk_decoder import (
        decode_upper_body_fk_points,
        upper_body_fk_feature_indices,
    )

    feature_names = feature_schema(1)
    feature_indices = upper_body_fk_feature_indices(feature_names)
    assert feature_indices is not None
    raw_features = torch.tensor([_feature_row(feature_names)], dtype=torch.float32)
    targets = torch.tensor([_identity_fk_targets()], dtype=torch.float32, requires_grad=True)

    points = decode_upper_body_fk_points(
        torch,
        targets,
        raw_features,
        feature_indices=feature_indices,
        target_names=FK_TARGET_NAMES,
    )
    loss = points.sum()
    loss.backward()

    assert targets.grad is not None
    assert torch.isfinite(targets.grad).all()


def _identity_fk_targets() -> list[float]:
    values = {name: 0.0 for name in FK_TARGET_NAMES}
    values["fk_hips_rel_y"] = -0.5
    identity_rotation6d = (0.0, 0.0, 1.0, 0.0, 1.0, 0.0)
    for name in FK_TARGET_NAMES:
        if "rotation6d" not in name:
            continue
        component = int(name.rsplit("_", maxsplit=1)[1])
        values[name] = identity_rotation6d[component]
    return [values[name] for name in FK_TARGET_NAMES]


def _feature_row(feature_names: list[str]) -> list[float]:
    values = {name: 0.0 for name in feature_names}
    values["t+00_head_position_y"] = 1.8
    values["calibration_human_height_m"] = 1.8
    values["calibration_head_to_neck_m"] = 0.10
    values["calibration_neck_to_chest_m"] = 0.14
    values["calibration_chest_to_spine_m"] = 0.20
    values["calibration_spine_to_hips_m"] = 0.22
    values["calibration_shoulder_width_m"] = 0.40
    values["calibration_upper_arm_m"] = 0.30
    return [values[name] for name in feature_names]
