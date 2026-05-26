from __future__ import annotations

import pytest
from pytest import approx

from validation.vr_pose_validation.geometry import Transform, look_rotation
from validation.vr_pose_validation.types import AvatarCalibration, HumanoidPoseFrame, JointPose
from validation.vr_pose_validation.upper_body_ml import (
    DECODED_REFERENCE_JOINTS,
    RUNTIME_DECODER_CONTEXT_JOINTS,
    TARGET_NAMES,
    UpperBodyLatentPrediction,
    apply_upper_body_latents,
    calibration_feature_values,
    feature_schema,
)


def test_torch_runtime_decoder_matches_python_latent_application() -> None:
    torch = pytest.importorskip("torch")

    from validation.vr_pose_validation.torch_upper_body_decoder import (
        decode_upper_body_runtime_points,
        runtime_decoder_feature_indices,
    )

    calibration = _calibration()
    base_pose = _base_pose()
    prediction = _prediction()
    adjusted = apply_upper_body_latents(base_pose, prediction, calibration=calibration)
    feature_names = feature_schema(1)
    feature_indices = runtime_decoder_feature_indices(feature_names)
    assert feature_indices is not None

    latents = torch.tensor(
        [[prediction.values[name] for name in TARGET_NAMES]],
        dtype=torch.float32,
    )
    raw_features = torch.tensor(
        [_feature_row(feature_names, base_pose, calibration)],
        dtype=torch.float32,
    )
    context_points, context_masks = _context_tensors(torch, base_pose)

    decoded, decoded_masks = decode_upper_body_runtime_points(
        torch,
        latents,
        raw_features,
        context_points,
        context_masks,
        feature_indices=feature_indices,
        target_names=TARGET_NAMES,
    )

    assert decoded_masks.tolist() == [[1.0] * len(DECODED_REFERENCE_JOINTS)]
    for joint_index, joint_name in enumerate(DECODED_REFERENCE_JOINTS):
        assert decoded[0, joint_index].tolist() == approx(
            adjusted.joints[joint_name].transform.position,
            abs=1e-5,
        )


def test_torch_runtime_decoder_keeps_pose_loss_differentiable() -> None:
    torch = pytest.importorskip("torch")

    from validation.vr_pose_validation.torch_upper_body_decoder import (
        decode_upper_body_runtime_points,
        runtime_decoder_feature_indices,
    )

    calibration = _calibration()
    base_pose = _base_pose()
    prediction = _prediction()
    feature_names = feature_schema(1)
    feature_indices = runtime_decoder_feature_indices(feature_names)
    assert feature_indices is not None
    latents = torch.tensor(
        [[prediction.values[name] for name in TARGET_NAMES]],
        dtype=torch.float32,
        requires_grad=True,
    )
    raw_features = torch.tensor(
        [_feature_row(feature_names, base_pose, calibration)],
        dtype=torch.float32,
    )
    context_points, context_masks = _context_tensors(torch, base_pose)

    decoded, decoded_masks = decode_upper_body_runtime_points(
        torch,
        latents,
        raw_features,
        context_points,
        context_masks,
        feature_indices=feature_indices,
        target_names=TARGET_NAMES,
    )
    loss = (decoded * decoded_masks[:, :, None]).sum()
    loss.backward()

    assert latents.grad is not None
    assert torch.isfinite(latents.grad).all()
    assert float(latents.grad.abs().sum()) > 0.0


def _base_pose() -> HumanoidPoseFrame:
    rotation = look_rotation((0.0, 0.0, 1.0))
    joints = {
        "hips": Transform(position=(0.0, 0.92, 0.0), rotation=rotation),
        "spine": Transform(position=(0.0, 1.12, 0.0), rotation=rotation),
        "chest": Transform(position=(0.0, 1.34, 0.0), rotation=rotation),
        "neck": Transform(position=(0.0, 1.48, 0.0), rotation=rotation),
        "head": Transform(position=(0.0, 1.62, 0.0), rotation=rotation),
        "left_shoulder": Transform(position=(0.21, 1.38, 0.0), rotation=rotation),
        "right_shoulder": Transform(position=(-0.21, 1.38, 0.0), rotation=rotation),
        "left_elbow": Transform(position=(0.34, 1.14, -0.08), rotation=rotation),
        "right_elbow": Transform(position=(-0.34, 1.14, -0.08), rotation=rotation),
        "left_hand": Transform(position=(0.46, 1.05, 0.18), rotation=rotation),
        "right_hand": Transform(position=(-0.43, 1.08, 0.16), rotation=rotation),
    }
    return HumanoidPoseFrame(
        timestamp_seconds=0.0,
        root=joints["hips"],
        joints={name: JointPose(transform=transform) for name, transform in joints.items()},
    )


def _prediction() -> UpperBodyLatentPrediction:
    values = {name: 0.0 for name in TARGET_NAMES}
    values.update(
        {
            "torso_heading_sin": 0.18,
            "torso_heading_cos": 0.9836666,
            "torso_lean_x": -0.08,
            "torso_lean_z": 0.22,
            "left_shoulder_lateral_delta": 0.03,
            "left_shoulder_elevation": 0.11,
            "left_shoulder_protraction": 0.06,
            "right_shoulder_lateral_delta": -0.02,
            "right_shoulder_elevation": 0.05,
            "right_shoulder_protraction": 0.04,
            "left_elbow_swivel_sin": 0.26,
            "left_elbow_swivel_cos": 0.965608,
            "right_elbow_swivel_sin": -0.18,
            "right_elbow_swivel_cos": 0.983667,
        }
    )
    return UpperBodyLatentPrediction(
        values=values,
        masks={name: 1.0 for name in TARGET_NAMES},
    )


def _calibration() -> AvatarCalibration:
    return AvatarCalibration(
        human_height_m=1.8,
        bone_lengths_m={
            "shoulder_width": 0.42,
            "upper_arm": 0.30,
            "lower_arm": 0.25,
            "arm": 0.55,
        },
    )


def _feature_row(
    feature_names: list[str],
    pose: HumanoidPoseFrame,
    calibration: AvatarCalibration,
) -> list[float]:
    values = {name: 0.0 for name in feature_names}
    for role, joint_name in (
        ("head", "head"),
        ("left_hand", "left_hand"),
        ("right_hand", "right_hand"),
    ):
        position = pose.joints[joint_name].transform.position
        values[f"t+00_{role}_position_x"] = position[0]
        values[f"t+00_{role}_position_y"] = position[1]
        values[f"t+00_{role}_position_z"] = position[2]
        values[f"t+00_{role}_valid"] = 1.0
    for name, value in zip(
        feature_names[-len(calibration_feature_values(calibration)) :],
        calibration_feature_values(calibration),
        strict=True,
    ):
        values[name] = value
    return [values[name] for name in feature_names]


def _context_tensors(
    torch,
    pose: HumanoidPoseFrame,
):
    points = []
    masks = []
    for joint_name in RUNTIME_DECODER_CONTEXT_JOINTS:
        joint = pose.joints.get(joint_name)
        points.append(joint.transform.position if joint is not None else (0.0, 0.0, 0.0))
        masks.append(1.0 if joint is not None else 0.0)
    return (
        torch.tensor([points], dtype=torch.float32),
        torch.tensor([masks], dtype=torch.float32),
    )
