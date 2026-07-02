from __future__ import annotations

import struct
from copy import deepcopy
from pathlib import Path
from typing import cast

import pytest

from validation.vr_pose_validation.geometry import (
    Transform,
    axis_angle_quaternion,
    distance,
    dot,
    look_rotation,
    rotate_vector,
)
from validation.vr_pose_validation.sparseposer_live import (
    SPARSEPOSER_INPUT_ROLES,
    SPARSEPOSER_LIVE_PROFILE,
    SPARSEPOSER_WINDOW_SIZE,
    SparsePoserBridgeConfig,
    SparsePoserLiveBackend,
    SparsePoserMissingRolesError,
    _apply_sparse_input_ik,
    _apply_sparseposer_role_offset,
    _calibrated_body_scale,
    _source_pose_to_target,
    _target_display_root_transform,
    godot_position_to_sparseposer,
    godot_quaternion_to_sparseposer,
    pack_sparseposer_window,
    source_tracker_position_to_sparseposer_input,
    sparseposer_position_to_godot,
    sparseposer_quaternion_to_godot,
    unpack_sparseposer_rotations,
)


def test_sparseposer_coordinate_conversion_round_trips() -> None:
    position = (0.25, 1.5, -0.75)
    rotation = (0.1, -0.2, 0.3, 0.9)

    assert sparseposer_position_to_godot(godot_position_to_sparseposer(position)) == position
    assert sparseposer_quaternion_to_godot(godot_quaternion_to_sparseposer(rotation)) == rotation


def test_sparseposer_live_input_position_conversion_matches_unity_z_flip() -> None:
    assert source_tracker_position_to_sparseposer_input((0.0, 0.0, 0.5)) == (
        0.0,
        0.0,
        -0.5,
    )


def test_sparseposer_window_pack_and_rotation_unpack() -> None:
    frame = tuple(
        ((float(index), 1.0, -1.0), (1.0, 0.0, 0.0, 0.0))
        for index, _role in enumerate(SPARSEPOSER_INPUT_ROLES)
    )
    packet = pack_sparseposer_window([frame for _index in range(SPARSEPOSER_WINDOW_SIZE)])
    assert len(packet) == SPARSEPOSER_WINDOW_SIZE * len(SPARSEPOSER_INPUT_ROLES) * 7 * 4

    payload = struct.pack("<8f", 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    assert unpack_sparseposer_rotations(payload, joint_count=2) == (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
    )


def test_sparseposer_backend_reports_missing_required_roles(tmp_path: Path) -> None:
    backend = SparsePoserLiveBackend(
        SparsePoserBridgeConfig(sparseposer_reference_bvh=_write_reference_bvh(tmp_path)),
        engine=_FakeEngine(),
    )

    with pytest.raises(SparsePoserMissingRolesError) as exc_info:
        backend.estimate_from_message(_message(include_optional=False, include_alignment=False))

    assert exc_info.value.missing_roles == ("pelvis", "left_foot", "right_foot")


def test_sparseposer_target_root_uses_playspace_forward_not_head_yaw() -> None:
    head_rotation = axis_angle_quaternion((0.0, 1.0, 0.0), 90.0)
    root = _target_display_root_transform(
        {
            "pelvis": Transform(position=(0.0, 0.9, 0.0)),
            "head": Transform(position=(0.0, 1.6, 0.0), rotation=head_rotation),
        }
    )
    root_forward = rotate_vector(root.rotation, (0.0, 0.0, 1.0))

    assert dot(root_forward, (0.0, 0.0, 1.0)) == pytest.approx(1.0)


def test_sparseposer_anatomical_source_negative_z_maps_to_target_forward() -> None:
    target_root = Transform(rotation=look_rotation((0.0, 0.0, -1.0)))
    target_forward = rotate_vector(target_root.rotation, (0.0, 0.0, 1.0))

    mapped = _source_pose_to_target(
        Transform(position=(0.0, 0.0, -0.5)),
        source_root=Transform(),
        target_root=target_root,
        source_to_target_rotation=(0.0, 0.0, 0.0, 1.0),
    )

    assert dot(mapped.position, target_forward) == pytest.approx(0.5)


def test_sparseposer_anatomical_source_left_maps_to_target_left() -> None:
    target_root = Transform(rotation=look_rotation((0.0, 0.0, 1.0)))

    mapped = _source_pose_to_target(
        Transform(position=(0.5, 0.0, 0.0)),
        source_root=Transform(),
        target_root=target_root,
        source_to_target_rotation=(0.0, 0.0, 0.0, 1.0),
    )

    assert dot(mapped.position, (1.0, 0.0, 0.0)) == pytest.approx(-0.5)


def test_sparseposer_backend_returns_tpose_before_alignment(tmp_path: Path) -> None:
    engine = _FakeEngine()
    backend = SparsePoserLiveBackend(
        SparsePoserBridgeConfig(
            sparseposer_reference_bvh=_write_reference_bvh(tmp_path),
            body_scale_mode="extent",
        ),
        engine=engine,
    )

    frame = backend.estimate_from_message(_message(include_optional=True, include_alignment=False))

    assert frame["skeleton_profile"] == SPARSEPOSER_LIVE_PROFILE
    assert frame["calibration_required"] == "sparseposer_alignment"
    assert "left_toe" in frame["joints"]
    assert engine.packets == []


def test_sparseposer_tpose_ignores_raw_pelvis_tracker_roll(tmp_path: Path) -> None:
    engine = _FakeEngine()
    backend = SparsePoserLiveBackend(
        SparsePoserBridgeConfig(
            sparseposer_reference_bvh=_write_reference_bvh(tmp_path),
            body_scale_mode="extent",
        ),
        engine=engine,
    )
    message = _message(include_optional=True, include_alignment=False)
    sample = cast(dict[str, object], message["sample"])
    optional = cast(dict[str, object], sample["optional_trackers"])
    pelvis = cast(dict[str, object], optional["pelvis"])
    transform = cast(dict[str, object], pelvis["transform"])
    transform["rotation"] = [0.0, 0.0, 0.70710678, 0.70710678]

    frame = backend.estimate_from_message(message)

    hips = frame["joints"]["hips"]["transform"]["position"]
    head = frame["joints"]["head"]["transform"]["position"]
    assert head[1] - hips[1] > 0.4
    assert abs(head[0] - hips[0]) < 0.05


def test_sparseposer_backend_calls_engine_after_alignment(tmp_path: Path) -> None:
    engine = _FakeEngine()
    backend = SparsePoserLiveBackend(
        SparsePoserBridgeConfig(sparseposer_reference_bvh=_write_reference_bvh(tmp_path)),
        engine=engine,
    )

    frame = backend.estimate_from_message(_message(include_optional=True, include_alignment=True))

    assert frame["skeleton_profile"] == SPARSEPOSER_LIVE_PROFILE
    assert "debug_bones" in frame
    assert "debug_markers" in frame
    assert "left_wrist" in frame["joints"]
    assert "left_hand" in frame["joints"]
    assert len(engine.packets) == 1


def test_sparseposer_backend_adds_controller_hand_endpoint_joints(tmp_path: Path) -> None:
    engine = _FakeEngine()
    backend = SparsePoserLiveBackend(
        SparsePoserBridgeConfig(sparseposer_reference_bvh=_write_reference_bvh(tmp_path)),
        engine=engine,
    )
    message = _message(include_optional=True, include_alignment=True)

    frame = backend.estimate_from_message(message)

    left_hand = frame["joints"]["left_hand"]["transform"]["position"]
    right_hand = frame["joints"]["right_hand"]["transform"]["position"]
    assert left_hand == pytest.approx([-0.4, 1.2, 0.0])
    assert right_hand == pytest.approx([0.4, 1.2, 0.0])
    assert ["left_wrist", "left_hand"] in frame["debug_bones"]
    assert ["right_wrist", "right_hand"] in frame["debug_bones"]


def test_sparseposer_backend_returns_input_markers_in_target_side_convention(
    tmp_path: Path,
) -> None:
    engine = _FakeEngine()
    backend = SparsePoserLiveBackend(
        SparsePoserBridgeConfig(sparseposer_reference_bvh=_write_reference_bvh(tmp_path)),
        engine=engine,
    )

    frame = backend.estimate_from_message(_message(include_optional=True, include_alignment=True))

    markers = {marker["role"]: marker for marker in frame["debug_markers"]}
    assert markers["left_hand"]["label"] == "sp left_wrist"
    assert markers["right_hand"]["label"] == "sp right_wrist"
    assert markers["left_hand"]["transform"]["position"][0] < 0.0
    assert markers["right_hand"]["transform"]["position"][0] > 0.0
    assert markers["left_foot"]["transform"]["position"][0] < 0.0
    assert markers["right_foot"]["transform"]["position"][0] > 0.0
    assert frame["joints"]["left_wrist"]["transform"]["position"][0] < 0.0
    assert frame["joints"]["right_wrist"]["transform"]["position"][0] > 0.0
    assert frame["joints"]["left_toe"]["transform"]["position"][0] < 0.0
    assert frame["joints"]["right_toe"]["transform"]["position"][0] > 0.0


def test_sparseposer_role_offset_is_local_to_tracker_rotation() -> None:
    tracker = Transform(
        position=(1.0, 0.0, 0.0),
        rotation=axis_angle_quaternion((0.0, 1.0, 0.0), 90.0),
    )
    offset = Transform(position=(0.0, 0.0, 0.5))

    retargeted = _apply_sparseposer_role_offset(
        tracker,
        offset,
        root_align_rotation=(0.0, 0.0, 0.0, 1.0),
    )

    assert retargeted.position == pytest.approx((1.5, 0.0, 0.0))


def test_sparseposer_backend_uses_calibrated_position_offsets_without_pinning(
    tmp_path: Path,
) -> None:
    engine = _FakeEngine()
    backend = SparsePoserLiveBackend(
        SparsePoserBridgeConfig(sparseposer_reference_bvh=_write_reference_bvh(tmp_path)),
        engine=engine,
    )
    message = _message(include_optional=True, include_alignment=True)
    sample = cast(dict[str, object], message["sample"])
    left_hand = cast(dict[str, object], sample["left_hand"])
    left_transform = cast(dict[str, object], left_hand["transform"])
    left_transform["position"] = [-0.35, 1.1, -0.25]
    optional = cast(dict[str, object], sample["optional_trackers"])
    right_foot = cast(dict[str, object], optional["right_foot"])
    right_foot_transform = cast(dict[str, object], right_foot["transform"])
    right_foot_transform["position"] = [0.18, 0.08, -0.3]

    frame = backend.estimate_from_message(message)
    last_frame = _unpack_last_sparseposer_input_frame(engine.packets[-1])

    assert frame["joints"]["left_wrist"]["transform"]["position"] != pytest.approx(
        [-0.35, 1.1, -0.25],
        abs=1e-4,
    )
    assert frame["joints"]["right_toe"]["transform"]["position"] != pytest.approx(
        [0.18, 0.08, -0.3],
        abs=1e-4,
    )
    assert backend._role_offsets["left_hand"].position != pytest.approx((0.0, 0.0, 0.0))
    assert backend._role_offsets["right_foot"].position != pytest.approx((0.0, 0.0, 0.0))
    assert abs(last_frame["left_hand"][0][0]) > 0.01


def test_sparseposer_backend_sends_target_left_to_unity_sparse_left(tmp_path: Path) -> None:
    engine = _FakeEngine()
    backend = SparsePoserLiveBackend(
        SparsePoserBridgeConfig(sparseposer_reference_bvh=_write_reference_bvh(tmp_path)),
        engine=engine,
    )

    backend.estimate_from_message(_message(include_optional=True, include_alignment=True))

    last_frame = _unpack_last_sparseposer_input_frame(engine.packets[-1])
    assert last_frame["left_hand"][0][0] > last_frame["right_hand"][0][0]
    assert last_frame["left_foot"][0][0] > last_frame["right_foot"][0][0]


def test_sparseposer_backend_live_motion_preserves_source_axis_signs(tmp_path: Path) -> None:
    engine = _FakeEngine()
    backend = SparsePoserLiveBackend(
        SparsePoserBridgeConfig(sparseposer_reference_bvh=_write_reference_bvh(tmp_path)),
        engine=engine,
    )
    base = _message(include_optional=True, include_alignment=True)
    backend.estimate_from_message(base)
    base_frame = _unpack_last_sparseposer_input_frame(engine.packets[-1])
    moved = _message(include_optional=True, include_alignment=True)
    moved["calibration"] = deepcopy(base["calibration"])
    sample = cast(dict[str, object], moved["sample"])
    left_hand = cast(dict[str, object], sample["left_hand"])
    left_hand_transform = cast(dict[str, object], left_hand["transform"])
    left_hand_transform["position"] = [-0.6, 1.0, 0.0]
    optional = cast(dict[str, object], sample["optional_trackers"])
    left_foot = cast(dict[str, object], optional["left_foot"])
    left_foot_transform = cast(dict[str, object], left_foot["transform"])
    left_foot_transform["position"] = [-0.3, 0.05, 0.1]
    right_hand = cast(dict[str, object], sample["right_hand"])
    right_hand_transform = cast(dict[str, object], right_hand["transform"])
    right_hand_transform["position"] = [0.4, 1.2, -0.2]

    backend.estimate_from_message(moved)

    moved_frame = _unpack_last_sparseposer_input_frame(engine.packets[-1])
    assert moved_frame["left_hand"][0][0] > base_frame["left_hand"][0][0]
    assert moved_frame["left_hand"][0][1] < base_frame["left_hand"][0][1]
    assert moved_frame["left_foot"][0][0] > base_frame["left_foot"][0][0]
    assert moved_frame["right_hand"][0][2] < base_frame["right_hand"][0][2]


def test_sparseposer_backend_pelvis_yaw_is_not_display_mirrored(tmp_path: Path) -> None:
    engine = _FakeEngine()
    backend = SparsePoserLiveBackend(
        SparsePoserBridgeConfig(sparseposer_reference_bvh=_write_reference_bvh(tmp_path)),
        engine=engine,
    )
    base = _message(include_optional=True, include_alignment=True)
    backend.estimate_from_message(base)
    turned = _message(include_optional=True, include_alignment=True)
    turned["calibration"] = deepcopy(base["calibration"])
    sample = cast(dict[str, object], turned["sample"])
    optional = cast(dict[str, object], sample["optional_trackers"])
    pelvis = cast(dict[str, object], optional["pelvis"])
    pelvis_transform = cast(dict[str, object], pelvis["transform"])
    pelvis_transform["rotation"] = list(axis_angle_quaternion((0.0, 1.0, 0.0), 30.0))

    frame = backend.estimate_from_message(turned)

    last_frame = _unpack_last_sparseposer_input_frame(engine.packets[-1])
    assert last_frame["pelvis"][1][2] < 0.0
    hips = frame["joints"]["hips"]["transform"]["position"]
    right_hip = frame["joints"]["right_hip"]["transform"]["position"]
    assert right_hip[2] < hips[2]


def test_sparseposer_backend_calibrates_body_scale_from_tpose_span(tmp_path: Path) -> None:
    engine = _FakeEngine()
    backend = SparsePoserLiveBackend(
        SparsePoserBridgeConfig(
            sparseposer_reference_bvh=_write_reference_bvh(tmp_path),
            body_scale_mode="extent",
        ),
        engine=engine,
    )
    message = _message(include_optional=True, include_alignment=True)
    calibration = cast(dict[str, object], message["calibration"])
    alignment = cast(dict[str, object], calibration["sparseposer_alignment"])
    role_transforms = cast(dict[str, object], alignment["role_transforms"])
    cast(dict[str, object], role_transforms["left_hand"])["position"] = [-0.8, 1.2, 0.0]
    cast(dict[str, object], role_transforms["right_hand"])["position"] = [0.8, 1.2, 0.0]
    sample = cast(dict[str, object], message["sample"])
    cast(dict[str, object], cast(dict[str, object], sample["left_hand"])["transform"])[
        "position"
    ] = [-0.8, 1.2, 0.0]
    cast(dict[str, object], cast(dict[str, object], sample["right_hand"])["transform"])[
        "position"
    ] = [0.8, 1.2, 0.0]

    backend.estimate_from_message(message)

    assert backend._body_scale > 1.0
    last_frame = _unpack_last_sparseposer_input_frame(engine.packets[-1])
    encoded_span = abs(last_frame["right_hand"][0][0] - last_frame["left_hand"][0][0])
    assert encoded_span < 1.6


def test_sparseposer_backend_treats_pelvis_tracker_as_mounted_offset(
    tmp_path: Path,
) -> None:
    engine = _FakeEngine()
    backend = SparsePoserLiveBackend(
        SparsePoserBridgeConfig(sparseposer_reference_bvh=_write_reference_bvh(tmp_path)),
        engine=engine,
    )
    message = _message(include_optional=True, include_alignment=True)
    sample = cast(dict[str, object], message["sample"])
    optional = cast(dict[str, object], sample["optional_trackers"])
    pelvis = cast(dict[str, object], optional["pelvis"])
    pelvis_transform = cast(dict[str, object], pelvis["transform"])
    pelvis_transform["position"] = [0.0, 0.9, 0.35]

    frame = backend.estimate_from_message(message)

    root_position = frame["root"]["position"]
    assert root_position != pytest.approx([0.0, 0.9, 0.35], abs=1e-4)
    assert backend._role_offsets["pelvis"].position != pytest.approx((0.0, 0.0, 0.0))
    assert frame["debug_metadata"]["pelvis_offset"] != pytest.approx([0.0, 0.0, 0.0])


def test_sparseposer_backend_accepts_nested_live_tracker_alignment(tmp_path: Path) -> None:
    engine = _FakeEngine()
    backend = SparsePoserLiveBackend(
        SparsePoserBridgeConfig(sparseposer_reference_bvh=_write_reference_bvh(tmp_path)),
        engine=engine,
    )
    message = _message(include_optional=True, include_alignment=True)
    calibration = cast(dict[str, object], message["calibration"])
    alignment = calibration.pop("sparseposer_alignment")
    calibration["live_tracker_calibration"] = {"sparseposer_alignment": alignment}

    frame = backend.estimate_from_message(message)

    assert frame["skeleton_profile"] == SPARSEPOSER_LIVE_PROFILE
    assert len(engine.packets) == 1


def test_sparseposer_backend_applies_unity_quaternion_continuity(tmp_path: Path) -> None:
    engine = _FakeEngine()
    backend = SparsePoserLiveBackend(
        SparsePoserBridgeConfig(sparseposer_reference_bvh=_write_reference_bvh(tmp_path)),
        engine=engine,
    )
    backend.estimate_from_message(_message(include_optional=True, include_alignment=True))
    first_values = struct.unpack(
        f"<{SPARSEPOSER_WINDOW_SIZE * len(SPARSEPOSER_INPUT_ROLES) * 7}f",
        engine.packets[-1],
    )
    message = _message(include_optional=True, include_alignment=True)
    sample = cast(dict[str, object], message["sample"])
    optional = cast(dict[str, object], sample["optional_trackers"])
    pelvis = cast(dict[str, object], optional["pelvis"])
    transform = cast(dict[str, object], pelvis["transform"])
    transform["rotation"] = [0.0, 0.0, 0.0, -1.0]

    backend.estimate_from_message(message)

    values = struct.unpack(
        f"<{SPARSEPOSER_WINDOW_SIZE * len(SPARSEPOSER_INPUT_ROLES) * 7}f",
        engine.packets[-1],
    )
    last_frame_offset = (SPARSEPOSER_WINDOW_SIZE - 1) * len(SPARSEPOSER_INPUT_ROLES) * 7
    pelvis_rotation_offset = last_frame_offset + 3
    first_rotation = first_values[pelvis_rotation_offset : pelvis_rotation_offset + 4]
    second_rotation = values[pelvis_rotation_offset : pelvis_rotation_offset + 4]
    assert sum(
        left * right for left, right in zip(first_rotation, second_rotation, strict=True)
    ) == pytest.approx(1.0)


def test_sparseposer_input_markers_show_calibrated_sparse_joint_offsets(tmp_path: Path) -> None:
    engine = _FakeEngine()
    backend = SparsePoserLiveBackend(
        SparsePoserBridgeConfig(sparseposer_reference_bvh=_write_reference_bvh(tmp_path)),
        engine=engine,
    )
    first_frame = backend.estimate_from_message(
        _message(include_optional=True, include_alignment=True)
    )
    first_markers = {marker["role"]: marker for marker in first_frame["debug_markers"]}
    first_left_hand = first_markers["left_hand"]["transform"]["position"]

    message = _message(include_optional=True, include_alignment=True)
    sample = cast(dict[str, object], message["sample"])
    left_hand = cast(dict[str, object], sample["left_hand"])
    transform = cast(dict[str, object], left_hand["transform"])
    transform["position"] = [-0.65, 0.95, 0.0]
    transform["rotation"] = [0.0, 0.0, 1.0, 0.0]

    frame = backend.estimate_from_message(message)
    markers = {marker["role"]: marker for marker in frame["debug_markers"]}
    moved_left_hand = markers["left_hand"]["transform"]["position"]

    assert moved_left_hand != pytest.approx([-0.65, 0.95, 0.0], abs=1e-4)
    assert moved_left_hand != pytest.approx(first_left_hand, abs=1e-4)


def test_sparseposer_calibrated_body_scale_uses_widest_sparse_extent() -> None:
    source = {
        "pelvis": Transform(position=(0.0, 0.9, 0.0)),
        "head": Transform(position=(0.0, 1.6, 0.0)),
        "left_hand": Transform(position=(-0.5, 1.2, 0.0)),
        "right_hand": Transform(position=(0.5, 1.2, 0.0)),
    }
    target = {
        "pelvis": Transform(position=(0.0, 0.9, 0.0)),
        "head": Transform(position=(0.0, 1.6, 0.0)),
        "left_hand": Transform(position=(-0.65, 1.2, 0.0)),
        "right_hand": Transform(position=(0.65, 1.2, 0.0)),
    }

    assert _calibrated_body_scale(target, source, mode="extent") == pytest.approx(
        1.25,
        abs=0.001,
    )


def test_sparseposer_endpoint_ik_preserves_arm_lengths() -> None:
    joints = _joint_map(
        {
            "left_shoulder": (0.0, 1.4, 0.0),
            "left_elbow": (-0.25, 1.1, 0.0),
            "left_wrist": (-0.5, 0.9, 0.0),
        }
    )
    upper_length = distance(
        _joint_position_for_test(joints, "left_shoulder"),
        _joint_position_for_test(joints, "left_elbow"),
    )
    lower_length = distance(
        _joint_position_for_test(joints, "left_elbow"),
        _joint_position_for_test(joints, "left_wrist"),
    )

    _apply_sparse_input_ik(
        joints,
        {"left_hand": Transform(position=(-0.2, 1.2, 0.45))},
    )

    assert _joint_position_for_test(joints, "left_wrist") == pytest.approx((-0.2, 1.2, 0.45))
    assert distance(
        _joint_position_for_test(joints, "left_shoulder"),
        _joint_position_for_test(joints, "left_elbow"),
    ) == pytest.approx(upper_length)
    assert distance(
        _joint_position_for_test(joints, "left_elbow"),
        _joint_position_for_test(joints, "left_wrist"),
    ) == pytest.approx(lower_length)


def test_sparseposer_endpoint_ik_preserves_leg_lengths() -> None:
    joints = _joint_map(
        {
            "right_hip": (0.1, 0.95, 0.0),
            "right_knee": (0.15, 0.5, 0.05),
            "right_ankle": (0.15, 0.1, 0.0),
            "right_toe": (0.15, 0.05, 0.15),
        }
    )
    thigh_length = distance(
        _joint_position_for_test(joints, "right_hip"),
        _joint_position_for_test(joints, "right_knee"),
    )
    lower_length = distance(
        _joint_position_for_test(joints, "right_knee"),
        _joint_position_for_test(joints, "right_ankle"),
    )

    _apply_sparse_input_ik(
        joints,
        {"right_foot": Transform(position=(0.2, 0.05, 0.45))},
    )

    assert _joint_position_for_test(joints, "right_toe") == pytest.approx((0.2, 0.05, 0.45))
    assert distance(
        _joint_position_for_test(joints, "right_hip"),
        _joint_position_for_test(joints, "right_knee"),
    ) == pytest.approx(thigh_length)
    assert distance(
        _joint_position_for_test(joints, "right_knee"),
        _joint_position_for_test(joints, "right_ankle"),
    ) == pytest.approx(lower_length)


def test_sparseposer_endpoint_ik_hard_targets_head() -> None:
    joints = _joint_map(
        {
            "chest4": (0.0, 1.35, 0.1),
            "neck": (0.0, 1.48, 0.12),
            "head": (0.0, 1.62, 0.15),
        }
    )
    target_rotation = axis_angle_quaternion((0.0, 1.0, 0.0), 35.0)

    _apply_sparse_input_ik(
        joints,
        {
            "head": Transform(
                position=(0.0, 1.6, -0.25),
                rotation=target_rotation,
            )
        },
    )

    assert _joint_position_for_test(joints, "head") == pytest.approx((0.0, 1.6, -0.25))
    assert cast(
        list[float],
        cast(dict[str, object], joints["head"]["transform"])["rotation"],
    ) == pytest.approx(list(target_rotation))


class _FakeEngine:
    def __init__(self) -> None:
        self.packets: list[bytes] = []

    def infer(self, packet: bytes) -> bytes:
        self.packets.append(packet)
        rotations = []
        for _index in range(22):
            rotations.extend((1.0, 0.0, 0.0, 0.0))
        return struct.pack(f"<{len(rotations)}f", *rotations)

    def close(self) -> None:
        return None


def _message(*, include_optional: bool, include_alignment: bool) -> dict[str, object]:
    identity = [0.0, 0.0, 0.0, 1.0]
    corrected_head = [0.0, 1.0, 0.0, 0.0]
    optional = {}
    if include_optional:
        optional = {
            "pelvis": _tracker("pelvis", [0.0, 0.9, 0.0], identity),
            "left_foot": _tracker("left_foot", [-0.15, 0.05, 0.1], identity),
            "right_foot": _tracker("right_foot", [0.15, 0.05, 0.1], identity),
        }
    calibration: dict[str, object] = {}
    if include_alignment:
        calibration["sparseposer_alignment"] = {
            "role_transforms": {
                "pelvis": optional["pelvis"]["transform"],
                "left_foot": optional["left_foot"]["transform"],
                "right_foot": optional["right_foot"]["transform"],
                "head": {"position": [0.0, 1.6, 0.0], "rotation": corrected_head},
                "left_hand": {"position": [-0.4, 1.2, 0.0], "rotation": identity},
                "right_hand": {"position": [0.4, 1.2, 0.0], "rotation": identity},
            }
        }
    return {
        "timestamp_seconds": 1.0,
        "sample": {
            "timestamp_seconds": 1.0,
            "hmd": _tracker("head", [0.0, 1.6, 0.0], corrected_head),
            "left_hand": _tracker("left_hand", [-0.4, 1.2, 0.0], identity),
            "right_hand": _tracker("right_hand", [0.4, 1.2, 0.0], identity),
            "optional_trackers": optional,
        },
        "calibration": calibration,
    }


def _tracker(role: str, position: list[float], rotation: list[float]) -> dict[str, object]:
    return {
        "role": role,
        "valid": True,
        "confidence": 1.0,
        "transform": {"position": position, "rotation": rotation},
    }


def _unpack_last_sparseposer_input_frame(
    packet: bytes,
) -> dict[str, tuple[tuple[float, float, float], tuple[float, float, float, float]]]:
    values = struct.unpack(
        f"<{SPARSEPOSER_WINDOW_SIZE * len(SPARSEPOSER_INPUT_ROLES) * 7}f",
        packet,
    )
    frame_offset = (SPARSEPOSER_WINDOW_SIZE - 1) * len(SPARSEPOSER_INPUT_ROLES) * 7
    result = {}
    for role_index, role in enumerate(SPARSEPOSER_INPUT_ROLES):
        offset = frame_offset + role_index * 7
        result[role] = (
            (values[offset], values[offset + 1], values[offset + 2]),
            (
                values[offset + 3],
                values[offset + 4],
                values[offset + 5],
                values[offset + 6],
            ),
        )
    return result


def _joint_map(
    positions: dict[str, tuple[float, float, float]],
) -> dict[str, dict[str, object]]:
    return {
        name: {
            "transform": {
                "position": list(position),
                "rotation": [0.0, 0.0, 0.0, 1.0],
            }
        }
        for name, position in positions.items()
    }


def _joint_position_for_test(
    joints: dict[str, dict[str, object]],
    name: str,
) -> tuple[float, float, float]:
    position = cast(list[float], cast(dict[str, object], joints[name]["transform"])["position"])
    return (position[0], position[1], position[2])


def _write_reference_bvh(tmp_path: Path) -> Path:
    path = tmp_path / "reference.bvh"
    joints = [
        "LeftHip",
        "LeftKnee",
        "LeftAnkle",
        "LeftToe",
        "RightHip",
        "RightKnee",
        "RightAnkle",
        "RightToe",
        "Chest",
        "Chest3",
        "Chest4",
        "Neck",
        "Head",
        "LeftCollar",
        "LeftShoulder",
        "LeftElbow",
        "LeftWrist",
        "RightCollar",
        "RightShoulder",
        "RightElbow",
        "RightWrist",
    ]
    offsets = {
        "LeftHip": (0.1, 0.0, 0.0),
        "LeftToe": (0.1, -0.9, 0.2),
        "RightHip": (-0.1, 0.0, 0.0),
        "RightToe": (-0.1, -0.9, 0.2),
        "Chest": (0.0, 0.2, 0.0),
        "Chest3": (0.0, 0.4, 0.0),
        "Chest4": (0.0, 0.5, 0.0),
        "Neck": (0.0, 0.6, 0.0),
        "Head": (0.0, 0.7, 0.0),
        "LeftWrist": (0.55, 0.45, 0.0),
        "RightWrist": (-0.55, 0.45, 0.0),
    }
    lines = [
        "HIERARCHY",
        "ROOT Hips",
        "{",
        "  OFFSET 0 0 0",
        "  CHANNELS 6 Xposition Yposition Zposition Yrotation Xrotation Zrotation",
    ]
    for index, name in enumerate(joints, start=1):
        offset_x = 0.05 * index
        offset = offsets.get(name, (offset_x, 0.0, 0.0))
        lines.extend(
            [
                f"  JOINT {name}",
                "  {",
                f"    OFFSET {offset[0]:.3f} {offset[1]:.3f} {offset[2]:.3f}",
                "    CHANNELS 3 Yrotation Xrotation Zrotation",
                "    End Site",
                "    {",
                "      OFFSET 0 0 0",
                "    }",
                "  }",
            ]
        )
    lines.extend(
        [
            "}",
            "MOTION",
            "Frames: 1",
            "Frame Time: 0.0166667",
            " ".join(["0"] * (6 + len(joints) * 3)),
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
