from __future__ import annotations

from pytest import approx

from validation.scripts.export_godot_replay_fixture import (
    WORLD_FORWARD,
    _head_rotation_from_reference,
    _selected_frame_indices,
    _with_visual_helper_joints,
)
from validation.vr_pose_validation.geometry import (
    Transform,
    look_rotation,
    normalize,
    rotate_vector,
)


def test_replay_hmd_rotation_preserves_source_head_pitch() -> None:
    source_head_rotation = look_rotation(normalize((0.0, -0.5, 1.0)))
    joints = {
        "left_shoulder": Transform(position=(-0.3, 1.3, 0.0)),
        "right_shoulder": Transform(position=(0.3, 1.3, 0.0)),
        "head": Transform(position=(0.0, 1.6, 0.0), rotation=source_head_rotation),
    }

    replay_rotation = _head_rotation_from_reference(joints)
    replay_forward = rotate_vector(replay_rotation, WORLD_FORWARD)

    assert replay_forward[1] == approx(-0.4472135955)
    assert replay_forward[2] == approx(0.8944271910)


def test_replay_hmd_rotation_uses_source_head_yaw_even_when_shoulders_are_reversed() -> None:
    source_head_rotation = look_rotation(WORLD_FORWARD)
    joints = {
        "left_shoulder": Transform(position=(0.3, 1.3, 0.0)),
        "right_shoulder": Transform(position=(-0.3, 1.3, 0.0)),
        "head": Transform(position=(0.0, 1.6, 0.0), rotation=source_head_rotation),
    }

    replay_rotation = _head_rotation_from_reference(joints)
    replay_forward = rotate_vector(replay_rotation, WORLD_FORWARD)

    assert replay_forward[2] > 0.9


def test_replay_visual_helpers_add_head_forward_marker() -> None:
    head_rotation = look_rotation(normalize((0.0, -0.5, 1.0)))
    joints = {
        "head": Transform(position=(0.0, 1.6, 0.0), rotation=head_rotation),
    }

    visual_joints = _with_visual_helper_joints(joints)

    assert "head_forward" in visual_joints
    assert visual_joints["head_forward"].position[1] < visual_joints["head"].position[1]
    assert visual_joints["head_forward"].position[2] > visual_joints["head"].position[2]


def test_replay_visual_helpers_add_reference_neck_when_missing() -> None:
    joints = {
        "chest": Transform(position=(0.0, 1.3, 0.0)),
        "head": Transform(position=(0.0, 1.6, 0.1)),
    }

    visual_joints = _with_visual_helper_joints(joints)

    assert "neck" in visual_joints
    assert visual_joints["chest"].position[1] < visual_joints["neck"].position[1]
    assert visual_joints["neck"].position[1] < visual_joints["head"].position[1]


def test_replay_visual_helpers_add_hand_forward_marker() -> None:
    hand_rotation = look_rotation(WORLD_FORWARD)
    joints = {
        "left_elbow": Transform(position=(-0.35, 1.2, 0.0)),
        "left_hand": Transform(position=(-0.55, 1.0, 0.0), rotation=hand_rotation),
    }

    visual_joints = _with_visual_helper_joints(joints)

    assert "left_hand_forward" in visual_joints
    assert visual_joints["left_hand_forward"].position[2] > visual_joints["left_hand"].position[2]


def test_replay_fixture_selects_strided_frame_indices() -> None:
    assert _selected_frame_indices(20, start_frame=3, max_frames=4, frame_stride=3) == [
        3,
        6,
        9,
        12,
    ]
