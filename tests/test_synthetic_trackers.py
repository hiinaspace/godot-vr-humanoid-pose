from __future__ import annotations

from pytest import approx

from validation.vr_pose_validation.geometry import Transform, look_rotation, rotate_vector
from validation.vr_pose_validation.synthetic_trackers import (
    WORLD_FORWARD,
    NoiseProfile,
    SyntheticTrackerConfig,
    generate_three_point_samples,
    pose_frame_from_positions,
)


def test_generate_three_point_samples_adds_headset_offset_and_timestamps() -> None:
    frames = [
        pose_frame_from_positions(
            {
                "head": (0, 1.6, 0),
                "left_hand": (-0.3, 1.2, -0.2),
                "right_hand": (0.3, 1.2, -0.2),
            }
        ),
        pose_frame_from_positions(
            {
                "head": (0, 1.61, 0),
                "left_hand": (-0.31, 1.2, -0.2),
                "right_hand": (0.31, 1.2, -0.2),
            }
        ),
    ]

    samples = generate_three_point_samples(frames)

    assert len(samples) == 2
    assert samples[0].timestamp_seconds == 0.0
    assert samples[1].timestamp_seconds == 1 / 120
    assert samples[0].hmd.transform.position == approx((0.0, 1.68, 0.02))
    assert samples[0].left_hand.transform.position == (-0.3, 1.2, -0.2)


def test_noise_and_dropout_are_seeded() -> None:
    frame = pose_frame_from_positions(
        {
            "head": (0, 1.6, 0),
            "left_hand": (-0.3, 1.2, -0.2),
            "right_hand": (0.3, 1.2, -0.2),
        }
    )
    config = SyntheticTrackerConfig(
        noise_profiles={"clean": NoiseProfile(position_std_m=0.01, dropout_probability=1.0)},
        seed=42,
    )

    first = generate_three_point_samples([frame], config)[0]
    second = generate_three_point_samples([frame], config)[0]

    assert first.hmd.transform.position == second.hmd.transform.position
    assert not first.hmd.valid
    assert first.hmd.confidence == 0.0


def test_head_tracker_uses_source_head_rotation_when_head_is_sideways_to_shoulders() -> None:
    head_rotation = look_rotation((-1.0, 0.0, 0.0))
    frames = [
        {
            "head": Transform(position=(0.0, 1.6, 0.0), rotation=head_rotation),
            "left_shoulder": Transform(position=(-0.3, 1.3, -0.01)),
            "right_shoulder": Transform(position=(0.3, 1.3, 0.01)),
            "left_hand": Transform(position=(-0.4, 1.2, 0.1)),
            "right_hand": Transform(position=(0.4, 1.2, 0.1)),
        },
        {
            "head": Transform(position=(0.0, 1.6, 0.0), rotation=head_rotation),
            "left_shoulder": Transform(position=(-0.3, 1.3, 0.01)),
            "right_shoulder": Transform(position=(0.3, 1.3, -0.01)),
            "left_hand": Transform(position=(-0.4, 1.2, 0.1)),
            "right_hand": Transform(position=(0.4, 1.2, 0.1)),
        },
    ]

    samples = generate_three_point_samples(frames)
    first_forward = rotate_vector(samples[0].hmd.transform.rotation, WORLD_FORWARD)
    second_forward = rotate_vector(samples[1].hmd.transform.rotation, WORLD_FORWARD)

    assert first_forward == approx((-1.0, 0.0, 0.0))
    assert second_forward == approx(first_forward)
