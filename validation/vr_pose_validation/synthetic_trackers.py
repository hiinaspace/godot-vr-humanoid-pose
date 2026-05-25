"""Synthetic VR tracker generation from canonical full-body pose frames."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from random import Random

from validation.vr_pose_validation.geometry import Transform, Vector3, add, look_rotation
from validation.vr_pose_validation.types import (
    TrackedPoseSample,
    TrackerAttachment,
    TrackerRole,
    TrackerState,
)

type PoseFrame = Mapping[str, Transform]
WORLD_UP: Vector3 = (0.0, 1.0, 0.0)
WORLD_FORWARD: Vector3 = (0.0, 0.0, 1.0)


@dataclass(frozen=True, slots=True)
class NoiseProfile:
    position_std_m: float = 0.0
    dropout_probability: float = 0.0
    invalid_confidence: float = 0.0


@dataclass(frozen=True, slots=True)
class SyntheticTrackerConfig:
    attachments: Mapping[TrackerRole, TrackerAttachment] = field(
        default_factory=lambda: {
            "head": TrackerAttachment(
                role="head",
                source_joint="head",
                local_transform=Transform(position=(0.0, 0.08, 0.02)),
            ),
            "left_hand": TrackerAttachment(role="left_hand", source_joint="left_hand"),
            "right_hand": TrackerAttachment(role="right_hand", source_joint="right_hand"),
        }
    )
    noise_profiles: Mapping[str, NoiseProfile] = field(
        default_factory=lambda: {"clean": NoiseProfile()}
    )
    seed: int = 0
    frame_rate_hz: float = 120.0


def generate_three_point_samples(
    frames: Iterable[PoseFrame],
    config: SyntheticTrackerConfig | None = None,
) -> list[TrackedPoseSample]:
    return list(iter_three_point_samples(frames, config))


def iter_three_point_samples(
    frames: Iterable[PoseFrame],
    config: SyntheticTrackerConfig | None = None,
) -> Iterable[TrackedPoseSample]:
    effective_config = config or SyntheticTrackerConfig()
    rng = Random(effective_config.seed)

    for frame_index, frame in enumerate(frames):
        timestamp = frame_index / effective_config.frame_rate_hz
        states = {
            role: _make_tracker_state(role, frame, effective_config, rng)
            for role in ("head", "left_hand", "right_hand")
        }
        yield TrackedPoseSample(
            timestamp_seconds=timestamp,
            hmd=states["head"],
            left_hand=states["left_hand"],
            right_hand=states["right_hand"],
        )


def _make_tracker_state(
    role: TrackerRole,
    frame: PoseFrame,
    config: SyntheticTrackerConfig,
    rng: Random,
) -> TrackerState:
    attachment = config.attachments[role]
    source = frame[attachment.source_joint]
    profile = config.noise_profiles.get(attachment.noise_profile_id, NoiseProfile())
    valid = rng.random() >= profile.dropout_probability
    confidence = attachment.confidence if valid else profile.invalid_confidence
    position = add(source.position, attachment.local_transform.position)
    if profile.position_std_m > 0.0:
        position = add(position, _gaussian_vector(rng, profile.position_std_m))
    rotation = head_tracker_rotation_from_pose_frame(frame) if role == "head" else source.rotation
    return TrackerState(
        role=role,
        transform=Transform(position=position, rotation=rotation),
        confidence=confidence,
        valid=valid,
        attachment=attachment,
    )


def head_tracker_rotation_from_pose_frame(frame: PoseFrame) -> tuple[float, float, float, float]:
    """Derive a VR-style HMD rotation from CMU head orientation.

    The sparse tracker stream should expose the user's head pose, not an inferred
    torso yaw. Earlier versions projected head pitch onto a shoulder-derived
    body forward, but clips where the head looks sideways can make the shoulder
    normal ambiguous and create adjacent-frame 180-degree HMD flips.
    """

    if "head" not in frame:
        return look_rotation(WORLD_FORWARD, WORLD_UP)
    return frame["head"].rotation


def _gaussian_vector(rng: Random, standard_deviation: float) -> Vector3:
    return (
        rng.gauss(0.0, standard_deviation),
        rng.gauss(0.0, standard_deviation),
        rng.gauss(0.0, standard_deviation),
    )


def pose_frame_from_positions(joints: Mapping[str, Sequence[float]]) -> PoseFrame:
    return {
        name: Transform(position=(float(values[0]), float(values[1]), float(values[2])))
        for name, values in joints.items()
    }
