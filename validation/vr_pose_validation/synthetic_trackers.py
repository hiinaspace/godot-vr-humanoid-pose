"""Synthetic VR tracker generation from canonical full-body pose frames."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from math import cos, pi
from random import Random
from zlib import crc32

from validation.vr_pose_validation.geometry import (
    Transform,
    Vector3,
    add,
    look_rotation,
    scale,
    subtract,
)
from validation.vr_pose_validation.types import (
    TrackedPoseSample,
    TrackerAttachment,
    TrackerRole,
    TrackerState,
)

type PoseFrame = Mapping[str, Transform]
WORLD_UP: Vector3 = (0.0, 1.0, 0.0)
WORLD_FORWARD: Vector3 = (0.0, 0.0, 1.0)
ZERO_VECTOR: Vector3 = (0.0, 0.0, 0.0)

type DeviceKind = str
HMD_KIND: DeviceKind = "hmd"
CONTROLLER_KIND: DeviceKind = "controller"
STRAP_TRACKER_KIND: DeviceKind = "strap_tracker"


@dataclass(frozen=True, slots=True)
class NoiseProfile:
    """Corruption terms for one tracker role.

    The first three fields are the original per-frame model and stay
    backward compatible: a profile built with only those fields behaves as
    before. The remaining fields add the fitted taxonomy (run-length dropout,
    excursion, calibration bias, drift, latency, prediction overshoot, and
    velocity-channel error).
    """

    position_std_m: float = 0.0
    dropout_probability: float = 0.0
    invalid_confidence: float = 0.0
    device_kind: DeviceKind = CONTROLLER_KIND
    dropout_mean_run_frames: float = 1.0
    dropout_hold_last_probability: float = 0.0
    excursion_magnitude_m: float = 0.0
    excursion_frames: int = 0
    calibration_bias_std_m: float = 0.0
    drift_amplitude_m: float = 0.0
    drift_period_seconds: float = 8.0
    latency_frames: int = 0
    overshoot_gain_seconds: float = 0.0
    velocity_std_mps: float = 0.0
    angular_velocity_std_radps: float = 0.0


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


DEFAULT_ROLE_DEVICE_KINDS: Mapping[TrackerRole, DeviceKind] = {
    "head": HMD_KIND,
    "left_hand": CONTROLLER_KIND,
    "right_hand": CONTROLLER_KIND,
    "pelvis": STRAP_TRACKER_KIND,
    "chest": STRAP_TRACKER_KIND,
    "left_foot": STRAP_TRACKER_KIND,
    "right_foot": STRAP_TRACKER_KIND,
    "left_elbow": STRAP_TRACKER_KIND,
    "right_elbow": STRAP_TRACKER_KIND,
    "left_knee": STRAP_TRACKER_KIND,
    "right_knee": STRAP_TRACKER_KIND,
}


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
    roles = ("head", "left_hand", "right_hand")
    dt = 1.0 / effective_config.frame_rate_hz
    corruptors = {
        role: _TrackerCorruptor(role, effective_config, _role_rng(effective_config.seed, role), dt)
        for role in roles
    }

    for frame_index, frame in enumerate(frames):
        timestamp = frame_index * dt
        states = {role: corruptors[role].step(frame, timestamp) for role in roles}
        yield TrackedPoseSample(
            timestamp_seconds=timestamp,
            hmd=states["head"],
            left_hand=states["left_hand"],
            right_hand=states["right_hand"],
        )


def generate_multi_point_samples(
    frames: Iterable[PoseFrame],
    config: SyntheticTrackerConfig | None = None,
) -> list[TrackedPoseSample]:
    return list(iter_multi_point_samples(frames, config))


def iter_multi_point_samples(
    frames: Iterable[PoseFrame],
    config: SyntheticTrackerConfig | None = None,
) -> Iterable[TrackedPoseSample]:
    effective_config = config or SyntheticTrackerConfig()
    roles = tuple(effective_config.attachments)
    dt = 1.0 / effective_config.frame_rate_hz
    corruptors = {
        role: _TrackerCorruptor(role, effective_config, _role_rng(effective_config.seed, role), dt)
        for role in roles
    }

    for frame_index, frame in enumerate(frames):
        timestamp = frame_index * dt
        states = {role: corruptors[role].step(frame, timestamp) for role in roles}
        optional: dict[TrackerRole, TrackerState] = {
            role: state
            for role, state in states.items()
            if role not in ("head", "left_hand", "right_hand")
        }
        yield TrackedPoseSample(
            timestamp_seconds=timestamp,
            hmd=states.get("head", _fallback_state("head")),
            left_hand=states.get("left_hand", _fallback_state("left_hand")),
            right_hand=states.get("right_hand", _fallback_state("right_hand")),
            optional_trackers=optional,
        )


class _TrackerCorruptor:
    """Stateful per-tracker corruption sampler over a frame sequence."""

    def __init__(
        self,
        role: TrackerRole,
        config: SyntheticTrackerConfig,
        rng: Random,
        dt: float,
    ) -> None:
        self.role = role
        self.attachment = config.attachments[role]
        self.profile = config.noise_profiles.get(self.attachment.noise_profile_id, NoiseProfile())
        self.rng = rng
        self.dt = dt
        self.calibration_bias = _sample_calibration_bias(self.profile, rng)
        self.drift_phase = rng.uniform(0.0, 2.0 * pi)
        self.latency_frames = max(
            self.profile.latency_frames,
            round(self.attachment.latency_seconds / dt) if dt > 0.0 else 0,
        )
        self._latency_buffer: deque[Transform] = deque()
        self._dropout_remaining = 0
        self._dropout_hold_last = False
        self._excursion_remaining = 0
        self._excursion_offset: Vector3 = ZERO_VECTOR
        self._was_valid = True
        self._last_valid_position: Vector3 | None = None
        self._prev_true_position: Vector3 | None = None
        self._prev_true_velocity: Vector3 | None = None

    def step(self, frame: PoseFrame, timestamp: float) -> TrackerState:
        source = frame[self.attachment.source_joint]
        rotation = (
            head_tracker_rotation_from_pose_frame(frame) if self.role == "head" else source.rotation
        )
        true_position = add(source.position, self.attachment.local_transform.position)

        latent_true = self._apply_latency(true_position)
        true_velocity = self._finite_difference(self._prev_true_position, latent_true)
        true_acceleration = self._finite_difference(self._prev_true_velocity, true_velocity)

        clean_position = self._clean_position(latent_true, timestamp, true_acceleration)
        tracking, hold_last = self._advance_dropout()
        self._advance_excursion(tracking)

        if tracking:
            reported_position = add(clean_position, self._excursion_offset)
            confidence = self.attachment.confidence
            self._last_valid_position = reported_position
            reported_velocity = self._reported_velocity(true_velocity)
        elif hold_last:
            # Device lost tracking but keeps streaming the last pose at full
            # confidence; downstream cannot tell from the valid flag alone.
            reported_position = self._last_valid_position or clean_position
            confidence = self.attachment.confidence
            reported_velocity = ZERO_VECTOR
        else:
            reported_position = self._last_valid_position or clean_position
            confidence = self.profile.invalid_confidence
            reported_velocity = ZERO_VECTOR
        reported_valid = tracking

        state = TrackerState(
            role=self.role,
            transform=Transform(position=reported_position, rotation=rotation),
            confidence=confidence,
            valid=reported_valid,
            attachment=self.attachment,
            linear_velocity=reported_velocity,
            angular_velocity=self._reported_angular_velocity() if tracking else ZERO_VECTOR,
        )

        self._prev_true_position = latent_true
        self._prev_true_velocity = true_velocity
        return state

    def _apply_latency(self, true_position: Vector3) -> Vector3:
        if self.latency_frames <= 0:
            return true_position
        self._latency_buffer.append(Transform(position=true_position))
        if len(self._latency_buffer) <= self.latency_frames:
            return self._latency_buffer[0].position
        return self._latency_buffer.popleft().position

    def _clean_position(
        self,
        true_position: Vector3,
        timestamp: float,
        acceleration: Vector3,
    ) -> Vector3:
        position = add(true_position, self.calibration_bias)
        if self.profile.drift_amplitude_m > 0.0 and self.profile.drift_period_seconds > 0.0:
            position = add(position, self._drift_offset(timestamp))
        if self.profile.overshoot_gain_seconds != 0.0:
            position = add(position, scale(acceleration, self.profile.overshoot_gain_seconds))
        if self.profile.position_std_m > 0.0:
            position = add(position, _gaussian_vector(self.rng, self.profile.position_std_m))
        return position

    def _drift_offset(self, timestamp: float) -> Vector3:
        angle = 2.0 * pi * timestamp / self.profile.drift_period_seconds + self.drift_phase
        amplitude = self.profile.drift_amplitude_m
        return (
            amplitude * cos(angle),
            amplitude * cos(angle + 2.094395),
            amplitude * cos(angle + 4.188790),
        )

    def _advance_dropout(self) -> tuple[bool, bool]:
        if self._dropout_remaining > 0:
            self._dropout_remaining -= 1
            return False, self._dropout_hold_last
        if self.rng.random() < self.profile.dropout_probability:
            run = _sample_run_length(self.profile.dropout_mean_run_frames, self.rng)
            self._dropout_remaining = run - 1
            self._dropout_hold_last = self.rng.random() < self.profile.dropout_hold_last_probability
            return False, self._dropout_hold_last
        return True, False

    def _advance_excursion(self, valid: bool) -> None:
        # Only reacquire (invalid -> valid) produces a reported excursion: invalid
        # frames hold the last pose and never add the offset, so an onset draw
        # would be dead RNG. Sampling only here keeps generator and fitter aligned.
        reacquire = not self._was_valid and valid
        self._was_valid = valid
        if reacquire and self.profile.excursion_frames > 0:
            self._excursion_remaining = self.profile.excursion_frames
            self._excursion_offset = _gaussian_vector(self.rng, self.profile.excursion_magnitude_m)
        if self._excursion_remaining > 0:
            self._excursion_remaining -= 1
            if self._excursion_remaining == 0:
                self._excursion_offset = ZERO_VECTOR
        else:
            self._excursion_offset = ZERO_VECTOR

    def _reported_velocity(self, true_velocity: Vector3) -> Vector3:
        native = true_velocity
        if self.profile.velocity_std_mps > 0.0:
            native = add(native, _gaussian_vector(self.rng, self.profile.velocity_std_mps))
        return native

    def _reported_angular_velocity(self) -> Vector3:
        if self.profile.angular_velocity_std_radps > 0.0:
            return _gaussian_vector(self.rng, self.profile.angular_velocity_std_radps)
        return ZERO_VECTOR

    def _finite_difference(
        self,
        previous: Vector3 | None,
        current: Vector3,
    ) -> Vector3:
        if previous is None or self.dt <= 0.0:
            return ZERO_VECTOR
        return scale(subtract(current, previous), 1.0 / self.dt)


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


def role_default_profile(
    role: TrackerRole | str,
    *,
    device_kind: DeviceKind | None = None,
) -> NoiseProfile:
    """Return an invented-prior profile keyed by device kind.

    These are placeholder priors until fitted profiles from real logs replace
    them (PLAN.md milestone 1). Dropout/excursion dominate; jitter is sub-mm.
    """

    kind = device_kind or _default_kind_for_role(role)
    if kind == HMD_KIND:
        return NoiseProfile(
            position_std_m=0.00025,
            dropout_probability=0.0005,
            invalid_confidence=0.0,
            device_kind=HMD_KIND,
            dropout_mean_run_frames=2.0,
            dropout_hold_last_probability=0.9,
            excursion_magnitude_m=0.01,
            excursion_frames=2,
            calibration_bias_std_m=0.001,
            drift_amplitude_m=0.0,
            latency_frames=1,
            overshoot_gain_seconds=0.004,
            velocity_std_mps=0.01,
            angular_velocity_std_radps=0.01,
        )
    if kind == CONTROLLER_KIND:
        return NoiseProfile(
            position_std_m=0.0003,
            dropout_probability=0.004,
            invalid_confidence=0.0,
            device_kind=CONTROLLER_KIND,
            dropout_mean_run_frames=6.0,
            dropout_hold_last_probability=0.4,
            excursion_magnitude_m=0.06,
            excursion_frames=4,
            calibration_bias_std_m=0.003,
            drift_amplitude_m=0.0,
            latency_frames=1,
            overshoot_gain_seconds=0.006,
            velocity_std_mps=0.03,
            angular_velocity_std_radps=0.05,
        )
    return NoiseProfile(
        position_std_m=0.0004,
        dropout_probability=0.006,
        invalid_confidence=0.0,
        device_kind=STRAP_TRACKER_KIND,
        dropout_mean_run_frames=10.0,
        dropout_hold_last_probability=0.3,
        excursion_magnitude_m=0.08,
        excursion_frames=5,
        calibration_bias_std_m=0.006,
        drift_amplitude_m=0.008,
        drift_period_seconds=12.0,
        latency_frames=2,
        overshoot_gain_seconds=0.005,
        velocity_std_mps=0.04,
        angular_velocity_std_radps=0.08,
    )


def _default_kind_for_role(role: TrackerRole | str) -> DeviceKind:
    for known_role, kind in DEFAULT_ROLE_DEVICE_KINDS.items():
        if known_role == role:
            return kind
    return CONTROLLER_KIND


def _role_rng(seed: int, role: TrackerRole | str) -> Random:
    """Derive a per-role RNG so each corruptor draws from an independent stream.

    A single shared ``Random(seed)`` made corruption order-sensitive: adding or
    reordering a role shifted every other role's stream, so the same config
    produced different corruption for shared roles between 3-point and
    multi-point generation. Deriving each role's seed from ``(seed, role)`` with
    a stable, process-independent hash (``crc32``, not the salted builtin
    ``hash``) keeps each role's stream fixed regardless of the role set.
    """

    role_hash = crc32(str(role).encode("utf-8"))
    return Random((seed * 0x9E3779B1 + role_hash) & 0xFFFFFFFF)


def _sample_calibration_bias(profile: NoiseProfile, rng: Random) -> Vector3:
    if profile.calibration_bias_std_m <= 0.0:
        return ZERO_VECTOR
    return _gaussian_vector(rng, profile.calibration_bias_std_m)


def _sample_run_length(mean_frames: float, rng: Random) -> int:
    if mean_frames <= 1.0:
        return 1
    p = 1.0 / mean_frames
    run = 1
    while rng.random() >= p:
        run += 1
        if run >= 4096:
            break
    return run


def _fallback_state(role: TrackerRole) -> TrackerState:
    return TrackerState(role=role, transform=Transform(), confidence=0.0, valid=False)


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


__all__ = [
    "CONTROLLER_KIND",
    "HMD_KIND",
    "STRAP_TRACKER_KIND",
    "WORLD_FORWARD",
    "WORLD_UP",
    "DeviceKind",
    "NoiseProfile",
    "PoseFrame",
    "SyntheticTrackerConfig",
    "generate_multi_point_samples",
    "generate_three_point_samples",
    "head_tracker_rotation_from_pose_frame",
    "iter_multi_point_samples",
    "iter_three_point_samples",
    "pose_frame_from_positions",
    "role_default_profile",
]
