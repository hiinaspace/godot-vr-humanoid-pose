"""Fitted tracker-noise profile schema, serialization, and config construction."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from validation.vr_pose_validation.synthetic_trackers import (
    DEFAULT_ROLE_DEVICE_KINDS,
    NoiseProfile,
    SyntheticTrackerConfig,
)
from validation.vr_pose_validation.types import TrackerAttachment, TrackerRole

FITTED_PROFILE_SCHEMA = "godot_vr_humanoid_pose.tracker_noise_profile.v1"

DEFAULT_ROLE_SOURCE_JOINTS: Mapping[TrackerRole, str] = {
    "head": "head",
    "left_hand": "left_hand",
    "right_hand": "right_hand",
    "pelvis": "hips",
    "chest": "chest",
    "left_foot": "left_foot",
    "right_foot": "right_foot",
    "left_elbow": "left_elbow",
    "right_elbow": "right_elbow",
    "left_knee": "left_knee",
    "right_knee": "right_knee",
}

_PROFILE_FIELDS: tuple[str, ...] = (
    "position_std_m",
    "dropout_probability",
    "invalid_confidence",
    "device_kind",
    "dropout_mean_run_frames",
    "dropout_hold_last_probability",
    "excursion_magnitude_m",
    "excursion_frames",
    "calibration_bias_std_m",
    "drift_amplitude_m",
    "drift_period_seconds",
    "latency_frames",
    "overshoot_gain_seconds",
    "velocity_std_mps",
    "angular_velocity_std_radps",
)


def noise_profile_to_dict(profile: NoiseProfile) -> dict[str, Any]:
    return {field_name: getattr(profile, field_name) for field_name in _PROFILE_FIELDS}


def noise_profile_from_dict(payload: Mapping[str, Any]) -> NoiseProfile:
    kwargs: dict[str, Any] = {}
    for field_name in _PROFILE_FIELDS:
        if field_name not in payload:
            continue
        value = payload[field_name]
        if field_name in ("excursion_frames", "latency_frames"):
            kwargs[field_name] = int(value)
        elif field_name == "device_kind":
            kwargs[field_name] = str(value)
        else:
            kwargs[field_name] = float(value)
    return NoiseProfile(**kwargs)


def fitted_profile_document(
    roles: Mapping[str, NoiseProfile],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema": FITTED_PROFILE_SCHEMA,
        "metadata": dict(metadata or {}),
        "roles": {role: noise_profile_to_dict(profile) for role, profile in roles.items()},
    }


def write_fitted_profile(
    path: Path,
    roles: Mapping[str, NoiseProfile],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    document = fitted_profile_document(roles, metadata=metadata)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_fitted_profiles(path: Path) -> dict[str, NoiseProfile]:
    document = json.loads(path.read_text(encoding="utf-8"))
    return fitted_profiles_from_document(document)


def fitted_profiles_from_document(document: Mapping[str, Any]) -> dict[str, NoiseProfile]:
    schema = document.get("schema")
    if schema != FITTED_PROFILE_SCHEMA:
        raise ValueError(f"Unsupported fitted-profile schema: {schema!r}")
    roles = document.get("roles")
    if not isinstance(roles, Mapping):
        raise ValueError("Fitted-profile document is missing a 'roles' mapping")
    return {
        str(role): noise_profile_from_dict(payload)
        for role, payload in roles.items()
        if isinstance(payload, Mapping)
    }


def config_from_fitted_profiles(
    profiles: Mapping[str, NoiseProfile],
    *,
    seed: int = 0,
    frame_rate_hz: float = 120.0,
) -> SyntheticTrackerConfig:
    """Build a SyntheticTrackerConfig where each role uses its own fitted profile."""

    attachments: dict[TrackerRole, TrackerAttachment] = {}
    noise_profiles: dict[str, NoiseProfile] = {"clean": NoiseProfile()}
    for role, profile in profiles.items():
        tracker_role = cast(TrackerRole, role)
        profile_id = f"fitted_{role}"
        source_joint = DEFAULT_ROLE_SOURCE_JOINTS.get(tracker_role, role)
        latency_seconds = profile.latency_frames / frame_rate_hz if frame_rate_hz > 0.0 else 0.0
        attachments[tracker_role] = TrackerAttachment(
            role=tracker_role,
            source_joint=source_joint,
            latency_seconds=latency_seconds,
            noise_profile_id=profile_id,
        )
        noise_profiles[profile_id] = profile
    return SyntheticTrackerConfig(
        attachments=attachments,
        noise_profiles=noise_profiles,
        seed=seed,
        frame_rate_hz=frame_rate_hz,
    )


def config_from_fitted_profile_file(
    path: Path,
    *,
    seed: int = 0,
    frame_rate_hz: float = 120.0,
) -> SyntheticTrackerConfig:
    return config_from_fitted_profiles(
        load_fitted_profiles(path),
        seed=seed,
        frame_rate_hz=frame_rate_hz,
    )


__all__ = [
    "DEFAULT_ROLE_DEVICE_KINDS",
    "DEFAULT_ROLE_SOURCE_JOINTS",
    "FITTED_PROFILE_SCHEMA",
    "config_from_fitted_profile_file",
    "config_from_fitted_profiles",
    "fitted_profile_document",
    "fitted_profiles_from_document",
    "load_fitted_profiles",
    "noise_profile_from_dict",
    "noise_profile_to_dict",
    "write_fitted_profile",
]
