"""Shared pose-estimation data contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, Protocol

from validation.vr_pose_validation.geometry import Transform, Vector3

type TrackerRole = Literal[
    "head",
    "left_hand",
    "right_hand",
    "pelvis",
    "chest",
    "left_foot",
    "right_foot",
    "left_elbow",
    "right_elbow",
    "left_knee",
    "right_knee",
]


@dataclass(frozen=True, slots=True)
class TrackerAttachment:
    """Body-local placement model for a physical or synthetic tracker."""

    role: TrackerRole
    source_joint: str
    local_transform: Transform = field(default_factory=Transform)
    confidence: float = 1.0
    latency_seconds: float = 0.0
    noise_profile_id: str = "clean"


@dataclass(frozen=True, slots=True)
class TrackerState:
    role: TrackerRole
    transform: Transform
    confidence: float = 1.0
    valid: bool = True
    attachment: TrackerAttachment | None = None


@dataclass(frozen=True, slots=True)
class TrackedPoseSample:
    """Sparse VR tracker observation at one point in time."""

    timestamp_seconds: float
    hmd: TrackerState
    left_hand: TrackerState
    right_hand: TrackerState
    optional_trackers: Mapping[TrackerRole, TrackerState] = field(default_factory=dict)
    character_velocity: Vector3 = (0.0, 0.0, 0.0)


@dataclass(frozen=True, slots=True)
class AvatarCalibration:
    """Avatar-specific proportions and offsets derived from a rest pose or calibration."""

    rest_pose: Mapping[str, Transform] = field(default_factory=dict)
    bone_lengths_m: Mapping[str, float] = field(default_factory=dict)
    avatar_scale: float = 1.0
    eye_to_head: Transform = field(default_factory=Transform)
    left_controller_to_wrist: Transform = field(default_factory=Transform)
    right_controller_to_wrist: Transform = field(default_factory=Transform)
    foot_height_m: float = 0.08
    human_height_m: float = 1.7


@dataclass(frozen=True, slots=True)
class GroundingContext:
    """Minimal scene context needed by data-only baselines."""

    floor_height_m: float = 0.0
    floor_normal: Vector3 = (0.0, 1.0, 0.0)
    character_velocity: Vector3 = (0.0, 0.0, 0.0)
    locomotion_state: str = "unknown"


@dataclass(frozen=True, slots=True)
class JointPose:
    transform: Transform
    confidence: float = 1.0
    estimated: bool = True


@dataclass(frozen=True, slots=True)
class HumanoidPoseFrame:
    """Canonical output pose for validation and future Godot application."""

    timestamp_seconds: float
    root: Transform
    joints: Mapping[str, JointPose]


class PoseEstimator(Protocol):
    def estimate(
        self,
        sample: TrackedPoseSample,
        calibration: AvatarCalibration,
        grounding: GroundingContext,
    ) -> HumanoidPoseFrame:
        """Estimate a full humanoid pose from sparse tracker input."""
