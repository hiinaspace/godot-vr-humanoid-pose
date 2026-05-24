"""Export a small CMU clip replay fixture for Godot visualization."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from validation.scripts.run_cached_validation import DEFAULT_SYNTHETIC_TRACKER_CONFIG
from validation.vr_pose_validation.baseline import EstimatorName, create_estimator
from validation.vr_pose_validation.geometry import (
    Transform,
    Vector3,
    add,
    normalize,
    rotate_vector,
    scale,
    subtract,
)
from validation.vr_pose_validation.metrics import infer_reference_floor_height
from validation.vr_pose_validation.parquet_cache import (
    CachedPoseFrame,
    clip_cache_path,
    iter_cached_pose_frames,
)
from validation.vr_pose_validation.synthetic_trackers import head_tracker_rotation_from_pose_frame
from validation.vr_pose_validation.types import (
    AvatarCalibration,
    GroundingContext,
    HumanoidPoseFrame,
    TrackedPoseSample,
    TrackerRole,
    TrackerState,
)

DEFAULT_CACHE_DIR = Path(".cache/cmu/parquet/canonical_full")
DEFAULT_OUTPUT = Path("godot/tests/fixtures/cmu_walk_replay_fixture.json")
WORLD_UP: Vector3 = (0.0, 1.0, 0.0)
WORLD_FORWARD: Vector3 = (0.0, 0.0, 1.0)
VISUAL_HAND_LENGTH_M = 0.09
VISUAL_HAND_FORWARD_LENGTH_M = 0.18
VISUAL_HEAD_FORWARD_LENGTH_M = 0.22
DEFAULT_BONES = (
    ("hips", "spine"),
    ("spine", "chest"),
    ("chest", "neck"),
    ("neck", "head"),
    ("head", "head_forward"),
    ("chest", "left_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("left_wrist", "left_hand"),
    ("left_hand", "left_hand_forward"),
    ("chest", "right_shoulder"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("right_wrist", "right_hand"),
    ("right_hand", "right_hand_forward"),
    ("hips", "left_knee"),
    ("left_knee", "left_foot"),
    ("hips", "right_knee"),
    ("right_knee", "right_foot"),
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--subject", default="02")
    parser.add_argument("--clip", default="02_01")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=90)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument(
        "--estimator",
        choices=(
            "trivial",
            "planted",
            "basis",
            "basis_knee_hint",
            "basis_footstep",
            "basis_arm_lookup",
            "basis_arm_swivel",
            "basis_shrug",
            "basis_research",
        ),
        default="basis",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    fixture = build_fixture(
        cache_dir=args.cache_dir,
        subject=args.subject,
        clip=args.clip,
        start_frame=args.start_frame,
        max_frames=args.max_frames,
        frame_stride=args.frame_stride,
        estimator_name=args.estimator,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes((json.dumps(fixture, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    print(f"Wrote {args.output}")
    return 0


def build_fixture(
    *,
    cache_dir: Path,
    subject: str,
    clip: str,
    start_frame: int,
    max_frames: int,
    frame_stride: int,
    estimator_name: EstimatorName,
) -> dict[str, Any]:
    source_path = clip_cache_path(cache_dir, subject, clip)
    if not source_path.exists():
        raise FileNotFoundError(source_path)

    source_frames = _select_frames(
        list(iter_cached_pose_frames(source_path)),
        start_frame=start_frame,
        max_frames=max_frames,
        frame_stride=frame_stride,
    )
    reference_frames = [frame.joints for frame in source_frames]
    floor_height_m = infer_reference_floor_height(reference_frames)
    grounding = GroundingContext(floor_height_m=floor_height_m)
    calibration = AvatarCalibration()
    estimator = create_estimator(estimator_name)
    estimated_frames = [
        estimator.estimate(_tracked_sample_from_replay_frame(frame), calibration, grounding)
        for frame in source_frames
    ]

    first_frame = source_frames[0]
    return {
        "schema_version": 1,
        "source": {
            "activity_class": first_frame.activity_class,
            "cache_path": source_path.as_posix(),
            "clip": clip,
            "environment_split": first_frame.environment_split,
            "estimator": estimator_name,
            "frame_stride": frame_stride,
            "motion_description": first_frame.motion_description,
            "start_frame": start_frame,
            "subject": subject,
        },
        "bone_pairs": [list(pair) for pair in DEFAULT_BONES],
        "floor_height_m": floor_height_m,
        "reference_frames": [
            _reference_frame_to_json(frame, source)
            for frame, source in zip(reference_frames, source_frames, strict=True)
        ],
        "estimated_frames": [_pose_frame_to_json(frame) for frame in estimated_frames],
    }


def _select_frames(
    frames: list[CachedPoseFrame],
    *,
    start_frame: int,
    max_frames: int,
    frame_stride: int,
) -> list[CachedPoseFrame]:
    if max_frames <= 0:
        raise ValueError("max_frames must be positive")
    if frame_stride <= 0:
        raise ValueError("frame_stride must be positive")
    selected = frames[start_frame::frame_stride][:max_frames]
    if not selected:
        raise ValueError("No frames selected")
    return selected


def _reference_frame_to_json(
    joints: dict[str, Transform],
    source: CachedPoseFrame,
) -> dict[str, Any]:
    visual_joints = _with_visual_helper_joints(joints)
    return {
        "frame_index": source.frame_index,
        "joints": {name: _transform_to_json(visual_joints[name]) for name in sorted(visual_joints)},
        "root": _transform_to_json(joints["hips"]),
        "timestamp_seconds": source.timestamp_s,
    }


def _pose_frame_to_json(frame: HumanoidPoseFrame) -> dict[str, Any]:
    joints = {name: pose.transform for name, pose in frame.joints.items()}
    visual_joints = _with_visual_helper_joints(joints)
    return {
        "joints": {name: _transform_to_json(visual_joints[name]) for name in sorted(visual_joints)},
        "root": _transform_to_json(frame.root),
        "timestamp_seconds": frame.timestamp_seconds,
    }


def _tracked_sample_from_replay_frame(frame: CachedPoseFrame) -> TrackedPoseSample:
    head_rotation = _head_rotation_from_reference(frame.joints)
    return TrackedPoseSample(
        timestamp_seconds=frame.timestamp_s,
        hmd=_tracker_state("head", frame, rotation_override=head_rotation),
        left_hand=_tracker_state("left_hand", frame),
        right_hand=_tracker_state("right_hand", frame),
    )


def _tracker_state(
    role: TrackerRole,
    frame: CachedPoseFrame,
    *,
    rotation_override: tuple[float, float, float, float] | None = None,
) -> TrackerState:
    attachment = DEFAULT_SYNTHETIC_TRACKER_CONFIG.attachments[role]
    source = frame.joints[attachment.source_joint]
    transform = Transform(
        position=add(source.position, attachment.local_transform.position),
        rotation=rotation_override or source.rotation,
    )
    return TrackerState(
        role=role,
        transform=transform,
        confidence=attachment.confidence,
        attachment=attachment,
    )


def _head_rotation_from_reference(
    joints: dict[str, Transform],
) -> tuple[float, float, float, float]:
    return head_tracker_rotation_from_pose_frame(joints)


def _with_visual_helper_joints(joints: dict[str, Transform]) -> dict[str, Transform]:
    visual_joints = dict(joints)
    if "neck" not in visual_joints and "chest" in visual_joints and "head" in visual_joints:
        head = visual_joints["head"]
        head_to_chest = subtract(visual_joints["chest"].position, head.position)
        neck_offset = min(0.12, _length(head_to_chest) * 0.35)
        visual_joints["neck"] = Transform(
            position=add(head.position, scale(normalize(head_to_chest), neck_offset)),
            rotation=head.rotation,
        )

    if "head" in visual_joints:
        head = visual_joints["head"]
        forward = normalize(rotate_vector(head.rotation, WORLD_FORWARD), fallback=WORLD_FORWARD)
        visual_joints["head_forward"] = Transform(
            position=add(head.position, scale(forward, VISUAL_HEAD_FORWARD_LENGTH_M)),
            rotation=head.rotation,
        )

    for side in ("left", "right"):
        elbow_name = f"{side}_elbow"
        hand_name = f"{side}_hand"
        wrist_name = f"{side}_wrist"
        if elbow_name not in visual_joints or hand_name not in visual_joints:
            continue
        elbow = visual_joints[elbow_name].position
        hand = visual_joints[hand_name].position
        hand_to_elbow = subtract(elbow, hand)
        wrist_offset = min(VISUAL_HAND_LENGTH_M, _length(hand_to_elbow) * 0.35)
        wrist = add(hand, scale(normalize(hand_to_elbow), wrist_offset))
        visual_joints[wrist_name] = Transform(
            position=wrist,
            rotation=visual_joints[hand_name].rotation,
        )
        hand_forward_name = f"{side}_hand_forward"
        hand = visual_joints[hand_name]
        hand_forward = normalize(
            rotate_vector(hand.rotation, WORLD_FORWARD),
            fallback=WORLD_FORWARD,
        )
        visual_joints[hand_forward_name] = Transform(
            position=add(hand.position, scale(hand_forward, VISUAL_HAND_FORWARD_LENGTH_M)),
            rotation=hand.rotation,
        )
    return visual_joints


def _transform_to_json(transform: Transform) -> dict[str, list[float]]:
    return {
        "position": list(transform.position),
        "rotation": list(transform.rotation),
    }


def _length(vector: Vector3) -> float:
    return sum(component * component for component in vector) ** 0.5


if __name__ == "__main__":
    raise SystemExit(main())
