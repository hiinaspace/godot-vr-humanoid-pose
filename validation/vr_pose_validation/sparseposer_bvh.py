"""SparsePoser Xsens BVH parsing and forward kinematics."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath

from validation.vr_pose_validation.geometry import (
    IDENTITY_QUATERNION,
    Quaternion,
    Transform,
    Vector3,
    add,
    euler_quaternion,
    multiply_quaternion,
    rotate_vector,
)

SPARSEPOSER_DATASET = "sparseposer_xsens"
SPARSEPOSER_FRAME_TIME_SECONDS = 1.0 / 60.0
SPARSEPOSER_CANONICAL_JOINTS = {
    "head": "Head",
    "chest": "Chest4",
    "spine": "Chest3",
    "left_shoulder": "LeftShoulder",
    "right_shoulder": "RightShoulder",
    "left_elbow": "LeftElbow",
    "right_elbow": "RightElbow",
    "left_hand": "LeftWrist",
    "right_hand": "RightWrist",
    "hips": "Hips",
    "left_knee": "LeftKnee",
    "right_knee": "RightKnee",
    "left_foot": "LeftAnkle",
    "right_foot": "RightAnkle",
}


@dataclass(slots=True)
class SparseposerBvhJoint:
    index: int
    name: str
    parent_index: int | None
    offset: Vector3
    channels: tuple[str, ...]
    channel_start: int


@dataclass(frozen=True, slots=True)
class SparseposerBvhMotion:
    joints: tuple[SparseposerBvhJoint, ...]
    source_frame_count: int
    frame_time_s: float
    frame_lines: tuple[str, ...]
    channel_count: int


@dataclass(frozen=True, slots=True)
class SparseposerClipIdentity:
    dataset_split: str
    subject: str
    clip: str
    source_clip: str


def sparseposer_clip_identity(path: str) -> SparseposerClipIdentity:
    pure_path = PurePosixPath(path)
    source_clip = pure_path.stem
    dataset_split = pure_path.parent.name if pure_path.parent.name in {"train", "eval"} else ""
    subject = source_clip.split("_", maxsplit=1)[0] if source_clip.startswith("S") else "named"
    clip = f"{dataset_split}_{source_clip}" if dataset_split else source_clip
    return SparseposerClipIdentity(
        dataset_split=dataset_split,
        subject=subject,
        clip=clip,
        source_clip=source_clip,
    )


def parse_sparseposer_bvh_text(text: str) -> SparseposerBvhMotion:
    lines = text.splitlines()
    try:
        motion_index = next(index for index, line in enumerate(lines) if line.strip() == "MOTION")
    except StopIteration as exc:
        raise ValueError("BVH text is missing MOTION section") from exc

    joints, channel_count = _parse_hierarchy("\n".join(lines[:motion_index]))
    source_frame_count = _parse_prefixed_int(lines[motion_index + 1], "Frames:")
    frame_time_s = _parse_prefixed_float(lines[motion_index + 2], "Frame Time:")
    frame_lines = tuple(line for line in lines[motion_index + 3 :] if line.strip())
    if len(frame_lines) < source_frame_count:
        raise ValueError(
            f"BVH declares {source_frame_count} frames but only {len(frame_lines)} frame rows exist"
        )
    return SparseposerBvhMotion(
        joints=tuple(joints),
        source_frame_count=source_frame_count,
        frame_time_s=frame_time_s,
        frame_lines=frame_lines[:source_frame_count],
        channel_count=channel_count,
    )


def iter_sparseposer_pose_frames(
    motion: SparseposerBvhMotion,
    *,
    frame_stride: int = 1,
    max_frames: int | None = None,
) -> Iterator[tuple[int, float, dict[str, Transform]]]:
    emitted = 0
    effective_stride = max(1, frame_stride)
    for row_index, line in enumerate(motion.frame_lines):
        if row_index % effective_stride != 0:
            continue
        values = tuple(float(value) for value in line.split())
        if len(values) != motion.channel_count:
            raise ValueError(
                f"BVH frame {row_index} has {len(values)} values; expected {motion.channel_count}"
            )
        yield (
            row_index,
            row_index * motion.frame_time_s,
            solve_sparseposer_pose_frame(
                motion.joints,
                values,
            ),
        )
        emitted += 1
        if max_frames is not None and emitted >= max_frames:
            return


def solve_sparseposer_pose_frame(
    joints: Sequence[SparseposerBvhJoint],
    frame_values: Sequence[float],
) -> dict[str, Transform]:
    positions: list[Vector3] = [(0.0, 0.0, 0.0)] * len(joints)
    rotations: list[Quaternion] = [IDENTITY_QUATERNION] * len(joints)
    transforms: dict[str, Transform] = {}

    for joint in joints:
        local_position = _local_position(joint, frame_values)
        local_rotation = _local_rotation(joint, frame_values)
        if joint.parent_index is None:
            position = add(joint.offset, local_position)
            rotation = local_rotation
        else:
            parent_position = positions[joint.parent_index]
            parent_rotation = rotations[joint.parent_index]
            position = add(
                parent_position,
                rotate_vector(parent_rotation, add(joint.offset, local_position)),
            )
            rotation = multiply_quaternion(parent_rotation, local_rotation)
        positions[joint.index] = position
        rotations[joint.index] = rotation
        transforms[joint.name] = Transform(position=position, rotation=rotation)
    return transforms


def canonical_pose_from_sparseposer(pose_frame: dict[str, Transform]) -> dict[str, Transform]:
    missing = [
        source for source in SPARSEPOSER_CANONICAL_JOINTS.values() if source not in pose_frame
    ]
    if missing:
        raise ValueError(f"SparsePoser pose is missing canonical joints: {', '.join(missing)}")
    return {
        canonical: pose_frame[source] for canonical, source in SPARSEPOSER_CANONICAL_JOINTS.items()
    }


def _parse_hierarchy(header_text: str) -> tuple[list[SparseposerBvhJoint], int]:
    tokens = _header_tokens(header_text)
    parser = _TokenParser(tokens)
    parser.expect("HIERARCHY")
    joints: list[SparseposerBvhJoint] = []
    channel_cursor = 0

    def parse_joint(parent_index: int | None) -> None:
        nonlocal channel_cursor
        kind = parser.next()
        if kind not in {"ROOT", "JOINT"}:
            raise ValueError(f"Expected ROOT or JOINT, got {kind!r}")
        name = parser.next()
        parser.expect("{")
        joint = SparseposerBvhJoint(
            index=len(joints),
            name=name,
            parent_index=parent_index,
            offset=(0.0, 0.0, 0.0),
            channels=(),
            channel_start=channel_cursor,
        )
        joints.append(joint)
        while True:
            token = parser.next()
            if token == "}":
                return
            if token == "OFFSET":
                joint.offset = (
                    float(parser.next()),
                    float(parser.next()),
                    float(parser.next()),
                )
                continue
            if token == "CHANNELS":
                channel_count = int(parser.next())
                joint.channel_start = channel_cursor
                joint.channels = tuple(parser.next() for _ in range(channel_count))
                channel_cursor += channel_count
                continue
            if token == "JOINT":
                parser.push_back(token)
                parse_joint(joint.index)
                continue
            if token == "End":
                parser.expect("Site")
                _skip_braced_block(parser)
                continue
            raise ValueError(f"Unexpected BVH hierarchy token {token!r}")

    parse_joint(None)
    return joints, channel_cursor


def _header_tokens(header_text: str) -> list[str]:
    return header_text.replace("{", " { ").replace("}", " } ").split()


def _skip_braced_block(parser: _TokenParser) -> None:
    parser.expect("{")
    depth = 1
    while depth:
        token = parser.next()
        if token == "{":
            depth += 1
        elif token == "}":
            depth -= 1


def _local_position(
    joint: SparseposerBvhJoint,
    frame_values: Sequence[float],
) -> Vector3:
    x = y = z = 0.0
    for offset, channel in enumerate(joint.channels):
        value = frame_values[joint.channel_start + offset]
        if channel == "Xposition":
            x = value
        elif channel == "Yposition":
            y = value
        elif channel == "Zposition":
            z = value
    return (x, y, z)


def _local_rotation(
    joint: SparseposerBvhJoint,
    frame_values: Sequence[float],
) -> Quaternion:
    channels: list[str] = []
    values: list[float] = []
    for offset, channel in enumerate(joint.channels):
        if channel.endswith("rotation"):
            channels.append(f"r{channel[0].lower()}")
            values.append(frame_values[joint.channel_start + offset])
    return euler_quaternion(tuple(channels), tuple(values))


def _parse_prefixed_int(line: str, prefix: str) -> int:
    return int(_parse_prefixed_value(line, prefix))


def _parse_prefixed_float(line: str, prefix: str) -> float:
    return float(_parse_prefixed_value(line, prefix))


def _parse_prefixed_value(line: str, prefix: str) -> str:
    stripped = line.strip()
    if not stripped.startswith(prefix):
        raise ValueError(f"Expected {prefix!r}, got {line!r}")
    return stripped[len(prefix) :].strip()


class _TokenParser:
    def __init__(self, tokens: Sequence[str]) -> None:
        self._tokens = tokens
        self._index = 0
        self._pushed_back: list[str] = []

    def next(self) -> str:
        if self._pushed_back:
            return self._pushed_back.pop()
        if self._index >= len(self._tokens):
            raise ValueError("Unexpected end of BVH hierarchy")
        token = self._tokens[self._index]
        self._index += 1
        return token

    def expect(self, expected: str) -> None:
        token = self.next()
        if token != expected:
            raise ValueError(f"Expected {expected!r}, got {token!r}")

    def push_back(self, token: str) -> None:
        self._pushed_back.append(token)
