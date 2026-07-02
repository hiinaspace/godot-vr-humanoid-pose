"""Generate a personalized SparsePoser Xsens reference BVH from live T-pose capture."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from validation.vr_pose_validation.geometry import Transform, distance
from validation.vr_pose_validation.sparseposer_live import (
    SPARSEPOSER_INPUT_JOINTS,
    SparsePoserBridgeConfig,
    SparsePoserLiveBackend,
    _alignment_role_transforms,
)

ARM_JOINTS = {
    "LeftCollar",
    "LeftShoulder",
    "LeftElbow",
    "LeftWrist",
    "RightCollar",
    "RightShoulder",
    "RightElbow",
    "RightWrist",
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-messages", type=Path, required=True)
    parser.add_argument("--reference-bvh", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--alignment",
        choices=("first", "last"),
        default="last",
        help="Which SparsePoser alignment event to use from the raw message log.",
    )
    parser.add_argument(
        "--max-arm-extra-scale",
        type=float,
        default=1.35,
        help="Clamp arm-only scale after height normalization.",
    )
    args = parser.parse_args(argv)

    alignment = _load_alignment(args.raw_messages, selection=args.alignment)
    target_roles = _alignment_role_transforms(alignment)
    source_roles = _source_roles(args.reference_bvh)

    height_scale = _height_scale(target_roles, source_roles)
    arm_extra_scale = _arm_extra_scale(
        target_roles,
        source_roles,
        height_scale=height_scale,
        max_scale=args.max_arm_extra_scale,
    )
    output_text = _scaled_reference_text(
        args.reference_bvh.read_text(encoding="utf-8"),
        height_scale=height_scale,
        arm_extra_scale=arm_extra_scale,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output_text, encoding="utf-8", newline="\n")

    print(f"wrote {args.output}")
    print(f"height_scale={height_scale:.6f}")
    print(f"arm_extra_scale={arm_extra_scale:.6f}")
    print(f"effective_arm_scale={height_scale * arm_extra_scale:.6f}")
    return 0


def _load_alignment(raw_messages: Path, *, selection: str) -> Mapping[str, Any]:
    alignments: list[Mapping[str, Any]] = []
    for line in raw_messages.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        message = json.loads(line)
        if message.get("type") != "estimate_request":
            continue
        calibration = _mapping_or_empty(message.get("calibration"))
        alignment = _mapping_or_empty(calibration.get("sparseposer_alignment"))
        if not alignment:
            live_calibration = _mapping_or_empty(calibration.get("live_tracker_calibration"))
            alignment = _mapping_or_empty(live_calibration.get("sparseposer_alignment"))
        if alignment:
            alignments.append(alignment)
    if not alignments:
        raise ValueError(f"{raw_messages} has no SparsePoser alignment messages")
    return alignments[0] if selection == "first" else alignments[-1]


def _source_roles(reference_bvh: Path) -> dict[str, Transform]:
    backend = SparsePoserLiveBackend(
        SparsePoserBridgeConfig(sparseposer_reference_bvh=reference_bvh),
        engine=_NoopEngine(),
    )
    try:
        return {
            role: backend._source_tpose_transform(role)
            for role in SPARSEPOSER_INPUT_JOINTS
        }
    finally:
        backend.close()


def _height_scale(
    target_roles: Mapping[str, Transform],
    source_roles: Mapping[str, Transform],
) -> float:
    target_floor = min(
        target_roles["left_foot"].position[1],
        target_roles["right_foot"].position[1],
    )
    source_floor = min(
        source_roles["left_foot"].position[1],
        source_roles["right_foot"].position[1],
    )
    target_height = target_roles["head"].position[1] - target_floor
    source_height = source_roles["head"].position[1] - source_floor
    if target_height <= 1e-6 or source_height <= 1e-6:
        return 1.0
    return target_height / source_height


def _arm_extra_scale(
    target_roles: Mapping[str, Transform],
    source_roles: Mapping[str, Transform],
    *,
    height_scale: float,
    max_scale: float,
) -> float:
    target_span = distance(target_roles["left_hand"].position, target_roles["right_hand"].position)
    source_span = distance(source_roles["left_hand"].position, source_roles["right_hand"].position)
    if target_span <= 1e-6 or source_span <= 1e-6 or height_scale <= 1e-6:
        return 1.0
    return min(max(target_span / (source_span * height_scale), 0.5), max_scale)


def _scaled_reference_text(
    text: str,
    *,
    height_scale: float,
    arm_extra_scale: float,
) -> str:
    output_lines: list[str] = []
    pending_joint: str | None = None
    in_motion = False
    motion_value_line_seen = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "MOTION":
            in_motion = True
            output_lines.append(line)
            continue
        if in_motion:
            if _is_motion_value_line(stripped) and not motion_value_line_seen:
                output_lines.append(_scale_motion_root_position_line(line, height_scale))
                motion_value_line_seen = True
            else:
                output_lines.append(line)
            continue
        if stripped.startswith("ROOT ") or stripped.startswith("JOINT "):
            parts = stripped.split()
            pending_joint = parts[1] if len(parts) >= 2 else None
            output_lines.append(line)
            continue
        if stripped.startswith("OFFSET ") and pending_joint is not None:
            joint_scale = height_scale * (arm_extra_scale if pending_joint in ARM_JOINTS else 1.0)
            output_lines.append(_scale_offset_line(line, joint_scale))
            pending_joint = None
            continue
        output_lines.append(line)
    return "\n".join(output_lines) + "\n"


def _scale_offset_line(line: str, scale: float) -> str:
    indent = line[: len(line) - len(line.lstrip())]
    values = [float(value) * scale for value in line.split()[1:4]]
    return f"{indent}OFFSET {values[0]:.6f} {values[1]:.6f} {values[2]:.6f}"


def _scale_motion_root_position_line(line: str, scale: float) -> str:
    values = [float(value) for value in line.split()]
    for index in range(min(3, len(values))):
        values[index] *= scale
    return " ".join(f"{value:.6f}" for value in values)


def _is_motion_value_line(stripped: str) -> bool:
    if not stripped or stripped[0].isalpha():
        return False
    try:
        float(stripped.split()[0])
    except ValueError:
        return False
    return True


def _mapping_or_empty(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


class _NoopEngine:
    def infer(self, packet: bytes) -> bytes:
        raise RuntimeError("reference-body generation does not run SparsePoser inference")

    def close(self) -> None:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
