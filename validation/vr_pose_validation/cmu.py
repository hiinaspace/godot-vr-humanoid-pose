"""Raw CMU ASF/AMC parsing helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

CMU_SCALE_TO_METERS = (1.0 / 0.45) * 2.54 / 100.0
CMU_FRAME_RATE_HZ = 120.0


@dataclass(frozen=True, slots=True)
class CmuRoot:
    order: tuple[str, ...] = ()
    axis_order: str = "XYZ"
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True, slots=True)
class CmuBone:
    bone_id: int
    name: str
    direction: tuple[float, float, float]
    length: float
    axis: tuple[float, float, float]
    axis_order: str
    dof: tuple[str, ...] = ()

    @property
    def length_m(self) -> float:
        return self.length * CMU_SCALE_TO_METERS


@dataclass(frozen=True, slots=True)
class CmuSkeleton:
    root: CmuRoot
    bones: dict[str, CmuBone]
    hierarchy: dict[str, tuple[str, ...]]
    units: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CmuFrame:
    index: int
    values: dict[str, tuple[float, ...]]


@dataclass(frozen=True, slots=True)
class CmuFrameSelection:
    frames: list[CmuFrame]
    source_frame_count: int


def load_asf(path: str | Path) -> CmuSkeleton:
    return parse_asf_text(Path(path).read_text(encoding="utf-8", errors="replace"))


def load_amc(path: str | Path) -> list[CmuFrame]:
    return parse_amc_text(Path(path).read_text(encoding="utf-8", errors="replace"))


def parse_asf_text(text: str) -> CmuSkeleton:
    lines = _clean_lines(text)
    root = CmuRoot()
    units: dict[str, str] = {}
    bones: dict[str, CmuBone] = {}
    hierarchy: dict[str, tuple[str, ...]] = {}

    index = 0
    while index < len(lines):
        line = lines[index]
        if not line.startswith(":"):
            index += 1
            continue

        section = line[1:].split(maxsplit=1)[0].lower()
        index += 1
        if section == "units":
            index, units = _parse_key_values_until_section(lines, index)
        elif section == "root":
            index, root = _parse_root(lines, index)
        elif section == "bonedata":
            index, bones = _parse_bonedata(lines, index)
        elif section == "hierarchy":
            index, hierarchy = _parse_hierarchy(lines, index)
        else:
            index = _skip_until_section(lines, index)

    return CmuSkeleton(root=root, bones=bones, hierarchy=hierarchy, units=units)


def parse_amc_text(text: str) -> list[CmuFrame]:
    return parse_amc_text_sampled(text).frames


def parse_amc_text_sampled(
    text: str,
    *,
    frame_stride: int = 1,
    max_frames: int | None = None,
) -> CmuFrameSelection:
    if frame_stride < 1:
        msg = f"frame_stride must be >= 1, got {frame_stride}"
        raise ValueError(msg)

    frames: list[CmuFrame] = []
    current_index: int | None = None
    current_values: dict[str, tuple[float, ...]] | None = None
    source_frame_count = 0
    selected_frame_count = 0

    for line in _clean_lines(text):
        if line.startswith(":"):
            continue
        tokens = line.split()
        if len(tokens) == 1 and tokens[0].lstrip("-").isdigit():
            if current_index is not None and current_values is not None:
                frames.append(CmuFrame(index=current_index, values=current_values))
            current_index = int(tokens[0])
            source_ordinal = source_frame_count
            source_frame_count += 1
            should_select = source_ordinal % frame_stride == 0 and (
                max_frames is None or selected_frame_count < max_frames
            )
            if should_select:
                current_values = {}
                selected_frame_count += 1
            else:
                current_values = None
            continue
        if current_index is None:
            msg = f"AMC joint row appeared before a frame number: {line!r}"
            raise ValueError(msg)
        if current_values is not None:
            current_values[tokens[0]] = tuple(float(value) for value in tokens[1:])

    if current_index is not None and current_values is not None:
        frames.append(CmuFrame(index=current_index, values=current_values))
    return CmuFrameSelection(frames=frames, source_frame_count=source_frame_count)


def scaled_root_translation(frame: CmuFrame, root: CmuRoot) -> tuple[float, float, float]:
    values = frame.values.get("root", ())
    result = [0.0, 0.0, 0.0]
    for axis_index, channel in enumerate(("TX", "TY", "TZ")):
        if channel in root.order:
            value_index = root.order.index(channel)
            if value_index < len(values):
                result[axis_index] = values[value_index] * CMU_SCALE_TO_METERS
    return (result[0], result[1], result[2])


def cmu_frame_timestamp_seconds(frame_index: int) -> float:
    """Return the zero-based timestamp for CMU's one-based AMC frame ids."""

    return (frame_index - 1) / CMU_FRAME_RATE_HZ


def ordered_joint_rotation(
    frame: CmuFrame, skeleton: CmuSkeleton, joint_name: str
) -> dict[str, float]:
    bone = skeleton.bones[joint_name]
    values = frame.values.get(joint_name, ())
    return {channel: values[index] for index, channel in enumerate(bone.dof) if index < len(values)}


def _clean_lines(text: str) -> list[str]:
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.split("#", maxsplit=1)[0].strip()
        if line:
            lines.append(line)
    return lines


def _skip_until_section(lines: list[str], index: int) -> int:
    while index < len(lines) and not lines[index].startswith(":"):
        index += 1
    return index


def _parse_key_values_until_section(lines: list[str], index: int) -> tuple[int, dict[str, str]]:
    values: dict[str, str] = {}
    while index < len(lines) and not lines[index].startswith(":"):
        tokens = lines[index].split(maxsplit=1)
        if tokens:
            values[tokens[0].lower()] = tokens[1] if len(tokens) > 1 else ""
        index += 1
    return index, values


def _parse_root(lines: list[str], index: int) -> tuple[int, CmuRoot]:
    root = CmuRoot()
    while index < len(lines) and not lines[index].startswith(":"):
        tokens = lines[index].split()
        key = tokens[0].lower()
        if key == "order":
            root = CmuRoot(
                order=tuple(tokens[1:]),
                axis_order=root.axis_order,
                position=root.position,
                orientation=root.orientation,
            )
        elif key == "axis":
            root = CmuRoot(
                order=root.order,
                axis_order=tokens[1],
                position=root.position,
                orientation=root.orientation,
            )
        elif key == "position":
            root = CmuRoot(
                order=root.order,
                axis_order=root.axis_order,
                position=_parse_float3(tokens[1:4]),
                orientation=root.orientation,
            )
        elif key == "orientation":
            root = CmuRoot(
                order=root.order,
                axis_order=root.axis_order,
                position=root.position,
                orientation=_parse_float3(tokens[1:4]),
            )
        index += 1
    return index, root


def _parse_bonedata(lines: list[str], index: int) -> tuple[int, dict[str, CmuBone]]:
    bones: dict[str, CmuBone] = {}
    while index < len(lines) and not lines[index].startswith(":"):
        if lines[index].lower() != "begin":
            index += 1
            continue

        index += 1
        fields: dict[str, list[str]] = {}
        while index < len(lines) and lines[index].lower() != "end":
            tokens = lines[index].split()
            fields[tokens[0].lower()] = tokens[1:]
            index += 1
        index += 1

        name = fields.get("name", [""])[0]
        if not name:
            continue
        axis_tokens = fields.get("axis", ["0", "0", "0", "XYZ"])
        bone = CmuBone(
            bone_id=int(fields.get("id", ["0"])[0]),
            name=name,
            direction=_parse_float3(fields.get("direction", ["0", "0", "0"])[:3]),
            length=float(fields.get("length", ["0"])[0]),
            axis=_parse_float3(axis_tokens[:3]),
            axis_order=axis_tokens[3] if len(axis_tokens) > 3 else "XYZ",
            dof=tuple(fields.get("dof", [])),
        )
        bones[name] = bone
    return index, bones


def _parse_hierarchy(lines: list[str], index: int) -> tuple[int, dict[str, tuple[str, ...]]]:
    hierarchy: dict[str, tuple[str, ...]] = {}
    while index < len(lines) and lines[index].lower() != "begin":
        index += 1
    index += 1
    while index < len(lines) and lines[index].lower() != "end":
        tokens = lines[index].split()
        if tokens:
            hierarchy[tokens[0]] = tuple(tokens[1:])
        index += 1
    return index + 1, hierarchy


def _parse_float3(values: list[str]) -> tuple[float, float, float]:
    padded = [*values, "0", "0", "0"]
    return (float(padded[0]), float(padded[1]), float(padded[2]))
