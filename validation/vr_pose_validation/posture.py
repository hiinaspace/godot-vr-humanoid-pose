"""Reference-pose posture slices for validation reports."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from statistics import fmean
from typing import Literal

from validation.vr_pose_validation.cmu_metadata import ActivityClass, EnvironmentSplit
from validation.vr_pose_validation.geometry import Transform, horizontal_distance, length, subtract
from validation.vr_pose_validation.metrics import infer_reference_floor_height

type PostureClass = Literal[
    "upright",
    "crouched",
    "sitting",
    "reclined",
    "floor_reclined",
    "inverted",
    "unknown",
]
type PostureSubset = Literal[
    "vr_stationary",
    "vr_locomotion",
    "vr_other_upright",
    "sitting",
    "floor_reclined_ood",
    "inverted_ood",
    "nonflat_ood",
    "dynamic_ood",
    "unknown_ood",
]

POSTURE_CLASSES: tuple[PostureClass, ...] = (
    "upright",
    "crouched",
    "sitting",
    "reclined",
    "floor_reclined",
    "inverted",
    "unknown",
)
VR_POSTURE_SUBSETS: frozenset[PostureSubset] = frozenset(
    ("vr_stationary", "vr_locomotion", "vr_other_upright")
)
INVERTED_OOD_PCT_THRESHOLD = 20.0


@dataclass(frozen=True, slots=True)
class FramePosture:
    posture_class: PostureClass
    head_height_m: float
    hip_height_m: float
    torso_verticality: float
    head_hip_horizontal_m: float


@dataclass(frozen=True, slots=True)
class PostureSummary:
    posture_class: PostureClass
    posture_subset: PostureSubset
    frame_count: int
    upright_frames: int
    crouched_frames: int
    sitting_frames: int
    reclined_frames: int
    floor_reclined_frames: int
    inverted_frames: int
    unknown_frames: int
    upright_pct: float
    crouched_pct: float
    sitting_pct: float
    reclined_pct: float
    floor_reclined_pct: float
    inverted_pct: float
    unknown_pct: float
    mean_head_height_m: float
    mean_hip_height_m: float
    mean_torso_verticality: float
    hip_horizontal_displacement_m: float
    hip_horizontal_path_m: float
    mean_hip_horizontal_speed_m_per_s: float

    @property
    def vr_candidate(self) -> bool:
        return self.posture_subset in VR_POSTURE_SUBSETS


def summarize_reference_posture(
    reference_frames: Iterable[dict[str, Transform]],
    *,
    environment_split: EnvironmentSplit = "unknown",
    activity_class: ActivityClass = "unknown",
    floor_height_m: float | None = None,
    timestamps_seconds: Iterable[float] | None = None,
) -> PostureSummary:
    frame_list = list(reference_frames)
    timestamp_list = list(timestamps_seconds) if timestamps_seconds is not None else None
    inferred_floor_height = (
        infer_reference_floor_height(frame_list) if floor_height_m is None else floor_height_m
    )
    frame_postures = [
        classify_reference_posture(frame, floor_height_m=inferred_floor_height)
        for frame in frame_list
    ]
    counts = Counter(posture.posture_class for posture in frame_postures)
    posture_class = _dominant_posture_class(counts)
    frame_count = len(frame_postures)
    hip_motion = _hip_motion_summary(frame_list, timestamp_list)
    posture_subset = _posture_subset(
        posture_class=posture_class,
        counts=counts,
        frame_count=frame_count,
        environment_split=environment_split,
        activity_class=activity_class,
        hip_horizontal_displacement_m=hip_motion.displacement_m,
        mean_hip_horizontal_speed_m_per_s=hip_motion.mean_speed_m_per_s,
    )

    return PostureSummary(
        posture_class=posture_class,
        posture_subset=posture_subset,
        frame_count=frame_count,
        upright_frames=counts["upright"],
        crouched_frames=counts["crouched"],
        sitting_frames=counts["sitting"],
        reclined_frames=counts["reclined"],
        floor_reclined_frames=counts["floor_reclined"],
        inverted_frames=counts["inverted"],
        unknown_frames=counts["unknown"],
        upright_pct=_percent(counts["upright"], frame_count),
        crouched_pct=_percent(counts["crouched"], frame_count),
        sitting_pct=_percent(counts["sitting"], frame_count),
        reclined_pct=_percent(counts["reclined"], frame_count),
        floor_reclined_pct=_percent(counts["floor_reclined"], frame_count),
        inverted_pct=_percent(counts["inverted"], frame_count),
        unknown_pct=_percent(counts["unknown"], frame_count),
        mean_head_height_m=_mean(posture.head_height_m for posture in frame_postures),
        mean_hip_height_m=_mean(posture.hip_height_m for posture in frame_postures),
        mean_torso_verticality=_mean(posture.torso_verticality for posture in frame_postures),
        hip_horizontal_displacement_m=hip_motion.displacement_m,
        hip_horizontal_path_m=hip_motion.path_m,
        mean_hip_horizontal_speed_m_per_s=hip_motion.mean_speed_m_per_s,
    )


def classify_reference_posture(
    reference_joints: dict[str, Transform],
    *,
    floor_height_m: float,
) -> FramePosture:
    head = reference_joints.get("head")
    hips = reference_joints.get("hips")
    if head is None or hips is None:
        return FramePosture(
            posture_class="unknown",
            head_height_m=0.0,
            hip_height_m=0.0,
            torso_verticality=0.0,
            head_hip_horizontal_m=0.0,
        )

    head_position = head.position
    hip_position = hips.position
    head_height = head_position[1] - floor_height_m
    hip_height = hip_position[1] - floor_height_m
    torso_vector = subtract(head_position, hip_position)
    torso_length = length(torso_vector)
    torso_verticality = abs(torso_vector[1]) / torso_length if torso_length > 1e-9 else 0.0
    head_hip_horizontal = horizontal_distance(head_position, hip_position)

    return FramePosture(
        posture_class=_classify_features(
            head_height_m=head_height,
            hip_height_m=hip_height,
            torso_verticality=torso_verticality,
            head_hip_horizontal_m=head_hip_horizontal,
        ),
        head_height_m=head_height,
        hip_height_m=hip_height,
        torso_verticality=torso_verticality,
        head_hip_horizontal_m=head_hip_horizontal,
    )


def _classify_features(
    *,
    head_height_m: float,
    hip_height_m: float,
    torso_verticality: float,
    head_hip_horizontal_m: float,
) -> PostureClass:
    if head_height_m <= 0.0 and hip_height_m <= 0.0:
        return "unknown"
    if head_height_m < hip_height_m - 0.10:
        return "inverted"
    if (
        head_height_m < 0.95
        or hip_height_m < 0.35
        or (torso_verticality < 0.38 and head_height_m < 1.25)
    ):
        return "floor_reclined"
    if torso_verticality < 0.55 or head_hip_horizontal_m > 0.75:
        return "reclined"
    if hip_height_m < 0.72 and head_height_m < 1.45:
        return "sitting"
    if hip_height_m < 0.82:
        return "crouched"
    if head_height_m >= 1.10 and torso_verticality >= 0.62:
        return "upright"
    return "unknown"


def _posture_subset(
    *,
    posture_class: PostureClass,
    counts: Counter[PostureClass],
    frame_count: int,
    environment_split: EnvironmentSplit,
    activity_class: ActivityClass,
    hip_horizontal_displacement_m: float,
    mean_hip_horizontal_speed_m_per_s: float,
) -> PostureSubset:
    floor_reclined_pct = _percent(
        counts["floor_reclined"] + counts["reclined"],
        frame_count,
    )
    inverted_pct = _percent(counts["inverted"], frame_count)
    sitting_pct = _percent(counts["sitting"], frame_count)
    upright_like_pct = _percent(counts["upright"] + counts["crouched"], frame_count)

    if environment_split == "nonflat":
        return "nonflat_ood"
    if inverted_pct >= INVERTED_OOD_PCT_THRESHOLD:
        return "inverted_ood"
    if activity_class == "floor_pose" or floor_reclined_pct >= 20.0:
        return "floor_reclined_ood"
    if activity_class == "sitting" or sitting_pct >= 30.0:
        return "sitting"
    if activity_class in {"dance_exercise", "sports", "swimming"}:
        return "dynamic_ood"
    if upright_like_pct >= 60.0:
        if (
            activity_class == "locomotion"
            or hip_horizontal_displacement_m >= 0.75
            or mean_hip_horizontal_speed_m_per_s >= 0.25
        ):
            return "vr_locomotion"
        if activity_class in {"idle_standing", "stationary_upper_body"}:
            return "vr_stationary"
        return "vr_other_upright"
    if posture_class == "unknown":
        return "unknown_ood"
    return "unknown_ood"


@dataclass(frozen=True, slots=True)
class _HipMotionSummary:
    displacement_m: float
    path_m: float
    mean_speed_m_per_s: float


def _hip_motion_summary(
    reference_frames: list[dict[str, Transform]],
    timestamps_seconds: list[float] | None,
) -> _HipMotionSummary:
    samples = [
        (_timestamp_for_frame(index, timestamps_seconds), frame["hips"].position)
        for index, frame in enumerate(reference_frames)
        if "hips" in frame
    ]
    if len(samples) < 2:
        return _HipMotionSummary(displacement_m=0.0, path_m=0.0, mean_speed_m_per_s=0.0)

    path = sum(
        horizontal_distance(current_position, previous_position)
        for (_previous_time, previous_position), (_current_time, current_position) in zip(
            samples,
            samples[1:],
            strict=False,
        )
    )
    duration = samples[-1][0] - samples[0][0]
    return _HipMotionSummary(
        displacement_m=horizontal_distance(samples[0][1], samples[-1][1]),
        path_m=path,
        mean_speed_m_per_s=path / duration if duration > 0.0 else 0.0,
    )


def _timestamp_for_frame(index: int, timestamps_seconds: list[float] | None) -> float:
    if timestamps_seconds is None or index >= len(timestamps_seconds):
        return index / 120.0
    return timestamps_seconds[index]


def _dominant_posture_class(counts: Counter[PostureClass]) -> PostureClass:
    if not counts:
        return "unknown"
    return max(POSTURE_CLASSES, key=lambda posture_class: counts[posture_class])


def _percent(count: int, total: int) -> float:
    return count / total * 100.0 if total > 0 else 0.0


def _mean(values: Iterable[float]) -> float:
    value_list = list(values)
    return fmean(value_list) if value_list else 0.0
