"""Basis-inspired sparse VR pose estimator phases.

This is a data-only port of the pieces that can be exercised before the Godot
runtime is wired in: virtual spine/hips synthesis, foot anchors, and two-bone
limb midpoint solves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, pi, sin
from typing import Literal

from validation.vr_pose_validation.geometry import (
    Quaternion,
    Transform,
    Vector3,
    add,
    cross,
    dot,
    horizontal_distance,
    lerp,
    look_rotation,
    normalize,
    project_on_plane,
    rotate_vector,
    scale,
    subtract,
)
from validation.vr_pose_validation.types import (
    AvatarCalibration,
    GroundingContext,
    HumanoidPoseFrame,
    JointPose,
    TrackedPoseSample,
    TrackerState,
)

WORLD_UP: Vector3 = (0.0, 1.0, 0.0)
WORLD_FORWARD: Vector3 = (0.0, 0.0, 1.0)
type FootPhase = Literal["unplanted", "planted", "swing", "settle"]


@dataclass(frozen=True, slots=True)
class BasisEstimatorSettings:
    """Tunable proportions matching the first Python-portable Basis phases."""

    head_to_hips_ratio: float = 0.34
    shoulder_width_ratio: float = 0.23
    upper_arm_ratio: float = 0.19
    lower_arm_ratio: float = 0.16
    upper_leg_ratio: float = 0.245
    lower_leg_ratio: float = 0.246
    head_to_neck_ratio: float = 0.04
    neck_to_chest_ratio: float = 0.09
    chest_to_spine_ratio: float = 0.10
    spine_to_hips_ratio: float = 0.11
    stance_width_ratio: float = 0.17
    hip_width_ratio: float = 0.11
    hips_forward_bias_m: float = 0.02
    hips_xz_follow_blend: float = 0.35
    max_anchor_stretch_m: float = 0.12
    plant_velocity_threshold_m_s: float = 0.01
    release_velocity_multiplier: float = 2.0
    foot_position_lerp_speed: float = 15.0
    planted_foot_lerp_speed: float = 25.0
    foot_normal_lerp_speed: float = 12.0
    step_duration_s: float = 0.18
    step_lift_height_m: float = 0.06
    planted_foot_confidence: float = 0.75


@dataclass(frozen=True, slots=True)
class _TorsoChainLengths:
    head_to_neck: float
    neck_to_chest: float
    chest_to_spine: float
    spine_to_hips: float

    @property
    def neck_to_spine(self) -> float:
        return self.neck_to_chest + self.chest_to_spine

    @property
    def neck_to_hips(self) -> float:
        return self.neck_to_spine + self.spine_to_hips


@dataclass(slots=True)
class _FootState:
    anchor_position: Vector3 | None = None
    anchor_normal: Vector3 = WORLD_UP
    filtered_normal: Vector3 = WORLD_UP
    current_position: Vector3 | None = None
    current_rotation: Quaternion | None = None
    previous_target: Vector3 | None = None
    previous_timestamp_seconds: float | None = None
    swing_start_position: Vector3 | None = None
    swing_start_time_seconds: float | None = None
    planted: bool = False
    phase: FootPhase = "unplanted"


@dataclass(frozen=True, slots=True)
class _FootTarget:
    position: Vector3
    rotation: tuple[float, float, float, float]


@dataclass(slots=True)
class BasisPortEstimator:
    """Incremental Python port of Basis' local sparse-body solver.

    Basis' runtime estimator is split across several Unity/Godot-side drivers.
    This class keeps the same broad phase ordering for the validation harness:
    derive a body target from HMD/controller trackers, synthesize a virtual
    torso, keep floor-projected feet sticky across sequential frames, then place
    knees and elbows using two-bone bend hints.
    """

    settings: BasisEstimatorSettings = field(default_factory=BasisEstimatorSettings)
    _foot_states: dict[str, _FootState] = field(default_factory=dict, init=False)

    def reset(self) -> None:
        self._foot_states.clear()

    def estimate(
        self,
        sample: TrackedPoseSample,
        calibration: AvatarCalibration,
        grounding: GroundingContext,
    ) -> HumanoidPoseFrame:
        head_source = _source_joint_position(sample.hmd)
        forward = _head_forward(sample.hmd)
        right = normalize(cross(WORLD_UP, forward), fallback=(1.0, 0.0, 0.0))

        torso_lengths = self._torso_chain_lengths(calibration)
        neck_position = subtract(head_source, scale(WORLD_UP, torso_lengths.head_to_neck))
        hips_position = self._estimate_hips(
            sample,
            grounding,
            neck_position,
            forward,
            torso_lengths.neck_to_hips,
        )
        chest_position = lerp(
            neck_position,
            hips_position,
            torso_lengths.neck_to_chest / torso_lengths.neck_to_hips,
        )
        spine_position = lerp(
            neck_position,
            hips_position,
            torso_lengths.neck_to_spine / torso_lengths.neck_to_hips,
        )
        torso_rotation = _yaw_rotation(forward)
        shoulder_width = _calibrated_length(
            calibration,
            ("shoulder_width", "left_shoulder_to_right_shoulder"),
            calibration.human_height_m * self.settings.shoulder_width_ratio,
        )
        left_shoulder = add(chest_position, scale(right, -shoulder_width * 0.5))
        right_shoulder = add(chest_position, scale(right, shoulder_width * 0.5))
        left_hand_source = _source_joint_position(sample.left_hand)
        right_hand_source = _source_joint_position(sample.right_hand)

        left_foot = self._estimate_foot(
            "left_foot",
            hips_position,
            right,
            -1.0,
            forward,
            calibration,
            grounding,
            sample.optional_trackers.get("left_foot"),
            sample.timestamp_seconds,
        )
        right_foot = self._estimate_foot(
            "right_foot",
            hips_position,
            right,
            1.0,
            forward,
            calibration,
            grounding,
            sample.optional_trackers.get("right_foot"),
            sample.timestamp_seconds,
        )
        left_hip = self._hip_socket(hips_position, right, -1.0, calibration)
        right_hip = self._hip_socket(hips_position, right, 1.0, calibration)

        joints = {
            "head": JointPose(sample.hmd.transform, sample.hmd.confidence, estimated=False),
            "left_hand": JointPose(
                sample.left_hand.transform,
                sample.left_hand.confidence,
                estimated=False,
            ),
            "right_hand": JointPose(
                sample.right_hand.transform,
                sample.right_hand.confidence,
                estimated=False,
            ),
            "hips": JointPose(
                Transform(position=hips_position, rotation=torso_rotation),
                estimated=True,
            ),
            "chest": JointPose(
                Transform(position=chest_position, rotation=torso_rotation),
                estimated=True,
            ),
            "spine": JointPose(
                Transform(position=spine_position, rotation=torso_rotation),
                estimated=True,
            ),
            "left_shoulder": JointPose(Transform(position=left_shoulder), estimated=True),
            "right_shoulder": JointPose(Transform(position=right_shoulder), estimated=True),
            "left_elbow": JointPose(
                Transform(
                    position=self._estimate_elbow(
                        left_shoulder,
                        left_hand_source,
                        right,
                        forward,
                        -1.0,
                        calibration,
                    )
                ),
                estimated=True,
            ),
            "right_elbow": JointPose(
                Transform(
                    position=self._estimate_elbow(
                        right_shoulder,
                        right_hand_source,
                        right,
                        forward,
                        1.0,
                        calibration,
                    )
                ),
                estimated=True,
            ),
            "left_foot": JointPose(
                Transform(position=left_foot.position, rotation=left_foot.rotation),
                confidence=self.settings.planted_foot_confidence,
                estimated=True,
            ),
            "right_foot": JointPose(
                Transform(position=right_foot.position, rotation=right_foot.rotation),
                confidence=self.settings.planted_foot_confidence,
                estimated=True,
            ),
        }
        joints["left_knee"] = JointPose(
            Transform(
                position=self._estimate_knee(
                    left_hip,
                    left_foot.position,
                    forward,
                    calibration,
                )
            ),
            estimated=True,
        )
        joints["right_knee"] = JointPose(
            Transform(
                position=self._estimate_knee(
                    right_hip,
                    right_foot.position,
                    forward,
                    calibration,
                )
            ),
            estimated=True,
        )

        return HumanoidPoseFrame(
            timestamp_seconds=sample.timestamp_seconds,
            root=Transform(position=hips_position, rotation=torso_rotation),
            joints=joints,
        )

    def _torso_chain_lengths(self, calibration: AvatarCalibration) -> _TorsoChainLengths:
        head_to_hips = _calibrated_length(
            calibration,
            ("head_to_hips", "hips_to_head"),
            calibration.human_height_m * self.settings.head_to_hips_ratio,
        )
        head_to_neck = min(
            head_to_hips * 0.5,
            _calibrated_length(
                calibration,
                ("head_to_neck", "neck_to_head"),
                calibration.human_height_m * self.settings.head_to_neck_ratio,
            ),
        )
        raw_segments = (
            _calibrated_length(
                calibration,
                ("neck_to_chest",),
                calibration.human_height_m * self.settings.neck_to_chest_ratio,
            ),
            _calibrated_length(
                calibration,
                ("chest_to_spine",),
                calibration.human_height_m * self.settings.chest_to_spine_ratio,
            ),
            _calibrated_length(
                calibration,
                ("spine_to_hips", "hips_to_spine"),
                calibration.human_height_m * self.settings.spine_to_hips_ratio,
            ),
        )
        raw_total = max(sum(raw_segments), 1e-4)
        target_neck_to_hips = max(head_to_hips - head_to_neck, 1e-4)
        scale_factor = target_neck_to_hips / raw_total
        neck_to_chest, chest_to_spine, spine_to_hips = (
            segment * scale_factor for segment in raw_segments
        )
        return _TorsoChainLengths(
            head_to_neck=head_to_neck,
            neck_to_chest=neck_to_chest,
            chest_to_spine=chest_to_spine,
            spine_to_hips=spine_to_hips,
        )

    def _estimate_hips(
        self,
        sample: TrackedPoseSample,
        grounding: GroundingContext,
        neck_position: Vector3,
        forward: Vector3,
        neck_to_hips_length: float,
    ) -> Vector3:
        floor_y = grounding.floor_height_m
        ideal = add(
            (
                neck_position[0],
                max(floor_y, neck_position[1] - neck_to_hips_length),
                neck_position[2],
            ),
            scale(forward, self.settings.hips_forward_bias_m),
        )

        pelvis_tracker = sample.optional_trackers.get("pelvis")
        if pelvis_tracker is None or not pelvis_tracker.valid:
            return ideal

        pelvis_position = _source_joint_position(pelvis_tracker)
        blend = self.settings.hips_xz_follow_blend
        return (
            ideal[0] + (pelvis_position[0] - ideal[0]) * blend,
            ideal[1],
            ideal[2] + (pelvis_position[2] - ideal[2]) * blend,
        )

    def _estimate_foot(
        self,
        name: str,
        hips_position: Vector3,
        right: Vector3,
        side: float,
        forward: Vector3,
        calibration: AvatarCalibration,
        grounding: GroundingContext,
        tracker: TrackerState | None,
        timestamp_seconds: float,
    ) -> _FootTarget:
        state = self._foot_states.setdefault(name, _FootState(anchor_normal=grounding.floor_normal))
        dt = _state_dt(state, timestamp_seconds)
        state.filtered_normal = _smooth_normal(
            state.filtered_normal,
            grounding.floor_normal,
            _exp_alpha(self.settings.foot_normal_lerp_speed, dt),
        )
        if tracker is not None and tracker.valid:
            position = _source_joint_position(tracker)
            state.anchor_position = position
            state.anchor_normal = grounding.floor_normal
            state.filtered_normal = grounding.floor_normal
            state.current_position = position
            state.current_rotation = _foot_rotation(forward, state.anchor_normal)
            state.previous_target = position
            state.previous_timestamp_seconds = timestamp_seconds
            state.planted = True
            state.phase = "planted"
            return _FootTarget(
                position=position,
                rotation=state.current_rotation,
            )

        stance_width = _calibrated_length(
            calibration,
            ("stance_width", "foot_separation"),
            calibration.human_height_m * self.settings.stance_width_ratio,
        )
        floor_y = grounding.floor_height_m + calibration.foot_height_m
        target = add(hips_position, scale(right, side * stance_width * 0.5))
        target = (target[0], floor_y, target[2])
        speed = _target_speed(state, target, timestamp_seconds)
        if state.current_position is None:
            state.current_position = target
        if state.current_rotation is None:
            state.current_rotation = _foot_rotation(forward, state.filtered_normal)

        if state.planted and state.anchor_position is not None:
            stretch = horizontal_distance(state.anchor_position, target)
            release_speed = (
                self.settings.plant_velocity_threshold_m_s
                * self.settings.release_velocity_multiplier
            )
            if stretch > self.settings.max_anchor_stretch_m or speed > release_speed:
                state.planted = False
                state.phase = "swing"
                state.swing_start_position = state.current_position
                state.swing_start_time_seconds = timestamp_seconds - dt
                state.anchor_position = None

        desired_position = target
        if state.phase == "swing" and state.swing_start_position is not None:
            progress = _step_progress(
                timestamp_seconds,
                state.swing_start_time_seconds,
                self.settings.step_duration_s,
            )
            desired_position = _swing_position(
                state.swing_start_position,
                target,
                progress,
                self.settings.step_lift_height_m,
            )
            if progress >= 1.0:
                state.phase = "settle"

        if state.phase == "settle":
            if horizontal_distance(state.current_position, target) <= 0.01:
                state.anchor_position = target
                state.anchor_normal = grounding.floor_normal
                state.planted = True
                state.phase = "planted"
            desired_position = target

        if not state.planted and state.phase == "unplanted":
            if speed <= self.settings.plant_velocity_threshold_m_s:
                state.anchor_position = target
                state.anchor_normal = grounding.floor_normal
                state.planted = True
                state.phase = "planted"

        if state.planted and state.anchor_position is not None:
            desired_position = state.anchor_position

        state.previous_target = target
        state.previous_timestamp_seconds = timestamp_seconds
        position_speed = (
            self.settings.planted_foot_lerp_speed
            if state.planted
            else self.settings.foot_position_lerp_speed
        )
        state.current_position = lerp(
            state.current_position,
            desired_position,
            _exp_alpha(position_speed, dt),
        )
        desired_rotation = _foot_rotation(
            forward, state.anchor_normal if state.planted else state.filtered_normal
        )
        state.current_rotation = _nlerp_quaternion(
            state.current_rotation,
            desired_rotation,
            _exp_alpha(self.settings.foot_position_lerp_speed, dt),
        )
        return _FootTarget(
            position=state.current_position,
            rotation=state.current_rotation,
        )

    def _hip_socket(
        self,
        hips_position: Vector3,
        right: Vector3,
        side: float,
        calibration: AvatarCalibration,
    ) -> Vector3:
        hip_width = _calibrated_length(
            calibration,
            ("hip_width", "pelvis_width"),
            calibration.human_height_m * self.settings.hip_width_ratio,
        )
        return add(hips_position, scale(right, side * hip_width * 0.5))

    def _estimate_knee(
        self,
        hip_position: Vector3,
        foot_position: Vector3,
        forward: Vector3,
        calibration: AvatarCalibration,
    ) -> Vector3:
        upper_leg = _calibrated_length(
            calibration,
            ("upper_leg", "thigh", "femur"),
            calibration.human_height_m * self.settings.upper_leg_ratio,
        )
        lower_leg = _calibrated_length(
            calibration,
            ("lower_leg", "shin", "tibia"),
            calibration.human_height_m * self.settings.lower_leg_ratio,
        )
        return _solve_two_bone_midpoint(hip_position, foot_position, upper_leg, lower_leg, forward)

    def _estimate_elbow(
        self,
        shoulder_position: Vector3,
        hand_position: Vector3,
        right: Vector3,
        forward: Vector3,
        side: float,
        calibration: AvatarCalibration,
    ) -> Vector3:
        upper_arm = _calibrated_length(
            calibration,
            ("upper_arm", "humerus"),
            calibration.human_height_m * self.settings.upper_arm_ratio,
        )
        lower_arm = _calibrated_length(
            calibration,
            ("lower_arm", "forearm", "radius"),
            calibration.human_height_m * self.settings.lower_arm_ratio,
        )
        outward = scale(right, side)
        hint = normalize(add(scale(forward, 0.7), scale(outward, 0.3)), fallback=forward)
        return _solve_two_bone_midpoint(
            shoulder_position,
            hand_position,
            upper_arm,
            lower_arm,
            hint,
        )


def _source_joint_position(tracker: TrackerState) -> Vector3:
    offset = (
        tracker.attachment.local_transform.position
        if tracker.attachment is not None
        else (0.0, 0.0, 0.0)
    )
    return subtract(tracker.transform.position, offset)


def _head_forward(tracker: TrackerState) -> Vector3:
    forward = rotate_vector(tracker.transform.rotation, WORLD_FORWARD)
    return normalize((forward[0], 0.0, forward[2]), fallback=WORLD_FORWARD)


def _yaw_rotation(forward: Vector3) -> Quaternion:
    return look_rotation(project_on_plane(forward, WORLD_UP), WORLD_UP)


def _foot_rotation(
    forward: Vector3,
    normal: Vector3,
) -> tuple[float, float, float, float]:
    foot_forward = project_on_plane(forward, normal)
    return look_rotation(foot_forward, normal)


def _target_speed(state: _FootState, target: Vector3, timestamp_seconds: float) -> float:
    if state.previous_target is None or state.previous_timestamp_seconds is None:
        return 0.0
    dt = max(timestamp_seconds - state.previous_timestamp_seconds, 1.0 / 120.0)
    return horizontal_distance(target, state.previous_target) / dt


def _state_dt(state: _FootState, timestamp_seconds: float) -> float:
    if state.previous_timestamp_seconds is None:
        return 1.0 / 120.0
    return max(timestamp_seconds - state.previous_timestamp_seconds, 1.0 / 120.0)


def _exp_alpha(speed: float, dt: float) -> float:
    if speed <= 0.0:
        return 1.0
    return 1.0 - exp(-speed * dt)


def _smooth_normal(current: Vector3, target: Vector3, alpha: float) -> Vector3:
    return normalize(lerp(current, target, alpha), fallback=WORLD_UP)


def _step_progress(
    timestamp_seconds: float,
    swing_start_time_seconds: float | None,
    step_duration_s: float,
) -> float:
    if swing_start_time_seconds is None:
        return 1.0
    return min(1.0, max(0.0, (timestamp_seconds - swing_start_time_seconds) / step_duration_s))


def _swing_position(
    start: Vector3,
    target: Vector3,
    progress: float,
    lift_height_m: float,
) -> Vector3:
    eased = progress * progress * (3.0 - 2.0 * progress)
    base = lerp(start, target, eased)
    lift = sin(pi * progress) * lift_height_m
    return (base[0], base[1] + lift, base[2])


def _nlerp_quaternion(a: Quaternion, b: Quaternion, alpha: float) -> Quaternion:
    bx, by, bz, bw = b
    if a[0] * bx + a[1] * by + a[2] * bz + a[3] * bw < 0.0:
        bx, by, bz, bw = -bx, -by, -bz, -bw
    blended = (
        a[0] + (bx - a[0]) * alpha,
        a[1] + (by - a[1]) * alpha,
        a[2] + (bz - a[2]) * alpha,
        a[3] + (bw - a[3]) * alpha,
    )
    magnitude = (
        blended[0] * blended[0]
        + blended[1] * blended[1]
        + blended[2] * blended[2]
        + blended[3] * blended[3]
    ) ** 0.5
    if magnitude <= 1e-9:
        return b
    return (
        blended[0] / magnitude,
        blended[1] / magnitude,
        blended[2] / magnitude,
        blended[3] / magnitude,
    )


def _solve_two_bone_midpoint(
    root: Vector3,
    tip: Vector3,
    root_to_mid_length: float,
    mid_to_tip_length: float,
    bend_hint: Vector3,
) -> Vector3:
    root_to_tip = subtract(tip, root)
    distance = max(0.0001, _length(root_to_tip))
    max_reach = max(0.0001, root_to_mid_length + mid_to_tip_length - 0.0001)
    min_reach = abs(root_to_mid_length - mid_to_tip_length) + 0.0001
    clamped_distance = min(max(distance, min_reach), max_reach)
    direction = normalize(root_to_tip, fallback=WORLD_UP)

    along = (
        root_to_mid_length * root_to_mid_length
        - mid_to_tip_length * mid_to_tip_length
        + clamped_distance * clamped_distance
    ) / (2.0 * clamped_distance)
    bend_height_squared = root_to_mid_length * root_to_mid_length - along * along
    bend_height = bend_height_squared**0.5 if bend_height_squared > 0.0 else 0.0

    bend_direction = project_on_plane(bend_hint, direction)
    bend_direction = normalize(bend_direction, fallback=_fallback_bend_direction(direction))
    return add(add(root, scale(direction, along)), scale(bend_direction, bend_height))


def _calibrated_length(
    calibration: AvatarCalibration,
    names: tuple[str, ...],
    fallback: float,
) -> float:
    for name in names:
        value = calibration.bone_lengths_m.get(name)
        if value is not None and value > 0.0:
            return value * calibration.avatar_scale
    return fallback * calibration.avatar_scale


def _length(vector: Vector3) -> float:
    return dot(vector, vector) ** 0.5


def _fallback_bend_direction(direction: Vector3) -> Vector3:
    fallback = cross(direction, WORLD_UP)
    if dot(fallback, fallback) <= 1e-8:
        fallback = cross(direction, (1.0, 0.0, 0.0))
    return normalize(fallback, fallback=WORLD_FORWARD)
