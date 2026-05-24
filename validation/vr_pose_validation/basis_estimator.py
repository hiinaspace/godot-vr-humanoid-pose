"""Basis-inspired sparse VR pose estimator phases.

This is a data-only port of the pieces that can be exercised before the Godot
runtime is wired in: virtual spine/hips synthesis, foot anchors, and two-bone
limb midpoint solves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, pi, radians, sin, tan
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
type ArmBendStrategy = Literal["default", "basis_lookup", "manikin_swivel"]


@dataclass(frozen=True, slots=True)
class BasisEstimatorSettings:
    """Tunable proportions matching the first Python-portable Basis phases."""

    head_to_hips_ratio: float = 0.34
    shoulder_width_ratio: float = 0.201
    upper_arm_ratio: float = 0.174
    lower_arm_ratio: float = 0.151
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
    support_hips_xz_follow_blend: float = 0.65
    max_torso_lean_degrees: float = 35.0
    controller_lateral_flip_threshold_m: float = 0.08
    controller_lateral_flip_min_head_pitch: float = 0.20
    neck_rotation_lerp_speed: float = 40.0
    chest_rotation_lerp_speed: float = 25.0
    spine_rotation_lerp_speed: float = 30.0
    hips_rotation_lerp_speed: float = 20.0
    max_anchor_stretch_m: float = 0.12
    plant_velocity_threshold_m_s: float = 0.01
    release_velocity_multiplier: float = 2.0
    foot_position_lerp_speed: float = 15.0
    planted_foot_lerp_speed: float = 25.0
    foot_normal_lerp_speed: float = 12.0
    step_duration_s: float = 0.18
    step_lift_height_m: float = 0.06
    planted_foot_confidence: float = 0.75
    footstep_planner_enabled: bool = False
    footstep_release_stretch_m: float = 0.24
    footstep_speed_release_reduction_s: float = 0.0
    footstep_forward_lead_m: float = 0.16
    footstep_speed_lead_s: float = 0.05
    footstep_max_forward_lead_m: float = 0.28
    footstep_max_step_length_m: float = 0.55
    footstep_swing_retarget_stretch_m: float = 0.22
    footstep_max_neutral_stretch_m: float = 0.60
    footstep_min_lateral_ratio: float = 0.35
    footstep_min_interval_s: float = 0.12
    footstep_startup_align_s: float = 0.0
    footstep_catchup_release_stretch_m: float = 0.52
    footstep_trailing_release_stretch_m: float = 0.34
    footstep_stop_speed_m_s: float = 0.10
    footstep_catchup_duration_s: float = 0.16
    footstep_speed_duration_reduction_s: float = 0.0
    knee_hint_enabled: bool = False
    knee_hint_lateral_weight: float = 0.12
    knee_hint_lerp_speed: float = 20.0
    arm_bend_strategy: ArmBendStrategy = "default"
    arm_bend_lookup_blend: float = 0.0
    arm_swivel_down_weight: float = 0.55
    arm_swivel_outward_weight: float = 0.35
    arm_swivel_backward_weight: float = 0.45
    arm_swivel_cross_body_inward_weight: float = 0.25
    arm_swivel_overhead_relief: float = 0.50
    shoulder_reach_engage_ratio: float = 0.7
    shoulder_elevation_gain_m: float = 0.05
    shoulder_protraction_gain_m: float = 0.04
    shoulder_cross_body_gain_m: float = 0.025
    shoulder_reach_slack_m: float = 0.02
    max_shoulder_reach_shift_m: float = 0.16


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
    swing_target_position: Vector3 | None = None
    swing_forward_lead_scale: float = 1.0
    swing_duration_s: float | None = None
    filtered_knee_hint: Vector3 | None = None
    first_timestamp_seconds: float | None = None
    planted: bool = False
    phase: FootPhase = "unplanted"


@dataclass(frozen=True, slots=True)
class _FootTarget:
    position: Vector3
    rotation: tuple[float, float, float, float]


@dataclass(slots=True)
class _TorsoState:
    neck_rotation: Quaternion | None = None
    chest_rotation: Quaternion | None = None
    spine_rotation: Quaternion | None = None
    hips_rotation: Quaternion | None = None
    previous_timestamp_seconds: float | None = None


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
    _torso_state: _TorsoState = field(default_factory=_TorsoState, init=False)
    _active_swing_foot: str | None = field(default=None, init=False)
    _last_stepped_foot: str | None = field(default=None, init=False)
    _last_step_time_seconds: float | None = field(default=None, init=False)

    def reset(self) -> None:
        self._foot_states.clear()
        self._torso_state = _TorsoState()
        self._active_swing_foot = None
        self._last_stepped_foot = None
        self._last_step_time_seconds = None

    def estimate(
        self,
        sample: TrackedPoseSample,
        calibration: AvatarCalibration,
        grounding: GroundingContext,
    ) -> HumanoidPoseFrame:
        dt = _state_dt(self._torso_state, sample.timestamp_seconds)
        self._torso_state.previous_timestamp_seconds = sample.timestamp_seconds
        head_source = _source_joint_position(sample.hmd)
        head_rotation = sample.hmd.transform.rotation
        forward = _disambiguate_body_forward(
            sample,
            _head_forward(sample.hmd),
            self.settings.controller_lateral_flip_threshold_m,
            self.settings.controller_lateral_flip_min_head_pitch,
        )
        right = normalize(cross(forward, WORLD_UP), fallback=(-1.0, 0.0, 0.0))

        torso_lengths = self._torso_chain_lengths(calibration)
        neck_position = _estimate_neck_position(
            head_source,
            head_rotation,
            torso_lengths.head_to_neck,
        )
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
        neck_rotation = _smooth_rotation(
            self._torso_state.neck_rotation,
            head_rotation,
            self.settings.neck_rotation_lerp_speed,
            dt,
        )
        hips_rotation = _smooth_rotation(
            self._torso_state.hips_rotation,
            _yaw_rotation(forward),
            self.settings.hips_rotation_lerp_speed,
            dt,
        )
        neck_yaw = _yaw_rotation(_horizontal_forward(neck_rotation))
        chest_rotation = _smooth_rotation(
            self._torso_state.chest_rotation,
            _nlerp_quaternion(
                neck_yaw,
                hips_rotation,
                torso_lengths.neck_to_chest / torso_lengths.neck_to_hips,
            ),
            self.settings.chest_rotation_lerp_speed,
            dt,
        )
        spine_rotation = _smooth_rotation(
            self._torso_state.spine_rotation,
            _nlerp_quaternion(
                neck_yaw,
                hips_rotation,
                torso_lengths.neck_to_spine / torso_lengths.neck_to_hips,
            ),
            self.settings.spine_rotation_lerp_speed,
            dt,
        )
        self._torso_state.neck_rotation = neck_rotation
        self._torso_state.chest_rotation = chest_rotation
        self._torso_state.spine_rotation = spine_rotation
        self._torso_state.hips_rotation = hips_rotation
        shoulder_width = _calibrated_length(
            calibration,
            ("shoulder_width", "left_shoulder_to_right_shoulder"),
            calibration.human_height_m * self.settings.shoulder_width_ratio,
        )
        left_shoulder = add(chest_position, scale(right, -shoulder_width * 0.5))
        right_shoulder = add(chest_position, scale(right, shoulder_width * 0.5))
        left_hand_source = _source_joint_position(sample.left_hand)
        right_hand_source = _source_joint_position(sample.right_hand)
        arm_length = self._arm_length(calibration)
        left_shoulder = self._estimate_shoulder_reach_offset(
            left_shoulder,
            left_hand_source,
            right,
            forward,
            -1.0,
            arm_length,
        )
        right_shoulder = self._estimate_shoulder_reach_offset(
            right_shoulder,
            right_hand_source,
            right,
            forward,
            1.0,
            arm_length,
        )

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
                Transform(position=hips_position, rotation=hips_rotation),
                estimated=True,
            ),
            "chest": JointPose(
                Transform(position=chest_position, rotation=chest_rotation),
                estimated=True,
            ),
            "neck": JointPose(
                Transform(position=neck_position, rotation=neck_rotation),
                estimated=True,
            ),
            "spine": JointPose(
                Transform(position=spine_position, rotation=spine_rotation),
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
                    self._knee_bend_hint("left_foot", forward, right, -1.0, dt),
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
                    self._knee_bend_hint("right_foot", forward, right, 1.0, dt),
                    calibration,
                )
            ),
            estimated=True,
        )

        return HumanoidPoseFrame(
            timestamp_seconds=sample.timestamp_seconds,
            root=Transform(position=hips_position, rotation=hips_rotation),
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
        support_center = self._support_center()
        if support_center is not None:
            blend = self.settings.support_hips_xz_follow_blend
            ideal = (
                ideal[0] + (support_center[0] - ideal[0]) * blend,
                ideal[1],
                ideal[2] + (support_center[2] - ideal[2]) * blend,
            )

        pelvis_tracker = sample.optional_trackers.get("pelvis")
        if pelvis_tracker is not None and pelvis_tracker.valid:
            pelvis_position = _source_joint_position(pelvis_tracker)
            blend = self.settings.hips_xz_follow_blend
            ideal = (
                ideal[0] + (pelvis_position[0] - ideal[0]) * blend,
                ideal[1],
                ideal[2] + (pelvis_position[2] - ideal[2]) * blend,
            )

        return _clamp_hips_lateral(
            neck_position,
            ideal,
            self.settings.max_torso_lean_degrees,
        )

    def _support_center(self) -> Vector3 | None:
        positions: list[Vector3] = []
        for name in ("left_foot", "right_foot"):
            state = self._foot_states.get(name)
            if state is None:
                continue
            position = state.anchor_position if state.planted else state.current_position
            if position is not None:
                positions.append(position)
        if not positions:
            return None
        return (
            sum(position[0] for position in positions) / len(positions),
            sum(position[1] for position in positions) / len(positions),
            sum(position[2] for position in positions) / len(positions),
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
        if self.settings.footstep_planner_enabled:
            return self._estimate_planned_foot(
                name,
                state,
                target,
                speed,
                right,
                side,
                forward,
                stance_width,
                grounding,
                timestamp_seconds,
                dt,
            )

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
            state.current_rotation or desired_rotation,
            desired_rotation,
            _exp_alpha(self.settings.foot_position_lerp_speed, dt),
        )
        return _FootTarget(
            position=state.current_position,
            rotation=state.current_rotation,
        )

    def _estimate_planned_foot(
        self,
        name: str,
        state: _FootState,
        neutral_target: Vector3,
        neutral_speed: float,
        right: Vector3,
        side: float,
        forward: Vector3,
        stance_width: float,
        grounding: GroundingContext,
        timestamp_seconds: float,
        dt: float,
    ) -> _FootTarget:
        if state.first_timestamp_seconds is None:
            state.first_timestamp_seconds = timestamp_seconds

        if state.phase == "unplanted":
            self._plant_planned_foot(state, neutral_target, grounding, forward)

        if (
            self.settings.footstep_startup_align_s > 0.0
            and state.first_timestamp_seconds is not None
            and timestamp_seconds - state.first_timestamp_seconds
            < self.settings.footstep_startup_align_s
        ):
            self._plant_planned_foot(state, neutral_target, grounding, forward)
            desired_position = neutral_target
            state.previous_target = neutral_target
            state.previous_timestamp_seconds = timestamp_seconds
            state.current_position = desired_position
            state.current_rotation = _foot_rotation(forward, state.anchor_normal)
            return _FootTarget(
                position=state.current_position,
                rotation=state.current_rotation,
            )

        if state.phase == "swing":
            self._retarget_planned_step(
                state,
                neutral_target,
                neutral_speed,
                right,
                side,
                forward,
                stance_width,
            )
            desired_position = self._planned_swing_position(state, timestamp_seconds)
        else:
            anchor_stretch = (
                horizontal_distance(state.anchor_position, neutral_target)
                if state.anchor_position is not None
                else 0.0
            )
            needs_catchup = anchor_stretch > self.settings.footstep_catchup_release_stretch_m
            needs_trailing_recovery = (
                neutral_speed <= self.settings.footstep_stop_speed_m_s
                and anchor_stretch > self.settings.footstep_trailing_release_stretch_m
            )
            needs_regular_step = anchor_stretch > self._planned_release_stretch(neutral_speed)
            force_recovery = needs_catchup or needs_trailing_recovery
            if (
                state.planted
                and state.anchor_position is not None
                and (force_recovery or needs_regular_step)
                and self._can_begin_planned_step(
                    name,
                    timestamp_seconds,
                    force_recovery=force_recovery,
                )
            ):
                self._begin_planned_step(
                    name,
                    state,
                    neutral_target,
                    neutral_speed,
                    right,
                    side,
                    forward,
                    stance_width,
                    timestamp_seconds,
                    catchup=needs_catchup,
                    forward_lead_scale=0.0 if needs_trailing_recovery else 1.0,
                )
                desired_position = self._planned_swing_position(state, timestamp_seconds)
            elif state.anchor_position is not None:
                desired_position = state.anchor_position
            else:
                state.anchor_position = neutral_target
                state.anchor_normal = grounding.floor_normal
                state.planted = True
                state.phase = "planted"
                desired_position = neutral_target

        desired_position = self._constrain_planned_foot_to_neutral(
            name,
            state,
            desired_position,
            neutral_target,
            grounding,
        )
        progress = _step_progress(
            timestamp_seconds,
            state.swing_start_time_seconds,
            state.swing_duration_s or self.settings.step_duration_s,
        )
        if state.phase == "swing" and progress >= 1.0 and state.swing_target_position is not None:
            desired_position = state.swing_target_position
            state.anchor_position = state.swing_target_position
            state.anchor_normal = grounding.floor_normal
            state.swing_start_position = None
            state.swing_start_time_seconds = None
            state.swing_target_position = None
            state.swing_forward_lead_scale = 1.0
            state.swing_duration_s = None
            state.planted = True
            state.phase = "planted"
            if self._active_swing_foot == name:
                self._active_swing_foot = None

        state.previous_target = neutral_target
        state.previous_timestamp_seconds = timestamp_seconds
        state.current_position = desired_position
        desired_rotation = _foot_rotation(
            forward, state.anchor_normal if state.planted else state.filtered_normal
        )
        state.current_rotation = _nlerp_quaternion(
            state.current_rotation or desired_rotation,
            desired_rotation,
            _exp_alpha(self.settings.foot_position_lerp_speed, dt),
        )
        return _FootTarget(
            position=state.current_position,
            rotation=state.current_rotation,
        )

    def _plant_planned_foot(
        self,
        state: _FootState,
        neutral_target: Vector3,
        grounding: GroundingContext,
        forward: Vector3,
    ) -> None:
        state.anchor_position = neutral_target
        state.anchor_normal = grounding.floor_normal
        state.filtered_normal = grounding.floor_normal
        state.current_position = neutral_target
        state.current_rotation = _foot_rotation(forward, state.anchor_normal)
        state.swing_start_position = None
        state.swing_start_time_seconds = None
        state.swing_target_position = None
        state.swing_forward_lead_scale = 1.0
        state.swing_duration_s = None
        state.planted = True
        state.phase = "planted"

    def _retarget_planned_step(
        self,
        state: _FootState,
        neutral_target: Vector3,
        neutral_speed: float,
        right: Vector3,
        side: float,
        forward: Vector3,
        stance_width: float,
    ) -> None:
        if state.swing_target_position is None:
            return
        if (
            horizontal_distance(state.swing_target_position, neutral_target)
            <= self.settings.footstep_swing_retarget_stretch_m
        ):
            return
        start = state.swing_start_position or state.current_position or neutral_target
        state.swing_target_position = self._planned_step_target(
            start,
            neutral_target,
            neutral_speed,
            right,
            side,
            forward,
            stance_width,
            state.swing_forward_lead_scale,
        )

    def _constrain_planned_foot_to_neutral(
        self,
        name: str,
        state: _FootState,
        desired_position: Vector3,
        neutral_target: Vector3,
        grounding: GroundingContext,
    ) -> Vector3:
        constrained = _clamp_horizontal_delta(
            neutral_target,
            desired_position,
            self.settings.footstep_max_neutral_stretch_m,
        )
        if constrained == desired_position:
            return desired_position

        if state.phase == "swing" and state.swing_target_position is not None:
            state.swing_target_position = _clamp_horizontal_delta(
                neutral_target,
                state.swing_target_position,
                self.settings.footstep_max_neutral_stretch_m,
            )
        elif state.phase == "planted":
            state.anchor_position = constrained
            state.anchor_normal = grounding.floor_normal

        return constrained

    def _can_begin_planned_step(
        self,
        name: str,
        timestamp_seconds: float,
        *,
        force_recovery: bool = False,
    ) -> bool:
        if self._active_swing_foot not in (None, name):
            return False

        other_state = self._foot_states.get(_opposite_foot_name(name))
        if other_state is not None and other_state.phase == "swing":
            return False

        if not force_recovery:
            if (
                self._last_step_time_seconds is not None
                and timestamp_seconds - self._last_step_time_seconds
                < self.settings.footstep_min_interval_s
            ):
                return False

            if self._last_stepped_foot == name and other_state is not None and other_state.planted:
                return False

        return True

    def _begin_planned_step(
        self,
        name: str,
        state: _FootState,
        neutral_target: Vector3,
        neutral_speed: float,
        right: Vector3,
        side: float,
        forward: Vector3,
        stance_width: float,
        timestamp_seconds: float,
        *,
        catchup: bool = False,
        forward_lead_scale: float = 1.0,
    ) -> None:
        start = state.current_position or state.anchor_position or neutral_target
        state.swing_start_position = start
        state.swing_start_time_seconds = timestamp_seconds
        state.swing_forward_lead_scale = forward_lead_scale
        state.swing_duration_s = self._planned_step_duration(neutral_speed, catchup=catchup)
        state.swing_target_position = self._planned_step_target(
            start,
            neutral_target,
            neutral_speed,
            right,
            side,
            forward,
            stance_width,
            forward_lead_scale,
        )
        state.anchor_position = None
        state.planted = False
        state.phase = "swing"
        self._active_swing_foot = name
        self._last_stepped_foot = name
        self._last_step_time_seconds = timestamp_seconds

    def _planned_step_target(
        self,
        start: Vector3,
        neutral_target: Vector3,
        neutral_speed: float,
        right: Vector3,
        side: float,
        forward: Vector3,
        stance_width: float,
        forward_lead_scale: float = 1.0,
    ) -> Vector3:
        forward_lead = (
            min(
                self.settings.footstep_max_forward_lead_m,
                self.settings.footstep_forward_lead_m
                + neutral_speed * self.settings.footstep_speed_lead_s,
            )
            * forward_lead_scale
        )
        target = add(neutral_target, scale(forward, forward_lead))
        target = (target[0], neutral_target[1], target[2])
        target = _clamp_horizontal_delta(
            start,
            target,
            self.settings.footstep_max_step_length_m,
        )

        lateral_offset = dot(subtract(target, neutral_target), right) * side
        min_lateral = stance_width * 0.5 * self.settings.footstep_min_lateral_ratio
        if lateral_offset < -min_lateral:
            target = add(target, scale(right, side * (-min_lateral - lateral_offset)))
        return (target[0], neutral_target[1], target[2])

    def _planned_release_stretch(self, neutral_speed: float) -> float:
        return max(
            0.08,
            self.settings.footstep_release_stretch_m
            - neutral_speed * self.settings.footstep_speed_release_reduction_s,
        )

    def _planned_step_duration(self, neutral_speed: float, *, catchup: bool) -> float:
        if catchup:
            return self.settings.footstep_catchup_duration_s
        return max(
            0.12,
            self.settings.step_duration_s
            - neutral_speed * self.settings.footstep_speed_duration_reduction_s,
        )

    def _planned_swing_position(
        self,
        state: _FootState,
        timestamp_seconds: float,
    ) -> Vector3:
        if state.swing_start_position is None or state.swing_target_position is None:
            return state.current_position or (0.0, 0.0, 0.0)
        return _swing_position(
            state.swing_start_position,
            state.swing_target_position,
            _step_progress(
                timestamp_seconds,
                state.swing_start_time_seconds,
                state.swing_duration_s or self.settings.step_duration_s,
            ),
            self.settings.step_lift_height_m,
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

    def _knee_bend_hint(
        self,
        foot_name: str,
        forward: Vector3,
        right: Vector3,
        side: float,
        dt: float,
    ) -> Vector3:
        if not self.settings.knee_hint_enabled:
            return forward

        raw_hint = normalize(
            add(
                forward,
                scale(right, side * self.settings.knee_hint_lateral_weight),
            ),
            fallback=forward,
        )
        state = self._foot_states.get(foot_name)
        if state is None:
            return raw_hint
        if state.filtered_knee_hint is None:
            state.filtered_knee_hint = raw_hint
            return raw_hint
        state.filtered_knee_hint = _smooth_direction(
            state.filtered_knee_hint,
            raw_hint,
            _exp_alpha(self.settings.knee_hint_lerp_speed, dt),
        )
        return state.filtered_knee_hint

    def _estimate_knee(
        self,
        hip_position: Vector3,
        foot_position: Vector3,
        bend_hint: Vector3,
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
        return _solve_two_bone_midpoint(
            hip_position,
            foot_position,
            upper_leg,
            lower_leg,
            bend_hint,
        )

    def _arm_length(self, calibration: AvatarCalibration) -> float:
        return _calibrated_length(
            calibration,
            ("upper_arm", "humerus"),
            calibration.human_height_m * self.settings.upper_arm_ratio,
        ) + _calibrated_length(
            calibration,
            ("lower_arm", "forearm", "radius"),
            calibration.human_height_m * self.settings.lower_arm_ratio,
        )

    def _estimate_shoulder_reach_offset(
        self,
        shoulder_position: Vector3,
        hand_position: Vector3,
        right: Vector3,
        forward: Vector3,
        side: float,
        arm_length: float,
    ) -> Vector3:
        if arm_length <= 1e-6:
            return shoulder_position

        shoulder_to_hand = subtract(hand_position, shoulder_position)
        reach = _length(shoulder_to_hand)
        if reach <= 1e-6:
            return shoulder_position

        reach_ratio = _shoulder_reach_ratio(
            reach,
            arm_length,
            self.settings.shoulder_reach_engage_ratio,
        )
        if reach_ratio <= 0.0:
            return shoulder_position

        local_dir = _chest_local_direction(shoulder_to_hand, right, forward)
        local_x, local_y, local_z = local_dir
        elevation = _clamp01(local_y) * reach_ratio
        protraction = _clamp01(local_z) * reach_ratio
        cross_body = _clamp01(local_x) if side < 0.0 else _clamp01(-local_x)

        shoulder = add(
            add(
                add(
                    shoulder_position,
                    scale(WORLD_UP, elevation * self.settings.shoulder_elevation_gain_m),
                ),
                scale(
                    forward,
                    (protraction + cross_body * 0.5)
                    * reach_ratio
                    * self.settings.shoulder_protraction_gain_m,
                ),
            ),
            scale(
                right,
                -side * cross_body * reach_ratio * self.settings.shoulder_cross_body_gain_m,
            ),
        )
        return self._shift_shoulder_to_reachable_hand(shoulder, hand_position, arm_length)

    def _shift_shoulder_to_reachable_hand(
        self,
        shoulder_position: Vector3,
        hand_position: Vector3,
        arm_length: float,
    ) -> Vector3:
        shoulder_to_hand = subtract(hand_position, shoulder_position)
        reach = _length(shoulder_to_hand)
        max_reach = arm_length + self.settings.shoulder_reach_slack_m
        excess_reach = reach - max_reach
        if excess_reach <= 0.0:
            return shoulder_position
        shift = min(excess_reach, self.settings.max_shoulder_reach_shift_m)
        return add(
            shoulder_position,
            scale(normalize(shoulder_to_hand, fallback=WORLD_FORWARD), shift),
        )

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
        default_hint = normalize(
            add(scale(forward, -0.7), scale(outward, 0.3)),
            fallback=scale(forward, -1.0),
        )
        if self.settings.arm_bend_strategy == "manikin_swivel":
            hint = _manikin_swivel_arm_bend_direction(
                shoulder_position,
                hand_position,
                right,
                forward,
                side,
                upper_arm + lower_arm,
                down_weight=self.settings.arm_swivel_down_weight,
                outward_weight=self.settings.arm_swivel_outward_weight,
                backward_weight=self.settings.arm_swivel_backward_weight,
                cross_body_inward_weight=self.settings.arm_swivel_cross_body_inward_weight,
                overhead_relief=self.settings.arm_swivel_overhead_relief,
            )
            return _solve_two_bone_midpoint(
                shoulder_position,
                hand_position,
                upper_arm,
                lower_arm,
                hint,
            )

        lookup_hint = _arm_bend_lookup_direction(
            shoulder_position,
            hand_position,
            right,
            forward,
            side,
            upper_arm + lower_arm,
        )
        lookup_blend = (
            1.0
            if self.settings.arm_bend_strategy == "basis_lookup"
            else (self.settings.arm_bend_lookup_blend)
        )
        hint = normalize(
            lerp(default_hint, lookup_hint, _clamp01(lookup_blend)), fallback=default_hint
        )
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


def _estimate_neck_position(
    head_position: Vector3,
    head_rotation: Quaternion,
    head_to_neck_length: float,
) -> Vector3:
    head_up = normalize(rotate_vector(head_rotation, WORLD_UP), fallback=WORLD_UP)
    return subtract(head_position, scale(head_up, head_to_neck_length))


def _head_forward(tracker: TrackerState) -> Vector3:
    forward = rotate_vector(tracker.transform.rotation, WORLD_FORWARD)
    return normalize((forward[0], 0.0, forward[2]), fallback=WORLD_FORWARD)


def _disambiguate_body_forward(
    sample: TrackedPoseSample,
    forward: Vector3,
    controller_lateral_flip_threshold_m: float,
    controller_lateral_flip_min_head_pitch: float,
) -> Vector3:
    head_forward = rotate_vector(sample.hmd.transform.rotation, WORLD_FORWARD)
    if abs(head_forward[1]) < controller_lateral_flip_min_head_pitch:
        return forward

    right = normalize(cross(forward, WORLD_UP), fallback=(-1.0, 0.0, 0.0))
    left_hand = _source_joint_position(sample.left_hand)
    right_hand = _source_joint_position(sample.right_hand)
    lateral_gap = dot(subtract(right_hand, left_hand), right)
    if lateral_gap < -controller_lateral_flip_threshold_m:
        return scale(forward, -1.0)
    return forward


def _horizontal_forward(rotation: Quaternion) -> Vector3:
    forward = rotate_vector(rotation, WORLD_FORWARD)
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


def _state_dt(state: _FootState | _TorsoState, timestamp_seconds: float) -> float:
    if state.previous_timestamp_seconds is None:
        return 1.0 / 120.0
    return max(timestamp_seconds - state.previous_timestamp_seconds, 1.0 / 120.0)


def _exp_alpha(speed: float, dt: float) -> float:
    if speed <= 0.0:
        return 1.0
    return 1.0 - exp(-speed * dt)


def _smooth_normal(current: Vector3, target: Vector3, alpha: float) -> Vector3:
    return normalize(lerp(current, target, alpha), fallback=WORLD_UP)


def _smooth_direction(current: Vector3, target: Vector3, alpha: float) -> Vector3:
    return normalize(lerp(current, target, alpha), fallback=target)


def _smooth_rotation(
    current: Quaternion | None,
    target: Quaternion,
    speed: float,
    dt: float,
) -> Quaternion:
    if current is None:
        return target
    return _nlerp_quaternion(current, target, _exp_alpha(speed, dt))


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


def _opposite_foot_name(name: str) -> str:
    return "right_foot" if name == "left_foot" else "left_foot"


def _clamp_horizontal_delta(start: Vector3, target: Vector3, max_length: float) -> Vector3:
    delta = (target[0] - start[0], 0.0, target[2] - start[2])
    delta_length = _length(delta)
    if delta_length <= max_length or delta_length <= 1e-9:
        return target
    scale_factor = max_length / delta_length
    return (
        start[0] + delta[0] * scale_factor,
        target[1],
        start[2] + delta[2] * scale_factor,
    )


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


def _clamp_hips_lateral(
    neck_position: Vector3,
    hips_position: Vector3,
    max_lean_degrees: float,
) -> Vector3:
    vertical_drop = max(abs(neck_position[1] - hips_position[1]), 0.05)
    max_lateral = tan(radians(max_lean_degrees)) * vertical_drop
    lateral = (
        hips_position[0] - neck_position[0],
        0.0,
        hips_position[2] - neck_position[2],
    )
    lateral_length = _length(lateral)
    if lateral_length <= max_lateral or lateral_length <= 1e-9:
        return hips_position
    scale_factor = max_lateral / lateral_length
    return (
        neck_position[0] + lateral[0] * scale_factor,
        hips_position[1],
        neck_position[2] + lateral[2] * scale_factor,
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


def _arm_bend_lookup_direction(
    shoulder_position: Vector3,
    hand_position: Vector3,
    right: Vector3,
    forward: Vector3,
    side: float,
    arm_length: float,
) -> Vector3:
    if arm_length <= 1e-6:
        return scale(right, side)

    shoulder_to_hand = subtract(hand_position, shoulder_position)
    local_pos = _chest_local_components(
        scale(shoulder_to_hand, 1.0 / arm_length),
        right,
        forward,
    )
    local_bend = _basis_arm_bend_lookup(local_pos, is_left=side < 0.0)
    return normalize(
        add(
            add(scale(right, local_bend[0]), scale(WORLD_UP, local_bend[1])),
            scale(forward, local_bend[2]),
        ),
        fallback=scale(right, side),
    )


def _manikin_swivel_arm_bend_direction(
    shoulder_position: Vector3,
    hand_position: Vector3,
    right: Vector3,
    forward: Vector3,
    side: float,
    arm_length: float,
    *,
    down_weight: float,
    outward_weight: float,
    backward_weight: float,
    cross_body_inward_weight: float,
    overhead_relief: float,
) -> Vector3:
    """Choose the elbow orbit direction for an analytic two-bone arm solve.

    MANIKIN parameterizes the extra arm degree of freedom as a swivel angle around
    the shoulder-wrist axis. This deterministic variant keeps the same analytic
    shape but uses a simple human prior instead of a learned swivel predictor:
    elbows prefer down, slightly out, and slightly behind the torso, with reduced
    downward bias when the hand is overhead.
    """

    if arm_length <= 1e-6:
        return scale(right, side)

    shoulder_to_hand = subtract(hand_position, shoulder_position)
    local_pos = _chest_local_components(
        scale(shoulder_to_hand, 1.0 / arm_length),
        right,
        forward,
    )
    local_x, local_y, local_z = local_pos
    overhead = _clamp01(local_y)
    cross_body = _clamp01(local_x) if side < 0.0 else _clamp01(-local_x)
    hand_behind = _clamp01(-local_z)

    down = -down_weight * (1.0 - overhead * _clamp01(overhead_relief))
    outward = side * outward_weight * (1.0 - cross_body * 0.5)
    inward = -side * cross_body * cross_body_inward_weight
    backward = -backward_weight * (1.0 - hand_behind * 0.7)
    return normalize(
        add(
            add(scale(right, outward + inward), scale(WORLD_UP, down)),
            scale(forward, backward),
        ),
        fallback=scale(forward, -1.0),
    )


def _basis_arm_bend_lookup(local_pos: Vector3, *, is_left: bool) -> Vector3:
    """Continuous form of BasisArmBendLookup.GenerateDefaultTable."""

    local_x, local_y, local_z = local_pos
    bend_dir = (0.0, -0.3, -1.0)
    bend_dir = lerp(bend_dir, (0.0, -1.0, -0.3), _clamp01(local_z) * 0.6)
    bend_dir = lerp(bend_dir, (0.0, -0.5, -1.0), _clamp01(local_y) * 0.5)

    inwardness = _clamp01(local_x) if is_left else _clamp01(-local_x)
    inward_bend = (-1.0 if is_left else 1.0, -0.5, 0.0)
    bend_dir = lerp(bend_dir, inward_bend, inwardness * 0.4)

    bend_dir = lerp(bend_dir, (0.0, 1.0, 0.0), _clamp01(-local_z) * 0.5)
    bend_dir = lerp(bend_dir, (0.0, 0.0, -1.0), _clamp01(-local_y) * 0.3)
    return normalize(bend_dir, fallback=(0.0, -0.3, -1.0))


def _chest_local_direction(vector: Vector3, right: Vector3, forward: Vector3) -> Vector3:
    return _chest_local_components(normalize(vector, fallback=forward), right, forward)


def _chest_local_components(vector: Vector3, right: Vector3, forward: Vector3) -> Vector3:
    return (
        dot(vector, right),
        dot(vector, WORLD_UP),
        dot(vector, forward),
    )


def _shoulder_reach_ratio(reach: float, arm_length: float, engage_ratio: float) -> float:
    raw_reach_ratio = _clamp01(reach / max(arm_length * 1.1, 1e-6))
    engage = _clamp01(engage_ratio)
    if raw_reach_ratio < engage:
        return 0.0
    if engage >= 1.0:
        return 1.0
    t = (raw_reach_ratio - engage) / (1.0 - engage)
    return t * t


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))


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
