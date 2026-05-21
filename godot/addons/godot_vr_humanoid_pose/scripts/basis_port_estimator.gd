extends RefCounted
class_name VRHumanoidBasisPortEstimator

const WORLD_UP = Vector3(0.0, 1.0, 0.0)
const WORLD_FORWARD = Vector3(0.0, 0.0, 1.0)

var settings = {
    "head_to_hips_ratio": 0.34,
    "shoulder_width_ratio": 0.23,
    "upper_arm_ratio": 0.19,
    "lower_arm_ratio": 0.16,
    "upper_leg_ratio": 0.245,
    "lower_leg_ratio": 0.246,
    "head_to_neck_ratio": 0.04,
    "neck_to_chest_ratio": 0.09,
    "chest_to_spine_ratio": 0.10,
    "spine_to_hips_ratio": 0.11,
    "stance_width_ratio": 0.17,
    "hip_width_ratio": 0.11,
    "hips_forward_bias_m": 0.02,
    "hips_xz_follow_blend": 0.35,
    "max_anchor_stretch_m": 0.12,
    "plant_velocity_threshold_m_s": 0.01,
    "release_velocity_multiplier": 2.0,
    "foot_position_lerp_speed": 15.0,
    "planted_foot_lerp_speed": 25.0,
    "foot_normal_lerp_speed": 12.0,
    "step_duration_s": 0.18,
    "step_lift_height_m": 0.06,
    "planted_foot_confidence": 0.75,
}

var _foot_states: Dictionary = {}


func reset() -> void:
    _foot_states.clear()


func estimate(sample: Dictionary, calibration: Dictionary, grounding: Dictionary) -> Dictionary:
    var head_source = _source_joint_position(sample["hmd"])
    var forward = _head_forward(sample["hmd"])
    var right = _normalize(WORLD_UP.cross(forward), Vector3(1.0, 0.0, 0.0))

    var torso_lengths = _torso_chain_lengths(calibration)
    var neck_position = head_source - WORLD_UP * torso_lengths["head_to_neck"]
    var hips_position = _estimate_hips(
        sample,
        grounding,
        neck_position,
        forward,
        torso_lengths["neck_to_hips"]
    )
    var chest_position = neck_position.lerp(
        hips_position,
        torso_lengths["neck_to_chest"] / torso_lengths["neck_to_hips"]
    )
    var spine_position = neck_position.lerp(
        hips_position,
        torso_lengths["neck_to_spine"] / torso_lengths["neck_to_hips"]
    )
    var torso_rotation = _yaw_rotation(forward)
    var shoulder_width = _calibrated_length(
        calibration,
        ["shoulder_width", "left_shoulder_to_right_shoulder"],
        calibration.get("human_height_m", 1.7) * settings["shoulder_width_ratio"]
    )
    var left_shoulder = chest_position + right * (-shoulder_width * 0.5)
    var right_shoulder = chest_position + right * (shoulder_width * 0.5)
    var left_hand_source = _source_joint_position(sample["left_hand"])
    var right_hand_source = _source_joint_position(sample["right_hand"])
    var optional_trackers: Dictionary = sample.get("optional_trackers", {})

    var left_foot = _estimate_foot(
        "left_foot",
        hips_position,
        right,
        -1.0,
        forward,
        calibration,
        grounding,
        optional_trackers.get("left_foot"),
        sample["timestamp_seconds"]
    )
    var right_foot = _estimate_foot(
        "right_foot",
        hips_position,
        right,
        1.0,
        forward,
        calibration,
        grounding,
        optional_trackers.get("right_foot"),
        sample["timestamp_seconds"]
    )
    var left_hip = _hip_socket(hips_position, right, -1.0, calibration)
    var right_hip = _hip_socket(hips_position, right, 1.0, calibration)

    var joints = {
        "head": {
            "transform": _tracker_transform(sample["hmd"]),
            "confidence": sample["hmd"].get("confidence", 1.0),
            "estimated": false,
        },
        "left_hand": {
            "transform": _tracker_transform(sample["left_hand"]),
            "confidence": sample["left_hand"].get("confidence", 1.0),
            "estimated": false,
        },
        "right_hand": {
            "transform": _tracker_transform(sample["right_hand"]),
            "confidence": sample["right_hand"].get("confidence", 1.0),
            "estimated": false,
        },
        "hips": {"transform": _transform(hips_position, torso_rotation), "estimated": true},
        "chest": {"transform": _transform(chest_position, torso_rotation), "estimated": true},
        "spine": {"transform": _transform(spine_position, torso_rotation), "estimated": true},
        "left_shoulder": {"transform": _transform(left_shoulder), "estimated": true},
        "right_shoulder": {"transform": _transform(right_shoulder), "estimated": true},
        "left_elbow": {
            "transform": _transform(
                _estimate_elbow(left_shoulder, left_hand_source, right, forward, -1.0, calibration)
            ),
            "estimated": true,
        },
        "right_elbow": {
            "transform": _transform(
                _estimate_elbow(right_shoulder, right_hand_source, right, forward, 1.0, calibration)
            ),
            "estimated": true,
        },
        "left_foot": {
            "transform": _transform(left_foot["position"], left_foot["rotation"]),
            "confidence": settings["planted_foot_confidence"],
            "estimated": true,
        },
        "right_foot": {
            "transform": _transform(right_foot["position"], right_foot["rotation"]),
            "confidence": settings["planted_foot_confidence"],
            "estimated": true,
        },
    }
    joints["left_knee"] = {
        "transform": _transform(
            _estimate_knee(left_hip, left_foot["position"], forward, calibration)
        ),
        "estimated": true,
    }
    joints["right_knee"] = {
        "transform": _transform(
            _estimate_knee(right_hip, right_foot["position"], forward, calibration)
        ),
        "estimated": true,
    }

    return {
        "timestamp_seconds": sample["timestamp_seconds"],
        "root": _transform(hips_position, torso_rotation),
        "joints": joints,
    }


func _torso_chain_lengths(calibration: Dictionary) -> Dictionary:
    var human_height = calibration.get("human_height_m", 1.7)
    var head_to_hips = _calibrated_length(
        calibration,
        ["head_to_hips", "hips_to_head"],
        human_height * settings["head_to_hips_ratio"]
    )
    var head_to_neck = minf(
        head_to_hips * 0.5,
        _calibrated_length(
            calibration,
            ["head_to_neck", "neck_to_head"],
            human_height * settings["head_to_neck_ratio"]
        )
    )
    var raw_neck_to_chest = _calibrated_length(
        calibration,
        ["neck_to_chest"],
        human_height * settings["neck_to_chest_ratio"]
    )
    var raw_chest_to_spine = _calibrated_length(
        calibration,
        ["chest_to_spine"],
        human_height * settings["chest_to_spine_ratio"]
    )
    var raw_spine_to_hips = _calibrated_length(
        calibration,
        ["spine_to_hips", "hips_to_spine"],
        human_height * settings["spine_to_hips_ratio"]
    )
    var raw_total = maxf(raw_neck_to_chest + raw_chest_to_spine + raw_spine_to_hips, 0.0001)
    var target_neck_to_hips = maxf(head_to_hips - head_to_neck, 0.0001)
    var scale_factor = target_neck_to_hips / raw_total
    var neck_to_chest = raw_neck_to_chest * scale_factor
    var chest_to_spine = raw_chest_to_spine * scale_factor
    var spine_to_hips = raw_spine_to_hips * scale_factor
    return {
        "head_to_neck": head_to_neck,
        "neck_to_chest": neck_to_chest,
        "chest_to_spine": chest_to_spine,
        "spine_to_hips": spine_to_hips,
        "neck_to_spine": neck_to_chest + chest_to_spine,
        "neck_to_hips": neck_to_chest + chest_to_spine + spine_to_hips,
    }


func _estimate_hips(
    sample: Dictionary,
    grounding: Dictionary,
    neck_position: Vector3,
    forward: Vector3,
    neck_to_hips_length: float
) -> Vector3:
    var floor_y = float(grounding.get("floor_height_m", 0.0))
    var ideal = Vector3(
        neck_position.x,
        maxf(floor_y, neck_position.y - neck_to_hips_length),
        neck_position.z
    ) + forward * settings["hips_forward_bias_m"]

    var pelvis_tracker = sample.get("optional_trackers", {}).get("pelvis")
    if pelvis_tracker == null or not pelvis_tracker.get("valid", true):
        return ideal

    var pelvis_position = _source_joint_position(pelvis_tracker)
    var blend = float(settings["hips_xz_follow_blend"])
    return Vector3(
        ideal.x + (pelvis_position.x - ideal.x) * blend,
        ideal.y,
        ideal.z + (pelvis_position.z - ideal.z) * blend
    )


func _estimate_foot(
    name: String,
    hips_position: Vector3,
    right: Vector3,
    side: float,
    forward: Vector3,
    calibration: Dictionary,
    grounding: Dictionary,
    tracker,
    timestamp_seconds: float
) -> Dictionary:
    if not _foot_states.has(name):
        _foot_states[name] = _new_foot_state(_vector3_from_json(grounding.get("floor_normal", [0.0, 1.0, 0.0])))
    var state: Dictionary = _foot_states[name]
    var dt = _state_dt(state, timestamp_seconds)
    var floor_normal = _vector3_from_json(grounding.get("floor_normal", [0.0, 1.0, 0.0]))
    state["filtered_normal"] = _smooth_normal(
        state["filtered_normal"],
        floor_normal,
        _exp_alpha(settings["foot_normal_lerp_speed"], dt)
    )

    if tracker != null and tracker.get("valid", true):
        var tracked_position = _source_joint_position(tracker)
        state["anchor_position"] = tracked_position
        state["anchor_normal"] = floor_normal
        state["filtered_normal"] = floor_normal
        state["current_position"] = tracked_position
        state["current_rotation"] = _foot_rotation(forward, state["anchor_normal"])
        state["previous_target"] = tracked_position
        state["previous_timestamp_seconds"] = timestamp_seconds
        state["planted"] = true
        state["phase"] = "planted"
        return {"position": tracked_position, "rotation": state["current_rotation"]}

    var stance_width = _calibrated_length(
        calibration,
        ["stance_width", "foot_separation"],
        calibration.get("human_height_m", 1.7) * settings["stance_width_ratio"]
    )
    var floor_y = float(grounding.get("floor_height_m", 0.0)) + calibration.get("foot_height_m", 0.08)
    var target = hips_position + right * (side * stance_width * 0.5)
    target = Vector3(target.x, floor_y, target.z)
    var speed = _target_speed(state, target, timestamp_seconds)

    if state["current_position"] == null:
        state["current_position"] = target
    if state["current_rotation"] == null:
        state["current_rotation"] = _foot_rotation(forward, state["filtered_normal"])

    if state["planted"] and state["anchor_position"] != null:
        var stretch = _horizontal_distance(state["anchor_position"], target)
        var release_speed = settings["plant_velocity_threshold_m_s"] * settings["release_velocity_multiplier"]
        if stretch > settings["max_anchor_stretch_m"] or speed > release_speed:
            state["planted"] = false
            state["phase"] = "swing"
            state["swing_start_position"] = state["current_position"]
            state["swing_start_time_seconds"] = timestamp_seconds - dt
            state["anchor_position"] = null

    var desired_position: Vector3 = target
    if state["phase"] == "swing" and state["swing_start_position"] != null:
        var progress = _step_progress(
            timestamp_seconds,
            state["swing_start_time_seconds"],
            settings["step_duration_s"]
        )
        desired_position = _swing_position(
            state["swing_start_position"],
            target,
            progress,
            settings["step_lift_height_m"]
        )
        if progress >= 1.0:
            state["phase"] = "settle"

    if state["phase"] == "settle":
        if _horizontal_distance(state["current_position"], target) <= 0.01:
            state["anchor_position"] = target
            state["anchor_normal"] = floor_normal
            state["planted"] = true
            state["phase"] = "planted"
        desired_position = target

    if not state["planted"] and state["phase"] == "unplanted":
        if speed <= settings["plant_velocity_threshold_m_s"]:
            state["anchor_position"] = target
            state["anchor_normal"] = floor_normal
            state["planted"] = true
            state["phase"] = "planted"

    if state["planted"] and state["anchor_position"] != null:
        desired_position = state["anchor_position"]

    state["previous_target"] = target
    state["previous_timestamp_seconds"] = timestamp_seconds
    var position_speed = settings["planted_foot_lerp_speed"] if state["planted"] else settings["foot_position_lerp_speed"]
    state["current_position"] = state["current_position"].lerp(
        desired_position,
        _exp_alpha(position_speed, dt)
    )
    var desired_rotation = _foot_rotation(
        forward,
        state["anchor_normal"] if state["planted"] else state["filtered_normal"]
    )
    state["current_rotation"] = _nlerp_quaternion(
        state["current_rotation"],
        desired_rotation,
        _exp_alpha(settings["foot_position_lerp_speed"], dt)
    )
    return {"position": state["current_position"], "rotation": state["current_rotation"]}


func _estimate_knee(
    hip_position: Vector3,
    foot_position: Vector3,
    forward: Vector3,
    calibration: Dictionary
) -> Vector3:
    var human_height = calibration.get("human_height_m", 1.7)
    var upper_leg = _calibrated_length(
        calibration,
        ["upper_leg", "thigh", "femur"],
        human_height * settings["upper_leg_ratio"]
    )
    var lower_leg = _calibrated_length(
        calibration,
        ["lower_leg", "shin", "tibia"],
        human_height * settings["lower_leg_ratio"]
    )
    return _solve_two_bone_midpoint(hip_position, foot_position, upper_leg, lower_leg, forward)


func _estimate_elbow(
    shoulder_position: Vector3,
    hand_position: Vector3,
    right: Vector3,
    forward: Vector3,
    side: float,
    calibration: Dictionary
) -> Vector3:
    var human_height = calibration.get("human_height_m", 1.7)
    var upper_arm = _calibrated_length(
        calibration,
        ["upper_arm", "humerus"],
        human_height * settings["upper_arm_ratio"]
    )
    var lower_arm = _calibrated_length(
        calibration,
        ["lower_arm", "forearm", "radius"],
        human_height * settings["lower_arm_ratio"]
    )
    var outward = right * side
    var hint = _normalize(forward * 0.7 + outward * 0.3, forward)
    return _solve_two_bone_midpoint(
        shoulder_position,
        hand_position,
        upper_arm,
        lower_arm,
        hint
    )


func _hip_socket(
    hips_position: Vector3,
    right: Vector3,
    side: float,
    calibration: Dictionary
) -> Vector3:
    var hip_width = _calibrated_length(
        calibration,
        ["hip_width", "pelvis_width"],
        calibration.get("human_height_m", 1.7) * settings["hip_width_ratio"]
    )
    return hips_position + right * (side * hip_width * 0.5)


func _solve_two_bone_midpoint(
    root: Vector3,
    tip: Vector3,
    root_to_mid_length: float,
    mid_to_tip_length: float,
    bend_hint: Vector3
) -> Vector3:
    var root_to_tip = tip - root
    var distance = maxf(0.0001, root_to_tip.length())
    var max_reach = maxf(0.0001, root_to_mid_length + mid_to_tip_length - 0.0001)
    var min_reach = absf(root_to_mid_length - mid_to_tip_length) + 0.0001
    var clamped_distance = minf(maxf(distance, min_reach), max_reach)
    var direction = _normalize(root_to_tip, WORLD_UP)

    var along = (
        root_to_mid_length * root_to_mid_length
        - mid_to_tip_length * mid_to_tip_length
        + clamped_distance * clamped_distance
    ) / (2.0 * clamped_distance)
    var bend_height_squared = root_to_mid_length * root_to_mid_length - along * along
    var bend_height = sqrt(bend_height_squared) if bend_height_squared > 0.0 else 0.0
    var bend_direction = _project_on_plane(bend_hint, direction)
    bend_direction = _normalize(bend_direction, _fallback_bend_direction(direction))
    return root + direction * along + bend_direction * bend_height


func _calibrated_length(calibration: Dictionary, names: Array, fallback: float) -> float:
    var bone_lengths: Dictionary = calibration.get("bone_lengths_m", {})
    for name in names:
        var value = bone_lengths.get(name)
        if value != null and value > 0.0:
            return value * calibration.get("avatar_scale", 1.0)
    return fallback * calibration.get("avatar_scale", 1.0)


func _source_joint_position(tracker: Dictionary) -> Vector3:
    var attachment = tracker.get("attachment")
    var offset = Vector3.ZERO
    if attachment != null:
        offset = _vector3_from_json(attachment["local_transform"]["position"])
    return _vector3_from_json(tracker["transform"]["position"]) - offset


func _tracker_transform(tracker: Dictionary) -> Dictionary:
    return _transform(
        _vector3_from_json(tracker["transform"]["position"]),
        _quaternion_from_json(tracker["transform"]["rotation"])
    )


func _head_forward(tracker: Dictionary) -> Vector3:
    var forward = _rotate_vector(_quaternion_from_json(tracker["transform"]["rotation"]), WORLD_FORWARD)
    return _normalize(Vector3(forward.x, 0.0, forward.z), WORLD_FORWARD)


func _yaw_rotation(forward: Vector3) -> Quaternion:
    return _look_rotation(_project_on_plane(forward, WORLD_UP), WORLD_UP)


func _foot_rotation(forward: Vector3, normal: Vector3) -> Quaternion:
    return _look_rotation(_project_on_plane(forward, normal), normal)


func _target_speed(state: Dictionary, target: Vector3, timestamp_seconds: float) -> float:
    if state["previous_target"] == null or state["previous_timestamp_seconds"] == null:
        return 0.0
    var dt = maxf(timestamp_seconds - state["previous_timestamp_seconds"], 1.0 / 120.0)
    return _horizontal_distance(target, state["previous_target"]) / dt


func _state_dt(state: Dictionary, timestamp_seconds: float) -> float:
    if state["previous_timestamp_seconds"] == null:
        return 1.0 / 120.0
    return maxf(timestamp_seconds - state["previous_timestamp_seconds"], 1.0 / 120.0)


func _exp_alpha(speed: float, dt: float) -> float:
    if speed <= 0.0:
        return 1.0
    return 1.0 - exp(-speed * dt)


func _smooth_normal(current: Vector3, target: Vector3, alpha: float) -> Vector3:
    return _normalize(current.lerp(target, alpha), WORLD_UP)


func _step_progress(
    timestamp_seconds: float,
    swing_start_time_seconds,
    step_duration_s: float
) -> float:
    if swing_start_time_seconds == null:
        return 1.0
    return minf(1.0, maxf(0.0, (timestamp_seconds - swing_start_time_seconds) / step_duration_s))


func _swing_position(
    start: Vector3,
    target: Vector3,
    progress: float,
    lift_height_m: float
) -> Vector3:
    var eased = progress * progress * (3.0 - 2.0 * progress)
    var base = start.lerp(target, eased)
    var lift = sin(PI * progress) * lift_height_m
    return Vector3(base.x, base.y + lift, base.z)


func _nlerp_quaternion(a: Quaternion, b: Quaternion, alpha: float) -> Quaternion:
    var bx = b.x
    var by = b.y
    var bz = b.z
    var bw = b.w
    if a.x * bx + a.y * by + a.z * bz + a.w * bw < 0.0:
        bx = -bx
        by = -by
        bz = -bz
        bw = -bw
    var blended = Quaternion(
        a.x + (bx - a.x) * alpha,
        a.y + (by - a.y) * alpha,
        a.z + (bz - a.z) * alpha,
        a.w + (bw - a.w) * alpha
    )
    var magnitude = sqrt(
        blended.x * blended.x
        + blended.y * blended.y
        + blended.z * blended.z
        + blended.w * blended.w
    )
    if magnitude <= 0.000000001:
        return b
    return Quaternion(
        blended.x / magnitude,
        blended.y / magnitude,
        blended.z / magnitude,
        blended.w / magnitude
    )


func _look_rotation(forward: Vector3, up = WORLD_UP) -> Quaternion:
    var f = _normalize(forward, WORLD_FORWARD)
    var r = _normalize(up.cross(f), Vector3(1.0, 0.0, 0.0))
    var u = f.cross(r)

    var m00 = r.x
    var m01 = u.x
    var m02 = f.x
    var m10 = r.y
    var m11 = u.y
    var m12 = f.y
    var m20 = r.z
    var m21 = u.z
    var m22 = f.z

    var trace = m00 + m11 + m22
    if trace > 0.0:
        var s = sqrt(trace + 1.0) * 2.0
        return Quaternion((m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s, 0.25 * s)

    if m00 > m11 and m00 > m22:
        var s = sqrt(1.0 + m00 - m11 - m22) * 2.0
        return Quaternion(0.25 * s, (m01 + m10) / s, (m02 + m20) / s, (m21 - m12) / s)
    if m11 > m22:
        var s = sqrt(1.0 + m11 - m00 - m22) * 2.0
        return Quaternion((m01 + m10) / s, 0.25 * s, (m12 + m21) / s, (m02 - m20) / s)

    var s = sqrt(1.0 + m22 - m00 - m11) * 2.0
    return Quaternion((m02 + m20) / s, (m12 + m21) / s, 0.25 * s, (m10 - m01) / s)


func _rotate_vector(rotation: Quaternion, vector: Vector3) -> Vector3:
    var rotated = _multiply_quaternion(
        _multiply_quaternion(rotation, Quaternion(vector.x, vector.y, vector.z, 0.0)),
        _conjugate_quaternion(rotation)
    )
    return Vector3(rotated.x, rotated.y, rotated.z)


func _multiply_quaternion(a: Quaternion, b: Quaternion) -> Quaternion:
    return Quaternion(
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z
    )


func _conjugate_quaternion(q: Quaternion) -> Quaternion:
    return Quaternion(-q.x, -q.y, -q.z, q.w)


func _project_on_plane(vector: Vector3, normal: Vector3) -> Vector3:
    var normalized_normal = _normalize(normal, WORLD_UP)
    return vector - normalized_normal * vector.dot(normalized_normal)


func _fallback_bend_direction(direction: Vector3) -> Vector3:
    var fallback = direction.cross(WORLD_UP)
    if fallback.dot(fallback) <= 0.00000001:
        fallback = direction.cross(Vector3(1.0, 0.0, 0.0))
    return _normalize(fallback, WORLD_FORWARD)


func _horizontal_distance(a: Vector3, b: Vector3) -> float:
    return Vector3(a.x - b.x, 0.0, a.z - b.z).length()


func _normalize(vector: Vector3, fallback = WORLD_UP) -> Vector3:
    var magnitude = vector.length()
    if magnitude <= 0.000000001:
        return fallback
    return vector / magnitude


func _transform(position: Vector3, rotation = Quaternion(0.0, 0.0, 0.0, 1.0)) -> Dictionary:
    return {"position": position, "rotation": rotation}


func _new_foot_state(floor_normal: Vector3) -> Dictionary:
    return {
        "anchor_position": null,
        "anchor_normal": floor_normal,
        "filtered_normal": floor_normal,
        "current_position": null,
        "current_rotation": null,
        "previous_target": null,
        "previous_timestamp_seconds": null,
        "swing_start_position": null,
        "swing_start_time_seconds": null,
        "planted": false,
        "phase": "unplanted",
    }


func _vector3_from_json(value: Array) -> Vector3:
    return Vector3(value[0], value[1], value[2])


func _quaternion_from_json(value: Array) -> Quaternion:
    return Quaternion(value[0], value[1], value[2], value[3])

