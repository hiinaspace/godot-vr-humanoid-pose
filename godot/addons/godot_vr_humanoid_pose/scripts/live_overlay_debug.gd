extends Node3D
class_name VRHumanoidLiveOverlayDebug

const BasisEstimator = preload("res://addons/godot_vr_humanoid_pose/scripts/basis_port_estimator.gd")
const DebugArmature = preload("res://addons/godot_vr_humanoid_pose/scripts/debug_armature.gd")
const StreamClient = preload("res://addons/godot_vr_humanoid_pose/scripts/live_capture_stream_client.gd")
const TrackerCalibration = preload("res://addons/godot_vr_humanoid_pose/scripts/live_tracker_calibration.gd")

const WORLD_FORWARD = Vector3(0.0, 0.0, 1.0)
const GODOT_VIEW_FORWARD = Vector3(0.0, 0.0, -1.0)
const WORLD_UP = Vector3(0.0, 1.0, 0.0)
const MIRROR_YAW_RADIANS = PI
const DEFAULT_BONES = [
	["hips", "spine"],
	["spine", "chest"],
	["chest", "neck"],
	["neck", "head"],
	["head", "head_forward"],
	["chest", "left_shoulder"],
	["left_shoulder", "left_elbow"],
	["left_elbow", "left_wrist"],
	["left_wrist", "left_hand"],
	["chest", "right_shoulder"],
	["right_shoulder", "right_elbow"],
	["right_elbow", "right_wrist"],
	["right_wrist", "right_hand"],
	["hips", "left_knee"],
	["left_knee", "left_foot"],
	["hips", "right_knee"],
	["right_knee", "right_foot"],
]

@export var stream_host = "127.0.0.1"
@export var stream_port = 8765
@export var mirror_distance_m = 2.0
@export var mirror_height_offset_m = -0.15
@export var calibration_target_distance_m = 0.0
@export var calibration_target_radius_m = 0.35
@export var calibration_dwell_seconds = 1.25
@export var calibration_countdown_seconds = 3.0
@export var post_calibration_settle_seconds = 2.0
@export var clip_duration_seconds = 10.0
@export var capture_sample_rate_hz = 30.0
@export var tracker_marker_radius_m = 0.028
@export var tracker_label_font_size = 10
@export var mirror_x_axis_enabled = true
@export var use_calibrated_trackers_for_estimation = false
@export var openvr_overlay_enabled = true
@export var synthetic_input_without_xr = true

var calibration = {
	"human_height_m": 1.7,
	"avatar_scale": 1.0,
	"bone_lengths_m": {},
	"foot_height_m": 0.08,
}
var grounding = {
	"floor_height_m": 0.0,
	"floor_normal": [0.0, 1.0, 0.0],
	"character_velocity": [0.0, 0.0, 0.0],
	"locomotion_state": "unknown",
}

var _estimator = BasisEstimator.new()
var _stream_client = StreamClient.new()
var _tracker_calibrator = TrackerCalibration.new()
var _session_id = ""
var _frame_index = 0
var _xr_interface
var _origin: XROrigin3D
var _camera: XRCamera3D
var _left_controller: XRController3D
var _right_controller: XRController3D
var _armature
var _tracker_markers = {}
var _calibration_target: MeshInstance3D
var _calibration_target_material: StandardMaterial3D
var _status: Label3D
var _session_message_sent = false
var _tracker_calibration: Dictionary = {}
var _calibration_dwell_s = 0.0
var _capture_phase = "idle"
var _phase_remaining_s = 0.0
var _last_stream_timestamp_s = -INF
var _clip_index = 0
var _last_flushed_clip_index = -1
var _requires_gaze_release = false
var _headlocked_prompt_root: Node3D
var _headlocked_prompt: Label3D


func _ready() -> void:
	_apply_user_args()
	_configure_live_estimator()
	_session_id = _new_session_id()
	_stream_client.configure(stream_host, stream_port)
	_setup_xr_nodes()
	_setup_debug_scene()
	if openvr_overlay_enabled:
		_initialize_openvr_overlay()


func _configure_live_estimator() -> void:
	_estimator.settings["footstep_planner_enabled"] = true
	_estimator.settings["stance_width_ratio"] = 0.22
	_estimator.settings["support_hips_xz_follow_blend"] = 0.0
	_estimator.settings["max_anchor_stretch_m"] = 0.14
	_estimator.settings["footstep_release_stretch_m"] = 0.20
	_estimator.settings["footstep_speed_release_reduction_s"] = 0.02
	_estimator.settings["footstep_forward_lead_m"] = 0.22
	_estimator.settings["footstep_speed_lead_s"] = 0.05
	_estimator.settings["footstep_max_forward_lead_m"] = 0.42
	_estimator.settings["footstep_max_step_length_m"] = 0.72
	_estimator.settings["footstep_swing_retarget_stretch_m"] = 0.24
	_estimator.settings["footstep_max_neutral_stretch_m"] = 0.58
	_estimator.settings["footstep_min_lateral_ratio"] = 0.45
	_estimator.settings["footstep_min_interval_s"] = 0.20
	_estimator.settings["footstep_catchup_release_stretch_m"] = 0.48
	_estimator.settings["footstep_trailing_release_stretch_m"] = 0.28
	_estimator.settings["footstep_stop_speed_m_s"] = 0.10
	_estimator.settings["footstep_catchup_duration_s"] = 0.28
	_estimator.settings["footstep_speed_duration_reduction_s"] = 0.0
	_estimator.settings["step_duration_s"] = 0.36
	_estimator.settings["step_lift_height_m"] = 0.075
	_estimator.settings["arm_bend_lookup_blend"] = 0.75


func _process(delta: float) -> void:
	_stream_client.poll_connection()
	if _stream_client.is_stream_connected() and not _session_message_sent:
		_session_message_sent = _stream_client.send_message(
			_stream_client.build_session_message(_session_id, "godot_live_overlay")
		)

	var devices = _available_devices()
	var sample = _current_sample(devices)
	if sample.is_empty():
		_status.text = "No XR poses"
		return
	if devices.is_empty():
		devices = _devices_from_sample(sample)

	_update_capture_state(sample, devices, delta)
	sample["optional_trackers"] = _tracker_calibrator.apply_optional_trackers(
		devices,
		_tracker_calibration
	)
	var estimator_sample = _estimator_sample(sample)

	var estimated_frame: Dictionary = _estimator.estimate(estimator_sample, calibration, grounding)
	var debug_frame = _debug_frame_from_estimate(estimated_frame)
	var display_forward = _display_forward(devices)
	var display_frame = _mirror_display_frame(debug_frame, display_forward)
	_position_live_debug(display_frame, devices)
	_armature.set_pose_frame(display_frame, DEFAULT_BONES, Color(1.0, 0.38, 0.16, 1.0))
	_update_tracker_markers(devices, estimated_frame, display_forward)
	_update_headlocked_prompt()
	_status.text = _status_text(devices)

	if _should_stream_frame(sample["timestamp_seconds"]):
		var message = _stream_client.build_frame_message(
			_session_id,
			_frame_index,
			sample["timestamp_seconds"],
			sample,
			estimated_frame,
			calibration,
			grounding,
			devices,
			"standing",
			_raw_to_standing_transform(),
			_clip_index
		)
		_stream_client.send_message(message)
		_last_stream_timestamp_s = sample["timestamp_seconds"]
		_frame_index += 1


func _setup_xr_nodes() -> void:
	_origin = XROrigin3D.new()
	_origin.name = "XROrigin3D"
	add_child(_origin)

	_camera = XRCamera3D.new()
	_camera.name = "XRCamera3D"
	_origin.add_child(_camera)

	_left_controller = XRController3D.new()
	_left_controller.name = "LeftController"
	_left_controller.tracker = &"left_hand"
	_origin.add_child(_left_controller)

	_right_controller = XRController3D.new()
	_right_controller.name = "RightController"
	_right_controller.tracker = &"right_hand"
	_origin.add_child(_right_controller)


func _setup_debug_scene() -> void:
	_armature = DebugArmature.new()
	_armature.name = "EstimatedMirrorArmature"
	add_child(_armature)

	_status = Label3D.new()
	_status.name = "Status"
	_status.billboard = BaseMaterial3D.BILLBOARD_ENABLED
	_status.font_size = 18
	add_child(_status)

	_headlocked_prompt_root = Node3D.new()
	_headlocked_prompt_root.name = "HeadlockedPrompt"
	_headlocked_prompt_root.position = Vector3(0.0, -0.18, -1.15)
	_headlocked_prompt_root.visible = false
	var prompt_quad = MeshInstance3D.new()
	prompt_quad.name = "Background"
	var prompt_mesh = QuadMesh.new()
	prompt_mesh.size = Vector2(0.95, 0.34)
	prompt_quad.mesh = prompt_mesh
	var prompt_material = StandardMaterial3D.new()
	prompt_material.albedo_color = Color(0.0, 0.0, 0.0, 0.58)
	prompt_material.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	prompt_material.cull_mode = BaseMaterial3D.CULL_DISABLED
	prompt_quad.material_override = prompt_material
	_headlocked_prompt_root.add_child(prompt_quad)
	_headlocked_prompt = Label3D.new()
	_headlocked_prompt.name = "Label3D"
	_headlocked_prompt.position = Vector3(0.0, 0.0, -0.01)
	_headlocked_prompt.font_size = 34
	_headlocked_prompt.outline_size = 8
	_headlocked_prompt.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	_headlocked_prompt.vertical_alignment = VERTICAL_ALIGNMENT_CENTER
	_headlocked_prompt_root.add_child(_headlocked_prompt)
	if _camera != null:
		_camera.add_child(_headlocked_prompt_root)
	else:
		add_child(_headlocked_prompt_root)

	_calibration_target = MeshInstance3D.new()
	_calibration_target.name = "CalibrationTarget"
	var target_mesh = CylinderMesh.new()
	target_mesh.top_radius = calibration_target_radius_m
	target_mesh.bottom_radius = calibration_target_radius_m
	target_mesh.height = 0.025
	_calibration_target.mesh = target_mesh
	_calibration_target_material = StandardMaterial3D.new()
	_calibration_target_material.albedo_color = Color(0.1, 0.85, 0.95, 0.75)
	_calibration_target.material_override = _calibration_target_material
	add_child(_calibration_target)

	var light = DirectionalLight3D.new()
	light.name = "KeyLight"
	light.rotation_degrees = Vector3(-45.0, 30.0, 0.0)
	add_child(light)


func _initialize_openvr_overlay() -> void:
	_xr_interface = XRServer.find_interface("OpenVR Overlay")
	if _xr_interface == null:
		push_warning(
			"OpenVR Overlay interface is not available. Symlink or copy the "
			+ "godot-openvr-overlay addon into this project for live VR overlay mode."
		)
		return
	if _xr_interface.is_initialized() or _xr_interface.initialize():
		get_viewport().use_xr = true
		print("OpenVR Overlay initialized")
	else:
		push_warning("OpenVR Overlay interface did not initialize")


func _current_sample(devices: Array) -> Dictionary:
	var timestamp_seconds = Time.get_ticks_msec() / 1000.0
	var device_sample = _sample_from_devices(timestamp_seconds, devices)
	if not device_sample.is_empty():
		return device_sample
	if _has_xr_pose():
		return {
			"timestamp_seconds": timestamp_seconds,
			"hmd": _tracker_state("head", _estimator_head_transform(_camera.global_transform), "hmd"),
			"left_hand": _tracker_state("left_hand", _left_controller.global_transform, "left_hand"),
			"right_hand": _tracker_state(
				"right_hand",
				_right_controller.global_transform,
				"right_hand"
			),
			"optional_trackers": {},
			"character_velocity": [0.0, 0.0, 0.0],
		}
	if synthetic_input_without_xr:
		return _synthetic_sample(timestamp_seconds)
	return {}


func _estimator_sample(sample: Dictionary) -> Dictionary:
	if use_calibrated_trackers_for_estimation:
		return sample
	var estimator_sample = sample.duplicate(true)
	estimator_sample["optional_trackers"] = {}
	return estimator_sample


func _sample_from_devices(timestamp_seconds: float, devices: Array) -> Dictionary:
	var hmd = _find_device(devices, "head", "HMD")
	var left_hand = _find_device(devices, "left_hand", "Controller")
	var right_hand = _find_device(devices, "right_hand", "Controller")
	if hmd.is_empty() or left_hand.is_empty() or right_hand.is_empty():
		return {}
	if not (
		bool(hmd.get("valid", true))
		and bool(left_hand.get("valid", true))
		and bool(right_hand.get("valid", true))
	):
		return {}
	return {
		"timestamp_seconds": timestamp_seconds,
		"hmd": _tracker_state_from_device(
			"head",
			hmd,
			"openvr_hmd",
			true
		),
		"left_hand": _tracker_state_from_device("left_hand", left_hand, "openvr_left_hand"),
		"right_hand": _tracker_state_from_device("right_hand", right_hand, "openvr_right_hand"),
		"optional_trackers": {},
		"character_velocity": [0.0, 0.0, 0.0],
	}


func _find_device(devices: Array, role: String, device_class: String) -> Dictionary:
	for item in devices:
		if not (item is Dictionary):
			continue
		if not bool(item.get("valid", true)):
			continue
		if str(item.get("role", "")) == role:
			return item
	for item in devices:
		if not (item is Dictionary):
			continue
		if not bool(item.get("valid", true)):
			continue
		if str(item.get("device_class", "")) == device_class:
			return item
	return {}


func _has_xr_pose() -> bool:
	return (
		_camera.global_transform.origin.length() > 0.0001
		or _left_controller.global_transform.origin.length() > 0.0001
		or _right_controller.global_transform.origin.length() > 0.0001
	)


func _synthetic_sample(timestamp_seconds: float) -> Dictionary:
	var sway = sin(timestamp_seconds * 1.7) * 0.12
	var head = Transform3D(Basis(), Vector3(sway, 1.62, 0.0))
	var left = Transform3D(Basis(), Vector3(-0.28 + sway, 1.22, 0.36))
	var right = Transform3D(Basis(), Vector3(0.28 + sway, 1.22, 0.36))
	return {
		"timestamp_seconds": timestamp_seconds,
		"hmd": _tracker_state("head", head, "synthetic_hmd"),
		"left_hand": _tracker_state("left_hand", left, "synthetic_left_hand"),
		"right_hand": _tracker_state("right_hand", right, "synthetic_right_hand"),
		"optional_trackers": {},
		"character_velocity": [0.0, 0.0, 0.0],
	}


func _tracker_state(role: String, transform: Transform3D, device_id: String) -> Dictionary:
	return {
		"role": role,
		"device_id": device_id,
		"valid": true,
		"confidence": 1.0,
		"transform": _transform_to_json(transform),
	}


func _tracker_state_from_device(
	role: String,
	device: Dictionary,
	fallback_device_id: String,
	apply_head_correction = false
) -> Dictionary:
	var transform = _transform3d_from_value(device.get("transform", Transform3D()))
	if apply_head_correction:
		transform = _estimator_head_transform(transform)
	return {
		"role": role,
		"device_id": str(device.get("device_id", fallback_device_id)),
		"valid": bool(device.get("valid", true)),
		"confidence": 1.0,
		"transform": _transform_to_json(transform),
	}


func _estimator_head_transform(transform: Transform3D) -> Transform3D:
	var corrected = transform
	corrected.basis = Basis(
		transform.basis.get_rotation_quaternion() * Quaternion(Vector3.UP, PI)
	)
	return corrected


func _devices_from_sample(sample: Dictionary) -> Array:
	return [
		_device_from_tracker(sample["hmd"], "HMD"),
		_device_from_tracker(sample["left_hand"], "Controller"),
		_device_from_tracker(sample["right_hand"], "Controller"),
	]


func _available_devices() -> Array:
	if _xr_interface != null and _xr_interface.has_method("get_tracked_devices"):
		var devices = _xr_interface.get_tracked_devices()
		if not devices.is_empty():
			return devices
	return []


func _device_from_tracker(tracker: Dictionary, device_class: String) -> Dictionary:
	var device = tracker.duplicate(true)
	device["device_class"] = device_class
	device["linear_velocity"] = [0.0, 0.0, 0.0]
	device["angular_velocity"] = [0.0, 0.0, 0.0]
	return device


func _raw_to_standing_transform():
	if _xr_interface == null or not _xr_interface.has_method("get_raw_zero_to_standing_transform"):
		return null
	var values = _xr_interface.get_raw_zero_to_standing_transform()
	if values == null or values.size() != 12:
		return null
	var result = []
	for value in values:
		result.append(value)
	return result


func _generic_tracker_count(devices: Array) -> int:
	var count = 0
	for item in devices:
		if (
			item is Dictionary
			and str(item.get("device_class", "")) == "GenericTracker"
			and bool(item.get("valid", true))
		):
			count += 1
	return count


func _debug_frame_from_estimate(frame: Dictionary) -> Dictionary:
	var root_position = _vector3_from_transform(frame["root"])
	var joints = {}
	for joint_name in frame["joints"].keys():
		var joint = frame["joints"][joint_name]
		var transform = joint["transform"]
		var position = _vector3_from_transform(transform) - root_position
		joints[joint_name] = {
			"position": position,
			"rotation": _quaternion_from_transform(transform),
		}

	_add_visual_helper_joints(joints)
	return {
		"timestamp_seconds": frame["timestamp_seconds"],
		"root": {"position": Vector3.ZERO, "rotation": frame["root"]["rotation"]},
		"joints": joints,
	}


func _mirror_display_frame(frame: Dictionary, display_forward: Vector3) -> Dictionary:
	var mirrored = frame.duplicate(true)
	if mirrored.has("root") and mirrored["root"] is Dictionary:
		var root: Dictionary = mirrored["root"]
		if root.has("position"):
			root["position"] = _display_relative_position(
				_vector3_from_value(root["position"]),
				display_forward
			)
	var joints: Dictionary = mirrored.get("joints", {})
	for joint_name in joints.keys():
		var joint = joints[joint_name]
		if joint is Dictionary and joint.has("position"):
			joint["position"] = _display_relative_position(
				_vector3_from_value(joint["position"]),
				display_forward
			)
	return mirrored


func _add_visual_helper_joints(joints: Dictionary) -> void:
	if joints.has("head"):
		var head = joints["head"]
		var forward = _rotate_vector(head["rotation"], WORLD_FORWARD).normalized()
		joints["head_forward"] = {
			"position": head["position"] + forward * 0.22,
			"rotation": head["rotation"],
		}

	for side in ["left", "right"]:
		var elbow_name = "%s_elbow" % side
		var hand_name = "%s_hand" % side
		var wrist_name = "%s_wrist" % side
		if not joints.has(elbow_name) or not joints.has(hand_name):
			continue
		var hand = joints[hand_name]["position"]
		var elbow = joints[elbow_name]["position"]
		var hand_to_elbow = elbow - hand
		var wrist_offset = minf(0.09, hand_to_elbow.length() * 0.35)
		joints[wrist_name] = {
			"position": hand + hand_to_elbow.normalized() * wrist_offset,
			"rotation": joints[hand_name]["rotation"],
		}


func _position_live_debug(debug_frame: Dictionary, devices: Array) -> void:
	var head_tracker = _current_head_tracker(devices)
	var head_position = _source_position(head_tracker)
	var head_rotation = _quaternion_from_tracker(head_tracker)
	var forward = _rotate_vector(head_rotation, GODOT_VIEW_FORWARD)
	forward = Vector3(forward.x, 0.0, forward.z)
	if forward.length() <= 0.0001:
		forward = GODOT_VIEW_FORWARD
	forward = forward.normalized()
	var mirror_origin = head_position + forward * mirror_distance_m
	mirror_origin.y += mirror_height_offset_m
	_armature.position = mirror_origin
	_armature.rotation = Vector3(0.0, _mirror_yaw_for_forward(forward), 0.0)
	_status.position = mirror_origin + Vector3(0.0, 1.2, 0.0)


func _update_capture_state(sample: Dictionary, devices: Array, delta: float) -> void:
	var head_tracker: Dictionary = _current_head_tracker(devices)
	var head_position = _source_position(head_tracker)
	var head_rotation = _quaternion_from_tracker(head_tracker)
	var forward = _horizontal(_rotate_vector(head_rotation, GODOT_VIEW_FORWARD))
	if forward.length() <= 0.0001:
		forward = GODOT_VIEW_FORWARD
	forward = forward.normalized()

	_calibration_target.position = Vector3(
		head_position.x + forward.x * calibration_target_distance_m,
		float(grounding.get("floor_height_m", 0.0)) + 0.02,
		head_position.z + forward.z * calibration_target_distance_m
	)
	var gaze_hits_target = _gaze_hits_calibration_target(head_tracker)

	if _capture_phase == "idle":
		_update_idle_capture_state(delta, gaze_hits_target, sample, devices)
	elif _capture_phase == "countdown":
		_update_countdown_capture_state(delta, sample, devices, head_tracker)
	elif _capture_phase == "settle":
		_update_timed_capture_phase(delta, "recording")
	elif _capture_phase == "recording":
		_update_timed_capture_phase(delta, "complete")
	elif _capture_phase == "complete" and not gaze_hits_target:
		_capture_phase = "idle"
		_requires_gaze_release = false

	_update_calibration_target_color()


func _update_idle_capture_state(
	delta: float,
	gaze_hits_target: bool,
	_sample: Dictionary,
	_devices: Array
) -> void:
	if _requires_gaze_release:
		if not gaze_hits_target:
			_requires_gaze_release = false
		return

	if gaze_hits_target:
		_calibration_dwell_s += delta
	else:
		_calibration_dwell_s = maxf(0.0, _calibration_dwell_s - delta * 2.0)

	if _calibration_dwell_s >= calibration_dwell_seconds:
		_capture_phase = "countdown"
		_phase_remaining_s = calibration_countdown_seconds
		_calibration_dwell_s = 0.0
		_requires_gaze_release = true


func _update_countdown_capture_state(
	delta: float,
	sample: Dictionary,
	devices: Array,
	head_tracker: Dictionary
) -> void:
	_phase_remaining_s -= delta
	if _phase_remaining_s > 0.0:
		return
	var calibration_sample = sample.duplicate(true)
	calibration_sample["hmd"] = head_tracker
	_tracker_calibration = _tracker_calibrator.build_calibration(calibration_sample, devices)
	_estimator.reset()
	_capture_phase = "settle"
	_phase_remaining_s = post_calibration_settle_seconds
	_clip_index += 1
	_frame_index = 0
	_last_stream_timestamp_s = -INF
	print(
		"Live tracker calibration captured with roles: ",
		_tracker_calibration.get("role_bindings", {})
	)


func _update_timed_capture_phase(delta: float, next_phase: String) -> void:
	_phase_remaining_s -= delta
	if _phase_remaining_s > 0.0:
		return
	if next_phase == "recording":
		_capture_phase = "recording"
		_phase_remaining_s = clip_duration_seconds
		_last_stream_timestamp_s = -INF
	elif next_phase == "complete":
		_capture_phase = "complete"
		_phase_remaining_s = 0.0
		_send_clip_flush()


func _send_clip_flush() -> void:
	if _last_flushed_clip_index == _clip_index:
		return
	if not _stream_client.is_stream_connected():
		return
	if _stream_client.send_message(
		_stream_client.build_flush_message(
			_session_id,
			_clip_index,
			_frame_index,
			"clip_complete"
		)
	):
		_last_flushed_clip_index = _clip_index


func _update_calibration_target_color() -> void:
	var progress = _capture_progress()
	_calibration_target_material.albedo_color = Color(
		0.1 + progress * 0.9,
		0.85 - progress * 0.35,
		0.95 - progress * 0.7,
		0.78
	)


func _capture_progress() -> float:
	if _capture_phase == "idle":
		return clampf(_calibration_dwell_s / calibration_dwell_seconds, 0.0, 1.0)
	if _capture_phase == "countdown":
		return 1.0 - clampf(_phase_remaining_s / calibration_countdown_seconds, 0.0, 1.0)
	if _capture_phase == "settle":
		return 1.0 - clampf(_phase_remaining_s / post_calibration_settle_seconds, 0.0, 1.0)
	if _capture_phase == "recording":
		return 1.0
	return 0.0


func _status_text(devices: Array) -> String:
	var recorder = "recorder connected" if _stream_client.is_stream_connected() else "recorder offline"
	var tracker_count = _generic_tracker_count(devices)
	var estimator_mode = "calibrated estimate" if use_calibrated_trackers_for_estimation else "3-point estimate"
	var prefix = "%s | live %d targets | %s" % [recorder, tracker_count, estimator_mode]
	if _capture_phase == "idle":
		if _requires_gaze_release:
			return "%s | release gaze target" % prefix
		return "%s | gaze target %.1fs" % [prefix, calibration_dwell_seconds - _calibration_dwell_s]
	if _capture_phase == "countdown":
		return "%s | T-pose in %.1fs" % [prefix, maxf(0.0, _phase_remaining_s)]
	if _capture_phase == "settle":
		return "%s | neutral stance %.1fs" % [prefix, maxf(0.0, _phase_remaining_s)]
	if _capture_phase == "recording":
		return "%s | recording clip %d %.1fs" % [
			prefix,
			_clip_index,
			maxf(0.0, _phase_remaining_s),
		]
	return "%s | clip %d complete" % [prefix, _clip_index]


func _update_headlocked_prompt() -> void:
	if _headlocked_prompt_root == null or _headlocked_prompt == null:
		return
	var text = _headlocked_prompt_text()
	_headlocked_prompt_root.visible = not text.is_empty()
	_headlocked_prompt.text = text


func _headlocked_prompt_text() -> String:
	if _capture_phase == "countdown":
		return "T-pose\n%.1f" % maxf(0.0, _phase_remaining_s)
	if _capture_phase == "settle":
		return "neutral stance\n%.1f" % maxf(0.0, _phase_remaining_s)
	if _capture_phase == "recording":
		return "recording\n%.1f" % maxf(0.0, _phase_remaining_s)
	if _capture_phase == "complete":
		return "clip saved"
	return ""


func _should_stream_frame(timestamp_seconds: float) -> bool:
	if _capture_phase != "recording":
		return false
	if capture_sample_rate_hz <= 0.0:
		return true
	var min_interval = 1.0 / capture_sample_rate_hz
	return (
		_last_stream_timestamp_s == -INF
		or timestamp_seconds - _last_stream_timestamp_s >= min_interval
	)


func _gaze_hits_calibration_target(head_tracker: Dictionary) -> bool:
	var origin = _source_position(head_tracker)
	var direction = _rotate_vector(_quaternion_from_tracker(head_tracker), GODOT_VIEW_FORWARD)
	if absf(direction.y) <= 0.0001:
		return false
	var floor_y = float(grounding.get("floor_height_m", 0.0)) + 0.02
	var t = (floor_y - origin.y) / direction.y
	if t <= 0.0:
		return false
	var hit = origin + direction * t
	var delta = hit - _calibration_target.position
	delta.y = 0.0
	return delta.length() <= calibration_target_radius_m


func _current_head_tracker(devices: Array = []) -> Dictionary:
	var device_head = _find_device(devices, "head", "HMD")
	if not device_head.is_empty():
		return _tracker_state_from_device("head", device_head, "openvr_hmd")
	if _has_xr_pose():
		return _tracker_state("head", _camera.global_transform, "hmd")
	return _synthetic_sample(Time.get_ticks_msec() / 1000.0)["hmd"]


func _display_forward(devices: Array) -> Vector3:
	var head_tracker = _current_head_tracker(devices)
	var forward = _horizontal(_rotate_vector(_quaternion_from_tracker(head_tracker), GODOT_VIEW_FORWARD))
	if forward.length() <= 0.0001:
		return GODOT_VIEW_FORWARD
	return forward.normalized()


func _update_tracker_markers(
	devices: Array,
	estimated_frame: Dictionary,
	display_forward: Vector3
) -> void:
	_hide_tracker_markers()
	var root_position = _vector3_from_transform(estimated_frame["root"])
	for item in devices:
		if not (item is Dictionary):
			continue
		var device: Dictionary = item
		if not _should_show_device_marker(device):
			continue
		var marker = _tracker_marker(_device_marker_key(device))
		var relative_position = _device_position(device) - root_position
		relative_position = _display_relative_position(relative_position, display_forward)
		marker.position = _armature.position + _armature.basis * relative_position
		marker.visible = true
		_set_tracker_marker_label(marker, _tracker_label_for_device(device))
		_set_tracker_marker_color(marker, _tracker_color_for_device(device))


func _hide_tracker_markers() -> void:
	for marker in _tracker_markers.values():
		marker.visible = false


func _should_show_device_marker(device: Dictionary) -> bool:
	if not bool(device.get("valid", true)):
		return false
	var device_class = str(device.get("device_class", ""))
	return device_class in ["HMD", "Controller", "GenericTracker"]


func _tracker_marker(key: String) -> Node3D:
	if _tracker_markers.has(key):
		return _tracker_markers[key]
	var marker = Node3D.new()
	marker.name = "%sMarker" % _safe_node_name(key)
	var mesh = SphereMesh.new()
	mesh.radius = tracker_marker_radius_m
	mesh.height = tracker_marker_radius_m * 2.0
	var sphere = MeshInstance3D.new()
	sphere.name = "Sphere"
	sphere.mesh = mesh
	var material = StandardMaterial3D.new()
	material.albedo_color = Color(0.2, 0.9, 0.35, 1.0)
	sphere.material_override = material
	marker.add_child(sphere)
	var label = Label3D.new()
	label.name = "Label3D"
	label.position = Vector3(0.0, tracker_marker_radius_m * 2.15, 0.0)
	label.billboard = BaseMaterial3D.BILLBOARD_ENABLED
	label.font_size = tracker_label_font_size
	label.outline_size = max(2, int(round(tracker_label_font_size * 0.25)))
	marker.add_child(label)
	add_child(marker)
	_tracker_markers[key] = marker
	return marker


func _set_tracker_marker_label(marker: Node3D, text: String) -> void:
	var label = marker.get_node("Label3D")
	label.text = text


func _set_tracker_marker_color(marker: Node3D, color: Color) -> void:
	var sphere = marker.get_node("Sphere")
	var material: StandardMaterial3D = sphere.material_override
	material.albedo_color = color


func _tracker_color_for_device(device: Dictionary) -> Color:
	var role = _calibrated_role_for_device(device)
	if not role.is_empty():
		return Color(1.0, 0.8, 0.18, 1.0)
	var device_class = str(device.get("device_class", ""))
	if device_class == "HMD":
		return Color(0.25, 0.65, 1.0, 1.0)
	if device_class == "Controller":
		return Color(0.2, 0.95, 0.4, 1.0)
	return Color(0.82, 0.82, 0.88, 1.0)


func _tracker_label_for_device(device: Dictionary) -> String:
	var calibrated_role = _calibrated_role_for_device(device)
	if not calibrated_role.is_empty():
		return calibrated_role
	var role = str(device.get("role", ""))
	if not role.is_empty() and role != "generic_tracker":
		return role
	if str(device.get("device_class", "")) == "GenericTracker":
		return "tracker %s" % _short_device_id(device)
	return str(device.get("device_class", "device"))


func _calibrated_role_for_device(device: Dictionary) -> String:
	if _tracker_calibration.is_empty():
		return ""
	var device_id = _calibration_device_id(device)
	var role_bindings: Dictionary = _tracker_calibration.get("role_bindings", {})
	for role in role_bindings.keys():
		if str(role_bindings[role]) == device_id:
			return str(role)
	return ""


func _device_marker_key(device: Dictionary) -> String:
	return _calibration_device_id(device)


func _calibration_device_id(device: Dictionary) -> String:
	var serial = str(device.get("serial_number", ""))
	if not serial.is_empty():
		return "openvr:%s" % serial
	return str(device.get("device_id", device.get("device_index", "")))


func _short_device_id(device: Dictionary) -> String:
	var value = str(device.get("serial_number", device.get("device_id", "")))
	if value.length() <= 4:
		return value
	return value.substr(value.length() - 4)


func _safe_node_name(value: String) -> String:
	var safe = ""
	for index in range(value.length()):
		var character = value[index]
		if character.is_valid_identifier() or character.is_valid_int():
			safe += character
		else:
			safe += "_"
	return safe


func _apply_user_args() -> void:
	var args = OS.get_cmdline_user_args()
	for index in range(args.size()):
		if args[index] == "--capture-host" and index + 1 < args.size():
			stream_host = args[index + 1]
		if args[index] == "--capture-port" and index + 1 < args.size():
			stream_port = int(args[index + 1])
		if args[index] == "--no-synthetic-live-input":
			synthetic_input_without_xr = false
		if args[index] == "--no-openvr-overlay":
			openvr_overlay_enabled = false
		if args[index] == "--use-calibrated-trackers-for-estimation":
			use_calibrated_trackers_for_estimation = true


func _new_session_id() -> String:
	return "%d" % Time.get_unix_time_from_system()


func _transform_to_json(transform: Transform3D) -> Dictionary:
	var rotation = transform.basis.get_rotation_quaternion()
	return {
		"position": [transform.origin.x, transform.origin.y, transform.origin.z],
		"rotation": [rotation.x, rotation.y, rotation.z, rotation.w],
	}


func _source_position(tracker: Dictionary) -> Vector3:
	return _vector3_from_transform(tracker["transform"])


func _device_position(device: Dictionary) -> Vector3:
	return _transform3d_from_value(device.get("transform", Transform3D())).origin


func _quaternion_from_tracker(tracker: Dictionary) -> Quaternion:
	return _quaternion_from_transform(tracker["transform"])


func _vector3_from_transform(transform: Dictionary) -> Vector3:
	return _vector3_from_value(transform["position"])


func _vector3_from_value(value) -> Vector3:
	if value is Vector3:
		return value
	return Vector3(value[0], value[1], value[2])


func _quaternion_from_transform(transform: Dictionary) -> Quaternion:
	var value = transform["rotation"]
	if value is Quaternion:
		return value
	return Quaternion(value[0], value[1], value[2], value[3])


func _transform3d_from_value(value) -> Transform3D:
	if value is Transform3D:
		return value
	if value is Dictionary:
		return Transform3D(Basis(_quaternion_from_transform(value)), _vector3_from_transform(value))
	return Transform3D()


func _rotate_vector(rotation: Quaternion, vector: Vector3) -> Vector3:
	return rotation * vector


func _horizontal(vector: Vector3) -> Vector3:
	return Vector3(vector.x, 0.0, vector.z)


func _display_relative_position(position: Vector3, display_forward: Vector3) -> Vector3:
	var forward = _horizontal(display_forward)
	if forward.length() <= 0.0001:
		forward = GODOT_VIEW_FORWARD
	forward = forward.normalized()
	var right = forward.cross(WORLD_UP)
	if right.length() <= 0.0001:
		right = Vector3.RIGHT
	right = right.normalized()
	var display_position = Vector3(
		position.dot(right),
		position.y,
		-position.dot(forward)
	)
	if not mirror_x_axis_enabled:
		return display_position
	return Vector3(-display_position.x, display_position.y, display_position.z)


func _mirror_yaw_for_forward(forward: Vector3) -> float:
	var horizontal_forward = _horizontal(forward)
	if horizontal_forward.length() <= 0.0001:
		return MIRROR_YAW_RADIANS
	horizontal_forward = horizontal_forward.normalized()
	return MIRROR_YAW_RADIANS - atan2(horizontal_forward.x, -horizontal_forward.z)
