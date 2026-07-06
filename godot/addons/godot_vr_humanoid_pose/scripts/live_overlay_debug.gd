extends Node3D
class_name VRHumanoidLiveOverlayDebug

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
@export var calibration_target_radius_m = 0.18
@export var calibration_dwell_seconds = 1.25
@export var calibration_countdown_seconds = 3.0
@export var post_calibration_settle_seconds = 2.0
@export var clip_duration_seconds = 10.0
@export var capture_sample_rate_hz = 30.0
@export var estimate_sample_rate_hz = 30.0
@export var tracker_marker_radius_m = 0.028
@export var tracker_label_font_size = 10
@export var mirror_x_axis_enabled = true
@export var use_calibrated_trackers_for_estimation = false
@export var sparseposer_calibration_enabled = false
@export var openvr_overlay_enabled = true
@export var synthetic_input_without_xr = true
@export var capture_director_enabled = false

# The redo action re-runs the current director run without advancing the run
# pointer. Trigger stays the accept action (via _controller_confirm_pressed);
# pick a distinct button here so one press cannot serve both.
const CAPTURE_DIRECTOR_REDO_ACTION = "grip"

# Ordered capture protocol for the in-VR director. Occlusion is intentionally
# omitted (decision 2026-07: skip as fit data until an inside-out rig exists).
# To add it later, append a run like:
#   {"name": "occlusion", "instruction": "Occlude controllers behind back",
#    "duration_s": 45.0, "kind": "occlusion"}
const CAPTURE_PROTOCOL = [
	{
		"name": "idle",
		"instruction": "Stand relaxed — arms at sides, then loosely folded",
		"duration_s": 60.0,
		"kind": "idle",
	},
	{
		"name": "range_of_motion",
		"instruction": "Slow full-body reach: arms overhead, torso twist, look around",
		"duration_s": 60.0,
		"kind": "motion",
	},
	{
		"name": "sit_stand",
		"instruction": "Sit, settle, stand — repeat at a normal chair",
		"duration_s": 45.0,
		"kind": "sit_stand",
	},
]

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

var _stream_client = StreamClient.new()
var _tracker_calibrator = TrackerCalibration.new()
var _session_id = ""
var _frame_index = 0
var _estimate_request_index = 0
var _xr_interface
var _origin: XROrigin3D
var _camera: XRCamera3D
var _left_controller: XRController3D
var _right_controller: XRController3D
var _armature
var _first_person_armature
var _tracker_markers = {}
var _calibration_target: MeshInstance3D
var _calibration_target_material: StandardMaterial3D
var _playspace_forward_indicator: Node3D
var _status: Label3D
var _session_message_sent = false
var _tracker_calibration: Dictionary = {}
var _calibration_dwell_s = 0.0
var _capture_phase = "idle"
var _phase_remaining_s = 0.0
var _last_stream_timestamp_s = -INF
var _last_estimate_request_timestamp_s = -INF
var _clip_index = 0
var _last_flushed_clip_index = -1
var _requires_gaze_release = false
var _requires_trigger_release = false
var _run_index = 0
var _requires_redo_release = false
var _alignment_gaze_dwell_s = 0.0
var _headlocked_prompt_root: Node3D
var _headlocked_prompt: Label3D
var _latest_estimated_frame: Dictionary = {}
var _last_estimate_error = ""
var _last_local_estimate_status = ""
var _estimate_request_in_flight = false


func _ready() -> void:
	_apply_user_args()
	_session_id = _new_session_id()
	_stream_client.configure(stream_host, stream_port)
	_setup_xr_nodes()
	_setup_debug_scene()
	if openvr_overlay_enabled:
		_initialize_openvr_overlay()


func _process(delta: float) -> void:
	for message in _stream_client.poll_messages():
		_handle_stream_message(message)
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

	sample["optional_trackers"] = _tracker_calibrator.apply_optional_trackers(
		devices,
		_tracker_calibration
	)
	_update_capture_state(sample, devices, delta)
	sample["optional_trackers"] = _tracker_calibrator.apply_optional_trackers(
		devices,
		_tracker_calibration
	)
	var estimator_sample = _estimator_sample(sample)
	_send_estimation_or_recording_message(sample, estimator_sample, devices)

	var estimated_frame: Dictionary = _latest_estimated_frame
	if estimated_frame.is_empty():
		_position_status_display(devices)
		_update_headlocked_prompt(_current_head_tracker(devices))
		_status.text = "%s | waiting for Python estimator" % _status_text(devices)
		return
	var debug_frame = _debug_frame_from_estimate(estimated_frame)
	var display_forward = _display_forward(devices)
	var display_frame = _mirror_display_frame(debug_frame, display_forward)
	_position_first_person_debug(estimated_frame)
	_first_person_armature.set_pose_frame(
		debug_frame,
		_debug_bones_for_estimate(estimated_frame),
		Color(0.2, 0.65, 1.0, 1.0)
	)
	_position_live_debug(devices, estimated_frame)
	_armature.set_pose_frame(
		display_frame,
		_debug_bones_for_estimate(estimated_frame),
		Color(1.0, 0.38, 0.16, 1.0)
	)
	_update_tracker_markers(devices, estimated_frame, display_forward)
	_update_headlocked_prompt(_current_head_tracker(devices))
	_status.text = _status_text(devices)


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
	_first_person_armature = DebugArmature.new()
	_first_person_armature.name = "EstimatedFirstPersonArmature"
	_first_person_armature.bone_radius = 0.018
	add_child(_first_person_armature)

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
	_headlocked_prompt_root.position = Vector3(0.0, 0.22, -1.25)
	_headlocked_prompt_root.visible = false
	var prompt_quad = MeshInstance3D.new()
	prompt_quad.name = "Background"
	var prompt_mesh = QuadMesh.new()
	prompt_mesh.size = Vector2(0.44, 0.16)
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
	_headlocked_prompt.font_size = 12
	_headlocked_prompt.outline_size = 3
	_headlocked_prompt.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	_headlocked_prompt.vertical_alignment = VERTICAL_ALIGNMENT_CENTER
	_headlocked_prompt_root.add_child(_headlocked_prompt)
	add_child(_headlocked_prompt_root)

	_calibration_target = MeshInstance3D.new()
	_calibration_target.name = "CalibrationTarget"
	var target_mesh = CylinderMesh.new()
	target_mesh.top_radius = calibration_target_radius_m
	target_mesh.bottom_radius = calibration_target_radius_m
	target_mesh.height = 0.012
	_calibration_target.mesh = target_mesh
	_calibration_target_material = StandardMaterial3D.new()
	_calibration_target_material.albedo_color = Color(0.1, 0.85, 0.95, 0.28)
	_calibration_target_material.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	_calibration_target_material.cull_mode = BaseMaterial3D.CULL_DISABLED
	_calibration_target.material_override = _calibration_target_material
	add_child(_calibration_target)

	_playspace_forward_indicator = _make_playspace_forward_indicator()
	add_child(_playspace_forward_indicator)

	var light = DirectionalLight3D.new()
	light.name = "KeyLight"
	light.rotation_degrees = Vector3(-45.0, 30.0, 0.0)
	add_child(light)


func _make_playspace_forward_indicator() -> Node3D:
	var root = Node3D.new()
	root.name = "PlayspaceForwardIndicator"

	var material = StandardMaterial3D.new()
	material.albedo_color = Color(0.2, 0.75, 1.0, 0.82)
	material.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	material.cull_mode = BaseMaterial3D.CULL_DISABLED

	var shaft = MeshInstance3D.new()
	shaft.name = "Shaft"
	var shaft_mesh = CylinderMesh.new()
	shaft_mesh.top_radius = 0.012
	shaft_mesh.bottom_radius = 0.012
	shaft_mesh.height = 0.5
	shaft.mesh = shaft_mesh
	shaft.material_override = material
	shaft.rotation_degrees = Vector3(90.0, 0.0, 0.0)
	shaft.position = Vector3(0.0, 0.0, 0.28)
	root.add_child(shaft)

	var head = MeshInstance3D.new()
	head.name = "ArrowHead"
	var head_mesh = CylinderMesh.new()
	head_mesh.top_radius = 0.0
	head_mesh.bottom_radius = 0.055
	head_mesh.height = 0.16
	head.mesh = head_mesh
	head.material_override = material
	head.rotation_degrees = Vector3(90.0, 0.0, 0.0)
	head.position = Vector3(0.0, 0.0, 0.58)
	root.add_child(head)

	var label = Label3D.new()
	label.name = "Label3D"
	label.text = "+Z"
	label.font_size = 18
	label.outline_size = 4
	label.billboard = BaseMaterial3D.BILLBOARD_ENABLED
	label.position = Vector3(0.0, 0.055, 0.78)
	root.add_child(label)
	return root


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
	if use_calibrated_trackers_for_estimation or sparseposer_calibration_enabled:
		return sample
	var estimator_sample = sample.duplicate(true)
	estimator_sample["optional_trackers"] = {}
	return estimator_sample


func _estimator_calibration() -> Dictionary:
	var estimator_calibration = calibration.duplicate(true)
	if _tracker_calibration.is_empty():
		return estimator_calibration
	estimator_calibration["live_tracker_calibration"] = _tracker_calibration
	if _tracker_calibration.has("sparseposer_alignment"):
		estimator_calibration["sparseposer_alignment"] = _tracker_calibration[
			"sparseposer_alignment"
		]
	return estimator_calibration


func _send_estimation_or_recording_message(
	sample: Dictionary,
	estimator_sample: Dictionary,
	devices: Array
) -> void:
	var timestamp_seconds = float(sample["timestamp_seconds"])
	var estimator_calibration = _estimator_calibration()
	if not _can_send_estimation_message(estimator_sample):
		return
	if _estimate_request_in_flight:
		return
	if _should_stream_frame(timestamp_seconds):
		var message = _stream_client.build_frame_message(
			_session_id,
			_frame_index,
			timestamp_seconds,
			estimator_sample,
			{},
			estimator_calibration,
			grounding,
			devices,
			"standing",
			_raw_to_standing_transform(),
			_clip_index,
			_current_run_name(),
			_current_run_kind()
		)
		if _stream_client.send_message(message):
			_estimate_request_in_flight = true
			_last_stream_timestamp_s = timestamp_seconds
			_frame_index += 1
			return

	if not _should_request_estimate(timestamp_seconds):
		return
	var request = _stream_client.build_estimate_request_message(
		_session_id,
		_estimate_request_index,
		_frame_index,
		timestamp_seconds,
		estimator_sample,
		estimator_calibration,
		grounding,
		"standing",
		_clip_index,
		devices
	)
	if _stream_client.send_message(request):
		_estimate_request_in_flight = true
		_last_estimate_request_timestamp_s = timestamp_seconds
		_estimate_request_index += 1


func _handle_stream_message(message: Dictionary) -> void:
	if str(message.get("session_id", "")) != _session_id:
		return
	var message_type = str(message.get("type", ""))
	if message_type == "estimate_response":
		_estimate_request_in_flight = false
		var estimated = message.get("estimated_frame", {})
		if estimated is Dictionary and not estimated.is_empty():
			_latest_estimated_frame = estimated
			_last_estimate_error = ""
			_last_local_estimate_status = ""
	elif message_type == "estimate_error":
		_estimate_request_in_flight = false
		var error = str(message.get("error", "estimate_error"))
		if error != _last_estimate_error:
			push_warning("Live estimator error: %s" % error)
		_last_estimate_error = error
		_latest_estimated_frame = {}
	elif message_type == "estimator_reset":
		_estimate_request_in_flight = false


func _sample_from_devices(timestamp_seconds: float, devices: Array) -> Dictionary:
	var hmd = _find_device(devices, "head", "HMD")
	var controllers = _controller_devices_for_sample(devices, hmd)
	var left_hand: Dictionary = controllers.get("left_hand", {})
	var right_hand: Dictionary = controllers.get("right_hand", {})
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
		"left_hand": _tracker_state_from_device(
			"left_hand",
			left_hand,
			"openvr_left_hand",
			false,
			_controller_wrist_transform(left_hand)
		),
		"right_hand": _tracker_state_from_device(
			"right_hand",
			right_hand,
			"openvr_right_hand",
			false,
			_controller_wrist_transform(right_hand)
		),
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


func _controller_devices_for_sample(devices: Array, head: Dictionary = {}) -> Dictionary:
	var bound = _controller_devices_from_calibration(devices)
	if not bound.is_empty():
		return bound
	return _bind_controllers(devices, head, sparseposer_calibration_enabled)


func _controller_devices_from_calibration(devices: Array) -> Dictionary:
	if _tracker_calibration.is_empty():
		return {}
	var bindings: Dictionary = _tracker_calibration.get("controller_bindings", {})
	if bindings.is_empty():
		return {}
	var left_hand = _find_device_by_calibration_id(devices, str(bindings.get("left_hand", "")))
	var right_hand = _find_device_by_calibration_id(devices, str(bindings.get("right_hand", "")))
	if left_hand.is_empty() or right_hand.is_empty():
		return {}
	return {
		"left_hand": left_hand,
		"right_hand": right_hand,
	}


# Assign the two hand controllers to left/right roles. Normal capture prefers
# each controller's self-reported OpenVR handedness; SparsePoser calibration
# forces geometric T-pose-side binding (prefer_geometry). The geometric
# fallback projects onto the head-relative body-right axis, never world/
# playspace X -- world X mislabels the hands whenever the user faces away from
# +Z at calibration, which is the observed left/right controller swap.
func _bind_controllers(devices: Array, head: Dictionary, prefer_geometry: bool) -> Dictionary:
	var controllers = []
	for item in devices:
		if not (item is Dictionary):
			continue
		var device: Dictionary = item
		if str(device.get("device_class", "")) != "Controller":
			continue
		if not bool(device.get("valid", true)):
			continue
		controllers.append(device)
	if controllers.size() < 2:
		return {}
	if not prefer_geometry:
		var by_role = _controllers_by_reported_hand(controllers)
		if not by_role.is_empty():
			return by_role
	return _controllers_by_body_side(controllers, head)


func _controllers_by_reported_hand(controllers: Array) -> Dictionary:
	var left_hand := {}
	var right_hand := {}
	for device in controllers:
		match str(device.get("role", "")):
			"left_hand":
				if left_hand.is_empty():
					left_hand = device
			"right_hand":
				if right_hand.is_empty():
					right_hand = device
	if left_hand.is_empty() or right_hand.is_empty():
		return {}
	if _calibration_device_id(left_hand) == _calibration_device_id(right_hand):
		return {}
	return {
		"left_hand": left_hand,
		"right_hand": right_hand,
	}


func _controllers_by_body_side(controllers: Array, head: Dictionary) -> Dictionary:
	var head_position = _device_position(head) if not head.is_empty() else Vector3.ZERO
	var right_axis = _head_right_axis(head)
	var ordered = controllers.duplicate()
	ordered.sort_custom(func(a, b): return _body_side_key(a, head_position, right_axis) < _body_side_key(b, head_position, right_axis))
	return {
		"left_hand": ordered[0],
		"right_hand": ordered[ordered.size() - 1],
	}


func _body_side_key(device: Dictionary, head_position: Vector3, right_axis: Vector3) -> float:
	return (_device_position(device) - head_position).dot(right_axis)


# Horizontal body-right axis derived from HMD yaw. Falls back to world +X only
# when no head pose is available or the head is looking straight up/down.
func _head_right_axis(head: Dictionary) -> Vector3:
	if head.is_empty():
		return Vector3.RIGHT
	var basis = _transform3d_from_value(head.get("transform", Transform3D())).basis
	var right = basis.x
	right.y = 0.0
	if right.length() < 0.0001:
		var forward = -basis.z
		forward.y = 0.0
		if forward.length() < 0.0001:
			return Vector3.RIGHT
		right = forward.normalized().cross(Vector3.UP)
	if right.length() < 0.0001:
		return Vector3.RIGHT
	return right.normalized()


func _controller_bindings_from_devices(devices: Array, head: Dictionary = {}) -> Dictionary:
	var controllers = _bind_controllers(devices, head, sparseposer_calibration_enabled)
	if controllers.is_empty():
		return {}
	return {
		"left_hand": _calibration_device_id(controllers["left_hand"]),
		"right_hand": _calibration_device_id(controllers["right_hand"]),
	}


func _find_device_by_calibration_id(devices: Array, device_id: String) -> Dictionary:
	if device_id.is_empty():
		return {}
	for item in devices:
		if not (item is Dictionary):
			continue
		var device: Dictionary = item
		if _calibration_device_id(device) == device_id:
			return device
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
	apply_head_correction = false,
	transform_override = null
) -> Dictionary:
	var transform_value = transform_override if transform_override != null else device.get("transform", Transform3D())
	var transform = _transform3d_from_value(transform_value)
	if apply_head_correction:
		transform = _estimator_head_transform(transform)
	return {
		"role": role,
		"device_id": str(device.get("device_id", fallback_device_id)),
		"valid": bool(device.get("valid", true)),
		"confidence": 1.0,
		"transform": _transform_to_json(transform),
	}


func _controller_wrist_transform(device: Dictionary):
	if device.has("wrist_transform"):
		return device["wrist_transform"]
	if device.has("grip_transform"):
		return device["grip_transform"]
	return device.get("transform", Transform3D())


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

	if str(frame.get("skeleton_profile", "")) != "sparseposer_xsens":
		_add_visual_helper_joints(joints)
	return {
		"timestamp_seconds": frame["timestamp_seconds"],
		"root": {"position": Vector3.ZERO, "rotation": frame["root"]["rotation"]},
		"joints": joints,
	}


func _debug_bones_for_estimate(frame: Dictionary) -> Array:
	var debug_bones = frame.get("debug_bones", [])
	if debug_bones is Array and not debug_bones.is_empty():
		return debug_bones
	return DEFAULT_BONES


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


func _position_first_person_debug(estimated_frame: Dictionary) -> void:
	if _first_person_armature == null:
		return
	_first_person_armature.position = _vector3_from_transform(estimated_frame["root"])
	_first_person_armature.rotation = Vector3.ZERO


func _position_live_debug(devices: Array, estimated_frame: Dictionary) -> void:
	var head_tracker = _current_head_tracker(devices)
	var head_position = _source_position(head_tracker)
	var head_rotation = _quaternion_from_tracker(head_tracker)
	var forward = _rotate_vector(head_rotation, GODOT_VIEW_FORWARD)
	forward = Vector3(forward.x, 0.0, forward.z)
	if forward.length() <= 0.0001:
		forward = GODOT_VIEW_FORWARD
	forward = forward.normalized()
	var mirror_origin = head_position + forward * mirror_distance_m
	mirror_origin.y = _vector3_from_transform(estimated_frame["root"]).y
	_armature.position = mirror_origin
	_armature.rotation = Vector3(0.0, _mirror_yaw_for_forward(forward), 0.0)
	_status.position = mirror_origin + Vector3(0.0, 1.2 + mirror_height_offset_m, 0.0)


func _position_status_display(devices: Array) -> void:
	var head_tracker = _current_head_tracker(devices)
	var head_position = _source_position(head_tracker)
	var head_rotation = _quaternion_from_tracker(head_tracker)
	var forward = _horizontal(_rotate_vector(head_rotation, GODOT_VIEW_FORWARD))
	if forward.length() <= 0.0001:
		forward = GODOT_VIEW_FORWARD
	forward = forward.normalized()
	var status_origin = head_position + forward * mirror_distance_m
	status_origin.y += mirror_height_offset_m + 1.2
	_status.position = status_origin


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
	_update_playspace_forward_indicator()
	var gaze_hits_target = _gaze_hits_calibration_target(head_tracker)

	if _capture_phase == "idle":
		_update_idle_capture_state(delta, gaze_hits_target, sample, devices)
	elif _capture_phase == "countdown":
		_update_countdown_capture_state(delta, sample, devices, head_tracker)
	elif _capture_phase == "align":
		_update_align_capture_state(_controller_confirm_pressed(), sample)
	elif _capture_phase == "align_countdown":
		_update_align_countdown_capture_state(delta, sample)
	elif _capture_phase == "settle":
		_update_timed_capture_phase(delta, "recording")
	elif _capture_phase == "recording":
		_update_timed_capture_phase(delta, "review" if capture_director_enabled else "complete")
	elif _capture_phase == "review":
		_update_review_capture_state()
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
		if sparseposer_calibration_enabled:
			_requires_trigger_release = _controller_confirm_pressed()


func _update_countdown_capture_state(
	delta: float,
	sample: Dictionary,
	devices: Array,
	head_tracker: Dictionary,
	confirm_pressed = null
) -> void:
	if sparseposer_calibration_enabled:
		if confirm_pressed == null:
			confirm_pressed = _controller_confirm_pressed()
		if _requires_trigger_release:
			if not confirm_pressed:
				_requires_trigger_release = false
			return
		if not confirm_pressed:
			return
	else:
		_phase_remaining_s -= delta
		if _phase_remaining_s > 0.0:
			return
	var calibration_sample = sample.duplicate(true)
	calibration_sample["hmd"] = head_tracker
	_tracker_calibration = _tracker_calibrator.build_calibration(calibration_sample, devices)
	var controller_bindings = _controller_bindings_from_devices(devices, head_tracker)
	if not controller_bindings.is_empty():
		_tracker_calibration["controller_bindings"] = controller_bindings
	_latest_estimated_frame = {}
	_last_estimate_request_timestamp_s = -INF
	_estimate_request_in_flight = false
	if _stream_client.send_message({
		"format": StreamClient.CAPTURE_FORMAT,
		"type": "reset_estimator",
		"session_id": _session_id,
		"request_id": _estimate_request_index,
	}):
		_estimate_request_in_flight = true
	_estimate_request_index += 1
	if sparseposer_calibration_enabled:
		_capture_phase = "align"
		_phase_remaining_s = 0.0
		_alignment_gaze_dwell_s = 0.0
		_requires_gaze_release = false
		_requires_trigger_release = _controller_confirm_pressed()
		_last_local_estimate_status = ""
	else:
		_capture_phase = "settle"
		_phase_remaining_s = post_calibration_settle_seconds
	_clip_index += 1
	_frame_index = 0
	_last_stream_timestamp_s = -INF
	print(
		"Live tracker calibration captured with roles: ",
		_tracker_calibration.get("role_bindings", {})
	)


func _update_align_capture_state(confirm_pressed: bool, sample: Dictionary) -> void:
	if _requires_trigger_release:
		if not confirm_pressed:
			_requires_trigger_release = false
		return

	if confirm_pressed:
		_capture_sparseposer_alignment(sample)


func _update_align_countdown_capture_state(delta: float, sample: Dictionary) -> void:
	_phase_remaining_s -= delta
	if _phase_remaining_s > 0.0:
		return
	_capture_sparseposer_alignment(sample)


func _capture_sparseposer_alignment(sample: Dictionary) -> void:
	var missing_roles = _sparseposer_missing_input_roles(sample)
	if not missing_roles.is_empty():
		_last_local_estimate_status = "missing roles: %s" % ", ".join(missing_roles)
		return
	_tracker_calibration["sparseposer_alignment"] = _sparseposer_alignment_from_sample(sample)
	_latest_estimated_frame = {}
	_last_estimate_request_timestamp_s = -INF
	_estimate_request_in_flight = false
	if _stream_client.send_message({
		"format": StreamClient.CAPTURE_FORMAT,
		"type": "reset_estimator",
		"session_id": _session_id,
		"request_id": _estimate_request_index,
	}):
		_estimate_request_in_flight = true
	_estimate_request_index += 1
	_capture_phase = "settle"
	_phase_remaining_s = post_calibration_settle_seconds
	_frame_index = 0
	_last_stream_timestamp_s = -INF
	_last_local_estimate_status = ""
	print("SparsePoser alignment captured")


func _sparseposer_alignment_from_sample(sample: Dictionary) -> Dictionary:
	var optional: Dictionary = sample.get("optional_trackers", {})
	var role_transforms = {
		"head": sample.get("hmd", {}).get("transform", {}),
		"left_hand": sample.get("left_hand", {}).get("transform", {}),
		"right_hand": sample.get("right_hand", {}).get("transform", {}),
	}
	for role in ["pelvis", "left_foot", "right_foot"]:
		if optional.has(role):
			role_transforms[role] = optional[role].get("transform", {})
	return {
		"schema_version": 1,
		"timestamp_seconds": sample.get("timestamp_seconds", 0.0),
		"role_transforms": role_transforms,
	}


func _update_timed_capture_phase(delta: float, next_phase: String) -> void:
	_phase_remaining_s -= delta
	if _phase_remaining_s > 0.0:
		return
	if next_phase == "recording":
		_capture_phase = "recording"
		_phase_remaining_s = _clip_duration_for_current_run()
		_last_stream_timestamp_s = -INF
	elif next_phase == "complete":
		_capture_phase = "complete"
		_phase_remaining_s = 0.0
		_send_clip_flush()
	elif next_phase == "review":
		_capture_phase = "review"
		_phase_remaining_s = 0.0
		_requires_trigger_release = _controller_confirm_pressed()
		_requires_redo_release = _controller_redo_pressed()
		_send_clip_flush()


func _update_review_capture_state(confirm_pressed = null, redo_pressed = null) -> void:
	if confirm_pressed == null:
		confirm_pressed = _controller_confirm_pressed()
	if redo_pressed == null:
		redo_pressed = _controller_redo_pressed()
	if _requires_trigger_release:
		if not confirm_pressed:
			_requires_trigger_release = false
	if _requires_redo_release:
		if not redo_pressed:
			_requires_redo_release = false

	if not _requires_redo_release and redo_pressed:
		_redo_current_run()
		return
	if not _requires_trigger_release and confirm_pressed:
		_accept_current_run()


func _accept_current_run() -> void:
	_run_index += 1
	_requires_trigger_release = _controller_confirm_pressed()
	_requires_redo_release = _controller_redo_pressed()
	if _run_index >= CAPTURE_PROTOCOL.size():
		_capture_phase = "done"
		_phase_remaining_s = 0.0
	else:
		_capture_phase = "idle"
		_requires_gaze_release = false
		_calibration_dwell_s = 0.0


# Re-run the same director run: back to settle (which flows into recording) with
# a fresh clip index, without advancing the run pointer.
func _redo_current_run() -> void:
	_capture_phase = "settle"
	_phase_remaining_s = post_calibration_settle_seconds
	_clip_index += 1
	_frame_index = 0
	_last_stream_timestamp_s = -INF
	_requires_trigger_release = false
	_requires_redo_release = false


func _current_run() -> Dictionary:
	if _run_index < 0 or _run_index >= CAPTURE_PROTOCOL.size():
		return {}
	return CAPTURE_PROTOCOL[_run_index]


func _current_run_name() -> String:
	if not capture_director_enabled:
		return ""
	return str(_current_run().get("name", ""))


func _current_run_kind() -> String:
	if not capture_director_enabled:
		return ""
	return str(_current_run().get("kind", ""))


func _clip_duration_for_current_run() -> float:
	if capture_director_enabled:
		var run = _current_run()
		if not run.is_empty():
			return float(run.get("duration_s", clip_duration_seconds))
	return clip_duration_seconds


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
			"clip_complete",
			_current_run_name(),
			_current_run_kind()
		)
	):
		_last_flushed_clip_index = _clip_index


func _update_calibration_target_color() -> void:
	var progress = _capture_progress()
	_calibration_target_material.albedo_color = Color(
		0.1 + progress * 0.9,
		0.85 - progress * 0.35,
		0.95 - progress * 0.7,
		0.28
	)


func _update_playspace_forward_indicator() -> void:
	if _playspace_forward_indicator == null:
		return
	_playspace_forward_indicator.visible = sparseposer_calibration_enabled
	_playspace_forward_indicator.position = Vector3(
		_calibration_target.position.x,
		float(grounding.get("floor_height_m", 0.0)) + 0.045,
		_calibration_target.position.z
	)
	_playspace_forward_indicator.rotation = Vector3.ZERO


func _capture_progress() -> float:
	if _capture_phase == "idle":
		return clampf(_calibration_dwell_s / calibration_dwell_seconds, 0.0, 1.0)
	if _capture_phase == "countdown":
		if sparseposer_calibration_enabled:
			return 0.0
		return 1.0 - clampf(_phase_remaining_s / calibration_countdown_seconds, 0.0, 1.0)
	if _capture_phase == "align":
		return 0.0
	if _capture_phase == "align_countdown":
		return 1.0 - clampf(_phase_remaining_s / calibration_countdown_seconds, 0.0, 1.0)
	if _capture_phase == "settle":
		return 1.0 - clampf(_phase_remaining_s / post_calibration_settle_seconds, 0.0, 1.0)
	if _capture_phase == "recording":
		return 1.0
	return 0.0


func _status_text(devices: Array) -> String:
	var recorder = "sidecar connected" if _stream_client.is_stream_connected() else "sidecar offline"
	var tracker_count = _generic_tracker_count(devices)
	var estimator_mode = "calibrated estimate" if use_calibrated_trackers_for_estimation else "3-point estimate"
	if sparseposer_calibration_enabled:
		estimator_mode = "SparsePoser 6-node"
	var estimator_status = "python estimate"
	if _latest_estimated_frame.is_empty():
		estimator_status = "waiting python estimate"
	if not _last_local_estimate_status.is_empty():
		estimator_status = _last_local_estimate_status
	if not _last_estimate_error.is_empty():
		estimator_status = "estimate error: %s" % _last_estimate_error
	var prefix = "%s | live %d targets | %s | %s" % [
		recorder,
		tracker_count,
		estimator_mode,
		estimator_status,
	]
	if _capture_phase == "idle":
		if _requires_gaze_release:
			return "%s | release gaze target" % prefix
		return "%s | gaze target %.1fs" % [prefix, calibration_dwell_seconds - _calibration_dwell_s]
	if _capture_phase == "countdown":
		if sparseposer_calibration_enabled:
			if _requires_trigger_release:
				return "%s | T-pose facing +Z arrow | release trigger" % prefix
			return "%s | T-pose facing +Z arrow | press trigger" % prefix
		return "%s | T-pose in %.1fs" % [prefix, maxf(0.0, _phase_remaining_s)]
	if _capture_phase == "align":
		if _requires_trigger_release:
			return "%s | align skeleton | release trigger" % prefix
		return "%s | align skeleton | press trigger" % prefix
	if _capture_phase == "align_countdown":
		return "%s | confirm alignment %.1fs" % [prefix, maxf(0.0, _phase_remaining_s)]
	if _capture_phase == "settle":
		return "%s | neutral stance %.1fs" % [prefix, maxf(0.0, _phase_remaining_s)]
	if _capture_phase == "recording":
		return "%s | recording clip %d %.1fs" % [
			prefix,
			_clip_index,
			maxf(0.0, _phase_remaining_s),
		]
	return "%s | clip %d complete" % [prefix, _clip_index]


func _update_headlocked_prompt(head_tracker: Dictionary = {}) -> void:
	if _headlocked_prompt_root == null or _headlocked_prompt == null:
		return
	var text = _headlocked_prompt_text()
	_headlocked_prompt_root.visible = not text.is_empty()
	_headlocked_prompt.text = text
	if not text.is_empty() and not head_tracker.is_empty():
		_position_headlocked_prompt(head_tracker)


func _headlocked_prompt_text() -> String:
	if (
		not _last_estimate_error.is_empty()
		and _latest_estimated_frame.is_empty()
		and _capture_phase in ["settle", "recording", "complete"]
	):
		return "estimate error\n%s" % _last_estimate_error.left(42)
	if _capture_phase == "countdown":
		if sparseposer_calibration_enabled:
			return "face +Z arrow\npress trigger"
		if capture_director_enabled:
			return "%s\nT-pose %.1f" % [_current_run().get("instruction", ""), maxf(0.0, _phase_remaining_s)]
		return "T-pose\n%.1f" % maxf(0.0, _phase_remaining_s)
	if _capture_phase == "align":
		return "align skeleton\npress trigger"
	if _capture_phase == "align_countdown":
		return "hold alignment\n%.1f" % maxf(0.0, _phase_remaining_s)
	if _capture_phase == "settle":
		return "neutral stance\n%.1f" % maxf(0.0, _phase_remaining_s)
	if _capture_phase == "recording":
		if capture_director_enabled:
			return "%s\n%.1f" % [_current_run().get("instruction", ""), maxf(0.0, _phase_remaining_s)]
		return "recording\n%.1f" % maxf(0.0, _phase_remaining_s)
	if _capture_phase == "review":
		return "%s done — [trigger] accept · [%s] redo" % [
			_current_run_name(),
			CAPTURE_DIRECTOR_REDO_ACTION,
		]
	if _capture_phase == "done":
		return "capture complete"
	if _capture_phase == "complete":
		return "clip saved"
	return ""


func _position_headlocked_prompt(head_tracker: Dictionary) -> void:
	var head_position = _source_position(head_tracker)
	var head_rotation = _quaternion_from_tracker(head_tracker)
	var basis = Basis(head_rotation)
	var local_offset = Vector3(0.0, 0.22, -1.25)
	_headlocked_prompt_root.transform = Transform3D(
		basis,
		head_position + basis * local_offset
	)


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


func _should_request_estimate(timestamp_seconds: float) -> bool:
	if estimate_sample_rate_hz <= 0.0:
		return true
	var min_interval = 1.0 / estimate_sample_rate_hz
	return (
		_last_estimate_request_timestamp_s == -INF
		or timestamp_seconds - _last_estimate_request_timestamp_s >= min_interval
	)


func _can_send_estimation_message(sample: Dictionary) -> bool:
	if not sparseposer_calibration_enabled:
		_last_local_estimate_status = ""
		return true
	var missing_roles = _sparseposer_missing_input_roles(sample)
	if not missing_roles.is_empty():
		_last_local_estimate_status = "missing roles: %s" % ", ".join(missing_roles)
		if _last_estimate_error.begins_with("missing SparsePoser roles"):
			_last_estimate_error = ""
		return false
	_last_local_estimate_status = ""
	return true


func _sparseposer_missing_input_roles(sample: Dictionary) -> Array:
	var missing = []
	if not sample.has("hmd"):
		missing.append("head")
	if not sample.has("left_hand"):
		missing.append("left_hand")
	if not sample.has("right_hand"):
		missing.append("right_hand")
	var optional: Dictionary = sample.get("optional_trackers", {})
	for role in ["pelvis", "left_foot", "right_foot"]:
		if not optional.has(role):
			missing.append(role)
	return missing


func _controller_confirm_pressed() -> bool:
	return _controller_trigger_pressed(_left_controller) or _controller_trigger_pressed(_right_controller)


func _controller_redo_pressed() -> bool:
	return (
		_controller_button_pressed(_left_controller, CAPTURE_DIRECTOR_REDO_ACTION)
		or _controller_button_pressed(_right_controller, CAPTURE_DIRECTOR_REDO_ACTION)
	)


func _controller_button_pressed(controller: XRController3D, action: String) -> bool:
	if controller == null:
		return false
	return controller.is_button_pressed(action)


func _controller_trigger_pressed(controller: XRController3D) -> bool:
	if controller == null:
		return false
	if controller.get_float("trigger") >= 0.85:
		return true
	for button in ["trigger_click", "trigger", "grip"]:
		if controller.is_button_pressed(button):
			return true
	return false


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
	for debug_marker in estimated_frame.get("debug_markers", []):
		if not (debug_marker is Dictionary):
			continue
		if str(debug_marker.get("kind", "")) != "sparseposer_input":
			continue
		var marker = _tracker_marker("debug:%s" % str(debug_marker.get("label", "")))
		var marker_transform = debug_marker.get("transform", {})
		var relative_position = _vector3_from_transform(marker_transform) - root_position
		relative_position = _display_relative_position(relative_position, display_forward)
		marker.position = _armature.position + _armature.basis * relative_position
		marker.visible = true
		_set_tracker_marker_label(marker, str(debug_marker.get("label", "")))
		_set_tracker_marker_color(marker, Color(0.95, 0.2, 1.0, 1.0))


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
	var controller_bindings: Dictionary = _tracker_calibration.get("controller_bindings", {})
	for role in controller_bindings.keys():
		if str(controller_bindings[role]) == device_id:
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
		if args[index] == "--sparseposer-calibration":
			sparseposer_calibration_enabled = true
			use_calibrated_trackers_for_estimation = true
		if args[index] == "--capture-director":
			capture_director_enabled = true


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
