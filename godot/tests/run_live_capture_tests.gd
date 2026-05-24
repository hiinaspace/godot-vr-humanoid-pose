extends SceneTree

const LiveOverlayDebug = preload("res://addons/godot_vr_humanoid_pose/scripts/live_overlay_debug.gd")
const StreamClient = preload("res://addons/godot_vr_humanoid_pose/scripts/live_capture_stream_client.gd")
const TrackerCalibration = preload("res://addons/godot_vr_humanoid_pose/scripts/live_tracker_calibration.gd")

var _failures: Array[String] = []


func _init() -> void:
	_test_stream_message_is_json_safe()
	_test_stream_estimate_request_is_json_safe()
	_test_tracker_calibration_infers_common_t_pose_roles()
	_test_live_estimator_sample_uses_three_point_input_by_default()
	_test_live_sample_prefers_openvr_devices()
	_test_live_tracker_marker_uses_calibrated_role_and_mirror_space()
	_test_live_debug_mirror_negates_display_x_only()
	_test_live_display_space_tracks_head_yaw()
	_test_live_stream_rate_limit_records_only_during_clip()
	_test_live_stream_flush_message_is_json_safe()
	_test_live_headlocked_prompt_names_capture_phase()
	_test_live_estimate_response_updates_latest_frame()
	_test_live_head_rotation_bridge_maps_estimator_forward_to_godot_view_forward()
	_test_live_debug_frame_flattens_estimator_output()
	_finish()


func _test_stream_message_is_json_safe() -> void:
	var client = StreamClient.new()
	var identity = Quaternion(0.0, 0.0, 0.0, 1.0)
	var transform = {"position": Vector3(0.0, 1.6, 0.0), "rotation": identity}
	var sample = {
		"timestamp_seconds": 1.0,
		"hmd": {"transform": transform},
		"left_hand": {"transform": transform},
		"right_hand": {"transform": transform},
		"optional_trackers": {},
	}
	var estimated = {
		"timestamp_seconds": 1.0,
		"root": transform,
		"joints": {"head": {"transform": transform, "estimated": false}},
	}
	var message = client.build_frame_message(
		"test",
		1,
		1.0,
		sample,
		estimated,
		{},
		{},
		[]
	)

	_assert_eq("message format", message["format"], StreamClient.CAPTURE_FORMAT)
	_assert_eq("clip index default", message["clip_index"], 0)
	_assert_close("vector conversion", message["sample"]["hmd"]["transform"]["position"][1], 1.6)
	_assert_eq("quaternion conversion", message["estimated_frame"]["root"]["rotation"][3], 1.0)


func _test_stream_estimate_request_is_json_safe() -> void:
	var client = StreamClient.new()
	var identity = Quaternion(0.0, 0.0, 0.0, 1.0)
	var transform = {"position": Vector3(0.0, 1.6, 0.0), "rotation": identity}
	var sample = {
		"timestamp_seconds": 1.0,
		"hmd": {"transform": transform},
		"left_hand": {"transform": transform},
		"right_hand": {"transform": transform},
		"optional_trackers": {},
	}
	var message = client.build_estimate_request_message("test", 4, 8, 1.0, sample, {}, {})

	_assert_eq("estimate request format", message["format"], StreamClient.CAPTURE_FORMAT)
	_assert_eq("estimate request type", message["type"], "estimate_request")
	_assert_eq("estimate request id", message["request_id"], 4)
	_assert_close("estimate request vector conversion", message["sample"]["hmd"]["transform"]["position"][1], 1.6)


func _test_live_stream_flush_message_is_json_safe() -> void:
	var client = StreamClient.new()
	var message = client.build_flush_message("test", 4, 120, "clip_complete")

	_assert_eq("flush message format", message["format"], StreamClient.CAPTURE_FORMAT)
	_assert_eq("flush message type", message["type"], "flush")
	_assert_eq("flush clip index", message["clip_index"], 4)
	_assert_eq("flush frame index", message["frame_index"], 120)
	_assert_eq("flush reason", message["reason"], "clip_complete")


func _test_tracker_calibration_infers_common_t_pose_roles() -> void:
	var calibrator = TrackerCalibration.new()
	var sample = {
		"timestamp_seconds": 2.0,
		"hmd": {
			"transform": {
				"position": [0.0, 1.65, 0.0],
				"rotation": [0.0, 0.0, 0.0, 1.0],
			},
		},
	}
	var devices = [
		_generic_device("left_foot", [-0.18, 0.08, -0.05]),
		_generic_device("right_foot", [0.18, 0.08, -0.05]),
		_generic_device("left_knee", [-0.18, 0.48, -0.04]),
		_generic_device("right_knee", [0.18, 0.48, -0.04]),
		_generic_device("pelvis", [0.0, 0.92, 0.0]),
		_generic_device("left_elbow", [-0.58, 1.18, -0.02]),
		_generic_device("right_elbow", [0.58, 1.18, -0.02]),
	]

	var calibration = calibrator.build_calibration(sample, devices)
	var bindings: Dictionary = calibration["role_bindings"]
	var optional = calibrator.apply_optional_trackers(devices, calibration)

	_assert_eq("left foot binding", bindings["left_foot"], "openvr:left_foot")
	_assert_eq("right foot binding", bindings["right_foot"], "openvr:right_foot")
	_assert_eq("pelvis binding", bindings["pelvis"], "openvr:pelvis")
	_assert_true("optional pelvis", optional.has("pelvis"))
	_assert_close("optional left foot x", optional["left_foot"]["transform"]["position"][0], -0.18)


func _test_live_estimator_sample_uses_three_point_input_by_default() -> void:
	var overlay = LiveOverlayDebug.new()
	var sample = {
		"timestamp_seconds": 1.0,
		"hmd": {"transform": {"position": [0.0, 1.6, 0.0], "rotation": [0.0, 0.0, 0.0, 1.0]}},
		"left_hand": {"transform": {"position": [-0.2, 1.2, 0.1], "rotation": [0.0, 0.0, 0.0, 1.0]}},
		"right_hand": {"transform": {"position": [0.2, 1.2, 0.1], "rotation": [0.0, 0.0, 0.0, 1.0]}},
		"optional_trackers": {"left_foot": {"role": "left_foot"}},
	}

	var estimator_sample = overlay._estimator_sample(sample)
	overlay.use_calibrated_trackers_for_estimation = true
	var full_sample = overlay._estimator_sample(sample)

	_assert_true("default strips optional trackers", estimator_sample["optional_trackers"].is_empty())
	_assert_true("capture sample keeps optional trackers", not sample["optional_trackers"].is_empty())
	_assert_true("optional estimator mode preserves trackers", not full_sample["optional_trackers"].is_empty())
	overlay.free()


func _test_live_debug_frame_flattens_estimator_output() -> void:
	var overlay = LiveOverlayDebug.new()
	var identity = Quaternion(0.0, 0.0, 0.0, 1.0)
	var frame = {
		"timestamp_seconds": 1.0,
		"root": {"position": Vector3(0.0, 1.0, 0.0), "rotation": identity},
		"joints": {
			"head": {
				"transform": {"position": Vector3(0.0, 1.6, 0.0), "rotation": identity}
			},
			"left_elbow": {
				"transform": {"position": Vector3(-0.25, 1.25, 0.2), "rotation": identity}
			},
			"left_hand": {
				"transform": {"position": Vector3(-0.35, 1.1, 0.35), "rotation": identity}
			},
		},
	}

	var debug_frame = overlay._debug_frame_from_estimate(frame)

	_assert_true("head forward helper", debug_frame["joints"].has("head_forward"))
	_assert_true("wrist helper", debug_frame["joints"].has("left_wrist"))
	_assert_close("root-relative head height", debug_frame["joints"]["head"]["position"].y, 0.6)
	overlay.free()


func _test_live_estimate_response_updates_latest_frame() -> void:
	var overlay = LiveOverlayDebug.new()
	overlay._session_id = "test-session"
	var identity = [0.0, 0.0, 0.0, 1.0]
	var estimated = {
		"timestamp_seconds": 1.0,
		"root": {"position": [0.0, 1.0, 0.0], "rotation": identity},
		"joints": {
			"head": {
				"transform": {"position": [0.0, 1.6, 0.0], "rotation": identity},
				"estimated": false,
			},
		},
	}

	overlay._handle_stream_message({
		"type": "estimate_response",
		"session_id": "test-session",
		"estimated_frame": estimated,
	})

	_assert_eq("latest estimate timestamp", overlay._latest_estimated_frame["timestamp_seconds"], 1.0)
	_assert_eq("estimate error cleared", overlay._last_estimate_error, "")
	overlay.free()


func _test_live_sample_prefers_openvr_devices() -> void:
	var overlay = LiveOverlayDebug.new()
	var devices = [
		_openvr_device("head", "HMD", [0.0, 1.65, 0.0]),
		_openvr_device("left_hand", "Controller", [-0.3, 1.2, -0.2]),
		_openvr_device("right_hand", "Controller", [0.3, 1.2, -0.2]),
	]

	var sample = overlay._sample_from_devices(3.0, devices)

	_assert_eq("device sample timestamp", sample["timestamp_seconds"], 3.0)
	_assert_close("device sample head height", sample["hmd"]["transform"]["position"][1], 1.65)
	_assert_close("device sample left hand x", sample["left_hand"]["transform"]["position"][0], -0.3)
	overlay.free()


func _test_live_head_rotation_bridge_maps_estimator_forward_to_godot_view_forward() -> void:
	var overlay = LiveOverlayDebug.new()
	var corrected = overlay._estimator_head_transform(Transform3D(Basis(), Vector3.ZERO))
	var forward = corrected.basis.get_rotation_quaternion() * Vector3(0.0, 0.0, 1.0)
	_assert_close("head bridge forward x", forward.x, 0.0)
	_assert_close("head bridge forward z", forward.z, -1.0)
	overlay.free()


func _test_live_tracker_marker_uses_calibrated_role_and_mirror_space() -> void:
	var overlay = LiveOverlayDebug.new()
	overlay._setup_debug_scene()
	overlay._tracker_calibration = {
		"role_bindings": {"left_foot": "openvr:LHR-TEST"},
	}
	overlay._armature.position = Vector3.ZERO
	overlay._armature.rotation = Vector3(0.0, PI, 0.0)
	var estimated_frame = {
		"root": {"position": Vector3.ZERO, "rotation": Quaternion(0.0, 0.0, 0.0, 1.0)}
	}
	var device = _generic_device("LHR-TEST", [0.25, 0.0, -1.0])

	overlay._update_tracker_markers([device], estimated_frame, Vector3(0.0, 0.0, -1.0))

	var marker = overlay._tracker_markers["openvr:LHR-TEST"]
	var label = marker.get_node("Label3D")
	_assert_eq("calibrated marker label", label.text, "left_foot")
	_assert_close("mirrored marker x", marker.position.x, 0.25)
	_assert_close("mirrored marker z", marker.position.z, 1.0)
	overlay.free()


func _test_live_debug_mirror_negates_display_x_only() -> void:
	var overlay = LiveOverlayDebug.new()
	var identity = Quaternion(0.0, 0.0, 0.0, 1.0)
	var frame = {
		"timestamp_seconds": 1.0,
		"root": {"position": Vector3.ZERO, "rotation": identity},
		"joints": {
			"head": {"position": Vector3(0.2, 0.6, -0.1), "rotation": identity},
			"left_hand": {"position": Vector3(-0.4, 0.1, 0.2), "rotation": identity},
		},
	}

	var mirrored = overlay._mirror_display_frame(frame, Vector3(0.0, 0.0, -1.0))

	_assert_close("mirrored head x", mirrored["joints"]["head"]["position"].x, -0.2)
	_assert_close("mirrored hand x", mirrored["joints"]["left_hand"]["position"].x, 0.4)
	_assert_close("original frame unchanged", frame["joints"]["head"]["position"].x, 0.2)
	overlay.free()


func _test_live_display_space_tracks_head_yaw() -> void:
	var overlay = LiveOverlayDebug.new()

	var default_forward = overlay._display_relative_position(
		Vector3(0.0, 0.0, -1.0),
		Vector3(0.0, 0.0, -1.0)
	)
	var turned_forward = overlay._display_relative_position(
		Vector3(1.0, 0.0, 0.0),
		Vector3(1.0, 0.0, 0.0)
	)
	var turned_right = overlay._display_relative_position(
		Vector3(0.0, 0.0, 1.0),
		Vector3(1.0, 0.0, 0.0)
	)

	_assert_close("default forward display z", default_forward.z, -1.0)
	_assert_close("turned forward display z", turned_forward.z, -1.0)
	_assert_close("turned right mirrored display x", turned_right.x, -1.0)
	_assert_close("turned mirror yaw", overlay._mirror_yaw_for_forward(Vector3(1.0, 0.0, 0.0)), PI * 0.5)
	overlay.free()


func _test_live_stream_rate_limit_records_only_during_clip() -> void:
	var overlay = LiveOverlayDebug.new()
	overlay.capture_sample_rate_hz = 30.0
	overlay._capture_phase = "idle"
	_assert_false("idle does not stream", overlay._should_stream_frame(1.0))

	overlay._capture_phase = "recording"
	overlay._last_stream_timestamp_s = -INF
	_assert_true("first recording frame streams", overlay._should_stream_frame(10.0))
	overlay._last_stream_timestamp_s = 10.0
	_assert_false("subsampled frame does not stream", overlay._should_stream_frame(10.01))
	_assert_true("next sample interval streams", overlay._should_stream_frame(10.04))
	overlay.free()


func _test_live_headlocked_prompt_names_capture_phase() -> void:
	var overlay = LiveOverlayDebug.new()

	overlay._capture_phase = "countdown"
	overlay._phase_remaining_s = 2.25
	_assert_true("countdown prompt", overlay._headlocked_prompt_text().begins_with("T-pose"))

	overlay._capture_phase = "recording"
	overlay._phase_remaining_s = 9.5
	_assert_true("recording prompt", overlay._headlocked_prompt_text().begins_with("recording"))

	overlay._capture_phase = "idle"
	_assert_eq("idle prompt hidden", overlay._headlocked_prompt_text(), "")
	overlay.free()


func _assert_false(label: String, value: bool) -> void:
	if value:
		_failures.append("%s expected false" % label)


func _assert_true(label: String, value: bool) -> void:
	if not value:
		_failures.append("%s expected true" % label)


func _assert_eq(label: String, actual, expected) -> void:
	if actual != expected:
		_failures.append("%s actual=%s expected=%s" % [label, actual, expected])


func _generic_device(serial: String, position: Array) -> Dictionary:
	return {
		"device_class": "GenericTracker",
		"serial_number": serial,
		"valid": true,
		"transform": {
			"position": position,
			"rotation": [0.0, 0.0, 0.0, 1.0],
		},
	}


func _openvr_device(role: String, device_class: String, position: Array) -> Dictionary:
	return {
		"role": role,
		"device_class": device_class,
		"device_id": "openvr_%s" % role,
		"valid": true,
		"transform": Transform3D(Basis(), Vector3(position[0], position[1], position[2])),
	}


func _three_point_sample(head_position: Vector3, timestamp_seconds: float) -> Dictionary:
	var identity = [0.0, 0.0, 0.0, 1.0]
	return {
		"timestamp_seconds": timestamp_seconds,
		"hmd": {
			"role": "head",
			"confidence": 1.0,
			"transform": {
				"position": [head_position.x, head_position.y, head_position.z],
				"rotation": identity,
			},
		},
		"left_hand": {
			"role": "left_hand",
			"confidence": 1.0,
			"transform": {
				"position": [head_position.x + 0.3, 1.2, head_position.z + 0.2],
				"rotation": identity,
			},
		},
		"right_hand": {
			"role": "right_hand",
			"confidence": 1.0,
			"transform": {
				"position": [head_position.x - 0.3, 1.2, head_position.z + 0.2],
				"rotation": identity,
			},
		},
		"optional_trackers": {},
	}


func _position_from_transform(transform: Dictionary) -> Vector3:
	var position = transform["position"]
	return Vector3(position[0], position[1], position[2])


func _assert_close(label: String, actual: float, expected: float) -> void:
	if absf(actual - expected) > 0.00001:
		_failures.append("%s actual=%s expected=%s" % [label, actual, expected])


func _finish() -> void:
	if _failures.is_empty():
		print("Godot live capture tests passed")
		quit(0)
		return

	for failure in _failures:
		push_error(failure)
	quit(1)

