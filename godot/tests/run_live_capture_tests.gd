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
	_test_live_sample_prefers_controller_wrist_pose()
	_test_live_sparseposer_controller_bindings_use_tpose_side()
	_test_live_controller_bindings_use_head_relative_side()
	_test_live_controller_bindings_prefer_reported_handedness()
	_test_live_tracker_marker_uses_calibrated_role_and_mirror_space()
	_test_live_tracker_marker_shows_sparseposer_input_markers()
	_test_live_first_person_armature_uses_estimated_root_position()
	_test_live_debug_mirror_negates_display_x_only()
	_test_live_display_space_tracks_head_yaw()
	_test_live_stream_rate_limit_records_only_during_clip()
	_test_live_stream_flush_message_is_json_safe()
	_test_live_headlocked_prompt_names_capture_phase()
	_test_live_sparseposer_forward_indicator_shows_playspace_z()
	_test_live_headlocked_prompt_uses_head_tracker_pose()
	_test_live_estimate_response_updates_latest_frame()
	_test_live_sparseposer_missing_roles_gate_estimation()
	_test_live_sparseposer_calibration_uses_raw_head_axes()
	_test_live_head_rotation_bridge_maps_estimator_forward_to_godot_view_forward()
	_test_live_debug_frame_flattens_estimator_output()
	_test_live_debug_bones_use_sparseposer_response_edges()
	_test_live_sparseposer_calibration_flow_reaches_alignment()
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
	var devices = [{"role": "tracking_reference", "transform": transform}]
	var message = client.build_estimate_request_message("test", 4, 8, 1.0, sample, {}, {}, "standing", 0, devices)

	_assert_eq("estimate request format", message["format"], StreamClient.CAPTURE_FORMAT)
	_assert_eq("estimate request type", message["type"], "estimate_request")
	_assert_eq("estimate request id", message["request_id"], 4)
	_assert_eq("estimate request devices", message["devices"].size(), 1)
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
	_assert_close("calibration body right x", calibration["body_axes"]["right"][0], 1.0)
	_assert_close("calibration body forward z", calibration["body_axes"]["forward"][2], 1.0)
	_assert_true("optional pelvis", optional.has("pelvis"))
	_assert_close("optional left foot x", optional["left_foot"]["transform"]["position"][0], -0.18)

	sample["hmd"]["transform"]["rotation"] = [0.70710678, 0.0, 0.0, 0.70710678]
	calibration = calibrator.build_calibration(sample, devices)
	_assert_eq(
		"downward gaze keeps left foot binding",
		calibration["role_bindings"]["left_foot"],
		"openvr:left_foot"
	)
	_assert_close("downward gaze body right x", calibration["body_axes"]["right"][0], 1.0)

	sample["hmd"]["transform"]["rotation"] = [0.0, 1.0, 0.0, 0.0]
	calibration = calibrator.build_calibration(sample, devices)
	_assert_eq(
		"head yaw does not flip left foot binding",
		calibration["role_bindings"]["left_foot"],
		"openvr:left_foot"
	)
	_assert_eq(
		"head yaw does not flip right foot binding",
		calibration["role_bindings"]["right_foot"],
		"openvr:right_foot"
	)


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


func _test_live_debug_bones_use_sparseposer_response_edges() -> void:
	var overlay = LiveOverlayDebug.new()
	var bones = overlay._debug_bones_for_estimate({
		"skeleton_profile": "sparseposer_xsens",
		"debug_bones": [["hips", "left_toe"], ["hips", "right_toe"]],
	})

	_assert_eq("sparseposer debug bone count", bones.size(), 2)
	_assert_eq("sparseposer debug bone target", bones[0][1], "left_toe")
	_assert_true(
		"default bones still available",
		overlay._debug_bones_for_estimate({}).size() > bones.size()
	)
	overlay.free()


func _test_live_sparseposer_calibration_flow_reaches_alignment() -> void:
	var overlay = LiveOverlayDebug.new()
	overlay.sparseposer_calibration_enabled = true
	overlay._session_id = "test-session"
	overlay._capture_phase = "countdown"
	overlay._phase_remaining_s = 0.0
	var sample = _three_point_sample(Vector3(0.0, 1.65, 0.0), 1.0)
	var devices = [
		_generic_device("left_foot", [-0.18, 0.08, -0.05]),
		_generic_device("right_foot", [0.18, 0.08, -0.05]),
		_generic_device("pelvis", [0.0, 0.92, 0.0]),
	]

	overlay._update_countdown_capture_state(1.0, sample, devices, sample["hmd"], false)
	_assert_eq("sparseposer waits for T-pose trigger", overlay._capture_phase, "countdown")
	overlay._update_countdown_capture_state(1.0, sample, devices, sample["hmd"], true)
	sample["optional_trackers"] = overlay._tracker_calibrator.apply_optional_trackers(
		devices,
		overlay._tracker_calibration
	)
	_assert_eq("sparseposer waits for alignment trigger", overlay._capture_phase, "align")
	overlay._update_align_capture_state(true, sample)

	var alignment: Dictionary = overlay._tracker_calibration.get("sparseposer_alignment", {})
	var estimator_calibration: Dictionary = overlay._estimator_calibration()
	var roles: Dictionary = alignment.get("role_transforms", {})
	_assert_eq("sparseposer phase after alignment", overlay._capture_phase, "settle")
	_assert_true("sparseposer estimator calibration", estimator_calibration.has("sparseposer_alignment"))
	_assert_true("sparseposer pelvis alignment", roles.has("pelvis"))
	_assert_true("sparseposer left foot alignment", roles.has("left_foot"))
	_assert_true("sparseposer right foot alignment", roles.has("right_foot"))
	_assert_true("sparseposer head alignment", roles.has("head"))
	_assert_true("sparseposer left hand alignment", roles.has("left_hand"))
	_assert_true("sparseposer right hand alignment", roles.has("right_hand"))
	overlay.free()


func _test_live_estimate_response_updates_latest_frame() -> void:
	var overlay = LiveOverlayDebug.new()
	overlay._session_id = "test-session"
	overlay._estimate_request_in_flight = true
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
	_assert_false("estimate response clears in-flight", overlay._estimate_request_in_flight)
	overlay.free()


func _test_live_sparseposer_missing_roles_gate_estimation() -> void:
	var overlay = LiveOverlayDebug.new()
	overlay.sparseposer_calibration_enabled = true
	overlay._last_estimate_error = "missing SparsePoser roles: pelvis, left_foot, right_foot"
	var sample = _three_point_sample(Vector3(0.0, 1.65, 0.0), 1.0)

	_assert_false("sparseposer blocks missing roles", overlay._can_send_estimation_message(sample))
	_assert_true("sparseposer missing status", overlay._last_local_estimate_status.contains("pelvis"))
	_assert_eq("sparseposer stale missing error cleared", overlay._last_estimate_error, "")

	sample["optional_trackers"] = {
		"pelvis": {"transform": {"position": [0.0, 0.9, 0.0], "rotation": [0.0, 0.0, 0.0, 1.0]}},
		"left_foot": {"transform": {"position": [-0.1, 0.1, 0.0], "rotation": [0.0, 0.0, 0.0, 1.0]}},
		"right_foot": {"transform": {"position": [0.1, 0.1, 0.0], "rotation": [0.0, 0.0, 0.0, 1.0]}},
	}
	_assert_true("sparseposer allows required roles", overlay._can_send_estimation_message(sample))
	_assert_eq("sparseposer local status cleared", overlay._last_local_estimate_status, "")
	overlay.free()


func _test_live_sparseposer_calibration_uses_raw_head_axes() -> void:
	var overlay = LiveOverlayDebug.new()
	overlay.sparseposer_calibration_enabled = true
	overlay._session_id = "test-session"
	overlay._capture_phase = "countdown"
	var sample = _three_point_sample(Vector3(0.0, 1.65, 0.0), 1.0)
	sample["hmd"]["transform"]["rotation"] = [0.0, 1.0, 0.0, 0.0]
	var raw_head_tracker = sample["hmd"].duplicate(true)
	raw_head_tracker["transform"]["rotation"] = [0.0, 0.0, 0.0, 1.0]
	var devices = [
		_generic_device("left_foot", [-0.18, 0.08, -0.05]),
		_generic_device("right_foot", [0.18, 0.08, -0.05]),
		_generic_device("pelvis", [0.0, 0.92, 0.0]),
	]

	overlay._update_countdown_capture_state(1.0, sample, devices, raw_head_tracker, true)

	var bindings: Dictionary = overlay._tracker_calibration.get("role_bindings", {})
	_assert_eq("calibration left foot uses raw axes", bindings["left_foot"], "openvr:left_foot")
	_assert_eq("calibration right foot uses raw axes", bindings["right_foot"], "openvr:right_foot")
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


func _test_live_sample_prefers_controller_wrist_pose() -> void:
	var overlay = LiveOverlayDebug.new()
	var left = _openvr_device("left_hand", "Controller", [-0.3, 1.2, -0.2])
	left["wrist_transform"] = Transform3D(Basis(), Vector3(-0.36, 1.16, -0.18))
	var right = _openvr_device("right_hand", "Controller", [0.3, 1.2, -0.2])
	right["grip_transform"] = Transform3D(Basis(), Vector3(0.34, 1.17, -0.18))
	var devices = [
		_openvr_device("head", "HMD", [0.0, 1.65, 0.0]),
		left,
		right,
	]

	var sample = overlay._sample_from_devices(3.0, devices)

	_assert_close("device sample left wrist x", sample["left_hand"]["transform"]["position"][0], -0.36)
	_assert_close("device sample right grip fallback x", sample["right_hand"]["transform"]["position"][0], 0.34)
	overlay.free()


func _test_live_sparseposer_controller_bindings_use_tpose_side() -> void:
	var overlay = LiveOverlayDebug.new()
	overlay.sparseposer_calibration_enabled = true
	var devices = [
		_openvr_device("head", "HMD", [0.0, 1.65, 0.0]),
		_openvr_device("right_hand", "Controller", [-0.45, 1.2, 0.0]),
		_openvr_device("left_hand", "Controller", [0.45, 1.2, 0.0]),
	]

	var sample = overlay._sample_from_devices(3.0, devices)
	overlay._tracker_calibration = {
		"controller_bindings": overlay._controller_bindings_from_devices(devices),
	}
	var crossed_devices = [
		_openvr_device("head", "HMD", [0.0, 1.65, 0.0]),
		_openvr_device("right_hand", "Controller", [0.25, 1.2, 0.0]),
		_openvr_device("left_hand", "Controller", [-0.25, 1.2, 0.0]),
	]
	var crossed_sample = overlay._sample_from_devices(4.0, crossed_devices)

	_assert_eq("sparseposer left controller by side id", sample["left_hand"]["device_id"], "openvr_right_hand")
	_assert_eq("sparseposer right controller by side id", sample["right_hand"]["device_id"], "openvr_left_hand")
	_assert_close("sparseposer initial left hand side", sample["left_hand"]["transform"]["position"][0], -0.45)
	_assert_eq("sparseposer bound left controller remains stable", crossed_sample["left_hand"]["device_id"], "openvr_right_hand")
	_assert_close("sparseposer bound left follows device", crossed_sample["left_hand"]["transform"]["position"][0], 0.25)
	overlay.free()


func _test_live_controller_bindings_use_head_relative_side() -> void:
	# SparsePoser mode binds by geometry. Facing -Z, the controller at world +X
	# is the user's LEFT hand; world-X sorting would mislabel it (the swap bug).
	var overlay = LiveOverlayDebug.new()
	overlay.sparseposer_calibration_enabled = true
	var head = _openvr_device("head", "HMD", [0.0, 1.65, 0.0])
	head["transform"] = Transform3D(Basis(Vector3.UP, PI), Vector3(0.0, 1.65, 0.0))
	var devices = [
		head,
		_openvr_device("right_hand", "Controller", [0.45, 1.2, 0.0]),
		_openvr_device("left_hand", "Controller", [-0.45, 1.2, 0.0]),
	]

	var bindings = overlay._controller_bindings_from_devices(devices, head)

	_assert_eq("facing -Z: +X controller is user left", bindings["left_hand"], "openvr_right_hand")
	_assert_eq("facing -Z: -X controller is user right", bindings["right_hand"], "openvr_left_hand")
	overlay.free()


func _test_live_controller_bindings_prefer_reported_handedness() -> void:
	# Normal capture trusts self-reported OpenVR handedness even when the two
	# controllers sit on the opposite physical side at calibration.
	var overlay = LiveOverlayDebug.new()
	var head = _openvr_device("head", "HMD", [0.0, 1.65, 0.0])
	var devices = [
		head,
		_openvr_device("left_hand", "Controller", [0.4, 1.2, 0.0]),
		_openvr_device("right_hand", "Controller", [-0.4, 1.2, 0.0]),
	]

	var bindings = overlay._controller_bindings_from_devices(devices, head)

	_assert_eq("reported left handedness wins", bindings["left_hand"], "openvr_left_hand")
	_assert_eq("reported right handedness wins", bindings["right_hand"], "openvr_right_hand")
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


func _test_live_tracker_marker_shows_sparseposer_input_markers() -> void:
	var overlay = LiveOverlayDebug.new()
	overlay._setup_debug_scene()
	overlay._armature.position = Vector3.ZERO
	overlay._armature.rotation = Vector3(0.0, PI, 0.0)
	var estimated_frame = {
		"root": {"position": Vector3.ZERO, "rotation": Quaternion(0.0, 0.0, 0.0, 1.0)},
		"debug_markers": [
			{
				"kind": "sparseposer_input",
				"label": "sp left_hand",
				"transform": {"position": [-0.2, 1.1, 0.0]},
			},
		],
	}

	overlay._update_tracker_markers([], estimated_frame, Vector3(0.0, 0.0, -1.0))

	var marker = overlay._tracker_markers["debug:sp left_hand"]
	var label = marker.get_node("Label3D")
	_assert_true("sparseposer marker visible", marker.visible)
	_assert_eq("sparseposer marker label", label.text, "sp left_hand")
	_assert_close("sparseposer marker mirrored x", marker.position.x, -0.2)
	overlay.free()


func _test_live_first_person_armature_uses_estimated_root_position() -> void:
	var overlay = LiveOverlayDebug.new()
	overlay._setup_debug_scene()
	var estimated_frame = {
		"root": {
			"position": [1.25, 0.92, -2.5],
			"rotation": [0.0, 0.0, 0.0, 1.0],
		},
	}

	overlay._position_first_person_debug(estimated_frame)

	_assert_close("first-person skeleton root x", overlay._first_person_armature.position.x, 1.25)
	_assert_close("first-person skeleton root y", overlay._first_person_armature.position.y, 0.92)
	_assert_close("first-person skeleton root z", overlay._first_person_armature.position.z, -2.5)
	_assert_close("first-person skeleton yaw unchanged", overlay._first_person_armature.rotation.y, 0.0)
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
	overlay.sparseposer_calibration_enabled = true
	_assert_true("sparseposer countdown prompt", overlay._headlocked_prompt_text().contains("press trigger"))
	overlay.sparseposer_calibration_enabled = false

	overlay._capture_phase = "recording"
	overlay._phase_remaining_s = 9.5
	_assert_true("recording prompt", overlay._headlocked_prompt_text().begins_with("recording"))

	overlay._capture_phase = "align"
	_assert_true("align prompt", overlay._headlocked_prompt_text().contains("press trigger"))

	overlay._capture_phase = "idle"
	_assert_eq("idle prompt hidden", overlay._headlocked_prompt_text(), "")
	overlay.free()


func _test_live_sparseposer_forward_indicator_shows_playspace_z() -> void:
	var overlay = LiveOverlayDebug.new()
	overlay.sparseposer_calibration_enabled = true
	overlay._setup_debug_scene()
	overlay._calibration_target.position = Vector3(1.0, 0.02, 2.0)

	overlay._update_playspace_forward_indicator()

	var label = overlay._playspace_forward_indicator.get_node("Label3D")
	var arrow_head = overlay._playspace_forward_indicator.get_node("ArrowHead")
	_assert_true("sparseposer +Z indicator visible", overlay._playspace_forward_indicator.visible)
	_assert_eq("sparseposer +Z label", label.text, "+Z")
	_assert_close("sparseposer +Z indicator x", overlay._playspace_forward_indicator.position.x, 1.0)
	_assert_close("sparseposer +Z indicator z", overlay._playspace_forward_indicator.position.z, 2.0)
	_assert_true("sparseposer +Z arrow points forward", arrow_head.position.z > 0.0)
	overlay.free()


func _test_live_headlocked_prompt_uses_head_tracker_pose() -> void:
	var overlay = LiveOverlayDebug.new()
	get_root().add_child(overlay)
	overlay._setup_xr_nodes()
	overlay._setup_debug_scene()
	overlay._capture_phase = "countdown"
	overlay._phase_remaining_s = 2.0
	var head = {
		"transform": {
			"position": [1.0, 2.0, 3.0],
			"rotation": [0.0, 0.0, 0.0, 1.0],
		},
	}

	overlay._update_headlocked_prompt(head)

	_assert_true("headlocked prompt visible", overlay._headlocked_prompt_root.visible)
	_assert_close("headlocked prompt x", overlay._headlocked_prompt_root.position.x, 1.0)
	_assert_close("headlocked prompt y", overlay._headlocked_prompt_root.position.y, 2.22)
	_assert_close("headlocked prompt z", overlay._headlocked_prompt_root.position.z, 1.75)
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

