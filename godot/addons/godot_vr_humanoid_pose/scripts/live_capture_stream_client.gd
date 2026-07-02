extends RefCounted
class_name VRHumanoidLiveCaptureStreamClient

const CAPTURE_FORMAT = "godot_vr_humanoid_pose.live_capture.v1"

var host = "127.0.0.1"
var port = 8765
var reconnect_interval_s = 1.0

var _peer := StreamPeerTCP.new()
var _next_reconnect_s = 0.0
var _receive_buffer = ""


func configure(stream_host: String, stream_port: int) -> void:
    host = stream_host
    port = stream_port


func poll_connection() -> void:
    _peer.poll()
    var status = _peer.get_status()
    if status == StreamPeerTCP.STATUS_CONNECTED or status == StreamPeerTCP.STATUS_CONNECTING:
        return

    var now = Time.get_ticks_msec() / 1000.0
    if now < _next_reconnect_s:
        return
    _next_reconnect_s = now + reconnect_interval_s
    _peer = StreamPeerTCP.new()
    _peer.connect_to_host(host, port)


func is_stream_connected() -> bool:
    _peer.poll()
    return _peer.get_status() == StreamPeerTCP.STATUS_CONNECTED


func send_message(message: Dictionary) -> bool:
    poll_connection()
    if not is_stream_connected():
        return false

    var payload = JSON.stringify(_json_safe(message), "", false) + "\n"
    var result = _peer.put_data(payload.to_utf8_buffer())
    return result == OK


func poll_messages() -> Array:
    poll_connection()
    var messages = []
    if not is_stream_connected():
        _receive_buffer = ""
        return messages

    while _peer.get_available_bytes() > 0:
        var result = _peer.get_partial_data(_peer.get_available_bytes())
        if result[0] != OK:
            break
        _receive_buffer += result[1].get_string_from_utf8()

    while true:
        var separator = _receive_buffer.find("\n")
        if separator < 0:
            break
        var line = _receive_buffer.substr(0, separator).strip_edges()
        _receive_buffer = _receive_buffer.substr(separator + 1)
        if line.is_empty():
            continue
        var parsed = JSON.parse_string(line)
        if parsed is Dictionary:
            messages.append(parsed)
    return messages


func build_session_message(session_id: String, source: String) -> Dictionary:
    return {
        "format": CAPTURE_FORMAT,
        "type": "session",
        "session_id": session_id,
        "source": source,
        "godot_version": Engine.get_version_info(),
    }


func build_flush_message(
    session_id: String,
    clip_index: int,
    frame_index: int,
    reason := "clip_complete"
) -> Dictionary:
    return {
        "format": CAPTURE_FORMAT,
        "type": "flush",
        "session_id": session_id,
        "clip_index": clip_index,
        "frame_index": frame_index,
        "reason": reason,
    }


func build_estimate_request_message(
    session_id: String,
    request_id: int,
    frame_index: int,
    timestamp_seconds: float,
    sample: Dictionary,
    calibration: Dictionary,
    grounding: Dictionary,
    pose_universe: String = "standing",
    clip_index: int = 0,
    devices: Array = []
) -> Dictionary:
    return _json_safe({
        "format": CAPTURE_FORMAT,
        "type": "estimate_request",
        "session_id": session_id,
        "request_id": request_id,
        "frame_index": frame_index,
        "clip_index": clip_index,
        "timestamp_seconds": timestamp_seconds,
        "source": "godot_live_overlay",
        "pose_universe": pose_universe,
        "sample": sample,
        "calibration": calibration,
        "grounding": grounding,
        "devices": devices,
    })


func build_frame_message(
    session_id: String,
    frame_index: int,
    timestamp_seconds: float,
    sample: Dictionary,
    estimated_frame: Dictionary,
    calibration: Dictionary,
    grounding: Dictionary,
    devices: Array,
    pose_universe := "standing",
    raw_to_standing = null,
    clip_index := 0
) -> Dictionary:
    return _json_safe({
        "format": CAPTURE_FORMAT,
        "type": "frame",
        "session_id": session_id,
        "frame_index": frame_index,
        "clip_index": clip_index,
        "timestamp_seconds": timestamp_seconds,
        "source": "godot_live_overlay",
        "pose_universe": pose_universe,
        "raw_to_standing": raw_to_standing,
        "sample": sample,
        "estimated_frame": estimated_frame,
        "calibration": calibration,
        "grounding": grounding,
        "devices": devices,
    })


func _json_safe(value):
    var value_type = typeof(value)
    if value_type == TYPE_VECTOR3:
        return [value.x, value.y, value.z]
    if value_type == TYPE_QUATERNION:
        return [value.x, value.y, value.z, value.w]
    if value_type == TYPE_BASIS:
        return _json_safe(value.get_rotation_quaternion())
    if value_type == TYPE_TRANSFORM3D:
        return _transform_to_json(value)
    if value_type == TYPE_DICTIONARY:
        var converted = {}
        for key in value.keys():
            converted[str(key)] = _json_safe(value[key])
        return converted
    if value_type == TYPE_ARRAY:
        var converted_array = []
        for item in value:
            converted_array.append(_json_safe(item))
        return converted_array
    return value


func _transform_to_json(transform: Transform3D) -> Dictionary:
    var rotation = transform.basis.get_rotation_quaternion()
    return {
        "position": [transform.origin.x, transform.origin.y, transform.origin.z],
        "rotation": [rotation.x, rotation.y, rotation.z, rotation.w],
    }
