extends RefCounted
class_name VRHumanoidLiveTrackerCalibration

const WORLD_UP = Vector3(0.0, 1.0, 0.0)
const PLAYSPACE_RIGHT = Vector3(1.0, 0.0, 0.0)
const PLAYSPACE_FORWARD = Vector3(0.0, 0.0, 1.0)


func build_calibration(sample: Dictionary, devices: Array) -> Dictionary:
    var head = sample.get("hmd", {})
    if head.is_empty():
        return {}

    var head_position = _tracker_position(head)
    var head_rotation = _tracker_rotation(head)
    var axes = _body_axes_from_tpose_devices(devices, head_position, head_rotation)
    var right: Vector3 = axes["right"]
    var forward: Vector3 = axes["forward"]

    var generic_trackers = _annotated_generic_trackers(devices, head_position, right, forward)
    var bindings = _infer_role_bindings(generic_trackers)
    var calibration_devices = {}
    for tracker in generic_trackers:
        var device: Dictionary = tracker["device"]
        var device_id = _device_id(device)
        calibration_devices[device_id] = {
            "role": bindings.get(device_id, ""),
            "device_class": str(device.get("device_class", "")),
            "rest_transform": _device_transform_json(device),
            "rest_local_position": [
                float(tracker["lateral_m"]),
                float(tracker["height_m"]),
                float(tracker["forward_m"]),
            ],
        }

    return {
        "schema_version": 1,
        "timestamp_seconds": sample.get("timestamp_seconds", 0.0),
        "head_transform": head.get("transform", {}),
        "body_axes": {
            "right": _vector_to_json(right),
            "forward": _vector_to_json(forward),
            "up": _vector_to_json(WORLD_UP),
        },
        "role_bindings": _bindings_by_role(bindings),
        "devices": calibration_devices,
    }


func _body_axes_from_tpose_devices(
    _devices: Array,
    _head_position: Vector3,
    _head_rotation: Quaternion
) -> Dictionary:
    return {
        "right": PLAYSPACE_RIGHT,
        "forward": PLAYSPACE_FORWARD,
    }


func apply_optional_trackers(devices: Array, calibration: Dictionary) -> Dictionary:
    if calibration.is_empty():
        return {}
    var role_bindings: Dictionary = calibration.get("role_bindings", {})
    if role_bindings.is_empty():
        return {}

    var devices_by_id = {}
    for item in devices:
        if item is Dictionary:
            devices_by_id[_device_id(item)] = item

    var optional_trackers = {}
    for role in role_bindings.keys():
        var device_id = str(role_bindings[role])
        if not devices_by_id.has(device_id):
            continue
        var device: Dictionary = devices_by_id[device_id]
        if not bool(device.get("valid", true)):
            continue
        optional_trackers[str(role)] = {
            "role": str(role),
            "device_id": device_id,
            "valid": true,
            "confidence": 1.0,
            "transform": _device_transform_json(device),
        }
    return optional_trackers


func _annotated_generic_trackers(
    devices: Array,
    head_position: Vector3,
    right: Vector3,
    forward: Vector3
) -> Array:
    var trackers = []
    for item in devices:
        if not (item is Dictionary):
            continue
        var device: Dictionary = item
        if str(device.get("device_class", "")) != "GenericTracker":
            continue
        if not bool(device.get("valid", true)):
            continue
        var position = _device_position(device)
        var relative = position - head_position
        trackers.append({
            "device": device,
            "device_id": _device_id(device),
            "height_m": position.y,
            "lateral_m": relative.dot(right),
            "forward_m": relative.dot(forward),
        })
    return trackers


func _infer_role_bindings(trackers: Array) -> Dictionary:
    var remaining = trackers.duplicate(true)
    var bindings = {}
    if remaining.is_empty():
        return bindings

    remaining.sort_custom(func(a, b): return a["height_m"] < b["height_m"])
    var feet = _take_lowest(remaining, 2)
    _assign_left_right(bindings, feet, "left_foot", "right_foot")

    if remaining.size() >= 2:
        var knees = _take_lowest(remaining, 2)
        _assign_left_right(bindings, knees, "left_knee", "right_knee")

    var central = remaining.filter(func(item): return absf(item["lateral_m"]) <= 0.32)
    central.sort_custom(func(a, b): return a["height_m"] < b["height_m"])
    if not central.is_empty():
        var pelvis = central[0]
        bindings[pelvis["device_id"]] = "pelvis"
        remaining.erase(pelvis)

    central = remaining.filter(func(item): return absf(item["lateral_m"]) <= 0.32)
    central.sort_custom(func(a, b): return a["height_m"] > b["height_m"])
    if not central.is_empty():
        var chest = central[0]
        bindings[chest["device_id"]] = "chest"
        remaining.erase(chest)

    remaining.sort_custom(func(a, b): return absf(a["lateral_m"]) > absf(b["lateral_m"]))
    var elbows = []
    for item in remaining:
        if elbows.size() >= 2:
            break
        elbows.append(item)
    _assign_left_right(bindings, elbows, "left_elbow", "right_elbow")
    return bindings


func _take_lowest(items: Array, count: int) -> Array:
    var taken = []
    while taken.size() < count and not items.is_empty():
        var item = items.pop_front()
        taken.append(item)
    return taken


func _assign_left_right(bindings: Dictionary, items: Array, left_role: String, right_role: String) -> void:
    if items.size() < 2:
        return
    items.sort_custom(func(a, b): return a["lateral_m"] < b["lateral_m"])
    bindings[items[0]["device_id"]] = left_role
    bindings[items[1]["device_id"]] = right_role


func _bindings_by_role(bindings_by_device: Dictionary) -> Dictionary:
    var by_role = {}
    for device_id in bindings_by_device.keys():
        var role = str(bindings_by_device[device_id])
        if role.is_empty():
            continue
        by_role[role] = str(device_id)
    return by_role


func _device_id(device: Dictionary) -> String:
    var serial = str(device.get("serial_number", ""))
    if not serial.is_empty():
        return "openvr:%s" % serial
    return str(device.get("device_id", device.get("device_index", "")))


func _device_position(device: Dictionary) -> Vector3:
    return _position_from_transform(device.get("transform", {}))


func _device_transform_json(device: Dictionary) -> Dictionary:
    var transform = device.get("transform", {})
    return {
        "position": _vector_to_json(_position_from_transform(transform)),
        "rotation": _quaternion_to_json(_rotation_from_transform(transform)),
    }


func _tracker_position(tracker: Dictionary) -> Vector3:
    return _position_from_transform(tracker.get("transform", {}))


func _tracker_rotation(tracker: Dictionary) -> Quaternion:
    return _rotation_from_transform(tracker.get("transform", {}))


func _position_from_transform(transform) -> Vector3:
    if transform is Transform3D:
        return transform.origin
    if transform is Dictionary:
        var value = transform.get("position", [0.0, 0.0, 0.0])
        if value is Vector3:
            return value
        return Vector3(value[0], value[1], value[2])
    return Vector3.ZERO


func _rotation_from_transform(transform) -> Quaternion:
    if transform is Transform3D:
        return transform.basis.get_rotation_quaternion()
    if transform is Dictionary:
        var value = transform.get("rotation", [0.0, 0.0, 0.0, 1.0])
        if value is Quaternion:
            return value
        return Quaternion(value[0], value[1], value[2], value[3])
    return Quaternion(0.0, 0.0, 0.0, 1.0)


func _vector_to_json(vector: Vector3) -> Array:
    return [vector.x, vector.y, vector.z]


func _quaternion_to_json(rotation: Quaternion) -> Array:
    return [rotation.x, rotation.y, rotation.z, rotation.w]
