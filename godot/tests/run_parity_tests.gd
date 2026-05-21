extends SceneTree

const BasisEstimator = preload("res://addons/godot_vr_humanoid_pose/scripts/basis_port_estimator.gd")

const FIXTURE_PATH = "res://tests/fixtures/basis_parity_fixture.json"
const POSITION_TOLERANCE = 0.00001
const ROTATION_DOT_TOLERANCE = 0.00001

var _failures: Array[String] = []


func _init() -> void:
    var fixture = _load_fixture(FIXTURE_PATH)
    if fixture.is_empty():
        _failures.append("Fixture is empty: %s" % FIXTURE_PATH)
        _finish()
        return

    var estimator = BasisEstimator.new()
    var samples: Array = fixture["samples"]
    var expected_frames: Array = fixture["expected_frames"]
    if samples.size() != expected_frames.size():
        _failures.append(
            "Sample count %d does not match expected frame count %d"
            % [samples.size(), expected_frames.size()]
        )
        _finish()
        return

    for index in range(samples.size()):
        var actual: Dictionary = estimator.estimate(
            samples[index],
            fixture["calibration"],
            fixture["grounding"]
        )
        _assert_pose_frame(index, actual, expected_frames[index])

    _finish()


func _load_fixture(path: String) -> Dictionary:
    var file = FileAccess.open(path, FileAccess.READ)
    if file == null:
        _failures.append("Unable to open fixture: %s" % path)
        return {}
    var parsed = JSON.parse_string(file.get_as_text())
    if typeof(parsed) != TYPE_DICTIONARY:
        _failures.append("Fixture did not parse as a dictionary: %s" % path)
        return {}
    return parsed


func _assert_pose_frame(index: int, actual: Dictionary, expected: Dictionary) -> void:
    _assert_close(
        "frame %d timestamp" % index,
        actual["timestamp_seconds"],
        expected["timestamp_seconds"],
        POSITION_TOLERANCE
    )
    _assert_transform("frame %d root" % index, actual["root"], expected["root"])

    var actual_joints: Dictionary = actual["joints"]
    var expected_joints: Dictionary = expected["joints"]
    for joint_name in expected_joints.keys():
        if not actual_joints.has(joint_name):
            _failures.append("frame %d missing joint %s" % [index, joint_name])
            continue
        _assert_transform(
            "frame %d joint %s" % [index, joint_name],
            actual_joints[joint_name]["transform"],
            expected_joints[joint_name]["transform"]
        )


func _assert_transform(label: String, actual: Dictionary, expected: Dictionary) -> void:
    var actual_position: Vector3 = actual["position"]
    var expected_position = _vector3_from_json(expected["position"])
    var position_error = actual_position.distance_to(expected_position)
    if position_error > POSITION_TOLERANCE:
        _failures.append(
            "%s position error %.9f: actual=%s expected=%s"
            % [label, position_error, actual_position, expected_position]
        )

    var actual_rotation: Quaternion = actual["rotation"]
    var expected_rotation = _quaternion_from_json(expected["rotation"])
    var rotation_dot = absf(
        actual_rotation.x * expected_rotation.x
        + actual_rotation.y * expected_rotation.y
        + actual_rotation.z * expected_rotation.z
        + actual_rotation.w * expected_rotation.w
    )
    if 1.0 - rotation_dot > ROTATION_DOT_TOLERANCE:
        _failures.append(
            "%s rotation dot %.9f: actual=%s expected=%s"
            % [label, rotation_dot, actual_rotation, expected_rotation]
        )


func _assert_close(label: String, actual: float, expected: float, tolerance: float) -> void:
    var error = absf(actual - expected)
    if error > tolerance:
        _failures.append("%s error %.9f: actual=%f expected=%f" % [label, error, actual, expected])


func _finish() -> void:
    if _failures.is_empty():
        print("Godot parity tests passed")
        quit(0)
        return

    for failure in _failures:
        push_error(failure)
    quit(1)


func _vector3_from_json(value: Array) -> Vector3:
    return Vector3(value[0], value[1], value[2])


func _quaternion_from_json(value: Array) -> Quaternion:
    return Quaternion(value[0], value[1], value[2], value[3])

