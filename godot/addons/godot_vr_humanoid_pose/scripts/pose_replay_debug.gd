extends Node3D
class_name VRHumanoidPoseReplayDebug

const DebugArmature = preload("res://addons/godot_vr_humanoid_pose/scripts/debug_armature.gd")
const DEFAULT_FIXTURE_PATH = "res://tests/fixtures/cmu_walk_replay_fixture.json"

@export var fixture_path = DEFAULT_FIXTURE_PATH
@export var playback_fps = 30.0
@export var lane_offset_m = 0.85
@export var view_row_gap_m = 1.9
@export var camera_padding = 1.30
@export var minimum_camera_size_m = 4.4
@export var center_on_pose = true
@export var fixed_window_camera = true

var fixture: Dictionary = {}
var frame_index = 0
var _elapsed = 0.0
var _reference_front_armature
var _estimated_front_armature
var _reference_side_armature
var _estimated_side_armature
var _camera: Camera3D
var _light: DirectionalLight3D
var _fixture_bounds: Dictionary = {}


func _ready() -> void:
    _apply_user_args()
    _setup_scene()
    load_fixture(fixture_path)
    set_frame_index(0)


func _process(delta: float) -> void:
    if fixture.is_empty():
        return
    _elapsed += delta
    var frame_step = 1.0 / maxf(playback_fps, 1.0)
    while _elapsed >= frame_step:
        _elapsed -= frame_step
        set_frame_index(frame_index + 1)


func load_fixture(path: String) -> void:
    var file = FileAccess.open(path, FileAccess.READ)
    if file == null:
        push_error("Unable to open replay fixture: %s" % path)
        fixture = {}
        return
    var parsed = JSON.parse_string(file.get_as_text())
    if typeof(parsed) != TYPE_DICTIONARY:
        push_error("Replay fixture did not parse as a dictionary: %s" % path)
        fixture = {}
        return
    fixture = parsed
    _fixture_bounds = {}


func set_frame_index(index: int) -> void:
    var reference_frames: Array = fixture.get("reference_frames", [])
    var estimated_frames: Array = fixture.get("estimated_frames", [])
    if reference_frames.is_empty() or estimated_frames.is_empty():
        return
    var frame_count = mini(reference_frames.size(), estimated_frames.size())
    frame_index = index % frame_count
    var pairs: Array = fixture.get("bone_pairs", [])
    var reference_frame: Dictionary = reference_frames[frame_index]
    var estimated_frame: Dictionary = estimated_frames[frame_index]
    var pose_origin = _pose_origin(reference_frame, estimated_frame)

    _position_armature(_reference_front_armature, -lane_offset_m, view_row_gap_m * 0.5, 0.0)
    _position_armature(_estimated_front_armature, lane_offset_m, view_row_gap_m * 0.5, 0.0)
    _position_armature(_reference_side_armature, -lane_offset_m, -view_row_gap_m * 0.5, 90.0)
    _position_armature(_estimated_side_armature, lane_offset_m, -view_row_gap_m * 0.5, 90.0)
    _set_pose_origin(_reference_front_armature, pose_origin)
    _set_pose_origin(_estimated_front_armature, pose_origin)
    _set_pose_origin(_reference_side_armature, pose_origin)
    _set_pose_origin(_estimated_side_armature, pose_origin)

    _reference_front_armature.set_pose_frame(reference_frame, pairs, Color(0.15, 0.55, 1.0, 1.0))
    _estimated_front_armature.set_pose_frame(estimated_frame, pairs, Color(1.0, 0.35, 0.18, 1.0))
    _reference_side_armature.set_pose_frame(reference_frame, pairs, Color(0.15, 0.55, 1.0, 1.0))
    _estimated_side_armature.set_pose_frame(estimated_frame, pairs, Color(1.0, 0.35, 0.18, 1.0))
    _update_camera(reference_frame, estimated_frame)


func _setup_scene() -> void:
    if _reference_front_armature == null:
        _reference_front_armature = _new_armature("GroundTruthFrontArmature")
    if _estimated_front_armature == null:
        _estimated_front_armature = _new_armature("EstimatedFrontArmature")
    if _reference_side_armature == null:
        _reference_side_armature = _new_armature("GroundTruthSideArmature")
    if _estimated_side_armature == null:
        _estimated_side_armature = _new_armature("EstimatedSideArmature")
    if _camera == null:
        _camera = Camera3D.new()
        _camera.name = "Camera3D"
        _camera.fov = 38.0
        add_child(_camera)
        _camera.make_current()
    if _light == null:
        _light = DirectionalLight3D.new()
        _light.name = "KeyLight"
        _light.rotation_degrees = Vector3(-45.0, 35.0, 0.0)
        add_child(_light)


func _update_camera(reference_frame: Dictionary, estimated_frame: Dictionary) -> void:
    var bounds = _camera_bounds(reference_frame, estimated_frame)
    var target = (bounds["min"] + bounds["max"]) * 0.5
    var span = bounds["max"] - bounds["min"]
    _camera.projection = Camera3D.PROJECTION_ORTHOGONAL
    var viewport_size = Vector2(1600.0, 1200.0)
    var viewport = get_viewport()
    if viewport != null:
        viewport_size = viewport.get_visible_rect().size
    var aspect = maxf(viewport_size.x / maxf(viewport_size.y, 1.0), 0.1)
    var vertical_size = span.y * camera_padding
    var horizontal_size = span.x / aspect * camera_padding
    _camera.size = maxf(maxf(vertical_size, horizontal_size), minimum_camera_size_m)
    _camera.position = target + Vector3(0.0, 0.0, 10.0)
    _camera.look_at_from_position(_camera.position, target, Vector3.UP)


func _joint_position(frame: Dictionary, joint_name: String) -> Vector3:
    var joints: Dictionary = frame.get("joints", {})
    if not joints.has(joint_name):
        return Vector3.ZERO
    var value = joints[joint_name]["position"]
    if value is Vector3:
        return value
    return Vector3(value[0], value[1], value[2])


func _apply_user_args() -> void:
    var args = OS.get_cmdline_user_args()
    for index in range(args.size()):
        if args[index] == "--fixture" and index + 1 < args.size():
            fixture_path = args[index + 1]


func _new_armature(armature_name: String):
    var armature = DebugArmature.new()
    armature.name = armature_name
    add_child(armature)
    return armature


func _position_armature(armature: Node3D, x: float, y: float, yaw_degrees: float) -> void:
    armature.position = Vector3(x, y, 0.0)
    armature.rotation_degrees = Vector3(0.0, yaw_degrees, 0.0)


func _set_pose_origin(armature: Node3D, pose_origin: Vector3) -> void:
    armature.set("pose_origin", pose_origin)


func _pose_origin(reference_frame: Dictionary, estimated_frame: Dictionary) -> Vector3:
    if not center_on_pose:
        return Vector3.ZERO
    var centers: Array[Vector3] = []
    _append_center_joint(centers, reference_frame, "hips")
    _append_center_joint(centers, estimated_frame, "hips")
    _append_center_joint(centers, reference_frame, "head")
    _append_center_joint(centers, estimated_frame, "head")
    if centers.is_empty():
        return Vector3.ZERO
    var total = Vector3.ZERO
    for center in centers:
        total += center
    var origin = total / float(centers.size())
    origin.y = 0.0
    return origin


func _append_center_joint(centers: Array[Vector3], frame: Dictionary, joint_name: String) -> void:
    var joints: Dictionary = frame.get("joints", {})
    if joints.has(joint_name):
        centers.append(_position_from_transform(joints[joint_name]))


func _pose_bounds(frame: Dictionary, armature: Node3D) -> Dictionary:
    var joints: Dictionary = frame.get("joints", {})
    var min_position = Vector3(INF, INF, INF)
    var max_position = Vector3(-INF, -INF, -INF)
    var pose_origin = _armature_pose_origin(armature)
    for joint in joints.values():
        var position = (
            armature.basis * (_position_from_transform(joint) - pose_origin)
            + armature.position
        )
        min_position = min_position.min(position)
        max_position = max_position.max(position)
    if joints.is_empty():
        min_position = armature.position
        max_position = armature.position
    return {"min": min_position, "max": max_position}


func _armature_pose_origin(armature: Node3D) -> Vector3:
    var value = armature.get("pose_origin")
    if value is Vector3:
        return value
    return Vector3.ZERO


func _camera_bounds(reference_frame: Dictionary, estimated_frame: Dictionary) -> Dictionary:
    if fixed_window_camera:
        if _fixture_bounds.is_empty():
            _fixture_bounds = _compute_fixture_bounds()
        if not _fixture_bounds.is_empty():
            return _fixture_bounds
    return _frame_pair_bounds(reference_frame, estimated_frame)


func _compute_fixture_bounds() -> Dictionary:
    var reference_frames: Array = fixture.get("reference_frames", [])
    var estimated_frames: Array = fixture.get("estimated_frames", [])
    var frame_count = mini(reference_frames.size(), estimated_frames.size())
    if frame_count <= 0:
        return {}
    var bounds = {}
    for index in range(frame_count):
        var frame_bounds = _frame_pair_bounds(reference_frames[index], estimated_frames[index])
        bounds = frame_bounds if bounds.is_empty() else _merge_bounds(bounds, frame_bounds)
    return bounds


func _frame_pair_bounds(reference_frame: Dictionary, estimated_frame: Dictionary) -> Dictionary:
    var pose_origin = _pose_origin(reference_frame, estimated_frame)
    _set_pose_origin(_reference_front_armature, pose_origin)
    _set_pose_origin(_estimated_front_armature, pose_origin)
    _set_pose_origin(_reference_side_armature, pose_origin)
    _set_pose_origin(_estimated_side_armature, pose_origin)
    var bounds = _pose_bounds(reference_frame, _reference_front_armature)
    bounds = _merge_bounds(bounds, _pose_bounds(estimated_frame, _estimated_front_armature))
    bounds = _merge_bounds(bounds, _pose_bounds(reference_frame, _reference_side_armature))
    return _merge_bounds(bounds, _pose_bounds(estimated_frame, _estimated_side_armature))


func _merge_bounds(a: Dictionary, b: Dictionary) -> Dictionary:
    return {
        "min": a["min"].min(b["min"]),
        "max": a["max"].max(b["max"]),
    }


func _position_from_transform(transform: Dictionary) -> Vector3:
    var value = transform["position"]
    if value is Vector3:
        return value
    return Vector3(value[0], value[1], value[2])
