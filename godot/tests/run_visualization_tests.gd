extends SceneTree

const DebugArmature = preload("res://addons/godot_vr_humanoid_pose/scripts/debug_armature.gd")
const ReplayDebug = preload("res://addons/godot_vr_humanoid_pose/scripts/pose_replay_debug.gd")
const FIXTURE_PATH = "res://tests/fixtures/cmu_walk_replay_fixture.json"

var _failures: Array[String] = []


func _init() -> void:
    var fixture = _load_fixture(FIXTURE_PATH)
    if fixture.is_empty():
        _finish()
        return
    _assert_true(fixture.get("reference_frames", []).size() > 0, "reference frames are present")
    _assert_true(fixture.get("estimated_frames", []).size() > 0, "estimated frames are present")
    _assert_true(fixture.get("bone_pairs", []).size() > 0, "bone pairs are present")

    var armature = DebugArmature.new()
    get_root().add_child(armature)
    armature.set_pose_frame(
        fixture["reference_frames"][0],
        fixture["bone_pairs"],
        Color(0.15, 0.55, 1.0, 1.0)
    )
    var mesh_instance = armature.get_node("ArmatureMesh")
    _assert_true(mesh_instance.mesh != null, "debug armature creates a mesh")
    _assert_true(mesh_instance.mesh.get_surface_count() > 0, "debug armature mesh has surfaces")

    var replay = ReplayDebug.new()
    get_root().add_child(replay)
    replay.fixture_path = FIXTURE_PATH
    replay._setup_scene()
    replay.load_fixture(FIXTURE_PATH)
    replay.set_frame_index(1)
    _assert_true(replay.frame_index == 1, "replay advances to requested frame")
    _assert_true(
        replay.has_node("GroundTruthFrontArmature"),
        "replay creates ground-truth front armature"
    )
    _assert_true(
        replay.has_node("EstimatedFrontArmature"),
        "replay creates estimated front armature"
    )
    _assert_true(
        replay.has_node("GroundTruthSideArmature"),
        "replay creates ground-truth side armature"
    )
    _assert_true(
        replay.has_node("EstimatedSideArmature"),
        "replay creates estimated side armature"
    )
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


func _assert_true(value: bool, label: String) -> void:
    if not value:
        _failures.append(label)


func _finish() -> void:
    if _failures.is_empty():
        print("Godot visualization tests passed")
        quit(0)
        return

    for failure in _failures:
        push_error(failure)
    quit(1)
