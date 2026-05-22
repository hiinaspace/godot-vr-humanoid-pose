@tool
extends Node3D
class_name VRHumanoidDebugArmature

const DEFAULT_BONE_RADIUS = 0.025

var bone_pairs: Array = []
var color: Color = Color(0.2, 0.65, 1.0, 1.0)
var bone_radius = DEFAULT_BONE_RADIUS
var _mesh_instance: MeshInstance3D


func _ready() -> void:
    _ensure_mesh_instance()


func set_pose_frame(frame: Dictionary, pairs: Array, armature_color: Color) -> void:
    bone_pairs = pairs
    color = armature_color
    _ensure_mesh_instance()
    _rebuild_mesh(frame)


func _ensure_mesh_instance() -> void:
    if _mesh_instance != null:
        return
    _mesh_instance = MeshInstance3D.new()
    _mesh_instance.name = "ArmatureMesh"
    var material = StandardMaterial3D.new()
    material.albedo_color = color
    material.roughness = 0.85
    _mesh_instance.material_override = material
    add_child(_mesh_instance)


func _rebuild_mesh(frame: Dictionary) -> void:
    var joints: Dictionary = frame.get("joints", {})
    var mesh = ImmediateMesh.new()
    mesh.surface_begin(Mesh.PRIMITIVE_TRIANGLES)
    for pair in bone_pairs:
        if pair.size() < 2:
            continue
        var start_name = pair[0]
        var end_name = pair[1]
        if not joints.has(start_name) or not joints.has(end_name):
            continue
        _add_octahedral_bone(
            mesh,
            _position_from_transform(joints[start_name]),
            _position_from_transform(joints[end_name]),
            bone_radius
        )
    mesh.surface_end()
    _mesh_instance.mesh = mesh
    var material: StandardMaterial3D = _mesh_instance.material_override
    material.albedo_color = color


func _add_octahedral_bone(mesh: ImmediateMesh, start: Vector3, finish: Vector3, radius: float) -> void:
    var axis = finish - start
    if axis.length() <= 0.000001:
        return
    var direction = axis.normalized()
    var side_a = direction.cross(Vector3.UP)
    if side_a.length() <= 0.000001:
        side_a = direction.cross(Vector3.RIGHT)
    side_a = side_a.normalized()
    var side_b = direction.cross(side_a).normalized()
    var center = (start + finish) * 0.5
    var ring = [
        center + side_a * radius,
        center + side_b * radius,
        center - side_a * radius,
        center - side_b * radius,
    ]

    for index in range(4):
        var next_index = (index + 1) % 4
        _triangle(mesh, start, ring[index], ring[next_index])
        _triangle(mesh, finish, ring[next_index], ring[index])


func _triangle(mesh: ImmediateMesh, a: Vector3, b: Vector3, c: Vector3) -> void:
    var normal = (b - a).cross(c - a)
    if normal.length() > 0.000001:
        mesh.surface_set_normal(normal.normalized())
    mesh.surface_add_vertex(a)
    mesh.surface_add_vertex(b)
    mesh.surface_add_vertex(c)


func _position_from_transform(transform: Dictionary) -> Vector3:
    var value = transform["position"]
    if value is Vector3:
        return value
    return Vector3(value[0], value[1], value[2])
