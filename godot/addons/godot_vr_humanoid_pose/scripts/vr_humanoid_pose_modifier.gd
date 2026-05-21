@tool
extends SkeletonModifier3D
class_name VRHumanoidPoseModifier

var pose_frame: Dictionary = {}


func set_pose_frame(frame: Dictionary) -> void:
    pose_frame = frame


func _process_modification() -> void:
    if pose_frame.is_empty():
        return
    # Bone mapping lands in the visualization/runtime slice once the pose schema is stable.
    pass
