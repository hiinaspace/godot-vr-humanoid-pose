from __future__ import annotations

from pytest import approx

from validation.vr_pose_validation.cmu import (
    CMU_SCALE_TO_METERS,
    ordered_joint_rotation,
    parse_amc_text,
    parse_asf_text,
    scaled_root_translation,
)

ASF_TEXT = """
:version 1.10
:units
mass 1.0
length 0.45
angle deg
:root
order TX TY TZ RX RY RZ
axis XYZ
position 0 0 0
orientation 0 0 0
:bonedata
begin
id 1
name lowerback
direction 0 1 0
length 10
axis 0 0 0 XYZ
dof rz rx ry
limits (-45 45) (-30 30) (-20 20)
end
:hierarchy
begin
root lowerback
end
"""


AMC_TEXT = """
:FULLY-SPECIFIED
:DEGREES
1
root 10 20 30 1 2 3
lowerback 4 5 6
2
root 11 21 31 2 3 4
lowerback 7 8 9
"""


def test_parse_asf_preserves_dof_order_and_scale() -> None:
    skeleton = parse_asf_text(ASF_TEXT)

    assert skeleton.root.order == ("TX", "TY", "TZ", "RX", "RY", "RZ")
    assert skeleton.units["length"] == "0.45"
    assert skeleton.bones["lowerback"].dof == ("rz", "rx", "ry")
    assert skeleton.bones["lowerback"].length_m == approx(10 * CMU_SCALE_TO_METERS)
    assert skeleton.hierarchy == {"root": ("lowerback",)}


def test_parse_amc_frames_and_ordered_joint_rotation() -> None:
    skeleton = parse_asf_text(ASF_TEXT)
    frames = parse_amc_text(AMC_TEXT)

    assert [frame.index for frame in frames] == [1, 2]
    assert scaled_root_translation(frames[0], skeleton.root) == approx(
        (
            10 * CMU_SCALE_TO_METERS,
            20 * CMU_SCALE_TO_METERS,
            30 * CMU_SCALE_TO_METERS,
        )
    )
    assert ordered_joint_rotation(frames[0], skeleton, "lowerback") == {
        "rz": 4.0,
        "rx": 5.0,
        "ry": 6.0,
    }
