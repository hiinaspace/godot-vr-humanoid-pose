from __future__ import annotations

from pytest import approx

from validation.vr_pose_validation.cmu import parse_amc_text, parse_asf_text
from validation.vr_pose_validation.cmu_fk import (
    canonical_pose_from_cmu,
    solve_cmu_pose_frame,
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
root 10 20 30 0 0 0
lowerback 4 5 6
"""


def test_solve_cmu_pose_frame_places_child_from_scaled_length() -> None:
    skeleton = parse_asf_text(ASF_TEXT)
    frame = parse_amc_text(AMC_TEXT)[0]

    pose = solve_cmu_pose_frame(skeleton, frame)

    assert pose["root"].position == approx((0.5644444444, 1.1288888889, 1.6933333333))
    assert pose["lowerback"].position[1] > pose["root"].position[1]


def test_solve_cmu_pose_frame_uses_axis_as_dof_basis_not_rest_rotation() -> None:
    skeleton = parse_asf_text("""
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
name upper
direction 0 1 0
length 10
axis 0 0 90 XYZ
dof rx
limits (-45 45)
end
begin
id 2
name lower
direction 1 0 0
length 10
axis 0 0 0 XYZ
end
:hierarchy
begin
root upper
upper lower
end
""")
    frame = parse_amc_text("""
:FULLY-SPECIFIED
:DEGREES
1
root 0 0 0 0 0 0
upper 0
""")[0]

    pose = solve_cmu_pose_frame(skeleton, frame)

    assert pose["lower"].position[0] > pose["upper"].position[0]
    assert pose["lower"].position[1] == approx(pose["upper"].position[1])


def test_solve_cmu_pose_frame_rotates_bone_endpoint_by_its_own_dof() -> None:
    skeleton = parse_asf_text("""
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
name upper
direction 0 1 0
length 10
axis 0 0 0 XYZ
dof rz
limits (-180 180)
end
:hierarchy
begin
root upper
end
""")
    frame = parse_amc_text("""
:FULLY-SPECIFIED
:DEGREES
1
root 0 0 0 0 0 0
upper 90
""")[0]

    pose = solve_cmu_pose_frame(skeleton, frame)

    assert pose["upper"].position[0] < pose["root"].position[0]
    assert pose["upper"].position[1] == approx(pose["root"].position[1])


def test_canonical_pose_from_cmu_uses_available_named_joints() -> None:
    pose = solve_cmu_pose_frame(parse_asf_text(ASF_TEXT), parse_amc_text(AMC_TEXT)[0])
    canonical = canonical_pose_from_cmu({"root": pose["root"]})

    assert set(canonical) == {"hips"}
