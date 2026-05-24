from __future__ import annotations

from pytest import approx

from validation.vr_pose_validation.geometry import euler_quaternion, rotate_vector


def test_euler_quaternion_composes_channels_as_fixed_axis_order() -> None:
    rotation = euler_quaternion(("rx", "ry"), (90.0, 90.0))

    assert rotate_vector(rotation, (0.0, 0.0, 1.0)) == approx((0.0, -1.0, 0.0))
    assert rotate_vector(rotation, (0.0, 1.0, 0.0)) == approx((1.0, 0.0, 0.0))
