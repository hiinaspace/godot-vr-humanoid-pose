"""Small transform math helpers for validation scaffolding."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, radians, sin, sqrt

type Vector3 = tuple[float, float, float]
type Quaternion = tuple[float, float, float, float]

ZERO_VECTOR: Vector3 = (0.0, 0.0, 0.0)
IDENTITY_QUATERNION: Quaternion = (0.0, 0.0, 0.0, 1.0)


@dataclass(frozen=True, slots=True)
class Transform:
    """Position and orientation in a named but caller-managed coordinate space."""

    position: Vector3 = ZERO_VECTOR
    rotation: Quaternion = IDENTITY_QUATERNION


def add(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def subtract(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def scale(a: Vector3, scalar: float) -> Vector3:
    return (a[0] * scalar, a[1] * scalar, a[2] * scalar)


def lerp(a: Vector3, b: Vector3, t: float) -> Vector3:
    return (
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    )


def length(a: Vector3) -> float:
    return sqrt(dot(a, a))


def distance(a: Vector3, b: Vector3) -> float:
    return length(subtract(a, b))


def dot(a: Vector3, b: Vector3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a: Vector3, b: Vector3) -> Vector3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def normalize(a: Vector3, fallback: Vector3 = (0.0, 1.0, 0.0)) -> Vector3:
    magnitude = length(a)
    if magnitude <= 1e-9:
        return fallback
    return scale(a, 1.0 / magnitude)


def horizontal(a: Vector3) -> Vector3:
    return (a[0], 0.0, a[2])


def horizontal_distance(a: Vector3, b: Vector3) -> float:
    return length(subtract(horizontal(a), horizontal(b)))


def with_y(position: Vector3, y: float) -> Vector3:
    return (position[0], y, position[2])


def project_on_plane(vector: Vector3, normal: Vector3) -> Vector3:
    normalized_normal = normalize(normal)
    return subtract(vector, scale(normalized_normal, dot(vector, normalized_normal)))


def conjugate_quaternion(q: Quaternion) -> Quaternion:
    return (-q[0], -q[1], -q[2], q[3])


def multiply_quaternion(a: Quaternion, b: Quaternion) -> Quaternion:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def axis_angle_quaternion(axis: Vector3, degrees: float) -> Quaternion:
    normalized_axis = normalize(axis)
    half_angle = radians(degrees) * 0.5
    sine = sin(half_angle)
    return (
        normalized_axis[0] * sine,
        normalized_axis[1] * sine,
        normalized_axis[2] * sine,
        cos(half_angle),
    )


def axis_name_quaternion(axis_name: str, degrees: float) -> Quaternion:
    half_angle = degrees * pi / 360.0
    sine = sin(half_angle)
    cosine = cos(half_angle)
    if axis_name == "x":
        return (sine, 0.0, 0.0, cosine)
    if axis_name == "y":
        return (0.0, sine, 0.0, cosine)
    if axis_name == "z":
        return (0.0, 0.0, sine, cosine)
    return IDENTITY_QUATERNION


def rotate_vector(rotation: Quaternion, vector: Vector3) -> Vector3:
    rotated = multiply_quaternion(
        multiply_quaternion(rotation, (vector[0], vector[1], vector[2], 0.0)),
        conjugate_quaternion(rotation),
    )
    return (rotated[0], rotated[1], rotated[2])


def look_rotation(
    forward: Vector3,
    up: Vector3 = (0.0, 1.0, 0.0),
) -> Quaternion:
    f = normalize(forward, fallback=(0.0, 0.0, 1.0))
    r = normalize(cross(up, f), fallback=(1.0, 0.0, 0.0))
    u = cross(f, r)

    m00, m01, m02 = r[0], u[0], f[0]
    m10, m11, m12 = r[1], u[1], f[1]
    m20, m21, m22 = r[2], u[2], f[2]

    trace = m00 + m11 + m22
    if trace > 0.0:
        s = sqrt(trace + 1.0) * 2.0
        return ((m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s, 0.25 * s)

    if m00 > m11 and m00 > m22:
        s = sqrt(1.0 + m00 - m11 - m22) * 2.0
        return (0.25 * s, (m01 + m10) / s, (m02 + m20) / s, (m21 - m12) / s)
    if m11 > m22:
        s = sqrt(1.0 + m11 - m00 - m22) * 2.0
        return ((m01 + m10) / s, 0.25 * s, (m12 + m21) / s, (m02 - m20) / s)

    s = sqrt(1.0 + m22 - m00 - m11) * 2.0
    return ((m02 + m20) / s, (m12 + m21) / s, 0.25 * s, (m10 - m01) / s)


def euler_quaternion(channels: tuple[str, ...], values: tuple[float, ...]) -> Quaternion:
    result = IDENTITY_QUATERNION
    for channel, value in zip(channels, values, strict=False):
        axis = _axis_for_channel(channel)
        if axis is None:
            continue
        result = multiply_quaternion(result, axis_angle_quaternion(axis, value))
    return result


def _axis_for_channel(channel: str) -> Vector3 | None:
    axis_name = channel.lower()[-1:]
    if axis_name == "x":
        return (1.0, 0.0, 0.0)
    if axis_name == "y":
        return (0.0, 1.0, 0.0)
    if axis_name == "z":
        return (0.0, 0.0, 1.0)
    return None
