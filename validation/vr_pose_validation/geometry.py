"""Small transform math helpers for validation scaffolding."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

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


def length(a: Vector3) -> float:
    return sqrt(dot(a, a))


def distance(a: Vector3, b: Vector3) -> float:
    return length(subtract(a, b))


def dot(a: Vector3, b: Vector3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def normalize(a: Vector3, fallback: Vector3 = (0.0, 1.0, 0.0)) -> Vector3:
    magnitude = length(a)
    if magnitude <= 1e-9:
        return fallback
    return scale(a, 1.0 / magnitude)


def with_y(position: Vector3, y: float) -> Vector3:
    return (position[0], y, position[2])
