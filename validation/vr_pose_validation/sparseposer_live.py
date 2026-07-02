"""Live bridge from Godot tracker samples to the official SparsePoser runtime."""

from __future__ import annotations

import socket
import struct
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from validation.vr_pose_validation.geometry import (
    IDENTITY_QUATERNION,
    Quaternion,
    Transform,
    Vector3,
    add,
    conjugate_quaternion,
    cross,
    distance,
    horizontal,
    look_rotation,
    multiply_quaternion,
    normalize,
    project_on_plane,
    rotate_vector,
    scale,
    subtract,
)
from validation.vr_pose_validation.sparseposer_bvh import (
    SparseposerBvhJoint,
    parse_sparseposer_bvh_text,
    solve_sparseposer_pose_frame,
)

SPARSEPOSER_LIVE_PROFILE = "sparseposer_xsens"
SPARSEPOSER_WINDOW_SIZE = 64
SPARSEPOSER_INPUT_ROLES = (
    "pelvis",
    "left_foot",
    "right_foot",
    "head",
    "left_hand",
    "right_hand",
)
SPARSEPOSER_INPUT_JOINTS = {
    "pelvis": "Hips",
    "left_foot": "LeftToe",
    "right_foot": "RightToe",
    "head": "Head",
    "left_hand": "LeftWrist",
    "right_hand": "RightWrist",
}
SPARSEPOSER_GODOT_FORWARD: Vector3 = (0.0, 0.0, 1.0)
SPARSEPOSER_GODOT_UP: Vector3 = (0.0, 1.0, 0.0)
SPARSEPOSER_XSENS_FORWARD_LOCAL: Vector3 = (0.0, 0.110432, -0.993884)
SPARSEPOSER_XSENS_UP_LOCAL: Vector3 = (0.0, 0.993884, 0.110432)
MIN_CALIBRATED_BODY_SCALE = 0.85
MAX_CALIBRATED_BODY_SCALE = 1.25
DEFAULT_SPARSEPOSER_PYTHON = Path(".cache/references/SparsePoser/python/.venv/Scripts/python.exe")
DEFAULT_SPARSEPOSER_ROOT = Path(".cache/references/SparsePoser/python")
DEFAULT_SPARSEPOSER_MODEL = Path(".cache/references/SparsePoser/python/models/model_xsens")
DEFAULT_SPARSEPOSER_REFERENCE_BVH = Path(
    ".cache/references/SparsePoser/SparsePoserUnity/Assets/Data/male_180_94.txt"
)
type SparsePoserBodyScaleMode = Literal["none", "height", "extent"]


class SparsePoserBridgeError(RuntimeError):
    """Base error for live SparsePoser bridge failures."""


class SparsePoserMissingRolesError(SparsePoserBridgeError):
    def __init__(self, missing_roles: Sequence[str]) -> None:
        self.missing_roles = tuple(missing_roles)
        super().__init__(f"missing SparsePoser roles: {', '.join(self.missing_roles)}")


class SparsePoserEngine(Protocol):
    def infer(self, packet: bytes) -> bytes:
        """Return one frame of 22 local Xsens rotations as little-endian wxyz floats."""

    def close(self) -> None:
        """Release engine resources."""


@dataclass(frozen=True, slots=True)
class SparsePoserBridgeConfig:
    sparseposer_python: Path = DEFAULT_SPARSEPOSER_PYTHON
    sparseposer_root: Path = DEFAULT_SPARSEPOSER_ROOT
    sparseposer_model: Path = DEFAULT_SPARSEPOSER_MODEL
    sparseposer_reference_bvh: Path = DEFAULT_SPARSEPOSER_REFERENCE_BVH
    host: str = "127.0.0.1"
    port: int = 2222
    connect_timeout_seconds: float = 45.0
    apply_endpoint_ik: bool = False
    body_scale_mode: SparsePoserBodyScaleMode = "height"


class SparsePoserSocketEngine:
    """Client for SparsePoser's official ``python/src/unity.py`` binary TCP server."""

    def __init__(self, config: SparsePoserBridgeConfig) -> None:
        self.config = config
        self._process: subprocess.Popen[bytes] | None = None
        self._socket: socket.socket | None = None

    def infer(self, packet: bytes) -> bytes:
        self._ensure_connected()
        assert self._socket is not None
        expected_bytes = 22 * 4 * 4
        try:
            self._socket.sendall(packet)
            return _recv_exact(self._socket, expected_bytes)
        except OSError:
            self._close_socket()
            self._ensure_connected(force_restart=True)
            assert self._socket is not None
            self._socket.sendall(packet)
            return _recv_exact(self._socket, expected_bytes)

    def prewarm(self) -> None:
        self._ensure_connected()

    def close(self) -> None:
        self._close_socket()
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
        self._process = None

    def _ensure_connected(self, *, force_restart: bool = False) -> None:
        if self._socket is not None and not force_restart:
            return
        if force_restart:
            self.close()
        if not self._try_connect(timeout_seconds=0.2):
            self._start_process()
            if not self._try_connect(timeout_seconds=self.config.connect_timeout_seconds):
                raise SparsePoserBridgeError(
                    f"could not connect to SparsePoser runtime on "
                    f"{self.config.host}:{self.config.port}"
                )

    def _start_process(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return
        command = [
            str(self.config.sparseposer_python.resolve()),
            "src/unity.py",
            str(self.config.sparseposer_model.resolve()),
            str(self.config.sparseposer_reference_bvh.resolve()),
            "ik",
        ]
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self._process = subprocess.Popen(
            command,
            cwd=self.config.sparseposer_root.resolve(),
            stdin=subprocess.DEVNULL,
            creationflags=creationflags,
        )

    def _try_connect(self, *, timeout_seconds: float) -> bool:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() <= deadline:
            try:
                self._socket = socket.create_connection(
                    (self.config.host, self.config.port),
                    timeout=1.0,
                )
                self._socket.settimeout(5.0)
                return True
            except OSError:
                time.sleep(0.1)
        return False

    def _close_socket(self) -> None:
        if self._socket is not None:
            try:
                self._socket.close()
            finally:
                self._socket = None


class SparsePoserLiveBackend:
    """Message-level live estimator backend for the existing Godot NDJSON socket."""

    name = SPARSEPOSER_LIVE_PROFILE

    def __init__(
        self,
        config: SparsePoserBridgeConfig,
        *,
        engine: SparsePoserEngine | None = None,
    ) -> None:
        self.config = config
        self.motion = parse_sparseposer_bvh_text(
            config.sparseposer_reference_bvh.read_text(encoding="utf-8")
        )
        self.joint_names = tuple(joint.name for joint in self.motion.joints)
        self.joint_keys = {joint.name: _joint_key(joint.name) for joint in self.motion.joints}
        self.debug_bones = tuple(
            (
                self.joint_keys[self.motion.joints[joint.parent_index].name],
                self.joint_keys[joint.name],
            )
            for joint in self.motion.joints
            if joint.parent_index is not None
        )
        self._rest_pose_bvh = solve_sparseposer_pose_frame(
            self.motion.joints,
            _first_frame_values(
                self.motion.frame_lines[0], channel_count=self.motion.channel_count
            ),
        )
        self._source_rest_to_display_rotation = _source_rest_to_display_rotation(
            self._source_tpose_transform("pelvis")
        )
        self._engine = engine or SparsePoserSocketEngine(config)
        self._window: list[tuple[tuple[Vector3, tuple[float, float, float, float]], ...]] = []
        self._alignment_cache_key = ""
        self._role_offsets: dict[str, Transform] = {}
        self._input_root_align_rotation = IDENTITY_QUATERNION
        self._body_scale = 1.0

    def reset(self) -> None:
        self._window = []
        self._alignment_cache_key = ""
        self._role_offsets = {}
        self._input_root_align_rotation = IDENTITY_QUATERNION
        self._body_scale = 1.0

    def close(self) -> None:
        self._engine.close()

    def prewarm(self) -> None:
        prewarm = getattr(self._engine, "prewarm", None)
        if callable(prewarm):
            prewarm()

    def estimate_from_message(self, message: Mapping[str, Any]) -> dict[str, Any]:
        sample = _mapping_or_empty(message.get("sample"))
        calibration = _mapping_or_empty(message.get("calibration"))
        role_transforms = _required_role_transforms(sample)
        alignment = _sparseposer_alignment_from_calibration(calibration)
        if not alignment:
            return self._tpose_frame(
                target_root=_target_display_root_transform(role_transforms),
                timestamp_seconds=_timestamp_seconds(message, sample),
                calibration_required="sparseposer_alignment",
            )

        self._ensure_role_offsets(alignment)
        target_display_root = _target_display_root_transform(
            role_transforms,
            pelvis_offset=self._role_offsets.get("pelvis"),
        )
        sparse_inputs = {
            role: _apply_sparseposer_role_offset(
                transform,
                self._role_offsets.get(role, Transform()),
                self._input_root_align_rotation,
            )
            for role, transform in role_transforms.items()
        }
        packet = self._append_and_pack_input(
            sparse_inputs,
            input_position_scale=1.0 / self._body_scale,
        )
        sparse_input_targets = {
            role: _apply_sparseposer_target_role_offset(
                transform,
                self._role_offsets.get(role, Transform()),
            )
            for role, transform in role_transforms.items()
        }
        rotations_wxyz = unpack_sparseposer_rotations(
            self._engine.infer(packet),
            joint_count=len(self.motion.joints),
        )
        frame = self._frame_from_rotations(
            rotations_wxyz,
            source_root=sparse_inputs["pelvis"],
            target_root=target_display_root,
            timestamp_seconds=_timestamp_seconds(message, sample),
            sparse_input_targets=sparse_input_targets,
        )
        _add_controller_hand_endpoints(frame, role_transforms)
        frame["debug_markers"] = self._debug_input_markers(sparse_input_targets)
        frame["debug_metadata"] = {
            "body_scale": self._body_scale,
            "pelvis_offset": list(self._role_offsets.get("pelvis", Transform()).position),
            "reference_bvh": str(self.config.sparseposer_reference_bvh),
        }
        return frame

    def _append_and_pack_input(
        self,
        role_transforms: Mapping[str, Transform],
        *,
        input_position_scale: float = 1.0,
    ) -> bytes:
        encoded = tuple(
            _encode_sparse_role_transforms(
                role_transforms,
                position_scale=input_position_scale,
            )
        )
        encoded = _with_quaternion_continuity(
            encoded,
            self._window[-1] if self._window else _identity_sparse_frame(),
        )
        if not self._window:
            self._window = [encoded for _ in range(SPARSEPOSER_WINDOW_SIZE)]
        else:
            self._window = [*self._window[-(SPARSEPOSER_WINDOW_SIZE - 1) :], encoded]
        return pack_sparseposer_window(self._window)

    def _ensure_role_offsets(self, alignment: Mapping[str, Any]) -> None:
        cache_key = str(alignment)
        if cache_key == self._alignment_cache_key:
            return
        alignment_roles = _alignment_role_transforms(alignment)
        missing = [role for role in SPARSEPOSER_INPUT_ROLES if role not in alignment_roles]
        if missing:
            raise SparsePoserMissingRolesError(missing)
        source_root = self._source_tpose_transform("pelvis")
        source_tpose_roles = {
            role: self._source_tpose_transform(role) for role in SPARSEPOSER_INPUT_ROLES
        }
        self._body_scale = _calibrated_body_scale(
            alignment_roles,
            source_tpose_roles,
            mode=self.config.body_scale_mode,
        )
        target_display_root = _target_display_root_transform(
            alignment_roles,
            inferred_position=_infer_target_display_root_position(
                alignment_roles,
                source_root=source_root,
                source_tpose_roles=source_tpose_roles,
                target_root_rotation=look_rotation(_target_body_forward(alignment_roles)),
                body_scale=self._body_scale,
            ),
        )
        self._input_root_align_rotation = _source_root_align_rotation(
            _target_source_alignment_root_rotation(target_display_root),
            source_root.rotation,
        )
        inverse_root_align = conjugate_quaternion(self._input_root_align_rotation)
        source_to_display_rotation = multiply_quaternion(
            target_display_root.rotation,
            self._source_rest_to_display_rotation,
        )
        self._role_offsets = {}
        for role, tracker_transform in alignment_roles.items():
            source_tpose = source_tpose_roles[role]
            source_tpose_relative_position = subtract(source_tpose.position, source_root.position)
            scaled_source_tpose = Transform(
                position=add(
                    source_root.position,
                    scale(source_tpose_relative_position, self._body_scale),
                ),
                rotation=source_tpose.rotation,
            )
            target_joint = _source_pose_to_target_display(
                scaled_source_tpose,
                source_root=source_root,
                target_root=target_display_root,
                source_to_target_rotation=source_to_display_rotation,
            )
            self._role_offsets[role] = Transform(
                position=rotate_vector(
                    conjugate_quaternion(tracker_transform.rotation),
                    subtract(target_joint.position, tracker_transform.position),
                ),
                rotation=multiply_quaternion(
                    conjugate_quaternion(tracker_transform.rotation),
                    multiply_quaternion(inverse_root_align, source_tpose.rotation),
                ),
            )
        self._alignment_cache_key = cache_key

    def _source_tpose_transform(self, role: str) -> Transform:
        joint_name = SPARSEPOSER_INPUT_JOINTS[role]
        rest = self._rest_pose_bvh[joint_name]
        return _sparseposer_transform_to_godot(rest)

    def _tpose_frame(
        self,
        *,
        target_root: Transform,
        timestamp_seconds: float,
        calibration_required: str,
    ) -> dict[str, Any]:
        frame = self._frame_from_local_xyzw_rotations(
            [IDENTITY_QUATERNION for _ in self.motion.joints],
            source_root=target_root,
            target_root=target_root,
            timestamp_seconds=timestamp_seconds,
        )
        frame["calibration_required"] = calibration_required
        return frame

    def _frame_from_rotations(
        self,
        rotations_wxyz: Sequence[tuple[float, float, float, float]],
        *,
        source_root: Transform,
        target_root: Transform,
        timestamp_seconds: float,
        sparse_input_targets: Mapping[str, Transform] | None = None,
    ) -> dict[str, Any]:
        local_xyzw = tuple((x, y, z, w) for w, x, y, z in rotations_wxyz)
        return self._frame_from_local_xyzw_rotations(
            local_xyzw,
            source_root=source_root,
            target_root=target_root,
            timestamp_seconds=timestamp_seconds,
            sparse_input_targets=sparse_input_targets,
        )

    def _frame_from_local_xyzw_rotations(
        self,
        local_rotations: Sequence[Quaternion],
        *,
        source_root: Transform,
        target_root: Transform,
        timestamp_seconds: float,
        sparse_input_targets: Mapping[str, Transform] | None = None,
    ) -> dict[str, Any]:
        root_position_bvh = godot_position_to_sparseposer(source_root.position)
        root_rotation_bvh = _wxyz_to_xyzw(godot_quaternion_to_sparseposer(source_root.rotation))
        effective_rotations = list(local_rotations)
        effective_rotations[0] = root_rotation_bvh
        pose_bvh = _fk_from_local_rotations(
            self.motion.joints,
            effective_rotations,
            root_position=root_position_bvh,
            root_rotation=root_rotation_bvh,
            offset_scale=self._body_scale,
        )
        source_pose_root = _sparseposer_transform_to_godot(pose_bvh["Hips"])
        source_to_target_rotation = conjugate_quaternion(self._input_root_align_rotation)
        mapped_root = _source_world_to_target_display(
            source_pose_root,
            source_to_target_rotation=source_to_target_rotation,
        )
        joints = {
            self.joint_keys[name]: {
                "transform": _transform_to_json(
                    _source_world_to_target_display(
                        _sparseposer_transform_to_godot(transform),
                        source_to_target_rotation=source_to_target_rotation,
                    )
                ),
                "confidence": 1.0,
                "estimated": True,
                "source_joint": name,
            }
            for name, transform in pose_bvh.items()
        }
        if self.config.apply_endpoint_ik and sparse_input_targets is not None:
            _apply_sparse_input_ik(joints, sparse_input_targets)
        return {
            "timestamp_seconds": timestamp_seconds,
            "root": _transform_to_json(mapped_root),
            "skeleton_profile": SPARSEPOSER_LIVE_PROFILE,
            "debug_bones": [list(pair) for pair in self.debug_bones],
            "joints": joints,
        }

    def _debug_input_markers(
        self,
        sparse_input_targets: Mapping[str, Transform],
    ) -> list[dict[str, Any]]:
        return [
            {
                "kind": "sparseposer_input",
                "role": role,
                "label": f"sp {self.joint_keys[SPARSEPOSER_INPUT_JOINTS[role]]}",
                "source_joint": SPARSEPOSER_INPUT_JOINTS[role],
                "transform": _transform_to_json(transform),
            }
            for role, transform in sparse_input_targets.items()
        ]


def pack_sparseposer_window(
    window: Sequence[Sequence[tuple[Vector3, tuple[float, float, float, float]]]],
) -> bytes:
    if len(window) != SPARSEPOSER_WINDOW_SIZE:
        raise ValueError(f"SparsePoser requires exactly {SPARSEPOSER_WINDOW_SIZE} frames")
    values: list[float] = []
    for frame in window:
        if len(frame) != len(SPARSEPOSER_INPUT_ROLES):
            raise ValueError(f"SparsePoser frames require {len(SPARSEPOSER_INPUT_ROLES)} roles")
        for position, rotation_wxyz in frame:
            values.extend(position)
            values.extend(rotation_wxyz)
    return struct.pack(f"<{len(values)}f", *values)


def unpack_sparseposer_rotations(
    payload: bytes,
    *,
    joint_count: int,
) -> tuple[tuple[float, float, float, float], ...]:
    expected = joint_count * 4 * 4
    if len(payload) != expected:
        raise ValueError(f"SparsePoser returned {len(payload)} bytes; expected {expected}")
    values = struct.unpack(f"<{joint_count * 4}f", payload)
    return tuple(
        (values[index], values[index + 1], values[index + 2], values[index + 3])
        for index in range(0, len(values), 4)
    )


def godot_position_to_sparseposer(position: Vector3) -> Vector3:
    return (position[0], position[1], -position[2])


def sparseposer_position_to_godot(position: Vector3) -> Vector3:
    return (position[0], position[1], -position[2])


def source_tracker_position_to_sparseposer_input(position: Vector3) -> Vector3:
    """Encode a live Ret* tracker position with SparsePoser's Unity wire format."""

    return godot_position_to_sparseposer(position)


def godot_quaternion_to_sparseposer(
    rotation_xyzw: Quaternion,
) -> tuple[float, float, float, float]:
    x, y, z, w = rotation_xyzw
    return (w, -x, -y, z)


def sparseposer_quaternion_to_godot(
    rotation_wxyz: tuple[float, float, float, float],
) -> Quaternion:
    w, x, y, z = rotation_wxyz
    return (-x, -y, z, w)


def _encode_sparse_role_transforms(
    role_transforms: Mapping[str, Transform],
    *,
    position_scale: float = 1.0,
) -> list[tuple[Vector3, tuple[float, float, float, float]]]:
    root = role_transforms["pelvis"]
    root_inverse_rotation = conjugate_quaternion(root.rotation)
    encoded = []
    for role in SPARSEPOSER_INPUT_ROLES:
        transform = role_transforms[role]
        if role == "pelvis":
            local_position = scale(transform.position, position_scale)
            local_rotation = transform.rotation
        else:
            local_position = scale(
                rotate_vector(
                    root_inverse_rotation,
                    subtract(transform.position, root.position),
                ),
                position_scale,
            )
            local_rotation = multiply_quaternion(root_inverse_rotation, transform.rotation)
        encoded.append(
            (
                source_tracker_position_to_sparseposer_input(local_position),
                godot_quaternion_to_sparseposer(local_rotation),
            )
        )
    return encoded


def _identity_sparse_frame() -> tuple[tuple[Vector3, tuple[float, float, float, float]], ...]:
    return tuple(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)) for _role in SPARSEPOSER_INPUT_ROLES)


def _with_quaternion_continuity(
    frame: Sequence[tuple[Vector3, tuple[float, float, float, float]]],
    previous_frame: Sequence[tuple[Vector3, tuple[float, float, float, float]]],
) -> tuple[tuple[Vector3, tuple[float, float, float, float]], ...]:
    result = []
    for index, (position, rotation) in enumerate(frame):
        previous_rotation = previous_frame[index][1]
        flipped = (-rotation[0], -rotation[1], -rotation[2], -rotation[3])
        if _quat_dot(previous_rotation, flipped) > _quat_dot(previous_rotation, rotation):
            rotation = flipped
        result.append((position, rotation))
    return tuple(result)


def _quat_dot(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2] + left[3] * right[3]


def _fk_from_local_rotations(
    joints: Sequence[SparseposerBvhJoint],
    local_rotations: Sequence[Quaternion],
    *,
    root_position: Vector3,
    root_rotation: Quaternion,
    offset_scale: float = 1.0,
) -> dict[str, Transform]:
    positions: list[Vector3] = [(0.0, 0.0, 0.0)] * len(joints)
    rotations: list[Quaternion] = [IDENTITY_QUATERNION] * len(joints)
    transforms: dict[str, Transform] = {}
    for joint in joints:
        if joint.parent_index is None:
            position = root_position
            rotation = root_rotation
        else:
            parent_position = positions[joint.parent_index]
            parent_rotation = rotations[joint.parent_index]
            position = add(
                parent_position,
                rotate_vector(parent_rotation, scale(joint.offset, offset_scale)),
            )
            rotation = multiply_quaternion(parent_rotation, local_rotations[joint.index])
        positions[joint.index] = position
        rotations[joint.index] = rotation
        transforms[joint.name] = Transform(position=position, rotation=rotation)
    return transforms


def _required_role_transforms(sample: Mapping[str, Any]) -> dict[str, Transform]:
    optional = _mapping_or_empty(sample.get("optional_trackers"))
    values: dict[str, Transform] = {}
    for role, sample_key in (
        ("head", "hmd"),
        ("left_hand", "left_hand"),
        ("right_hand", "right_hand"),
    ):
        tracker = _mapping_or_empty(sample.get(sample_key))
        if tracker:
            values[role] = _tracker_transform(tracker)
    for role in ("pelvis", "left_foot", "right_foot"):
        tracker = _mapping_or_empty(optional.get(role))
        if tracker:
            values[role] = _tracker_transform(tracker)
    missing = [role for role in SPARSEPOSER_INPUT_ROLES if role not in values]
    if missing:
        raise SparsePoserMissingRolesError(missing)
    return values


def _target_body_forward(role_transforms: Mapping[str, Transform]) -> Vector3:
    return SPARSEPOSER_GODOT_FORWARD


def _target_display_root_transform(
    role_transforms: Mapping[str, Transform],
    *,
    pelvis_offset: Transform | None = None,
    inferred_position: Vector3 | None = None,
) -> Transform:
    root = role_transforms["pelvis"]
    position = root.position
    if inferred_position is not None:
        position = inferred_position
    elif pelvis_offset is not None:
        position = add(root.position, rotate_vector(root.rotation, pelvis_offset.position))
    return Transform(
        position=position,
        rotation=look_rotation(_target_body_forward(role_transforms)),
    )


def _target_source_alignment_root_rotation(target_display_root: Transform) -> Quaternion:
    return target_display_root.rotation


def _infer_target_display_root_position(
    alignment_roles: Mapping[str, Transform],
    *,
    source_root: Transform,
    source_tpose_roles: Mapping[str, Transform],
    target_root_rotation: Quaternion,
    body_scale: float,
) -> Vector3:
    target_root_at_origin = Transform(rotation=target_root_rotation)
    source_to_target_rotation = multiply_quaternion(
        target_root_rotation,
        _source_rest_to_display_rotation(source_root),
    )
    weighted_sum = (0.0, 0.0, 0.0)
    total_weight = 0.0
    for role, weight in (
        ("head", 0.35),
        ("left_foot", 0.25),
        ("right_foot", 0.25),
        ("left_hand", 0.075),
        ("right_hand", 0.075),
    ):
        if role not in alignment_roles:
            continue
        source_tpose = source_tpose_roles[role]
        source_relative = subtract(source_tpose.position, source_root.position)
        scaled_source = Transform(
            position=scale(source_relative, body_scale),
            rotation=source_tpose.rotation,
        )
        target_relative = _source_pose_to_target_display(
            scaled_source,
            source_root=Transform(),
            target_root=target_root_at_origin,
            source_to_target_rotation=source_to_target_rotation,
        ).position
        candidate = subtract(alignment_roles[role].position, target_relative)
        weighted_sum = add(weighted_sum, scale(candidate, weight))
        total_weight += weight
    if total_weight <= 1e-6:
        return alignment_roles["pelvis"].position
    return scale(weighted_sum, 1.0 / total_weight)


def _source_pose_to_target_display(
    transform: Transform,
    *,
    source_root: Transform,
    target_root: Transform,
    source_to_target_rotation: Quaternion,
) -> Transform:
    source_root_relative_position = subtract(transform.position, source_root.position)
    target_forward = normalize(
        horizontal(rotate_vector(target_root.rotation, SPARSEPOSER_GODOT_FORWARD)),
        fallback=SPARSEPOSER_GODOT_FORWARD,
    )
    target_right = normalize(
        cross(SPARSEPOSER_GODOT_UP, target_forward),
        fallback=(1.0, 0.0, 0.0),
    )
    target_relative_position = add(
        add(
            scale(target_right, -source_root_relative_position[0]),
            scale(SPARSEPOSER_GODOT_UP, source_root_relative_position[1]),
        ),
        scale(target_forward, -source_root_relative_position[2]),
    )
    return Transform(
        position=add(target_root.position, target_relative_position),
        rotation=multiply_quaternion(source_to_target_rotation, transform.rotation),
    )


def _source_pose_to_target(
    transform: Transform,
    *,
    source_root: Transform,
    target_root: Transform,
    source_to_target_rotation: Quaternion,
) -> Transform:
    return _source_pose_to_target_display(
        transform,
        source_root=source_root,
        target_root=target_root,
        source_to_target_rotation=source_to_target_rotation,
    )


def _source_world_to_target_display(
    transform: Transform,
    *,
    source_to_target_rotation: Quaternion,
) -> Transform:
    return Transform(
        position=rotate_vector(source_to_target_rotation, transform.position),
        rotation=multiply_quaternion(source_to_target_rotation, transform.rotation),
    )


def _source_rest_to_display_rotation(source_root: Transform) -> Quaternion:
    source_forward = rotate_vector(source_root.rotation, SPARSEPOSER_XSENS_FORWARD_LOCAL)
    source_up = rotate_vector(source_root.rotation, SPARSEPOSER_XSENS_UP_LOCAL)
    source_look = look_rotation(source_forward, source_up)
    target_look = look_rotation(SPARSEPOSER_GODOT_FORWARD, SPARSEPOSER_GODOT_UP)
    # Unity's SparsePoser demo applies a static hips correction when drawing the
    # source Xsens skeleton onto a target avatar. This is the inverse correction:
    # source/Xsens rest space -> default Godot target rest space. Live target yaw
    # is layered on by _target_display_root_transform.
    return conjugate_quaternion(multiply_quaternion(source_look, conjugate_quaternion(target_look)))


def _source_root_align_rotation(
    target_root_rotation: Quaternion,
    source_root_rotation: Quaternion,
) -> Quaternion:
    target_forward = rotate_vector(target_root_rotation, SPARSEPOSER_GODOT_FORWARD)
    target_up = rotate_vector(target_root_rotation, SPARSEPOSER_GODOT_UP)
    source_forward = rotate_vector(source_root_rotation, SPARSEPOSER_XSENS_FORWARD_LOCAL)
    source_up = rotate_vector(source_root_rotation, SPARSEPOSER_XSENS_UP_LOCAL)
    return multiply_quaternion(
        look_rotation(source_forward, source_up),
        conjugate_quaternion(look_rotation(target_forward, target_up)),
    )


def _calibrated_body_scale(
    alignment_roles: Mapping[str, Transform],
    source_tpose_roles: Mapping[str, Transform],
    *,
    mode: SparsePoserBodyScaleMode = "height",
) -> float:
    if mode == "none":
        return 1.0
    if mode == "height":
        scale_ratio = _head_to_floor_scale_ratio(alignment_roles, source_tpose_roles)
        return min(max(scale_ratio, MIN_CALIBRATED_BODY_SCALE), MAX_CALIBRATED_BODY_SCALE)

    ratios = []
    for left_role, right_role in (
        ("left_hand", "right_hand"),
        ("pelvis", "left_hand"),
        ("pelvis", "right_hand"),
        ("pelvis", "head"),
    ):
        target_distance = distance(
            alignment_roles[left_role].position,
            alignment_roles[right_role].position,
        )
        source_distance = distance(
            source_tpose_roles[left_role].position,
            source_tpose_roles[right_role].position,
        )
        if source_distance > 1e-6 and target_distance > 1e-6:
            ratios.append(target_distance / source_distance)
    if not ratios:
        return 1.0
    # Prefer matching the user's widest sparse extent. Head-to-pelvis height is
    # useful, but arm reach dominates visible VR discomfort when the skeleton is
    # too small.
    return min(max(max(ratios), MIN_CALIBRATED_BODY_SCALE), MAX_CALIBRATED_BODY_SCALE)


def _head_to_floor_scale_ratio(
    alignment_roles: Mapping[str, Transform],
    source_tpose_roles: Mapping[str, Transform],
) -> float:
    target_head_y = alignment_roles["head"].position[1]
    target_floor_y = min(
        alignment_roles["left_foot"].position[1],
        alignment_roles["right_foot"].position[1],
    )
    source_head_y = source_tpose_roles["head"].position[1]
    source_floor_y = min(
        source_tpose_roles["left_foot"].position[1],
        source_tpose_roles["right_foot"].position[1],
    )
    target_height = target_head_y - target_floor_y
    source_height = source_head_y - source_floor_y
    if target_height <= 1e-6 or source_height <= 1e-6:
        return 1.0
    return target_height / source_height


def _alignment_role_transforms(alignment: Mapping[str, Any]) -> dict[str, Transform]:
    role_transforms = _mapping_or_empty(alignment.get("role_transforms"))
    return {
        str(role): _transform_from_message(_mapping_or_empty(transform))
        for role, transform in role_transforms.items()
        if isinstance(transform, Mapping) and str(role) in SPARSEPOSER_INPUT_ROLES
    }


def _add_controller_hand_endpoints(
    frame: dict[str, Any],
    role_transforms: Mapping[str, Transform],
) -> None:
    joints = frame.get("joints")
    if not isinstance(joints, dict):
        return
    debug_bones = frame.get("debug_bones")
    if not isinstance(debug_bones, list):
        debug_bones = []
        frame["debug_bones"] = debug_bones

    for side in ("left", "right"):
        role = f"{side}_hand"
        hand = role_transforms.get(role)
        if hand is None:
            continue
        endpoint_name = f"{side}_hand"
        wrist_name = f"{side}_wrist"
        joints[endpoint_name] = {
            "transform": _transform_to_json(hand),
            "confidence": 1.0,
            "estimated": False,
            "source_joint": f"{side}_controller",
        }
        bone = [wrist_name, endpoint_name]
        if wrist_name in joints and bone not in debug_bones:
            debug_bones.append(bone)


def _sparseposer_alignment_from_calibration(calibration: Mapping[str, Any]) -> Mapping[str, Any]:
    alignment = _mapping_or_empty(calibration.get("sparseposer_alignment"))
    if alignment:
        return alignment
    live_tracker_calibration = _mapping_or_empty(calibration.get("live_tracker_calibration"))
    return _mapping_or_empty(live_tracker_calibration.get("sparseposer_alignment"))


def _tracker_transform(tracker: Mapping[str, Any]) -> Transform:
    return _transform_from_message(_mapping_or_empty(tracker.get("transform")))


def _transform_from_message(value: Mapping[str, Any]) -> Transform:
    return Transform(
        position=_vector_from_value(value.get("position")),
        rotation=_quaternion_from_value(value.get("rotation")),
    )


def _sparseposer_transform_to_godot(transform: Transform) -> Transform:
    return Transform(
        position=sparseposer_position_to_godot(transform.position),
        rotation=sparseposer_quaternion_to_godot(_xyzw_to_wxyz(transform.rotation)),
    )


def _apply_sparseposer_role_offset(
    tracker: Transform,
    offset: Transform,
    root_align_rotation: Quaternion,
) -> Transform:
    target_joint = _apply_sparseposer_target_role_offset(tracker, offset)
    return Transform(
        position=rotate_vector(root_align_rotation, target_joint.position),
        rotation=multiply_quaternion(
            root_align_rotation,
            multiply_quaternion(tracker.rotation, offset.rotation),
        ),
    )


def _apply_sparseposer_target_role_offset(
    tracker: Transform,
    offset: Transform,
) -> Transform:
    return Transform(
        position=add(tracker.position, rotate_vector(tracker.rotation, offset.position)),
        rotation=multiply_quaternion(tracker.rotation, offset.rotation),
    )


def _apply_sparse_input_ik(
    joints: dict[str, dict[str, Any]],
    sparse_input_targets: Mapping[str, Transform],
) -> None:
    head = sparse_input_targets.get("head")
    if head is not None:
        _apply_head_endpoint_ik(joints, target=head)

    for side in ("left", "right"):
        hand = sparse_input_targets.get(f"{side}_hand")
        if hand is not None:
            _apply_two_bone_endpoint_ik(
                joints,
                root_name=f"{side}_shoulder",
                mid_name=f"{side}_elbow",
                tip_name=f"{side}_wrist",
                target=hand.position,
            )

        foot = sparse_input_targets.get(f"{side}_foot")
        if foot is not None:
            _apply_leg_toe_endpoint_ik(
                joints,
                root_name=f"{side}_hip",
                mid_name=f"{side}_knee",
                ankle_name=f"{side}_ankle",
                toe_name=f"{side}_toe",
                target=foot.position,
            )


def _apply_head_endpoint_ik(
    joints: dict[str, dict[str, Any]],
    *,
    target: Transform,
) -> None:
    if "head" not in joints:
        return
    original_head = _joint_position(joints, "head")
    if "chest4" in joints and "neck" in joints:
        chest = _joint_position(joints, "chest4")
        neck = _joint_position(joints, "neck")
        chest_to_neck = distance(chest, neck)
        neck_to_head = distance(neck, original_head)
        clamped_target = _clamp_target_to_chain(
            chest,
            target.position,
            chest_to_neck + neck_to_head,
        )
        solved_neck = _solve_two_bone_midpoint(
            root=chest,
            tip=clamped_target,
            upper_length=chest_to_neck,
            lower_length=neck_to_head,
            bend_hint=subtract(neck, chest),
        )
        _set_joint_position(joints, "neck", solved_neck)
    _set_joint_position(joints, "head", target.position)
    _set_joint_rotation(joints, "head", target.rotation)


def _apply_two_bone_endpoint_ik(
    joints: dict[str, dict[str, Any]],
    *,
    root_name: str,
    mid_name: str,
    tip_name: str,
    target: Vector3,
) -> None:
    if root_name not in joints or mid_name not in joints or tip_name not in joints:
        return
    root = _joint_position(joints, root_name)
    mid = _joint_position(joints, mid_name)
    tip = _joint_position(joints, tip_name)
    upper_length = distance(root, mid)
    lower_length = distance(mid, tip)
    clamped_target = _clamp_target_to_chain(root, target, upper_length + lower_length)
    solved_mid = _solve_two_bone_midpoint(
        root=root,
        tip=clamped_target,
        upper_length=upper_length,
        lower_length=lower_length,
        bend_hint=subtract(mid, root),
    )
    _set_joint_position(joints, mid_name, solved_mid)
    _set_joint_position(joints, tip_name, clamped_target)


def _apply_leg_toe_endpoint_ik(
    joints: dict[str, dict[str, Any]],
    *,
    root_name: str,
    mid_name: str,
    ankle_name: str,
    toe_name: str,
    target: Vector3,
) -> None:
    if (
        root_name not in joints
        or mid_name not in joints
        or ankle_name not in joints
        or toe_name not in joints
    ):
        return
    root = _joint_position(joints, root_name)
    mid = _joint_position(joints, mid_name)
    ankle = _joint_position(joints, ankle_name)
    toe = _joint_position(joints, toe_name)
    upper_length = distance(root, mid)
    lower_length = distance(mid, ankle)
    foot_length = distance(ankle, toe)
    lower_chain_length = lower_length + foot_length
    clamped_target = _clamp_target_to_chain(
        root,
        target,
        upper_length + lower_chain_length,
    )
    solved_mid = _solve_two_bone_midpoint(
        root=root,
        tip=clamped_target,
        upper_length=upper_length,
        lower_length=lower_chain_length,
        bend_hint=subtract(mid, root),
    )
    knee_to_toe = normalize(
        subtract(clamped_target, solved_mid),
        fallback=normalize(subtract(toe, mid), fallback=(0.0, -1.0, 0.0)),
    )
    solved_ankle = add(solved_mid, scale(knee_to_toe, lower_length))
    _set_joint_position(joints, mid_name, solved_mid)
    _set_joint_position(joints, ankle_name, solved_ankle)
    _set_joint_position(joints, toe_name, clamped_target)


def _solve_two_bone_midpoint(
    *,
    root: Vector3,
    tip: Vector3,
    upper_length: float,
    lower_length: float,
    bend_hint: Vector3,
) -> Vector3:
    root_to_tip = subtract(tip, root)
    chain_distance = distance(root, tip)
    if chain_distance <= 1e-6 or upper_length <= 1e-6 or lower_length <= 1e-6:
        return add(root, scale(normalize(bend_hint), upper_length))

    direction = scale(root_to_tip, 1.0 / chain_distance)
    along = (
        upper_length * upper_length - lower_length * lower_length + chain_distance * chain_distance
    ) / (2.0 * chain_distance)
    perpendicular_squared = max(upper_length * upper_length - along * along, 0.0)
    perpendicular_length = perpendicular_squared**0.5
    pole = normalize(
        project_on_plane(bend_hint, direction),
        fallback=_fallback_bend_pole(direction),
    )
    return add(add(root, scale(direction, along)), scale(pole, perpendicular_length))


def _fallback_bend_pole(direction: Vector3) -> Vector3:
    horizontal_pole = project_on_plane((0.0, -1.0, 0.0), direction)
    if distance(horizontal_pole, (0.0, 0.0, 0.0)) > 1e-6:
        return normalize(horizontal_pole)
    return normalize(project_on_plane((1.0, 0.0, 0.0), direction), fallback=(1.0, 0.0, 0.0))


def _clamp_target_to_chain(root: Vector3, target: Vector3, max_length: float) -> Vector3:
    root_to_target = subtract(target, root)
    target_distance = distance(root, target)
    if target_distance <= 1e-6:
        return target
    reach = max(max_length - 1e-5, 0.0)
    if target_distance <= reach:
        return target
    return add(root, scale(root_to_target, reach / target_distance))


def _joint_position(joints: Mapping[str, Mapping[str, Any]], name: str) -> Vector3:
    transform = _mapping_or_empty(joints[name].get("transform"))
    return _vector_from_value(transform.get("position"))


def _set_joint_position(joints: dict[str, dict[str, Any]], name: str, position: Vector3) -> None:
    transform = joints[name].get("transform")
    if not isinstance(transform, dict):
        return
    transform["position"] = list(position)


def _set_joint_rotation(joints: dict[str, dict[str, Any]], name: str, rotation: Quaternion) -> None:
    transform = joints[name].get("transform")
    if not isinstance(transform, dict):
        return
    transform["rotation"] = list(rotation)


def _xyzw_to_wxyz(rotation: Quaternion) -> tuple[float, float, float, float]:
    x, y, z, w = rotation
    return (w, x, y, z)


def _wxyz_to_xyzw(rotation: tuple[float, float, float, float]) -> Quaternion:
    w, x, y, z = rotation
    return (x, y, z, w)


def _timestamp_seconds(message: Mapping[str, Any], sample: Mapping[str, Any]) -> float:
    return _float_or_default(
        sample.get("timestamp_seconds"),
        _float_or_default(message.get("timestamp_seconds"), 0.0),
    )


def _first_frame_values(line: str, *, channel_count: int) -> tuple[float, ...]:
    values = tuple(float(value) for value in line.split())
    if len(values) != channel_count:
        raise ValueError(f"SparsePoser T-pose has {len(values)} values; expected {channel_count}")
    return values


def _transform_to_json(transform: Transform) -> dict[str, list[float]]:
    return {
        "position": list(transform.position),
        "rotation": list(transform.rotation),
    }


def _vector_from_value(value: Any) -> Vector3:
    if isinstance(value, list | tuple) and len(value) == 3:
        return (float(value[0]), float(value[1]), float(value[2]))
    return (0.0, 0.0, 0.0)


def _quaternion_from_value(value: Any) -> Quaternion:
    if isinstance(value, list | tuple) and len(value) == 4:
        return (float(value[0]), float(value[1]), float(value[2]), float(value[3]))
    return IDENTITY_QUATERNION


def _float_or_default(value: Any, default: float) -> float:
    if value is None:
        return default
    return float(value)


def _mapping_or_empty(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _joint_key(name: str) -> str:
    chars: list[str] = []
    for index, char in enumerate(name):
        if char.isupper() and index > 0 and not name[index - 1].isupper():
            chars.append("_")
        chars.append(char.lower())
    return "".join(chars)


def _recv_exact(source: socket.socket, byte_count: int) -> bytes:
    chunks: list[bytes] = []
    received = 0
    while received < byte_count:
        chunk = source.recv(byte_count - received)
        if not chunk:
            raise SparsePoserBridgeError("SparsePoser runtime closed the socket")
        chunks.append(chunk)
        received += len(chunk)
    return b"".join(chunks)
