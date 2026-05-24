"""Live VR pose capture protocol and Parquet recorder helpers."""

from __future__ import annotations

import json
import socket
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from threading import Event
from typing import Any, Literal
from uuid import uuid4

import polars as pl

CAPTURE_FORMAT = "godot_vr_humanoid_pose.live_capture.v1"
TRANSFORM_COMPONENTS = ("px", "py", "pz", "qx", "qy", "qz", "qw")
RAW_TO_STANDING_COMPONENTS = tuple(f"raw_to_standing_{index:02d}" for index in range(12))
type ParquetCompression = Literal["lz4", "uncompressed", "snappy", "gzip", "brotli", "zstd"]


@dataclass(frozen=True, slots=True)
class LiveCapturePaths:
    session_dir: Path
    session_metadata_path: Path
    frames_path: Path
    devices_path: Path
    joints_path: Path


@dataclass(frozen=True, slots=True)
class LiveCaptureSummary:
    session_id: str
    session_dir: Path
    frames: int
    devices: int
    joints: int


def new_session_id() -> str:
    return time.strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:8]


def live_capture_paths(output_dir: Path, session_id: str) -> LiveCapturePaths:
    session_dir = output_dir / f"session={session_id}"
    return LiveCapturePaths(
        session_dir=session_dir,
        session_metadata_path=session_dir / "session.json",
        frames_path=session_dir / "frames.parquet",
        devices_path=session_dir / "devices.parquet",
        joints_path=session_dir / "joints.parquet",
    )


class LiveCaptureParquetWriter:
    """Collect live NDJSON frame messages and flush them into Parquet tables."""

    def __init__(
        self,
        output_dir: Path,
        *,
        session_id: str | None = None,
        compression: ParquetCompression = "zstd",
    ) -> None:
        self.session_id = session_id or new_session_id()
        self.paths = live_capture_paths(output_dir, self.session_id)
        self.compression = compression
        self._metadata: dict[str, Any] = {
            "format": CAPTURE_FORMAT,
            "session_id": self.session_id,
            "created_unix_seconds": time.time(),
        }
        self._frames: list[dict[str, Any]] = []
        self._devices: list[dict[str, Any]] = []
        self._joints: list[dict[str, Any]] = []

    def append_message(self, message: Mapping[str, Any]) -> None:
        message_type = str(message.get("type", "frame"))
        if message_type == "session":
            self._append_session_metadata(message)
            return
        if message_type == "flush":
            self._append_flush_metadata(message)
            self.flush()
            return
        if message_type != "frame":
            raise ValueError(f"Unsupported live capture message type: {message_type}")
        self.append_frame(message)

    def append_frame(self, message: Mapping[str, Any]) -> None:
        if message.get("format") not in (None, CAPTURE_FORMAT):
            raise ValueError(f"Unsupported live capture format: {message.get('format')}")

        frame_index = _int_or_default(message.get("frame_index"), len(self._frames))
        clip_index = _int_or_default(message.get("clip_index"), 0)
        session_id = str(message.get("session_id") or self.session_id)
        timestamp_seconds = _float_or_default(message.get("timestamp_seconds"), 0.0)
        received_monotonic_ns = time.monotonic_ns()
        raw_to_standing = _float_sequence_or_none(message.get("raw_to_standing"), 12)
        estimated_frame = _mapping_or_empty(message.get("estimated_frame"))
        root = _mapping_or_empty(estimated_frame.get("root"))

        frame_row: dict[str, Any] = {
            "capture_format": CAPTURE_FORMAT,
            "session_id": session_id,
            "clip_index": clip_index,
            "frame_index": frame_index,
            "timestamp_seconds": timestamp_seconds,
            "received_monotonic_ns": received_monotonic_ns,
            "source": str(message.get("source", "")),
            "pose_universe": str(message.get("pose_universe", "standing")),
            "calibration_json": _stable_json(message.get("calibration", {})),
            "grounding_json": _stable_json(message.get("grounding", {})),
        }
        frame_row.update(_transform_columns("root", root))
        frame_row.update(
            {
                component: raw_to_standing[index] if raw_to_standing is not None else None
                for index, component in enumerate(RAW_TO_STANDING_COMPONENTS)
            }
        )
        self._frames.append(frame_row)

        for device_row in self._device_rows(
            message,
            session_id,
            clip_index,
            frame_index,
            timestamp_seconds,
        ):
            self._devices.append(device_row)
        for joint_row in self._joint_rows(
            estimated_frame,
            session_id,
            clip_index,
            frame_index,
            timestamp_seconds,
        ):
            self._joints.append(joint_row)

    def flush(self) -> LiveCaptureSummary:
        self.paths.session_dir.mkdir(parents=True, exist_ok=True)
        self.paths.session_metadata_path.write_bytes(
            (json.dumps(self._metadata, indent=2, sort_keys=True) + "\n").encode("utf-8")
        )
        _write_parquet(self.paths.frames_path, self._frames, compression=self.compression)
        _write_parquet(self.paths.devices_path, self._devices, compression=self.compression)
        _write_parquet(self.paths.joints_path, self._joints, compression=self.compression)
        return LiveCaptureSummary(
            session_id=self.session_id,
            session_dir=self.paths.session_dir,
            frames=len(self._frames),
            devices=len(self._devices),
            joints=len(self._joints),
        )

    def _append_session_metadata(self, message: Mapping[str, Any]) -> None:
        for key, value in message.items():
            if key == "type":
                continue
            self._metadata[str(key)] = value

    def _append_flush_metadata(self, message: Mapping[str, Any]) -> None:
        flushes = self._metadata.setdefault("flushes", [])
        if not isinstance(flushes, list):
            flushes = []
            self._metadata["flushes"] = flushes
        flushes.append(
            {
                "received_unix_seconds": time.time(),
                "session_id": str(message.get("session_id", self.session_id)),
                "clip_index": _int_or_default(message.get("clip_index"), 0),
                "frame_index": _int_or_default(message.get("frame_index"), len(self._frames)),
                "reason": str(message.get("reason", "")),
            }
        )

    def _device_rows(
        self,
        message: Mapping[str, Any],
        session_id: str,
        clip_index: int,
        frame_index: int,
        timestamp_seconds: float,
    ) -> Iterable[dict[str, Any]]:
        sample = _mapping_or_empty(message.get("sample"))
        devices = message.get("devices")
        if isinstance(devices, list):
            source_devices = [device for device in devices if isinstance(device, Mapping)]
            source_devices.extend(_calibrated_devices_from_sample(sample))
        else:
            source_devices = _devices_from_sample(sample)

        for device_order, device in enumerate(source_devices):
            if not isinstance(device, Mapping):
                continue
            transform = _mapping_or_empty(device.get("transform"))
            row: dict[str, Any] = {
                "session_id": session_id,
                "clip_index": clip_index,
                "frame_index": frame_index,
                "timestamp_seconds": timestamp_seconds,
                "device_order": device_order,
                "role": str(device.get("role", "")),
                "device_id": str(device.get("device_id", "")),
                "device_class": str(device.get("device_class", "")),
                "serial_hash": _serial_hash(device),
                "valid": bool(device.get("valid", True)),
                "confidence": _float_or_default(device.get("confidence"), 1.0),
            }
            row.update(_transform_columns("device", transform))
            row.update(_vector_columns("linear_velocity", device.get("linear_velocity")))
            row.update(_vector_columns("angular_velocity", device.get("angular_velocity")))
            yield row

    def _joint_rows(
        self,
        estimated_frame: Mapping[str, Any],
        session_id: str,
        clip_index: int,
        frame_index: int,
        timestamp_seconds: float,
    ) -> Iterable[dict[str, Any]]:
        joints = estimated_frame.get("joints")
        if not isinstance(joints, Mapping):
            return

        for joint_name, joint in joints.items():
            if not isinstance(joint, Mapping):
                continue
            transform = _mapping_or_empty(joint.get("transform", joint))
            row: dict[str, Any] = {
                "session_id": session_id,
                "clip_index": clip_index,
                "frame_index": frame_index,
                "timestamp_seconds": timestamp_seconds,
                "joint": str(joint_name),
                "estimated": bool(joint.get("estimated", True)),
                "confidence": _float_or_default(joint.get("confidence"), 1.0),
            }
            row.update(_transform_columns("joint", transform))
            yield row


def record_ndjson_lines(
    lines: Iterable[bytes | str],
    writer: LiveCaptureParquetWriter,
) -> LiveCaptureSummary:
    for raw_line in lines:
        line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
        line = line.strip()
        if not line:
            continue
        parsed = json.loads(line)
        if not isinstance(parsed, Mapping):
            raise ValueError("Live capture NDJSON records must be JSON objects")
        writer.append_message(parsed)
    return writer.flush()


def run_tcp_recorder(
    *,
    host: str,
    port: int,
    writer: LiveCaptureParquetWriter,
    stop_after_client: bool = False,
    stop_event: Event | None = None,
    accept_timeout_seconds: float = 0.5,
) -> LiveCaptureSummary:
    with socket.create_server((host, port), reuse_port=False) as server:
        server.settimeout(accept_timeout_seconds)
        while True:
            if stop_event is not None and stop_event.is_set():
                return writer.flush()
            try:
                client, _address = server.accept()
            except TimeoutError:
                continue
            with client:
                client.settimeout(accept_timeout_seconds)
                buffer = b""
                try:
                    while True:
                        if stop_event is not None and stop_event.is_set():
                            return writer.flush()
                        try:
                            chunk = client.recv(65536)
                        except TimeoutError:
                            continue
                        if not chunk:
                            break
                        buffer += chunk
                        while b"\n" in buffer:
                            raw_line, buffer = buffer.split(b"\n", 1)
                            _append_socket_line(raw_line, writer)
                except (ConnectionResetError, BrokenPipeError):
                    pass
                finally:
                    if buffer.strip():
                        _append_socket_line(buffer, writer)
                    writer.flush()
            if stop_after_client:
                return writer.flush()


def _append_socket_line(raw_line: bytes, writer: LiveCaptureParquetWriter) -> None:
    line = raw_line.strip()
    if not line:
        return
    parsed = json.loads(line.decode("utf-8"))
    if not isinstance(parsed, Mapping):
        raise ValueError("Live capture socket records must be JSON objects")
    writer.append_message(parsed)


def _devices_from_sample(sample: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    devices: list[Mapping[str, Any]] = []
    for role, sample_key, device_class in (
        ("head", "hmd", "HMD"),
        ("left_hand", "left_hand", "Controller"),
        ("right_hand", "right_hand", "Controller"),
    ):
        tracker = sample.get(sample_key)
        if isinstance(tracker, Mapping):
            devices.append({"role": role, "device_class": device_class, **tracker})

    devices.extend(_calibrated_devices_from_sample(sample))
    return devices


def _calibrated_devices_from_sample(sample: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    devices: list[Mapping[str, Any]] = []
    optional_trackers = sample.get("optional_trackers")
    if not isinstance(optional_trackers, Mapping):
        return devices
    for role, tracker in optional_trackers.items():
        if isinstance(tracker, Mapping):
            devices.append({"device_class": "CalibratedTracker", **tracker, "role": str(role)})
    return devices


def _transform_columns(prefix: str, transform: Mapping[str, Any]) -> dict[str, float | None]:
    values = _transform_values(transform)
    return {
        f"{prefix}_{component}": values[index] if values is not None else None
        for index, component in enumerate(TRANSFORM_COMPONENTS)
    }


def _transform_values(
    transform: Mapping[str, Any],
) -> tuple[float, float, float, float, float, float, float] | None:
    position = _float_sequence_or_none(transform.get("position"), 3)
    rotation = _float_sequence_or_none(transform.get("rotation"), 4)
    if position is None or rotation is None:
        return None
    return (
        position[0],
        position[1],
        position[2],
        rotation[0],
        rotation[1],
        rotation[2],
        rotation[3],
    )


def _vector_columns(prefix: str, value: Any) -> dict[str, float | None]:
    vector = _float_sequence_or_none(value, 3)
    return {
        f"{prefix}_{component}": vector[index] if vector is not None else None
        for index, component in enumerate(("x", "y", "z"))
    }


def _float_sequence_or_none(value: Any, length: int) -> tuple[float, ...] | None:
    if not isinstance(value, list | tuple) or len(value) != length:
        return None
    return tuple(float(component) for component in value)


def _mapping_or_empty(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _int_or_default(value: Any, default: int) -> int:
    if value is None:
        return default
    return int(value)


def _float_or_default(value: Any, default: float) -> float:
    if value is None:
        return default
    return float(value)


def _serial_hash(device: Mapping[str, Any]) -> str:
    existing = str(device.get("serial_hash", ""))
    if existing:
        return existing
    serial = str(device.get("serial_number", ""))
    if not serial:
        return ""
    return sha256(serial.encode("utf-8")).hexdigest()


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _write_parquet(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    compression: ParquetCompression,
) -> None:
    if not rows:
        return
    dataframe = pl.DataFrame(rows)
    dataframe.write_parquet(path, compression=compression)
