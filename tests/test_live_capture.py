from __future__ import annotations

import json
import socket
import struct
import time
from threading import Event, Thread
from typing import cast

import polars as pl

from validation.vr_pose_validation.live_capture import (
    CAPTURE_FORMAT,
    LiveCaptureParquetWriter,
    LiveCaptureSummary,
    record_ndjson_lines,
    run_tcp_recorder,
)


def test_live_capture_writer_splits_frames_devices_and_joints(tmp_path) -> None:
    writer = LiveCaptureParquetWriter(tmp_path, session_id="test-session")
    writer.append_message(_session_message())
    writer.append_message(_frame_message(frame_index=0, timestamp_seconds=1.25))

    summary = writer.flush()

    assert summary.frames == 1
    assert summary.devices == 3
    assert summary.joints == 2
    assert summary.session_id == "test-session"

    frames = pl.read_parquet(summary.session_dir / "frames.parquet")
    devices = pl.read_parquet(summary.session_dir / "devices.parquet")
    joints = pl.read_parquet(summary.session_dir / "joints.parquet")

    assert frames["session_id"].to_list() == ["test-session"]
    assert frames["clip_index"].to_list() == [2]
    assert frames["root_px"].to_list() == [0.0]
    assert frames["raw_to_standing_11"].to_list() == [0.0]
    assert devices["role"].to_list() == ["head", "left_hand", "right_hand"]
    assert devices["clip_index"].to_list() == [2, 2, 2]
    assert devices["device_px"].to_list() == [0.0, -0.2, 0.2]
    assert len(devices["serial_hash"].to_list()[0]) == 64
    assert joints["joint"].to_list() == ["head", "hips"]
    assert joints["clip_index"].to_list() == [2, 2]
    assert joints["joint_py"].to_list() == [1.6, 1.0]


def test_live_capture_ndjson_records_sample_devices_when_devices_are_omitted(tmp_path) -> None:
    writer = LiveCaptureParquetWriter(tmp_path, session_id="sample-devices")
    message = _frame_message(frame_index=0, timestamp_seconds=0.0)
    message.pop("devices")

    summary = record_ndjson_lines([json.dumps(message)], writer)

    devices = pl.read_parquet(summary.session_dir / "devices.parquet")
    assert devices["role"].to_list() == ["head", "left_hand", "right_hand"]
    assert devices["device_class"].to_list() == ["HMD", "Controller", "Controller"]


def test_live_capture_writer_adds_calibrated_optional_tracker_rows(tmp_path) -> None:
    writer = LiveCaptureParquetWriter(tmp_path, session_id="calibrated-trackers")
    message = _frame_message(frame_index=0, timestamp_seconds=0.0)
    sample = cast(dict[str, object], message["sample"])
    sample["optional_trackers"] = {
        "left_elbow": _tracker("left_elbow", [-0.45, 1.2, 0.15], [0.0, 0.0, 0.0, 1.0])
    }

    summary = record_ndjson_lines([json.dumps(message)], writer)

    devices = pl.read_parquet(summary.session_dir / "devices.parquet")
    calibrated = devices.filter(pl.col("device_class") == "CalibratedTracker")
    assert calibrated["role"].to_list() == ["left_elbow"]
    assert calibrated["device_px"].to_list() == [-0.45]


def test_live_capture_flush_message_persists_without_stopping_recorder(tmp_path) -> None:
    writer = LiveCaptureParquetWriter(tmp_path, session_id="flush-message")
    writer.append_message(_frame_message(frame_index=0, timestamp_seconds=0.0))
    writer.append_message(
        {
            "format": CAPTURE_FORMAT,
            "type": "flush",
            "session_id": "test-session",
            "clip_index": 2,
            "frame_index": 1,
            "reason": "clip_complete",
        }
    )

    frames = pl.read_parquet(writer.paths.frames_path)
    metadata = json.loads(writer.paths.session_metadata_path.read_text(encoding="utf-8"))

    assert frames["frame_index"].to_list() == [0]
    assert metadata["flushes"][0]["clip_index"] == 2
    assert metadata["flushes"][0]["reason"] == "clip_complete"


def test_tcp_recorder_flushes_after_abrupt_client_disconnect(tmp_path) -> None:
    writer = LiveCaptureParquetWriter(tmp_path, session_id="reset-client")
    port = _free_tcp_port()
    result: dict[str, object] = {}

    def run_server() -> None:
        result["summary"] = run_tcp_recorder(
            host="127.0.0.1",
            port=port,
            writer=writer,
            stop_after_client=True,
        )

    thread = Thread(target=run_server, daemon=True)
    thread.start()
    with socket.create_connection(("127.0.0.1", port), timeout=5.0) as client:
        payload = json.dumps(_frame_message(frame_index=0, timestamp_seconds=0.0)) + "\n"
        client.sendall(payload.encode())
        time.sleep(0.05)
        client.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("hh", 1, 0))

    thread.join(timeout=5.0)

    assert "summary" in result
    summary = cast(LiveCaptureSummary, result["summary"])
    assert summary.frames == 1
    assert (summary.session_dir / "frames.parquet").exists()


def test_tcp_recorder_stop_event_exits_with_connected_client(tmp_path) -> None:
    writer = LiveCaptureParquetWriter(tmp_path, session_id="stop-event")
    stop_event = Event()
    port = _free_tcp_port()
    result: dict[str, object] = {}

    def run_server() -> None:
        result["summary"] = run_tcp_recorder(
            host="127.0.0.1",
            port=port,
            writer=writer,
            stop_event=stop_event,
            accept_timeout_seconds=0.05,
        )

    thread = Thread(target=run_server, daemon=True)
    thread.start()
    with socket.create_connection(("127.0.0.1", port), timeout=5.0) as client:
        assert client is not None
        stop_event.set()
        thread.join(timeout=5.0)

    assert "summary" in result
    summary = cast(LiveCaptureSummary, result["summary"])
    assert summary.frames == 0


def test_tcp_recorder_replies_to_live_estimate_requests(tmp_path) -> None:
    writer = LiveCaptureParquetWriter(tmp_path, session_id="estimate-request")
    port = _free_tcp_port()
    result: dict[str, object] = {}
    request = _frame_message(frame_index=7, timestamp_seconds=1.0)
    request["type"] = "estimate_request"
    request["request_id"] = 42
    request.pop("estimated_frame")

    def run_server() -> None:
        result["summary"] = run_tcp_recorder(
            host="127.0.0.1",
            port=port,
            writer=writer,
            estimator_name="basis",
            stop_after_client=True,
        )

    thread = Thread(target=run_server, daemon=True)
    thread.start()
    with socket.create_connection(("127.0.0.1", port), timeout=5.0) as client:
        client.sendall((json.dumps(request) + "\n").encode())
        response = json.loads(client.makefile("r", encoding="utf-8").readline())

    thread.join(timeout=5.0)

    assert response["type"] == "estimate_response"
    assert response["request_id"] == 42
    assert "hips" in response["estimated_frame"]["joints"]
    summary = cast(LiveCaptureSummary, result["summary"])
    assert summary.frames == 0


def test_tcp_recorder_enriches_recorded_frames_with_python_estimates(tmp_path) -> None:
    writer = LiveCaptureParquetWriter(tmp_path, session_id="estimate-frame")
    port = _free_tcp_port()
    result: dict[str, object] = {}
    request = _frame_message(frame_index=0, timestamp_seconds=1.0)
    request.pop("estimated_frame")

    def run_server() -> None:
        result["summary"] = run_tcp_recorder(
            host="127.0.0.1",
            port=port,
            writer=writer,
            estimator_name="basis",
            stop_after_client=True,
        )

    thread = Thread(target=run_server, daemon=True)
    thread.start()
    with socket.create_connection(("127.0.0.1", port), timeout=5.0) as client:
        client.sendall((json.dumps(request) + "\n").encode())
        response = json.loads(client.makefile("r", encoding="utf-8").readline())

    thread.join(timeout=5.0)

    assert response["type"] == "estimate_response"
    summary = cast(LiveCaptureSummary, result["summary"])
    assert summary.frames == 1
    joints = pl.read_parquet(summary.session_dir / "joints.parquet")
    assert "hips" in joints["joint"].to_list()


def _session_message() -> dict[str, object]:
    return {
        "format": CAPTURE_FORMAT,
        "type": "session",
        "source": "test",
        "user_note": "unit-test",
    }


def _frame_message(frame_index: int, timestamp_seconds: float) -> dict[str, object]:
    identity = [0.0, 0.0, 0.0, 1.0]
    raw_to_standing = [
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
    ]
    return {
        "format": CAPTURE_FORMAT,
        "type": "frame",
        "session_id": "test-session",
        "frame_index": frame_index,
        "clip_index": 2,
        "timestamp_seconds": timestamp_seconds,
        "source": "godot_live_overlay",
        "pose_universe": "standing",
        "raw_to_standing": raw_to_standing,
        "calibration": {"human_height_m": 1.7},
        "grounding": {"floor_height_m": 0.0, "floor_normal": [0.0, 1.0, 0.0]},
        "sample": {
            "timestamp_seconds": timestamp_seconds,
            "hmd": _tracker("head", [0.0, 1.6, 0.0], identity),
            "left_hand": _tracker("left_hand", [-0.2, 1.2, 0.4], identity),
            "right_hand": _tracker("right_hand", [0.2, 1.2, 0.4], identity),
        },
        "devices": [
            _device("head", "HMD", [0.0, 1.6, 0.0], identity),
            _device("left_hand", "Controller", [-0.2, 1.2, 0.4], identity),
            _device("right_hand", "Controller", [0.2, 1.2, 0.4], identity),
        ],
        "estimated_frame": {
            "timestamp_seconds": timestamp_seconds,
            "root": {"position": [0.0, 1.0, 0.0], "rotation": identity},
            "joints": {
                "head": {
                    "transform": {"position": [0.0, 1.6, 0.0], "rotation": identity},
                    "estimated": False,
                    "confidence": 1.0,
                },
                "hips": {
                    "transform": {"position": [0.0, 1.0, 0.0], "rotation": identity},
                    "estimated": True,
                    "confidence": 1.0,
                },
            },
        },
    }


def _tracker(role: str, position: list[float], rotation: list[float]) -> dict[str, object]:
    return {
        "role": role,
        "valid": True,
        "confidence": 1.0,
        "transform": {"position": position, "rotation": rotation},
    }


def _device(
    role: str,
    device_class: str,
    position: list[float],
    rotation: list[float],
) -> dict[str, object]:
    value = _tracker(role, position, rotation)
    value["device_class"] = device_class
    value["device_id"] = role
    value["serial_number"] = f"{role}-serial"
    value["linear_velocity"] = [0.0, 0.0, 0.0]
    value["angular_velocity"] = [0.0, 0.0, 0.0]
    return value


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])
