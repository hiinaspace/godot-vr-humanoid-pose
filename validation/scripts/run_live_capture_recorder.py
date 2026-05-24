"""Record live Godot VR pose capture frames from a localhost NDJSON socket."""

from __future__ import annotations

import argparse
import signal
from collections.abc import Sequence
from pathlib import Path
from threading import Event
from typing import cast

from validation.vr_pose_validation.baseline import EstimatorName
from validation.vr_pose_validation.live_capture import (
    LiveCaptureParquetWriter,
    run_tcp_recorder,
)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_OUTPUT_DIR = Path(".cache/live_capture")
ESTIMATOR_CHOICES = (
    "trivial",
    "planted",
    "basis",
    "basis_knee_hint",
    "basis_footstep",
    "basis_arm_lookup",
    "basis_arm_swivel",
    "basis_shrug",
    "basis_research",
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--session-id")
    parser.add_argument(
        "--estimator",
        choices=ESTIMATOR_CHOICES,
        default="basis_research",
        help="Python estimator used for live estimate_response messages.",
    )
    parser.add_argument(
        "--no-estimation",
        action="store_true",
        help="Record incoming frames without answering live estimate requests.",
    )
    parser.add_argument(
        "--stop-after-client",
        action="store_true",
        help="Flush and exit after the first client disconnects.",
    )
    args = parser.parse_args(argv)

    writer = LiveCaptureParquetWriter(args.output_dir, session_id=args.session_id)
    stop_event = Event()

    def request_stop(_signum: int, _frame: object) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    print(
        f"Listening for live capture frames on {args.host}:{args.port}; session={writer.session_id}"
    )
    try:
        summary = run_tcp_recorder(
            host=args.host,
            port=args.port,
            writer=writer,
            estimator_name=None if args.no_estimation else cast(EstimatorName, args.estimator),
            stop_after_client=args.stop_after_client,
            stop_event=stop_event,
        )
    except KeyboardInterrupt:
        stop_event.set()
        summary = writer.flush()

    print(
        f"Wrote {summary.frames} frames, {summary.devices} devices, "
        f"{summary.joints} joints to {summary.session_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
