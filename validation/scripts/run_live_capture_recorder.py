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
from validation.vr_pose_validation.sparseposer_live import (
    DEFAULT_SPARSEPOSER_MODEL,
    DEFAULT_SPARSEPOSER_PYTHON,
    DEFAULT_SPARSEPOSER_REFERENCE_BVH,
    DEFAULT_SPARSEPOSER_ROOT,
    SparsePoserBridgeConfig,
    SparsePoserLiveBackend,
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
        "--backend",
        choices=("basis", "sparseposer_xsens", "none"),
        default="basis",
        help="Live estimation backend. 'basis' uses --estimator; 'sparseposer_xsens' "
        "bridges the official SparsePoser runtime.",
    )
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
    parser.add_argument("--sparseposer-python", type=Path, default=DEFAULT_SPARSEPOSER_PYTHON)
    parser.add_argument("--sparseposer-root", type=Path, default=DEFAULT_SPARSEPOSER_ROOT)
    parser.add_argument("--sparseposer-model", type=Path, default=DEFAULT_SPARSEPOSER_MODEL)
    parser.add_argument(
        "--sparseposer-reference-bvh",
        type=Path,
        default=DEFAULT_SPARSEPOSER_REFERENCE_BVH,
    )
    parser.add_argument("--sparseposer-host", default="127.0.0.1")
    parser.add_argument("--sparseposer-port", type=int, default=2222)
    parser.add_argument(
        "--sparseposer-endpoint-ik",
        action="store_true",
        help="Apply this repo's post-SparsePoser endpoint IK correction. Disabled by default "
        "to match the official SparsePoser Unity demo more closely.",
    )
    parser.add_argument(
        "--sparseposer-body-scale-mode",
        choices=("none", "height", "extent"),
        default="height",
        help="How live T-pose calibration scales the rendered/input Xsens body. 'height' "
        "matches head-to-floor height; 'extent' also considers arm reach; 'none' uses the "
        "reference BVH exactly.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print live protocol events for debugging Godot/sidecar handshakes.",
    )
    parser.add_argument(
        "--stop-after-client",
        action="store_true",
        help="Flush and exit after the first client disconnects.",
    )
    args = parser.parse_args(argv)

    writer = LiveCaptureParquetWriter(args.output_dir, session_id=args.session_id)
    stop_event = Event()
    backend = None
    estimator_name = (
        None
        if args.no_estimation or args.backend == "none"
        else cast(EstimatorName, args.estimator)
    )
    if not args.no_estimation and args.backend == "sparseposer_xsens":
        backend = SparsePoserLiveBackend(
            SparsePoserBridgeConfig(
                sparseposer_python=args.sparseposer_python,
                sparseposer_root=args.sparseposer_root,
                sparseposer_model=args.sparseposer_model,
                sparseposer_reference_bvh=args.sparseposer_reference_bvh,
                host=args.sparseposer_host,
                port=args.sparseposer_port,
                apply_endpoint_ik=args.sparseposer_endpoint_ik,
                body_scale_mode=args.sparseposer_body_scale_mode,
            )
        )
        print("Prewarming SparsePoser runtime...", flush=True)
        backend.prewarm()
        estimator_name = None

    def request_stop(_signum: int, _frame: object) -> None:
        stop_event.set()

    def log_event(message: str) -> None:
        print(f"[live-capture] {message}", flush=True)

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    print(
        f"Listening for live capture frames on {args.host}:{args.port}; "
        f"session={writer.session_id}",
        flush=True,
    )
    try:
        summary = run_tcp_recorder(
            host=args.host,
            port=args.port,
            writer=writer,
            estimator_name=estimator_name,
            estimator_backend=backend,
            stop_after_client=args.stop_after_client,
            stop_event=stop_event,
            event_logger=log_event if args.verbose else None,
        )
    except KeyboardInterrupt:
        stop_event.set()
        summary = writer.flush()

    print(
        f"Wrote {summary.frames} frames, {summary.devices} devices, "
        f"{summary.joints} joints to {summary.session_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
