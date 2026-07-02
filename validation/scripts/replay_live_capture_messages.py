"""Replay raw live-capture NDJSON messages through a live estimator backend."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from validation.vr_pose_validation.live_capture import CAPTURE_FORMAT
from validation.vr_pose_validation.sparseposer_live import (
    DEFAULT_SPARSEPOSER_MODEL,
    DEFAULT_SPARSEPOSER_PYTHON,
    DEFAULT_SPARSEPOSER_REFERENCE_BVH,
    DEFAULT_SPARSEPOSER_ROOT,
    SparsePoserBridgeConfig,
    SparsePoserLiveBackend,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_messages", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-messages", type=int)
    parser.add_argument("--sparseposer-python", type=Path, default=DEFAULT_SPARSEPOSER_PYTHON)
    parser.add_argument("--sparseposer-root", type=Path, default=DEFAULT_SPARSEPOSER_ROOT)
    parser.add_argument("--sparseposer-model", type=Path, default=DEFAULT_SPARSEPOSER_MODEL)
    parser.add_argument(
        "--sparseposer-reference-bvh",
        type=Path,
        default=DEFAULT_SPARSEPOSER_REFERENCE_BVH,
    )
    args = parser.parse_args(argv)

    backend = SparsePoserLiveBackend(
        SparsePoserBridgeConfig(
            sparseposer_python=args.sparseposer_python,
            sparseposer_root=args.sparseposer_root,
            sparseposer_model=args.sparseposer_model,
            sparseposer_reference_bvh=args.sparseposer_reference_bvh,
        )
    )
    responses: list[dict[str, Any]] = []
    counts = {"messages": 0, "estimates": 0, "resets": 0, "errors": 0}
    try:
        for message in _iter_messages(args.raw_messages):
            if args.max_messages is not None and counts["messages"] >= args.max_messages:
                break
            counts["messages"] += 1
            message_type = str(message.get("type", "frame"))
            if message_type == "reset_estimator":
                backend.reset()
                counts["resets"] += 1
                continue
            if message_type not in {"estimate_request", "frame"}:
                continue
            try:
                estimated_frame = backend.estimate_from_message(message)
            except Exception as exc:  # noqa: BLE001 - replay should preserve bad messages.
                counts["errors"] += 1
                responses.append(
                    {
                        "format": CAPTURE_FORMAT,
                        "type": "estimate_error",
                        "session_id": str(message.get("session_id", "")),
                        "frame_index": int(message.get("frame_index", 0)),
                        "clip_index": int(message.get("clip_index", 0)),
                        "error": str(exc),
                    }
                )
                continue
            counts["estimates"] += 1
            responses.append(
                {
                    "format": CAPTURE_FORMAT,
                    "type": "estimate_response",
                    "session_id": str(message.get("session_id", "")),
                    "frame_index": int(message.get("frame_index", 0)),
                    "clip_index": int(message.get("clip_index", 0)),
                    "estimated_frame": estimated_frame,
                }
            )
    finally:
        backend.close()

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            for response in responses:
                handle.write(json.dumps(response, separators=(",", ":"), sort_keys=True))
                handle.write("\n")
    print(json.dumps(counts, indent=2, sort_keys=True))
    return 0


def _iter_messages(path: Path) -> Sequence[dict[str, Any]]:
    messages = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parsed = json.loads(line)
        if not isinstance(parsed, dict):
            raise ValueError(f"{path} contains a non-object JSON line")
        messages.append(parsed)
    return messages


if __name__ == "__main__":
    raise SystemExit(main())
