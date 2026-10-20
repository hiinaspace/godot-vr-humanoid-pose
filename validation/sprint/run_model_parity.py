# ruff: noqa: E501
"""One command: export -> fixture -> start the `learned` sidecar -> headless Godot parity test -> tear down.

    CUDA_VISIBLE_DEVICES= uv run python -m validation.sprint.run_model_parity \
        --ckpt .cache/sprint/runs/e1/best.pt [--threads 2] [--frames 900] [--port 8791]

Work files go to ``--work`` (default ``.cache/r4a/parity_<run>``): ``model.onnx``, ``fixture/``,
``sidecar.log``, ``godot.log``, ``report.json``. Exit code is Godot's (0 = all assertions passed).
The sidecar is our own child process: it is stopped with SIGTERM (then SIGKILL after 10 s) in a
``finally``; nothing is left running.
"""

from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

from validation.sprint import export_onnx
from validation.sprint import parity as P
from validation.sprint import splits as S

REPO = S.REPO


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_port(port: int, proc: subprocess.Popen, log: Path, timeout: float = 60.0) -> bool:
    """True once *our* sidecar (still alive, 'Listening' in its log) accepts connections."""
    end = time.time() + timeout
    while time.time() < end:
        if proc.poll() is not None:
            return False
        if "Listening for live capture" in log.read_text():
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                    return proc.poll() is None
            except OSError:
                pass
        time.sleep(0.2)
    return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument(
        "--onnx", type=Path, help="reuse an existing export (default: export into --work)"
    )
    ap.add_argument("--work", type=Path)
    ap.add_argument(
        "--threads", type=int, default=1, help="onnxruntime intra-op threads in the sidecar"
    )
    ap.add_argument("--frames", type=int, default=900)
    ap.add_argument("--port", type=int, default=0, help="0 = pick a free port")
    ap.add_argument("--godot", default="godot")
    ap.add_argument("--paced-seconds", type=float, default=6.0)
    ap.add_argument("--max-p99-ms", type=float, default=30.0)
    ap.add_argument("--clip")
    ap.add_argument(
        "--paced-hz", type=float, default=90.0, help="request tick rate of the paced phase"
    )
    ap.add_argument(
        "--protocol-floor",
        action="store_true",
        help="only the paced round-trip phase, against the trivial estimator (protocol + JSON cost, no model)",
    )
    a = ap.parse_args(argv)

    work = (a.work or REPO / ".cache/r4a" / f"parity_{a.ckpt.parent.name}").resolve()
    work.mkdir(parents=True, exist_ok=True)
    onnx = a.onnx or work / "model.onnx"
    if not onnx.exists() and not a.protocol_floor:
        print("exporting", a.ckpt, "->", onnx, flush=True)
        export_onnx.export(a.ckpt, onnx)
    fix = work / "fixture"
    if not (a.protocol_floor and fix.exists()):
        print("building fixture", flush=True)
        P.fixture(
            argparse.Namespace(
                ckpt=a.ckpt, onnx=onnx, out=fix, frames=a.frames, threads=1, clip=a.clip
            )
        )

    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    a.port = a.port or free_port()
    side_log = (work / "sidecar.log").open("w", buffering=1)
    sidecar = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "validation.scripts.run_live_capture_recorder",
            "--estimator",
            "trivial" if a.protocol_floor else "learned",
            "--learned-model",
            str(onnx),
            "--learned-frame",
            "model",
            "--learned-threads",
            str(a.threads),
            "--port",
            str(a.port),
            "--no-raw-log",
            "--output-dir",
            str(work / "capture"),
            "--session-id",
            "parity",
        ],
        cwd=REPO,
        env=env,
        stdout=side_log,
        stderr=subprocess.STDOUT,
    )
    code = 1
    try:
        if not wait_port(a.port, sidecar, work / "sidecar.log"):
            print("sidecar did not start; see", work / "sidecar.log")
            return 2
        print(
            f"sidecar up (pid {sidecar.pid}, {a.threads} ORT thread(s)); running Godot", flush=True
        )
        with (work / "godot.log").open("w") as glog:
            r = subprocess.run(
                [
                    a.godot,
                    "--headless",
                    "--path",
                    str(REPO / "godot"),
                    "--script",
                    "res://tests/run_model_parity.gd",
                    "--",
                    f"--fixture={fix}",
                    f"--port={a.port}",
                    f"--paced-seconds={a.paced_seconds}",
                    f"--max-p99-ms={a.max_p99_ms}",
                    f"--report={work / 'report.json'}",
                    f"--paced-hz={a.paced_hz:g}",
                    f"--latency-only={int(a.protocol_floor)}",
                ],
                cwd=REPO,
                env=env,
                stdout=glog,
                stderr=subprocess.STDOUT,
                timeout=900,
            )
        code = r.returncode
        log = (work / "godot.log").read_text()
        for line in log.splitlines():
            if (
                line.startswith(("PARITY", "RESET", "ROUNDTRIP", "Godot model parity"))
                or "ERROR: " in line
                and "openvr" not in line.lower()
                and "autoload" not in line.lower()
            ):
                print(line)
        print("GODOT EXIT", code, "->", "PASS" if code == 0 else "FAIL")
    finally:
        if sidecar.poll() is None:
            sidecar.send_signal(signal.SIGTERM)
            try:
                sidecar.wait(timeout=10)
            except subprocess.TimeoutExpired:
                sidecar.kill()
                sidecar.wait()
        side_log.close()
    return code


if __name__ == "__main__":
    sys.exit(main())
