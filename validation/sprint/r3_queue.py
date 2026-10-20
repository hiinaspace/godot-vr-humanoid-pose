# ruff: noqa: E501
"""Round 3 job queue: trains runs from ``.cache/sprint/r3/jobs.json`` one at a time, then evaluates them.

Each job: ``{"name": str, "train": [train.py args...], "evals": [{"tag", "ckpt"?, "flags"?, "split"?,
"variant"?}, ...], "after": "name"?}``. The file is re-read after every step, so jobs can be appended
while the queue runs. GPU steps run as ``flock /tmp/sayu-gpu.lock ...``, each at most ``TRAIN_MIN``
minutes (train.py pauses and resumes from last.pt), and never start while a VR runtime is up.
Launch: ``systemd-run --user tmux new -d -s r3q "python3 validation/sprint/r3_queue.py"``.
Stop cleanly by creating ``.cache/sprint/r3/STOP`` (checked between steps).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
R3 = REPO / ".cache/sprint/r3"
RUNS = REPO / ".cache/sprint/runs"
EVAL_OUT = R3 / "eval"
TRAIN_MIN = 110
ENV = {
    **os.environ,
    "UV_PROJECT_ENVIRONMENT": str(Path.home() / ".cache/venvs/godot-vr-humanoid-pose"),
    "LD_LIBRARY_PATH": "/run/opengl-driver/lib",
    "PYTHONPATH": str(REPO),
}


def vr_up() -> bool:
    out = subprocess.run(["ps", "-eo", "comm"], capture_output=True, text=True).stdout.split()
    return any(c.startswith(("monado", "vrcompositor", "vrserver")) for c in out)


def gpu(args: list[str], log: Path) -> int:
    while vr_up():
        print("VR runtime up; waiting", flush=True)
        time.sleep(120)
    cmd = ["flock", "/tmp/sayu-gpu.lock", "uv", "run", "python", *args]
    print("RUN", " ".join(cmd[3:]), flush=True)
    with log.open("a") as f:
        return subprocess.run(cmd, cwd=REPO, env=ENV, stdout=f, stderr=subprocess.STDOUT).returncode


def pending() -> tuple[dict, str] | None:
    jobs = json.loads((R3 / "jobs.json").read_text())
    for j in jobs:
        run = RUNS / j["name"]
        if not (run / "elapsed_gpu_s").exists() and "train" in j:
            return j, "train"
        for e in j.get("evals", []):
            if not (EVAL_OUT / f"{j['name']}__{e['tag']}.json").exists():
                return j, "eval:" + e["tag"]
    return None


def main() -> int:
    EVAL_OUT.mkdir(parents=True, exist_ok=True)
    logs = R3 / "logs"
    logs.mkdir(exist_ok=True)
    while not (R3 / "STOP").exists():
        p = pending()
        if p is None:
            print("queue empty", flush=True)
            return 0
        j, what = p
        name = j["name"]
        if what == "train":
            rc = gpu(
                [
                    "-m",
                    "validation.sprint.train",
                    "train",
                    "--run",
                    name,
                    "--time-min",
                    str(TRAIN_MIN),
                    *j["train"],
                ],
                logs / f"{name}.train.log",
            )
            if rc != 0:
                print("train failed", name, rc, flush=True)
                return rc
        else:
            e = next(x for x in j["evals"] if x["tag"] == what[5:])
            ckpt = e.get("ckpt", f".cache/sprint/runs/{e.get('run', name)}/best.pt")
            who = ["--method", e["method"]] if "method" in e else ["--ckpt", ckpt]
            args = [
                "-m",
                "validation.sprint.eval",
                *who,
                "--split",
                e.get("split", "val"),
                "--variant",
                e.get("variant", "both"),
                "--tag",
                f"{name}__{e['tag']}",
                "--out",
                str(EVAL_OUT),
                "--workers",
                "14" if "method" in e else "1",
                *e.get("flags", []),
            ]
            rc = gpu(args, logs / f"{name}.eval.log")
            if rc != 0:
                print("eval failed", name, e["tag"], rc, flush=True)
                return rc
    print("STOP file seen", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
