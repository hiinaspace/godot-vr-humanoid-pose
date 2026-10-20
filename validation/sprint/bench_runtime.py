# ruff: noqa: E501
"""Per-frame latency of the ONNX runtime path on CPU (Round 4a).

    uv run python -m validation.sprint.bench_runtime --onnx MODEL.onnx --frames 10000 [--threads 1 2 4]

Reports p50 / p99 / max in milliseconds over ``--frames`` steps of a real tracker stream (a val clip
looped), for each stage of one runtime step and for the whole sidecar-side request:

  window    rate conversion + 41-frame window build (numpy)
  ort       onnxruntime session.run: feature construction + network + decode, all in the graph
  fk        numpy forward kinematics (14 joints) + tracker bookkeeping
  step      window + ort + fk  (``LearnedRuntime.step``)
  request   JSON parse of an estimate_request, sample unpack, estimator.estimate, response JSON dump
            (everything the sidecar does per message except the socket and the raw-message log)
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from validation.sprint import parity as P
from validation.sprint import runtime as RT
from validation.vr_pose_validation import live_capture as LC
from validation.vr_pose_validation.learned_estimator import LearnedEstimator


def pct(a, q):
    return float(np.percentile(np.asarray(a) * 1e3, q))


def summarize(name: str, a) -> dict:
    return dict(
        stage=name,
        p50_ms=pct(a, 50),
        p99_ms=pct(a, 99),
        max_ms=float(np.max(a) * 1e3),
        mean_ms=float(np.mean(a) * 1e3),
    )


def stream(n: int):
    clip_id = P.pick_clips(5)[-1]
    from validation.sprint import conventions as C
    from validation.sprint import oracle as O
    from validation.sprint import rest as R
    from validation.sprint import splits as S

    clip = C.load_clip(S.clip_path(clip_id))
    clip.positions, clip.rotations = clip.positions[:1500], clip.rotations[:1500]
    inputs = O.make_inputs(clip, "clean")
    calib = O.make_calib(clip)
    lengths = R.bone_lengths(clip.positions)
    reps = int(np.ceil(n / len(inputs)))
    # loop the clip; keep head/hands continuous enough for realistic feature magnitudes
    raw = np.concatenate([inputs] * reps)[:n]
    return raw, calib, lengths


def message(k: int, raw: np.ndarray, t: float) -> dict:
    def tr(role, ps, qs):
        return {
            "role": role,
            "confidence": 1.0,
            "valid": True,
            "transform": {"position": raw[ps].tolist(), "rotation": raw[qs].tolist()},
        }

    return {
        "format": LC.CAPTURE_FORMAT,
        "type": "estimate_request",
        "session_id": "bench",
        "request_id": k,
        "frame_index": k,
        "timestamp_seconds": t,
        "sample": {
            "timestamp_seconds": t,
            "hmd": tr("head", *RT.TRACKERS[0]),
            "left_hand": tr("left_hand", *RT.TRACKERS[1]),
            "right_hand": tr("right_hand", *RT.TRACKERS[2]),
            "optional_trackers": {},
        },
        "calibration": {"human_height_m": 1.7, "bone_lengths_m": {}},
        "grounding": {},
    }


def bench(onnx: Path, threads: int, n: int, hz: float, warm: int = 200) -> dict:
    raw, calib, lengths = stream(n + warm)
    rt = RT.LearnedRuntime(onnx, threads=threads)
    rt.set_calibration((calib.height_m, calib.arm_length_m, calib.shoulder_width_m))
    win, ort_, fk_, tot = [], [], [], []
    for k in range(n + warm):
        r = rt.step(k / hz, raw[k], lengths)
        if k >= warm:
            (
                win.append(r.timing["window"]),
                ort_.append(r.timing["ort"]),
                fk_.append(r.timing["fk"]),
                tot.append(r.timing["total"]),
            )
    est = LearnedEstimator(onnx, threads=threads, frame="model")
    req = []
    cal = LC._calibration_from_message({"human_height_m": 1.7})
    del cal
    for k in range(n + warm):
        line = json.dumps(message(k, raw[k], k / hz)).encode()  # client side, not timed
        t0 = time.perf_counter()
        m = json.loads(line)
        out = LC._estimate_frame_from_message(m, est)
        _ = json.dumps(
            LC._estimate_response(m, out, estimator_name="learned"), separators=(",", ":")
        )
        if k >= warm:
            req.append(time.perf_counter() - t0)
    return dict(
        threads=threads,
        hz=hz,
        frames=n,
        stages=[
            summarize("window", win),
            summarize("ort", ort_),
            summarize("fk", fk_),
            summarize("step", tot),
            summarize("request", req),
        ],
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--frames", type=int, default=10000)
    ap.add_argument("--threads", type=int, nargs="+", default=[1])
    ap.add_argument(
        "--hz", type=float, default=60.0, help="timestamp spacing of the replayed stream"
    )
    ap.add_argument("--json", type=Path)
    a = ap.parse_args(argv)
    res = []
    for th in a.threads:
        r = bench(a.onnx, th, a.frames, a.hz)
        res.append(r)
        print(f"threads={th} hz={a.hz:g} frames={a.frames} loadavg={os.getloadavg()[0]:.1f}")
        for s in r["stages"]:
            print(
                f"  {s['stage']:8s} p50 {s['p50_ms']:7.3f}  p99 {s['p99_ms']:7.3f}  max {s['max_ms']:7.3f}  mean {s['mean_ms']:7.3f} ms"
            )
    if a.json:
        a.json.write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
