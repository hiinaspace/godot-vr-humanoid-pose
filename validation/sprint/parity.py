# ruff: noqa: E501
"""ONNX <-> torch parity gate and parity-fixture export (Round 4a).

    uv run python -m validation.sprint.parity gate --ckpt CKPT --onnx MODEL.onnx [--clips 5 --frames 1000]
    uv run python -m validation.sprint.parity fixture --ckpt CKPT --onnx MODEL.onnx --out DIR

``gate``: for val clips (2 CMU + 3 Xsens), the frame-by-frame ONNX runtime (ring buffer, no lookahead,
numpy FK) must reproduce ``train.ModelPredictor.predict_clip_meta`` (the full-sequence torch path the
oracle evaluates) -> FK with the clip's GT bone lengths (oracle-v2 placement). Gate: max |dpos| <= 1e-4 m.

``fixture``: one clip's tracker stream + torch expected outputs (+ a 90 Hz resampled stream with the
runtime's own expected output) as little-endian float32 ``.bin`` files for the Godot test.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from validation.sprint import conventions as C
from validation.sprint import oracle as O
from validation.sprint import rest as R
from validation.sprint import runtime as RT
from validation.sprint import splits as S

GATE_M = 1e-4


def pick_clips(n_clips: int = 5, min_frames: int = 600) -> list[str]:
    """2 CMU val + 3 Xsens val clips (deterministic: evenly spaced, long enough)."""
    out: list[str] = []
    for split, k in (("cmu_val", 2), ("xsens_val", 3)):
        ids = S.clip_ids(split)
        ids = [i for i in ids[:: max(1, len(ids) // 24)]]
        got = []
        for i in ids:
            if len(got) == k:
                break
            clip = C.load_clip(S.clip_path(i))
            if clip.n_frames >= min_frames:
                got.append(i)
        out += got
    return out[:n_clips]


def torch_reference(predictor, clip_id: str, frames: int):
    """Full-sequence torch path -> (inputs, calib, lengths, local quats, world positions, global quats)."""
    clip = C.load_clip(S.clip_path(clip_id))
    clip.positions, clip.rotations = clip.positions[:frames], clip.rotations[:frames]
    calib, inputs = O.make_calib(clip), O.make_inputs(clip, "clean")
    pred = predictor.predict_clip_meta(inputs, calib, {"dataset": clip.dataset})
    lengths = R.bone_lengths(clip.positions)
    pos = R.fk(pred.rotations, pred.root_pos, lengths)
    gq = C.global_rotations(pred.rotations)
    return clip, inputs, calib, lengths, pred, pos, gq


def run_onnx(rt: RT.LearnedRuntime, inputs, calib, lengths, dataset: str, hz: float = 60.0):
    rt.reset()
    rt.skel_id = None if rt.skel_id is None else (0 if dataset == "cmu" else 1)
    rt.set_calibration((calib.height_m, calib.arm_length_m, calib.shoulder_width_m))
    pos, loc, glob, hips, times = [], [], [], [], []
    for k in range(len(inputs)):
        r = rt.step(k / hz, inputs[k], lengths)
        (
            pos.append(r.positions),
            loc.append(r.local_rot),
            glob.append(r.global_rot),
            hips.append(r.hips),
        )
        times.append(r.timing)
    return np.stack(pos), np.stack(loc), np.stack(glob), np.stack(hips), times


def gate(args) -> int:
    from validation.sprint import train as T

    predictor = T.ModelPredictor.load(args.ckpt, dev="cpu")
    rt = RT.LearnedRuntime(args.onnx, threads=args.threads)
    rows, worst = [], 0.0
    for cid in args.clip_ids or pick_clips(args.clips):
        clip, inputs, calib, lengths, pred, pos_t, _ = torch_reference(predictor, cid, args.frames)
        pos_o, loc_o, _, hips_o, _ = run_onnx(rt, inputs, calib, lengths, clip.dataset)
        q_o = RT.rotmat_to_quat(loc_o)
        d_pos = np.abs(pos_o - pos_t).max()
        d_hip = np.abs(hips_o - pred.root_pos).max()
        d_deg = float(C.qgeodesic_deg(q_o, pred.rotations).max())
        rows.append((cid, len(inputs), d_pos, d_hip, d_deg))
        worst = max(worst, d_pos)
        print(
            f"{cid:44s} {len(inputs):5d} fr  max|dpos| {d_pos:.2e} m  max|dhips| {d_hip:.2e} m  max rot {d_deg:.4f} deg",
            flush=True,
        )
    ok = worst <= GATE_M
    print(
        f"PARITY {'PASS' if ok else 'FAIL'}: worst max|dpos| {worst:.3e} m (gate {GATE_M:g}), {len(rows)} clips"
    )
    if args.json:
        Path(args.json).write_text(
            json.dumps(dict(worst_pos_m=worst, gate_m=GATE_M, clips=rows), indent=1)
        )
    return 0 if ok else 1


def resample_stream(
    raw60: np.ndarray, hz_out: float, hz_in: float = 60.0
) -> tuple[np.ndarray, np.ndarray]:
    """Resample an on-grid stream to ``hz_out`` (linear / nlerp) -> (times, raw). Used for the 90 Hz variant."""
    n = len(raw60)
    t_in = np.arange(n) / hz_in
    t_out = np.arange(0.0, t_in[-1] + 1e-9, 1.0 / hz_out)
    idx = np.clip((t_out * hz_in).astype(int), 0, n - 2)
    w = ((t_out - idx / hz_in) * hz_in)[:, None]
    x0, x1 = raw60[idx], raw60[np.minimum(idx + 1, n - 1)]
    out = x0 * (1 - w) + x1 * w
    for _, qs in RT.TRACKERS:
        out[:, qs] = RT._nlerp_quat(x0[:, qs], x1[:, qs], w)
    out[:, 21:] = x0[:, 21:]
    return t_out, out


def fixture(args) -> int:
    from validation.sprint import train as T

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    predictor = T.ModelPredictor.load(args.ckpt, dev="cpu")
    rt = RT.LearnedRuntime(args.onnx, threads=args.threads)
    cid = args.clip or pick_clips(5)[-1]
    clip, inputs, calib, lengths, pred, pos_t, gq_t = torch_reference(predictor, cid, args.frames)
    n = len(inputs)
    # sanity: the runtime reproduces torch on exactly this stream before we call it a fixture
    pos_o, *_ = run_onnx(rt, inputs, calib, lengths, clip.dataset)
    err = float(np.abs(pos_o - pos_t).max())
    print(f"fixture clip {cid}: {n} frames, runtime-vs-torch max|dpos| {err:.2e} m")
    t90, raw90 = resample_stream(inputs, 90.0)
    pos90, _, glob90 = run_onnx_times(rt, raw90, t90, calib, lengths, clip.dataset)
    f32 = lambda a: np.ascontiguousarray(a, "<f4")  # noqa: E731
    f32(inputs).tofile(out / "stream60.bin")
    f32(pos_t).tofile(out / "expected_pos60.bin")
    f32(gq_t).tofile(out / "expected_quat60.bin")
    f32(raw90).tofile(out / "stream90.bin")
    f32(t90).astype("<f8").tofile(out / "t90.bin")
    f32(pos90).tofile(out / "expected_pos90.bin")
    f32(RT.rotmat_to_quat(glob90)).tofile(out / "expected_quat90.bin")
    meta = dict(
        clip=cid,
        frames60=n,
        frames90=len(t90),
        nj=C.NJ,
        joints=list(C.JOINTS),
        raw_dim=RT.RAW_DIM,
        calib=dict(
            height_m=calib.height_m,
            arm_length_m=calib.arm_length_m,
            shoulder_width_m=calib.shoulder_width_m,
        ),
        bone_lengths_m={n_: float(lengths[i]) for i, n_ in enumerate(C.JOINTS)},
        dataset=clip.dataset,
        onnx=str(args.onnx),
        ckpt=str(args.ckpt),
        runtime_vs_torch_max_dpos_m=err,
        tol_pos_m=1e-3,
        tol_rot_deg=0.05,
    )
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print("wrote", out)
    return 0


def run_onnx_times(rt, raw, times, calib, lengths, dataset):
    rt.reset()
    rt.skel_id = None if rt.skel_id is None else (0 if dataset == "cmu" else 1)
    rt.set_calibration((calib.height_m, calib.arm_length_m, calib.shoulder_width_m))
    pos, loc, glob = [], [], []
    for k in range(len(raw)):
        r = rt.step(float(times[k]), raw[k], lengths)
        pos.append(r.positions), loc.append(r.local_rot), glob.append(r.global_rot)
    return np.stack(pos), np.stack(loc), np.stack(glob)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("gate", "fixture"):
        p = sub.add_parser(name)
        p.add_argument("--ckpt", type=Path, required=True)
        p.add_argument("--onnx", type=Path, required=True)
        p.add_argument("--frames", type=int, default=1000)
        p.add_argument("--threads", type=int, default=1)
    g = sub.choices["gate"]
    g.add_argument("--clips", type=int, default=5)
    g.add_argument("--clip-ids", nargs="*")
    g.add_argument("--json")
    f = sub.choices["fixture"]
    f.add_argument("--out", type=Path, required=True)
    f.add_argument("--clip")
    a = ap.parse_args(argv)
    return {"gate": gate, "fixture": fixture}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
