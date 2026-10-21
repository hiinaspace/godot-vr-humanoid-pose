# ruff: noqa: E501
"""One-Euro output filter sweep: jitter ratio, MPJPE and measured added latency (frames at 60 Hz).

``python -m validation.sprint.r3_latency CKPT [--out FILE]``: runs the model on a val subsample
(CMU val every 8th clip, <= 4000 frames; all S09 clips, <= 4000 frames), applies
``post.one_euro(min_cutoff, beta)`` to the model's local rotations and root position, FKs with the
clip's GT lengths (as the oracle does) and reports for each setting: jitter / GT jitter, MPJPE upper /
lower, and the added latency = the fractional delay s that makes the *unfiltered* positions delayed by
s frames best match the filtered ones (min mean joint distance; linear interpolation).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from validation.sprint import conventions as C
from validation.sprint import oracle as O
from validation.sprint import post as P
from validation.sprint import rest as R
from validation.sprint import splits as S
from validation.sprint import train as T

GRID = [(c, b) for c in (3.0, 6.0, 12.0) for b in (0.0, 0.05, 0.3)]


def delayed(pos: np.ndarray, s: float) -> np.ndarray:
    i = np.arange(len(pos)) - s
    lo = np.clip(np.floor(i).astype(int), 0, len(pos) - 1)
    hi = np.clip(lo + 1, 0, len(pos) - 1)
    w = (np.clip(i, 0, len(pos) - 1) - lo)[:, None, None]
    return pos[lo] * (1 - w) + pos[hi] * w


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--variant", default="clean")
    ap.add_argument("--dev", default="cuda")
    args = ap.parse_args()
    import torch

    torch.set_num_threads(3)
    pred = T.ModelPredictor.load(args.ckpt, dev=args.dev)
    ids = S.clip_ids("cmu_val")[::8] + S.clip_ids("xsens_val")
    acc = {(c, b): dict(jp=0.0, jg=0.0, n=0, up=[], lo=[], lag=[]) for c, b in [(0, 0), *GRID]}
    for cid in ids:
        clip = C.load_clip(S.clip_path(cid))
        n = min(clip.n_frames, 4000)
        clip.positions, clip.rotations = clip.positions[:n], clip.rotations[:n]
        if n < O.MIN_FRAMES:
            continue
        calib, inp = O.make_calib(clip), O.make_inputs(clip, args.variant)
        p = pred.predict_clip_meta(inp, calib, {"dataset": clip.dataset})
        ln = R.bone_lengths(clip.positions)
        raw = R.fk(p.rotations, p.root_pos, ln)
        jg = O.jerk_norms(clip.positions)
        for key in acc:
            if key == (0, 0):
                pos = raw
            else:
                rot, root = P.one_euro(p.rotations, p.root_pos, *key)
                pos = R.fk(rot, root, ln)
            a = acc[key]
            a["jp"] += O.jerk_norms(pos).sum()
            a["jg"] += jg.sum()
            a["n"] += jg.size
            e = np.linalg.norm(pos - clip.positions, axis=-1)
            a["up"].append(e[:, C.UPPER_IDX].mean())
            a["lo"].append(e[:, C.LOWER_IDX].mean())
            if key != (0, 0):
                lags = np.arange(0, 8.01, 0.25)
                errs = [np.linalg.norm(delayed(raw, s)[8:] - pos[8:], axis=-1).mean() for s in lags]
                a["lag"].append(float(lags[int(np.argmin(errs))]))
    lines = [
        "| min cutoff Hz | beta | jitter / GT | MPJPE upper cm | lower cm | added latency (frames) | (ms) |",
        "|---|---|---|---|---|---|---|",
    ]
    out = []
    for key, a in acc.items():
        lag = float(np.mean(a["lag"])) if a["lag"] else 0.0
        row = dict(
            cutoff=key[0],
            beta=key[1],
            jit_ratio=a["jp"] / a["jg"],
            upper_cm=100 * float(np.mean(a["up"])),
            lower_cm=100 * float(np.mean(a["lo"])),
            lag_frames=lag,
        )
        out.append(row)
        name = "none" if key == (0, 0) else f"{key[0]:g}"
        lines.append(
            f"| {name} | {key[1]:g} | {row['jit_ratio']:.2f} | {row['upper_cm']:.2f} | {row['lower_cm']:.2f} | {lag:.2f} | {lag * 1000 / 60:.0f} |"
        )
    print("\n".join(lines))
    if args.out:
        args.out.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
