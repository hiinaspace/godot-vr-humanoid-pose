# ruff: noqa: E501
"""Evaluation oracle v3 = oracle-v2 (imported unchanged) + a per-frame GT-validity mask. LOCKED once tagged.

Some GT clips contain broken frames (Xsens S09 has 118 one-frame discontinuities > 0.5 m, Round 1).
Frames are masked out of every metric:

* a frame is **invalid** if any joint jumps by more than 0.5 m relative to the previous or next frame, or
  any bone length (parent->child distance) deviates more than 35 % from the clip median
  (CMU's virtual joints are not rigid, so a tighter bound flags legitimate poses);
* the invalid mask is dilated by +-5 frames.

Masked frames are excluded from MPJPE-type columns (full / upper / lower / root-rel / hand / MPJRE /
bone-dir / p95 / joint errors). Jitter uses only 4-frame windows that are fully valid; skate / pen /
float use contact frames whose neighbours t-1, t, t+1 are valid (the floor percentile is also taken over
valid frames only); pops use transitions with both frames valid; swivel uses valid frames. The
predictor still sees the whole clip. Per-set and per-clip masked fractions are reported
(``masked_frac``; ``masked_clips`` lists the top clips when a set exceeds 2 %).
The splits are splits-v2 (dedupe) as in Round 3.
"""

from __future__ import annotations

import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from validation.sprint import conventions as C
from validation.sprint import oracle as O
from validation.sprint import oracle_v2 as O2
from validation.sprint import rest as R
from validation.sprint import splits as S

ORACLE_VERSION = "oracle-v3"
CanonPrediction = O2.CanonPrediction
JUMP_M, BONE_TOL, DILATE, WARN_FRAC = 0.5, 0.35, 5, 0.02


def valid_mask(pos: np.ndarray) -> np.ndarray:
    """[T] bool, True = GT frame usable (see module doc)."""
    bad = np.zeros(len(pos), bool)
    step = np.linalg.norm(pos[1:] - pos[:-1], axis=-1).max(1) > JUMP_M
    bad[1:] |= step
    bad[:-1] |= step
    bl = np.linalg.norm(pos[:, 1:] - pos[:, C.PARENT[1:]], axis=-1)
    med = np.median(bl, axis=0)
    bad |= (np.abs(bl - med) > BONE_TOL * np.maximum(med, 1e-6)).any(1)
    if bad.any():
        k = np.ones(2 * DILATE + 1)
        bad = np.convolve(bad.astype(float), k, mode="same") > 0
    return ~bad


def _hspeed(p):
    return O2._hspeed(p)


def contact_metrics(gt, pr, ok):
    gf, pf = gt[:, O2.FEET], pr[:, O2.FEET]
    okc = ok[1:-1] & ok[:-2] & ok[2:]
    lows = gf[..., 1].min(1)
    floor = np.percentile(lows[ok] if ok.any() else lows, 2)
    contact = (gf[1:-1, :, 1] < floor + O2.CONTACT_H) & (_hspeed(gf) < O2.CONTACT_V) & okc[:, None]
    skate = _hspeed(pf)[contact] * 100.0
    signed = (pf[1:-1, :, 1] - floor)[contact] * 100.0
    return {
        "skate_vals": skate.astype(np.float32),
        "ground_vals": signed.astype(np.float32),
        "skate_clip": float(skate.mean()) if skate.size else np.nan,
    }


def pop_count(gt, pr, ok):
    lg, lp = (
        R.swing_from_positions(gt)[:, O2.POP_JOINTS],
        R.swing_from_positions(pr)[:, O2.POP_JOINTS],
    )
    dg = C.qgeodesic_deg(lg[1:], lg[:-1])
    dp = C.qgeodesic_deg(lp[1:], lp[:-1])
    ev = (dp > O2.POP_PRED_DEG) & (dg < O2.POP_GT_DEG) & (ok[1:] & ok[:-1])[:, None]
    start = ev.copy()
    start[1:] &= ~ev[:-1]
    return int(start.sum())


def swivel_errors(gt, pr, ok):
    errs, total = [], 0
    for side in ("left", "right"):
        ag, flex, vert = O2._swivel(gt, side)
        ap, _, _ = O2._swivel(pr, side)
        sel = (flex > O2.SWIVEL_MIN_FLEX_DEG) & (vert >= O2.SWIVEL_MIN_VERT_DEG) & ok
        d = (ap - ag + 180.0) % 360.0 - 180.0
        errs.append(np.abs(d[sel]))
        total += int(ok.sum())
    return np.concatenate(errs).astype(np.float32), total


def clip_metrics(clip: C.ClipData, pos: np.ndarray, pred: O.Prediction) -> dict:
    ok = valid_mask(clip.positions)
    m = O2.clip_metrics(clip, pos, pred)  # full-clip per-frame arrays (v1 + v2 keys)
    for k in ("full", "upper", "lower", "rootrel", "hand", "legacy_upper", "bonedir", "mpjre"):
        if k in m:
            m[k] = m[k][ok]
    err = np.linalg.norm(pos - clip.positions, axis=-1)
    m["joint_err"] = err[ok].mean(0) if ok.any() else np.zeros(C.NJ)
    n_all = clip.n_frames
    m["n"] = int(ok.sum())
    # jitter: only fully valid 4-frame windows
    wok = ok[3:] & ok[2:-1] & ok[1:-2] & ok[:-3]
    jp, jg = O.jerk_norms(pos)[wok], O.jerk_norms(clip.positions)[wok]
    m["jit_pred_sum"], m["jit_gt_sum"], m["jit_n"] = float(jp.sum()), float(jg.sum()), jp.size
    m.update(contact_metrics(clip.positions, pos, ok))
    m["pops"] = pop_count(clip.positions, pos, ok)
    m["swivel_vals"], m["swivel_frames"] = swivel_errors(clip.positions, pos, ok)
    m["n_total"] = n_all
    return m


def evaluate_clip(args: tuple) -> dict | None:
    clip_id, variant, predictor, max_frames = args
    clip = C.load_clip(S.clip_path(clip_id))
    if max_frames:
        clip.positions, clip.rotations = clip.positions[:max_frames], clip.rotations[:max_frames]
    if clip.n_frames < O.MIN_FRAMES:
        return None
    if isinstance(predictor, str):
        from validation.sprint import baselines

        predictor = baselines.create(predictor)
    calib, inputs = O.make_calib(clip), O.make_inputs(clip, variant)
    if getattr(predictor, "needs_gt", False):
        pred = predictor.predict_clip_with_gt(inputs, calib, clip)
    elif getattr(predictor, "needs_meta", False):
        pred = predictor.predict_clip_meta(inputs, calib, {"dataset": clip.dataset})
    else:
        pred = predictor.predict_clip(inputs, calib)
    m = clip_metrics(clip, O2.predicted_positions(pred, clip, calib), pred)
    m.update(clip_id=clip_id, activity=clip.activity_class, dataset=clip.dataset)
    return m


def aggregate(results: list[dict]) -> dict:
    results = [r for r in results if r["n"] > 0]
    agg = O2.aggregate(results)
    tot = sum(r["n_total"] for r in results)
    agg["masked_frac"] = float(1 - sum(r["n"] for r in results) / max(1, tot))
    agg["frames_total"] = int(tot)
    return agg


def evaluate_set(
    predictor, split: str, variant: str, *, workers: int = 1, quick: bool = False
) -> dict:
    ids = S.clip_ids(split)
    max_frames = None
    if quick:
        ids = ids[:: max(1, len(ids) // 6)][:6]
        max_frames = 1800
    t0 = time.perf_counter()
    jobs = [(i, variant, predictor, max_frames) for i in ids]
    if workers > 1 and (isinstance(predictor, str) or getattr(predictor, "parallel_safe", False)):
        with ProcessPoolExecutor(workers) as ex:
            results = list(ex.map(evaluate_clip, jobs, chunksize=1))
    else:
        results = [evaluate_clip(j) for j in jobs]
    kept = [r for r in results if r is not None and r["n"] > 0]
    agg = aggregate(kept)
    by_act: dict[str, list[dict]] = defaultdict(list)
    for r in kept:
        by_act[r["activity"]].append(r)
    agg["per_activity"] = {a: aggregate(rs) for a, rs in sorted(by_act.items())}
    agg["per_clip"] = [
        {
            "clip": r["clip_id"],
            "frames": r["n"],
            "masked_frac": float(1 - r["n"] / r["n_total"]),
            "activity": r["activity"],
            "full_cm": float(r["full"].mean() * 100),
            "upper_cm": float(r["upper"].mean() * 100),
            "lower_cm": float(r["lower"].mean() * 100),
        }
        for r in kept
    ]
    if agg["masked_frac"] > WARN_FRAC:
        agg["masked_clips"] = sorted(
            ((c["masked_frac"], c["clip"]) for c in agg["per_clip"] if c["masked_frac"] > 0),
            reverse=True,
        )[:10]
    agg["skipped_short_clips"] = len(results) - len(kept)
    agg["seconds"] = time.perf_counter() - t0
    agg["split"], agg["variant"], agg["quick"] = split, variant, quick
    print(
        f"  masked {100 * agg['masked_frac']:.2f} % of {agg['frames_total']} frames in {split}",
        flush=True,
    )
    return agg


COLS = O2.COLS + [("masked_frac", "masked %", "{:.2f}")]


def render_markdown(method: str, results: list[dict]) -> str:
    lines = [
        f"# Oracle table: `{method}` ({ORACLE_VERSION}; v2 columns + GT-validity mask)",
        "",
        "Frames with a GT joint jump > 0.5 m or a bone > 35 % off the clip median (dilated +-5 frames) are excluded "
        "from every metric; `masked %` is the excluded fraction of the set. Rotation-output methods are placed with "
        "the clip's GT bone lengths unless noted.",
        "",
        "| set | variant | " + " | ".join(c[1] for c in COLS) + " |",
        "|---|---|" + "---|" * len(COLS),
    ]
    for r in results:
        cells = [
            O._cell(r, k, f) if k != "masked_frac" else f"{100 * r['masked_frac']:.2f}"
            for k, _, f in COLS
        ]
        lines.append(f"| {r['split']} | {r['variant']} | " + " | ".join(cells) + " |")
    for r in results:
        if "masked_clips" in r:
            lines.append(
                f"\nmasked > 2 % in `{r['split']}`: "
                + ", ".join(f"{c} ({100 * f:.0f} %)" for f, c in r["masked_clips"][:6])
            )
    lines += [
        "",
        "wall-clock: "
        + ", ".join(f"{r['split']}/{r['variant']} {r['seconds']:.0f}s" for r in results),
    ]
    return "\n".join(lines) + "\n"


def write_report(
    method: str, results: list[dict], out_dir: Path, tag: str, extra: dict | None = None
) -> Path:
    import json

    p = O2.write_report(method, results, out_dir, tag, extra)  # json with meta
    meta = json.loads((out_dir / f"{tag}.json").read_text())["meta"]
    meta.update(oracle=ORACLE_VERSION, v2="oracle-v2")
    (out_dir / f"{tag}.md").write_text(render_markdown(method, results))
    (out_dir / f"{tag}.json").write_text(
        json.dumps({"method": method, "meta": meta, "results": results}, indent=1)
    )
    return p
