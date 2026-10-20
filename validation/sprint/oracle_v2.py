# ruff: noqa: E501
"""Evaluation oracle v2 = oracle-v1 (imported unchanged) + extra columns. LOCKED once tagged.

Adds, without touching any v1 column or the v1 input generator / splits / conventions:

(a) Rotation-output methods return a ``CanonPrediction`` (local rotations in the canonical rest pose
    of ``rest.py``). Their positions come from FK with the clip's GT per-bone lengths (median over the
    clip) instead of the three noisy calibration scalars: the avatar's skeleton equals the user's, as
    AvatarPoser / HMD-Poser evaluate with GT body shape. The model's *input* calibration is unchanged.
    The FK floor of this path (GT canonical rotations, median lengths) is the ``gt_canon`` method.
(b) Perceptual columns (perceptual-eval memo 3a), all contact / validity labels from GT, never from
    the prediction. Adaptation to the 14-joint skeleton (no toe, no pelvis/upper-leg joints):
    * contact joint = ``left_foot`` / ``right_foot`` (ankle for Xsens, mid-foot end for CMU). Floor =
      2nd percentile of the lowest GT foot-joint height in the clip. Contact frame: GT foot height <
      floor + 5 cm and GT horizontal speed (central difference) < 10 cm/s.
    * ``skate_cm_s``: mean horizontal speed of the predicted foot over GT-contact frames, mean over
      clips (clips without contact skipped); ``skate_p95`` pooled over contact frames.
    * ``pen_p95`` / ``float_p95`` (cm): 95th percentile of max(0, floor - h) and max(0, h - floor), h =
      predicted foot height over GT-contact frames. GT itself floats up to 5 cm by definition, so the
      GT row is reported, not required to be 0.
    * ``pop_per_min``: rotations are swing-only rotations derived from *positions* (twist is invisible
      in positions, and it makes position-only methods such as Basis comparable). Joints = shoulder,
      elbow, hip, knee, i.e. the incoming-bone locals of left/right_elbow, left/right_hand,
      left/right_knee, left/right_foot. Event = predicted per-frame geodesic change > 20 deg while the
      GT change < 5 deg; consecutive frames of one joint merge into one event.
    * ``swivel_err_deg``: angle of the shoulder-elbow-wrist plane about the shoulder->wrist axis, from
      the plane containing that axis and the pose's own torso-down vector (chest -> hips). Error =
      |wrap(pred - GT)| on frames where GT elbow flexion > 20 deg AND the GT arm axis is >= 15 deg
      away from vertical (with a vertical arm "torso down" does not define a plane). Reports mean and
      p95 over both arms.
    ``mpjre`` for canonical predictions is measured against the GT canonical local rotations.
"""

from __future__ import annotations

import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from validation.sprint import conventions as C
from validation.sprint import oracle as O
from validation.sprint import rest as R
from validation.sprint import splits as S

ORACLE_VERSION = "oracle-v2"
Prediction = O.Prediction
VARIANTS = O.VARIANTS

FEET = np.array([C.J["left_foot"], C.J["right_foot"]])
POP_JOINTS = np.array(
    [C.J[n] for n in ("left_elbow", "left_hand", "right_elbow", "right_hand")]
    + [C.J[n] for n in ("left_knee", "left_foot", "right_knee", "right_foot")]
)
CONTACT_H, CONTACT_V = 0.05, 0.10
POP_PRED_DEG, POP_GT_DEG = 20.0, 5.0
SWIVEL_MIN_FLEX_DEG, SWIVEL_MIN_VERT_DEG = 20.0, 15.0


@dataclass(slots=True)
class CanonPrediction(Prediction):
    """Local rotations in the canonical rest pose (rest.py); root entry global; ``root_pos`` set."""


def predicted_positions(pred: Prediction, clip: C.ClipData, calib: O.Calib) -> np.ndarray:
    if isinstance(pred, CanonPrediction):
        assert pred.rotations is not None and pred.root_pos is not None
        return R.fk(pred.rotations, pred.root_pos, R.bone_lengths(clip.positions))
    return O.predicted_positions(pred, clip, calib)


# --- perceptual columns ---------------------------------------------------------------------


def _unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def _hspeed(p: np.ndarray) -> np.ndarray:
    v = (p[2:] - p[:-2]) * (C.FPS / 2.0)
    return np.linalg.norm(v[..., [0, 2]], axis=-1)


def contact_metrics(gt: np.ndarray, pr: np.ndarray) -> dict:
    gf, pf = gt[:, FEET], pr[:, FEET]
    floor = np.percentile(gf[..., 1].min(1), 2)
    contact = (gf[1:-1, :, 1] < floor + CONTACT_H) & (_hspeed(gf) < CONTACT_V)
    skate = _hspeed(pf)[contact] * 100.0
    signed = (pf[1:-1, :, 1] - floor)[contact] * 100.0
    return {
        "skate_vals": skate.astype(np.float32),
        "ground_vals": signed.astype(np.float32),
        "skate_clip": float(skate.mean()) if skate.size else np.nan,
    }


def pop_count(gt: np.ndarray, pr: np.ndarray) -> int:
    lg, lp = R.swing_from_positions(gt)[:, POP_JOINTS], R.swing_from_positions(pr)[:, POP_JOINTS]
    dg = C.qgeodesic_deg(lg[1:], lg[:-1])
    dp = C.qgeodesic_deg(lp[1:], lp[:-1])
    ev = (dp > POP_PRED_DEG) & (dg < POP_GT_DEG)
    start = ev.copy()
    start[1:] &= ~ev[:-1]
    return int(start.sum())


def _swivel(p: np.ndarray, side: str):
    sh, el, wr = (p[:, C.J[f"{side}_{n}"]] for n in ("shoulder", "elbow", "hand"))
    a = _unit(wr - sh)
    down = _unit(p[:, C.J["hips"]] - p[:, C.J["chest"]])
    ref = down - np.sum(down * a, -1, keepdims=True) * a
    e = (el - sh) - np.sum((el - sh) * a, -1, keepdims=True) * a
    ang = np.degrees(np.arctan2(np.sum(a * np.cross(ref, e), -1), np.sum(ref * e, -1)))
    flex = 180.0 - np.degrees(
        np.arccos(np.clip(np.sum(_unit(sh - el) * _unit(wr - el), -1), -1, 1))
    )
    vert = np.degrees(np.arccos(np.clip(np.abs(np.sum(a * down, -1)), 0, 1)))
    return ang, flex, vert


def swivel_errors(gt: np.ndarray, pr: np.ndarray) -> tuple[np.ndarray, int]:
    errs, total = [], 0
    for side in ("left", "right"):
        ag, flex, vert = _swivel(gt, side)
        ap, _, _ = _swivel(pr, side)
        ok = (flex > SWIVEL_MIN_FLEX_DEG) & (vert >= SWIVEL_MIN_VERT_DEG)
        d = (ap - ag + 180.0) % 360.0 - 180.0
        errs.append(np.abs(d[ok]))
        total += len(d)
    return np.concatenate(errs).astype(np.float32), total


def clip_metrics(clip: C.ClipData, pos: np.ndarray, pred: Prediction) -> dict:
    canon = isinstance(pred, CanonPrediction)
    m = O.clip_metrics(clip, pos, None if canon else pred.rotations)
    if canon and pred.rotations is not None:
        gl = R.canonical_local(clip)
        m["mpjre"] = C.qgeodesic_deg(pred.rotations[:, 1:], gl[:, 1:]).mean(1)
    m.update(contact_metrics(clip.positions, pos))
    m["pops"] = pop_count(clip.positions, pos)
    m["swivel_vals"], m["swivel_frames"] = swivel_errors(clip.positions, pos)
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
    m = clip_metrics(clip, predicted_positions(pred, clip, calib), pred)
    m.update(clip_id=clip_id, activity=clip.activity_class, dataset=clip.dataset)
    return m


# --- aggregation ----------------------------------------------------------------------------


def aggregate(results: list[dict]) -> dict:
    agg = O.aggregate(results)
    sk = [r["skate_clip"] for r in results if not np.isnan(r["skate_clip"])]
    agg["skate_cm_s"] = float(np.mean(sk)) if sk else float("nan")
    sv = np.concatenate([r["skate_vals"] for r in results])
    gv = np.concatenate([r["ground_vals"] for r in results])
    agg["skate_p95"] = float(np.percentile(sv, 95)) if sv.size else float("nan")
    agg["pen_p95_cm"] = float(np.percentile(np.maximum(0, -gv), 95)) if gv.size else float("nan")
    agg["float_p95_cm"] = float(np.percentile(np.maximum(0, gv), 95)) if gv.size else float("nan")
    agg["contact_frames"] = int(sv.size)
    agg["pop_per_min"] = float(sum(r["pops"] for r in results) / (agg["frames"] / C.FPS / 60.0))
    ev = np.concatenate([r["swivel_vals"] for r in results])
    agg["swivel_err_deg"] = float(ev.mean()) if ev.size else float("nan")
    agg["swivel_p95_deg"] = float(np.percentile(ev, 95)) if ev.size else float("nan")
    agg["swivel_used_frac"] = float(ev.size / max(1, sum(r["swivel_frames"] for r in results)))
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
    kept = [r for r in results if r is not None]
    agg = aggregate(kept)
    by_act: dict[str, list[dict]] = defaultdict(list)
    for r in kept:
        by_act[r["activity"]].append(r)
    agg["per_activity"] = {a: aggregate(rs) for a, rs in sorted(by_act.items())}
    agg["per_clip"] = [
        {
            "clip": r["clip_id"],
            "frames": r["n"],
            "activity": r["activity"],
            "full_cm": float(r["full"].mean() * 100),
            "upper_cm": float(r["upper"].mean() * 100),
            "lower_cm": float(r["lower"].mean() * 100),
        }
        for r in kept
    ]
    agg["skipped_short_clips"] = len(results) - len(kept)
    agg["seconds"] = time.perf_counter() - t0
    agg["split"], agg["variant"], agg["quick"] = split, variant, quick
    return agg


# --- reporting ------------------------------------------------------------------------------

COLS = O.COLS + [
    ("skate_cm_s", "skate cm/s", "{:.2f}"),
    ("pen_p95_cm", "pen p95 cm", "{:.2f}"),
    ("float_p95_cm", "float p95 cm", "{:.2f}"),
    ("pop_per_min", "pops/min", "{:.2f}"),
    ("swivel_err_deg", "swivel deg", "{:.1f}"),
]


def render_markdown(method: str, results: list[dict]) -> str:
    lines = [
        f"# Oracle table: `{method}` ({ORACLE_VERSION}; v1 columns unchanged, see oracle_v2.py)",
        "",
        "Columns as oracle-v1 plus skate (cm/s, foot speed on GT contact frames), pen/float p95 "
        "(cm, foot height vs GT-derived floor on GT contact frames), pops per minute (>20 deg "
        "per-frame swing change where GT <5 deg), swivel error (deg, GT-gated). Rotation-output "
        "methods are placed with the clip's GT bone lengths.",
        "",
        "| set | variant | " + " | ".join(c[1] for c in COLS) + " |",
        "|---|---|" + "---|" * len(COLS),
    ]
    for r in results:
        lines.append(
            f"| {r['split']} | {r['variant']} | "
            + " | ".join(O._cell(r, k, f) for k, _, f in COLS)
            + " |"
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

    out_dir.mkdir(parents=True, exist_ok=True)
    O.write_report(method, results, out_dir, tag)  # v1 writer: json with meta; overwritten below
    meta = json.loads((out_dir / f"{tag}.json").read_text())["meta"]
    meta.update(oracle=ORACLE_VERSION, v1="oracle-v1", rest=R.REST_VERSION, **(extra or {}))
    (out_dir / f"{tag}.md").write_text(render_markdown(method, results))
    (out_dir / f"{tag}.json").write_text(
        json.dumps({"method": method, "meta": meta, "results": results}, indent=1)
    )
    return out_dir / f"{tag}.md"
