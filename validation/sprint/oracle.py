"""Frozen evaluation oracle (oracle-v1). LOCKED: see LOCK.md; changes mean oracle-v2.

Predictor interface (model-agnostic)::

    class Predictor(Protocol):
        name: str
        parallel_safe: bool            # True -> may be re-created in worker processes
        def predict_clip(self, inputs: np.ndarray, calib: Calib) -> Prediction: ...

``inputs`` is ``[T, INPUT_DIM]`` (see conventions) at 60 Hz, ``calib`` the static calibration.
A predictor must be causal: frame t may use only ``inputs[: t + 1]`` (the oracle cannot enforce
this; ``windowed_predictor`` is a helper that makes the contract explicit for window models).
``Prediction`` carries global ``positions [T,NJ,3]`` and/or parent-relative ``rotations
[T,NJ,4]`` (incoming-bone convention, root entry global) plus ``root_pos [T,3]``. If positions
are absent the oracle runs forward kinematics from the rotations on a calibration-scaled template.

Calibration: ``make_calib`` measures height, arm length and shoulder width on the clip's
calibration pose (Xsens: the leading T-pose frame; CMU: first 10 frames, rigid lengths only) and
perturbs each by an independent seeded +-5 % factor. Nothing is read from later frames.

Inputs: ``make_inputs`` drives repo's ``synthetic_trackers`` with a per-clip seed derived from
(INPUT_SEED, clip_id, variant). ``clean`` = fixed tracker offset only (the pre-sprint
behaviour); ``corrupted`` = the fitted corruption model (dropout runs, calibration bias,
drift, 1-frame
latency, re-acquire excursions) from ``validation/reports/fitted_tracker_noise.json`` with
Gaussian jitter and the overshoot term disabled (see ``_tracker_config``). The fitted drift
amplitudes (head 3.4 cm, hands 9-14 cm) are documented over-estimates and are used as is.
Tracker rotations are not corrupted by that model.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import time
import zlib
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import numpy as np

from validation.sprint import conventions as C
from validation.sprint import splits as S

ORACLE_VERSION = "oracle-v1"
INPUT_GEN_VERSION = "inputs-v1"
INPUT_SEED = 20261019
CALIB_NOISE = 0.05
MIN_FRAMES = 16
FITTED_NOISE_PATH = S.REPO / "validation/reports/fitted_tracker_noise.json"
VARIANTS = ("clean", "corrupted")


@dataclass(frozen=True, slots=True)
class Calib:
    height_m: float
    arm_length_m: float  # shoulder->elbow->hand, mean of both arms
    shoulder_width_m: float
    floor_y_m: float = 0.0  # VR playspace floor; both datasets have the floor at Y = 0


@dataclass(slots=True)
class Prediction:
    positions: np.ndarray | None = None  # [T, NJ, 3] global
    rotations: np.ndarray | None = None  # [T, NJ, 4] parent-relative; root entry global
    root_pos: np.ndarray | None = None  # [T, 3], required when positions is None


class Predictor(Protocol):
    name: str
    parallel_safe: bool

    def predict_clip(self, inputs: np.ndarray, calib: Calib) -> Prediction: ...


def windowed_predictor(
    fn: Callable[[np.ndarray, Calib], Prediction], window: int = 40
) -> Callable[[np.ndarray, Calib], Prediction]:
    """Adapt ``fn(window[W, INPUT_DIM], calib) -> Prediction (last frame only, T == 1)``
    to a clip predictor with causal left-padding (first frame repeated)."""

    def run(inputs: np.ndarray, calib: Calib) -> Prediction:
        pad = np.repeat(inputs[:1], window - 1, axis=0)
        padded = np.concatenate([pad, inputs])
        outs = [fn(padded[t : t + window], calib) for t in range(len(inputs))]

        def cat(name: str):
            vals = [getattr(o, name) for o in outs]
            return None if vals[0] is None else np.concatenate(vals)

        return Prediction(cat("positions"), cat("rotations"), cat("root_pos"))

    return run


# --- calibration and inputs -----------------------------------------------------------------


def _seed(*parts: object) -> int:
    return zlib.crc32(":".join(str(p) for p in (INPUT_SEED, *parts)).encode())


def make_calib(clip: C.ClipData) -> Calib:
    """Pose-invariant body measurements on the calibration pose, each x (1 +- 5 %).

    Xsens: the BVH's leading T-pose frame (dropped from the evaluated clip). CMU: the first 10
    frames (clips start in arbitrary poses, so only rigid bone lengths are measured, never
    heights above the floor). height = bone chain head-chest-spine-hips-knee-foot + foot height,
    scaled by a per-dataset constant so it matches standing head height (conventions)."""
    p, J = clip.calib_positions, C.J
    dist = lambda a, b: np.median(np.linalg.norm(p[:, J[a]] - p[:, J[b]], axis=-1))  # noqa: E731
    sides = ("left", "right")
    chain = (
        dist("head", "chest")
        + dist("chest", "spine")
        + dist("spine", "hips")
        + C.FOOT_HEIGHT_M
        + 0.5 * sum(dist("hips", f"{s}_knee") + dist(f"{s}_knee", f"{s}_foot") for s in sides)
    )
    height = chain * C.HEIGHT_FROM_CHAIN[clip.dataset]
    arm = 0.5 * sum(
        dist(f"{s}_shoulder", f"{s}_elbow") + dist(f"{s}_elbow", f"{s}_hand") for s in sides
    )
    shoulder = dist("left_shoulder", "right_shoulder")
    rng = np.random.default_rng(_seed(clip.clip_id, "calib"))
    f = 1.0 + rng.uniform(-CALIB_NOISE, CALIB_NOISE, size=3)
    return Calib(float(height * f[0]), float(arm * f[1]), float(shoulder * f[2]))


def _tracker_config(variant: str, clip_id: str):
    from validation.vr_pose_validation.noise_profile_io import (
        config_from_fitted_profiles,
        load_fitted_profiles,
    )
    from validation.vr_pose_validation.synthetic_trackers import SyntheticTrackerConfig

    seed = _seed(clip_id, variant)
    default = SyntheticTrackerConfig()
    if variant == "clean":
        return dataclasses.replace(default, seed=seed, frame_rate_hz=C.FPS)
    # Two terms of the fitted model are switched off: Gaussian position jitter (the memo skips it)
    # and "overshoot" (position += 6 ms x finite-difference acceleration), which turns mocap noise
    # into metre-scale hand steps (3 m on an Xsens eval clip whose clean hand step is 0.2 m).
    profiles = {
        r: dataclasses.replace(p, overshoot_gain_seconds=0.0, position_std_m=0.0)
        for r, p in load_fitted_profiles(FITTED_NOISE_PATH).items()
    }
    cfg = config_from_fitted_profiles(profiles, seed=seed, frame_rate_hz=C.FPS)
    atts = {  # the fitted file carries no tracker mounting offset; keep the clean one
        r: dataclasses.replace(a, local_transform=default.attachments[r].local_transform)
        for r, a in cfg.attachments.items()
        if r in default.attachments  # drops the fitted file's unused 'generic_tracker' role
    }
    return dataclasses.replace(cfg, attachments=atts)


def make_inputs(clip: C.ClipData, variant: str) -> np.ndarray:
    """[T, INPUT_DIM] tracker stream (inputs-v1)."""
    from validation.vr_pose_validation.geometry import Quaternion, Transform, Vector3
    from validation.vr_pose_validation.synthetic_trackers import iter_three_point_samples

    if variant not in VARIANTS:
        raise ValueError(variant)
    cfg = _tracker_config(variant, clip.clip_id)
    names = {n: C.J[n] for n in C.INPUT_JOINTS}
    frames = (
        {
            n: Transform(
                cast("Vector3", tuple(map(float, clip.positions[t, i]))),
                cast("Quaternion", tuple(map(float, clip.rotations[t, i]))),
            )
            for n, i in names.items()
        }
        for t in range(clip.n_frames)
    )
    out = np.zeros((clip.n_frames, C.INPUT_DIM))
    for t, s in enumerate(iter_three_point_samples(frames, cfg)):
        for k, st in enumerate((s.hmd, s.left_hand, s.right_hand)):
            out[t, 7 * k : 7 * k + 3] = st.transform.position
            out[t, 7 * k + 3 : 7 * k + 7] = st.transform.rotation
            out[t, 21 + k] = 1.0 if st.confidence > 0.5 else 0.0
    return out


# --- metrics --------------------------------------------------------------------------------


def jerk_norms(pos: np.ndarray) -> np.ndarray:
    """AGRoL jitter integrand: |3rd finite difference| * fps^3 / 100, shape [T-3, NJ]."""
    d3 = pos[3:] - 3.0 * pos[2:-1] + 3.0 * pos[1:-2] - pos[:-3]
    return np.linalg.norm(d3, axis=-1) * C.FPS**3 / 100.0


def predicted_positions(pred: Prediction, clip: C.ClipData, calib: Calib) -> np.ndarray:
    if pred.positions is not None:
        return np.asarray(pred.positions, dtype=np.float64)
    assert pred.rotations is not None and pred.root_pos is not None
    off = C.template_offsets(clip.dataset, calib.height_m, calib.arm_length_m)
    return C.forward_kinematics(pred.rotations, pred.root_pos, off)


def clip_metrics(clip: C.ClipData, pos: np.ndarray, local_rot: np.ndarray | None) -> dict:
    gt = clip.positions
    assert pos.shape == gt.shape, (pos.shape, gt.shape)
    err = np.linalg.norm(pos - gt, axis=-1)
    rel = np.linalg.norm((pos - pos[:, :1]) - (gt - gt[:, :1]), axis=-1)
    full = err[:, C.NONINPUT_IDX].mean(1)
    cb = pos[:, 1:] - pos[:, C.PARENT[1:]]
    gb = gt[:, 1:] - gt[:, C.PARENT[1:]]
    # angle via atan2(|cross|, dot): accurate at 0 (arccos of a cosine is not)
    bonedir = np.degrees(
        np.arctan2(np.linalg.norm(np.cross(cb, gb), axis=-1), np.sum(cb * gb, -1))
    ).mean(1)
    out = {
        "n": clip.n_frames,
        "full": full,
        "upper": err[:, C.UPPER_IDX].mean(1),
        "lower": err[:, C.LOWER_IDX].mean(1),
        "rootrel": rel[:, C.ROOTREL_IDX].mean(1),
        "hand": err[:, [C.J["left_hand"], C.J["right_hand"]]].mean(1),
        "legacy_upper": err[:, C.LEGACY_UPPER_IDX].mean(1),
        "bonedir": bonedir,
        "joint_err": err.mean(0),
    }
    if local_rot is not None:
        gl = C.local_rotations(clip.canonical_rotations())
        out["mpjre"] = C.qgeodesic_deg(local_rot[:, 1:], gl[:, 1:]).mean(1)
    jp, jg = jerk_norms(pos), jerk_norms(gt)
    out["jit_pred_sum"], out["jit_gt_sum"], out["jit_n"] = float(jp.sum()), float(jg.sum()), jp.size
    out["gt_glitch_frames"] = int((np.linalg.norm(np.diff(gt, axis=0), axis=-1).max(1) > 0.5).sum())
    return out


def evaluate_clip(args: tuple) -> dict | None:
    """Worker: (clip_id, variant, predictor-or-method, max_frames) -> metric dict."""
    clip_id, variant, predictor, max_frames = args
    clip = C.load_clip(S.clip_path(clip_id))
    if max_frames:
        clip.positions, clip.rotations = clip.positions[:max_frames], clip.rotations[:max_frames]
    if clip.n_frames < MIN_FRAMES:
        return None
    if isinstance(predictor, str):
        from validation.sprint import baselines

        predictor = baselines.create(predictor)
    calib, inputs = make_calib(clip), make_inputs(clip, variant)
    if getattr(predictor, "needs_gt", False):
        pred = predictor.predict_clip_with_gt(inputs, calib, clip)
    else:
        pred = predictor.predict_clip(inputs, calib)
    pos = predicted_positions(pred, clip, calib)
    m = clip_metrics(clip, pos, pred.rotations)
    m.update(clip_id=clip_id, activity=clip.activity_class, dataset=clip.dataset)
    return m


# --- aggregation ----------------------------------------------------------------------------

METRIC_KEYS = ("full", "upper", "lower", "rootrel", "hand", "legacy_upper", "bonedir", "mpjre")


def aggregate(results: list[dict]) -> dict:
    """Frame-pooled means (headline) and clip-macro means for the same metrics."""
    n = sum(r["n"] for r in results)
    agg: dict = {"clips": len(results), "frames": n}
    for k in METRIC_KEYS:
        rs = [r for r in results if k in r]
        if not rs:
            continue
        agg[k + "_m"] = float(sum(r[k].sum() for r in rs) / sum(r["n"] for r in rs))
        agg[k + "_macro_m"] = float(np.mean([r[k].mean() for r in rs]))
    allfull = np.concatenate([r["full"] for r in results])
    agg["p95_full_m"] = float(np.percentile(allfull, 95))
    jn = sum(r["jit_n"] for r in results)
    agg["jitter"] = float(sum(r["jit_pred_sum"] for r in results) / jn)
    agg["jitter_gt"] = float(sum(r["jit_gt_sum"] for r in results) / jn)
    agg["gt_glitch_frames"] = int(
        sum(r["gt_glitch_frames"] for r in results)
    )  # GT steps > 0.5 m/frame
    agg["joint_err_m"] = {
        j: float(sum(r["joint_err"][i] * r["n"] for r in results) / n)
        for i, j in enumerate(C.JOINTS)
    }
    return agg


def evaluate_set(
    predictor: Predictor | str, split: str, variant: str, *, workers: int = 1, quick: bool = False
) -> dict:
    ids = S.clip_ids(split)
    max_frames = None
    if quick:
        ids = ids[:: max(1, len(ids) // 6)][:6]
        max_frames = 1800
    t0 = time.perf_counter()
    jobs = [(i, variant, predictor, max_frames) for i in ids]
    if workers > 1:
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

COLS = [
    ("clips", "clips", "{:d}"),
    ("frames", "frames", "{:d}"),
    ("full_m", "MPJPE-full cm", "{:.2f}"),
    ("upper_m", "upper cm", "{:.2f}"),
    ("lower_m", "lower cm", "{:.2f}"),
    ("rootrel_m", "root-rel cm", "{:.2f}"),
    ("mpjre_m", "MPJRE deg", "{:.2f}"),
    ("bonedir_m", "bone-dir deg", "{:.2f}"),
    ("jitter", "jitter (GT)", "{:.2f}"),
    ("p95_full_m", "p95 full cm", "{:.2f}"),
    ("hand_m", "hand cm", "{:.2f}"),
    ("legacy_upper_m", "legacy-upper cm", "{:.2f}"),
]
CM = {"full_m", "upper_m", "lower_m", "rootrel_m", "p95_full_m", "hand_m", "legacy_upper_m"}


def _cell(agg: dict, key: str, fmt: str) -> str:
    if key not in agg:
        return "n/a"
    v = agg[key]
    if key in CM:
        v *= 100.0
    s = fmt.format(v)
    if key == "jitter":
        s += f" ({agg['jitter_gt']:.2f})"
    return s


def render_markdown(method: str, results: list[dict], meta: dict) -> str:
    lines = [
        f"# Oracle table: `{method}` ({ORACLE_VERSION})",
        "",
        f"inputs `{INPUT_GEN_VERSION}` seed {INPUT_SEED}; splits `{S.SPLITS_VERSION}`; "
        f"conventions `{C.CONVENTIONS_VERSION}`; 60 Hz; global frame (head+hands are inputs); "
        "MPJPE over the 11 non-input joints (upper 6, lower 5); root-rel excludes hips; "
        "jitter = mean |3rd diff| x fps^3 in 1e2 m/s^3 over all 14 joints, GT in brackets; "
        "MPJRE over 13 non-root joints (n/a if the method gives no rotations); "
        "bone-dir = angle between predicted and GT parent->child vectors; "
        "legacy-upper = pre-sprint 'upper_body' joint set "
        "(head, chest, spine, shoulders, elbows, hands).",
        "",
        "| set | variant | " + " | ".join(c[1] for c in COLS) + " |",
        "|---|---|" + "---|" * len(COLS),
    ]
    for r in results:
        lines.append(
            f"| {r['split']} | {r['variant']} | "
            + " | ".join(_cell(r, k, f) for k, _, f in COLS)
            + " |"
        )
    for r in results:
        if r["split"].startswith("cmu") and r["per_activity"]:
            lines += [
                "",
                f"## Per-activity ({r['split']}, {r['variant']})",
                "",
                "| activity | clips | frames | full cm | upper cm | lower cm | jitter (GT) |",
                "|---|---|---|---|---|---|---|",
            ]
            for a, ag in r["per_activity"].items():
                lines.append(
                    f"| {a} | {ag['clips']} | {ag['frames']} | {ag['full_m'] * 100:.2f} "
                    f"| {ag['upper_m'] * 100:.2f} "
                    f"| {ag['lower_m'] * 100:.2f} | {ag['jitter']:.2f} ({ag['jitter_gt']:.2f}) |"
                )
    x = meta.get("xsens_eval_overlap")
    if x:
        lines += [
            "",
            f"Note: official Xsens eval shares subjects {x} with official train "
            "(named clips: unknown subject).",
        ]
    lines += [
        "",
        "GT discontinuities (a joint moving > 0.5 m in one frame, kept in the data): "
        + ", ".join(f"{r['split']} {r['gt_glitch_frames']}" for r in results[:: len(VARIANTS)]),
    ]
    lines += [
        "",
        "wall-clock: "
        + ", ".join(f"{r['split']}/{r['variant']} {r['seconds']:.0f}s" for r in results),
    ]
    return "\n".join(lines) + "\n"


def write_report(method: str, results: list[dict], out_dir: Path, tag: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = S.load_splits()
    meta = {
        "oracle": ORACLE_VERSION,
        "inputs": INPUT_GEN_VERSION,
        "seed": INPUT_SEED,
        "splits": S.SPLITS_VERSION,
        "conventions": C.CONVENTIONS_VERSION,
        "fitted_noise_sha256": hashlib.sha256(FITTED_NOISE_PATH.read_bytes()).hexdigest(),
        "xsens_eval_overlap": splits["xsens"]["eval_subject_overlap_with_train"],
        "xsens_val_subject": splits["xsens"]["val_subject"],
    }
    (out_dir / f"{tag}.md").write_text(render_markdown(method, results, meta))
    (out_dir / f"{tag}.json").write_text(
        json.dumps({"method": method, "meta": meta, "results": results}, indent=1)
    )
    return out_dir / f"{tag}.md"
