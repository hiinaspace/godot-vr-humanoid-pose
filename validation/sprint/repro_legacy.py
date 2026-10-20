"""Reproduce the pre-sprint documented Basis numbers through the oracle's own loaders/metrics.

    uv run python -m validation.sprint.repro_legacy            # ~2 min

Documented (validation/reports, 2026-05-24 / 2026-05-27, macro mean over clips of the old
"upper_body" set = head, chest, spine, shoulders, elbows, hands): CMU all clips 7.67 cm,
Xsens official eval 9.03 cm. The old pipeline used whole-clip GT calibration (p95 head height,
exact bone lengths, floor = 5th percentile of ankle height) and did not drop the Xsens T-pose
reference frame; both are reproduced here with ``legacy=True``. Findings are in ROUNDS.md.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import polars as pl

from validation.sprint import conventions as C
from validation.sprint import oracle as O
from validation.sprint import splits as S
from validation.vr_pose_validation.baseline import create_estimator
from validation.vr_pose_validation.calibration import calibration_from_reference_frames
from validation.vr_pose_validation.geometry import Transform
from validation.vr_pose_validation.metrics import infer_reference_floor_height
from validation.vr_pose_validation.synthetic_trackers import SyntheticTrackerConfig
from validation.vr_pose_validation.types import GroundingContext, TrackedPoseSample, TrackerState


def legacy_clip(args: tuple[str, bool]) -> tuple[float, float, int]:
    clip_id, keep_tpose_frame = args
    path = S.clip_path(clip_id)
    clip = C.load_clip(path)
    if keep_tpose_frame and clip.dataset == "xsens":
        f0 = pl.read_parquet(path).head(1)
        r0 = np.stack(
            [
                f0.select([f"{n}_qx", f"{n}_qy", f"{n}_qz", f"{n}_qw"]).to_numpy()[0]
                for n in C.JOINTS
            ]
        )[None]
        clip.positions = np.concatenate([clip.calib_positions[:1], clip.positions])
        clip.rotations = np.concatenate([r0, clip.rotations])
    inputs = O.make_inputs(clip, "clean")
    frames = [
        {n: Transform(tuple(clip.positions[t, i])) for i, n in enumerate(C.JOINTS)}
        for t in range(clip.n_frames)
    ]
    floor = infer_reference_floor_height(frames)
    cal = calibration_from_reference_frames(frames, floor_height_m=floor)
    grounding = GroundingContext(floor_height_m=floor)
    est = create_estimator("basis_research")
    atts = SyntheticTrackerConfig().attachments
    pos = np.zeros((clip.n_frames, C.NJ, 3))
    for t in range(clip.n_frames):
        states = []
        for k, role in enumerate(("head", "left_hand", "right_hand")):
            x = inputs[t, 7 * k : 7 * k + 7]
            states.append(
                TrackerState(
                    role=role,  # ty: ignore[invalid-argument-type]
                    transform=Transform(tuple(x[:3]), tuple(x[3:])),
                    attachment=atts[role],  # ty: ignore[invalid-argument-type]
                )
            )
        out = est.estimate(TrackedPoseSample(t / C.FPS, *states), cal, grounding)
        for i, n in enumerate(C.JOINTS):
            pos[t, i] = out.joints[n].transform.position
    m = O.clip_metrics(clip, pos, None)
    return float(m["legacy_upper"].mean()), float(m["lower"].mean()), clip.n_frames


def macro(split: str, keep_tpose_frame: bool) -> tuple[float, float]:
    ids = S.clip_ids(split)
    with ProcessPoolExecutor(16) as ex:
        r = np.array(list(ex.map(legacy_clip, [(i, keep_tpose_frame) for i in ids], chunksize=4)))
    return float(r[:, 0].mean() * 100), float(r[:, 1].mean() * 100)


if __name__ == "__main__":
    up, lo = macro("xsens_eval", True)
    print(
        f"xsens_eval legacy calib, T-pose frame KEPT : upper(legacy set) {up:.2f} cm "
        f"lower {lo:.2f}  (documented 9.03 / 27.28)"
    )
    up, lo = macro("xsens_eval", False)
    print(
        f"xsens_eval legacy calib, T-pose frame DROPPED: upper(legacy set) {up:.2f} cm "
        f"lower {lo:.2f}"
    )
    up, lo = macro("cmu_all", False)
    print(
        f"cmu_all    legacy calib, 60 Hz              : upper(legacy set) {up:.2f} cm "
        f"lower {lo:.2f}  (documented 7.67 / 28.6)"
    )
