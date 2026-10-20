# ruff: noqa: E501
"""Baselines behind the oracle's Predictor interface: GT passthrough variants and Basis."""

from __future__ import annotations

import json
from typing import cast

import numpy as np

from validation.sprint import conventions as C
from validation.sprint.oracle import Calib, Prediction


class GroundTruth:
    """Returns the ground truth (positions + canonical rotations). Must score 0 everywhere."""

    name = "gt"
    parallel_safe = True
    needs_gt = True

    def predict_clip(self, inputs, calib):  # pragma: no cover - the oracle calls the _with_gt form
        raise RuntimeError("GroundTruth needs the clip")

    def predict_clip_with_gt(self, inputs, calib, clip: C.ClipData) -> Prediction:
        return Prediction(
            positions=clip.positions.copy(),
            rotations=C.local_rotations(clip.canonical_rotations()),
            root_pos=clip.positions[:, 0].copy(),
        )


class GroundTruthRotations(GroundTruth):
    """GT rotations through the oracle's calibration-scaled template FK (no GT positions).

    Not a method anyone ships: it measures the floor that template FK puts under any
    rotation-output model (bone lengths come from 3 noisy calibration scalars)."""

    name = "gt_rot"

    def predict_clip_with_gt(self, inputs, calib, clip):
        return Prediction(
            positions=None,
            rotations=C.local_rotations(clip.canonical_rotations()),
            root_pos=clip.positions[:, 0].copy(),
        )


class GroundTruthCanon(GroundTruth):
    """GT canonical-rest rotations through oracle-v2 FK with GT bone lengths (oracle-v2 only).

    Measures the FK floor of the rotation path (clip-median lengths vs per-frame lengths)."""

    name = "gt_canon"

    def predict_clip_with_gt(self, inputs, calib, clip):
        from validation.sprint import oracle_v2, rest

        return oracle_v2.CanonPrediction(
            rotations=rest.canonical_local(clip), root_pos=clip.positions[:, 0].copy()
        )


class Basis:
    """The in-repo procedural Basis port (``basis_research``), untuned and run as is."""

    name = "basis"
    parallel_safe = True
    needs_gt = False
    ESTIMATOR = "basis_research"
    # Basis's own default arm proportions; used to split the measured arm length.
    UPPER_ARM_RATIO, LOWER_ARM_RATIO = 0.174, 0.151

    def predict_clip(self, inputs: np.ndarray, calib: Calib) -> Prediction:
        from validation.vr_pose_validation.baseline import EstimatorName, create_estimator
        from validation.vr_pose_validation.geometry import Transform
        from validation.vr_pose_validation.synthetic_trackers import SyntheticTrackerConfig
        from validation.vr_pose_validation.types import (
            AvatarCalibration,
            GroundingContext,
            TrackedPoseSample,
            TrackerRole,
            TrackerState,
        )

        atts = SyntheticTrackerConfig().attachments
        arm_ratio = self.UPPER_ARM_RATIO / (self.UPPER_ARM_RATIO + self.LOWER_ARM_RATIO)
        cal = AvatarCalibration(
            bone_lengths_m={
                "upper_arm": calib.arm_length_m * arm_ratio,
                "lower_arm": calib.arm_length_m * (1 - arm_ratio),
                "shoulder_width": calib.shoulder_width_m,
            },
            human_height_m=calib.height_m,
        )
        grounding = GroundingContext(floor_height_m=calib.floor_y_m)
        est = create_estimator(cast("EstimatorName", self.ESTIMATOR))
        roles: tuple[TrackerRole, ...] = ("head", "left_hand", "right_hand")
        T = inputs.shape[0]
        pos = np.zeros((T, C.NJ, 3))
        names = [n for n in C.JOINTS]
        for t in range(T):
            states = []
            for k, role in enumerate(roles):
                x = inputs[t, 7 * k : 7 * k + 7]
                valid = bool(inputs[t, 21 + k] > 0.5)
                states.append(
                    TrackerState(
                        role=role,
                        transform=Transform(tuple(x[:3]), tuple(x[3:])),
                        confidence=1.0 if valid else 0.0,
                        valid=valid,
                        attachment=atts[role],
                    )
                )
            sample = TrackedPoseSample(t / C.FPS, *states)
            frame = est.estimate(sample, cal, grounding)
            for i, n in enumerate(names):
                pos[t, i] = frame.joints[n].transform.position
        return Prediction(positions=pos)


def _torso_frame(pos: np.ndarray) -> np.ndarray:
    """[T,3,3] columns (lateral, up, forward) from a predicted pose: lateral = left - right shoulder,
    up = head - hips; re-orthogonalised. Uses only Basis's own output."""
    j = C.J
    x = pos[:, j["left_shoulder"]] - pos[:, j["right_shoulder"]]
    y = pos[:, j["head"]] - pos[:, j["hips"]]
    x = x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)
    y = y - np.sum(y * x, -1, keepdims=True) * x
    y = y / np.maximum(np.linalg.norm(y, axis=-1, keepdims=True), 1e-9)
    z = np.cross(x, y)
    return np.stack([x, y, z], -1)


def _fit_one(args):
    (cid,) = args
    from validation.sprint import oracle as O
    from validation.sprint import oracle_v3 as O3
    from validation.sprint import splits as S

    clip = C.load_clip(S.clip_path(cid))
    inputs, calib = O.make_inputs(clip, "clean"), O.make_calib(clip)
    pred = Basis().predict_clip(inputs, calib).positions
    ok = O3.valid_mask(clip.positions)
    Rm = _torso_frame(pred)
    res = np.einsum("tji,tkj->tki", Rm, clip.positions - pred)[ok]  # residual in the torso frame
    return res.sum(0), int(ok.sum())


class BasisDebiased(Basis):
    """Basis plus a constant per-dataset, per-joint offset in Basis's own torso frame, fitted on the
    train split (mean residual, <= 200 clips per dataset). Removes the dataset skeleton-convention bias
    (Basis's fixed torso ratios vs each dataset's joint definitions) so the learned model is compared
    against the best a constant-offset Basis can do. Inputs are never used for the fit beyond Basis's own
    output; offsets are fixed constants at eval time."""

    name = "basis_debiased"
    needs_meta = True
    CACHE = None

    def __init__(self):
        self.off = self._offsets()

    @classmethod
    def _offsets(cls):
        from validation.sprint import splits as S

        path = S.REPO / ".cache/sprint/basis_debias_v1.json"
        if path.exists():
            return {k: np.array(v) for k, v in json.loads(path.read_text()).items()}
        from concurrent.futures import ProcessPoolExecutor

        doc = S.load_splits()
        out = {}
        for ds, ids in (
            ("cmu", doc["cmu"]["clips"]["train"]),
            ("xsens", [c for c in doc["xsens"]["clips"]["train"] if not c.endswith("_mirror")]),
        ):
            sel = ids[:: max(1, len(ids) // 200)][:200]
            with ProcessPoolExecutor(8) as ex:
                parts = list(ex.map(_fit_one, [(c,) for c in sel]))
            tot = sum(p[0] for p in parts) / sum(p[1] for p in parts)
            tot[[C.J[n] for n in C.INPUT_JOINTS]] = 0.0  # inputs are placed by the trackers
            out[ds] = tot
        path.write_text(json.dumps({k: v.tolist() for k, v in out.items()}))
        return out

    def predict_clip_meta(self, inputs, calib, meta):
        pos = super().predict_clip(inputs, calib).positions
        Rm = _torso_frame(pos)
        return Prediction(positions=pos + np.einsum("tij,kj->tki", Rm, self.off[meta["dataset"]]))


_REGISTRY = {
    "gt": GroundTruth,
    "gt_rot": GroundTruthRotations,
    "gt_canon": GroundTruthCanon,
    "basis": Basis,
    "basis_debiased": BasisDebiased,
}


def create(name: str):
    return _REGISTRY[name]()
