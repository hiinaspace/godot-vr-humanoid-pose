"""Baselines behind the oracle's Predictor interface: GT passthrough variants and Basis."""

from __future__ import annotations

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


_REGISTRY = {"gt": GroundTruth, "gt_rot": GroundTruthRotations, "basis": Basis}


def create(name: str):
    return _REGISTRY[name]()
