"""``learned`` live estimator: ONNX HMD-Poser-design model behind the live-capture sidecar.

Registered next to ``basis_research`` (``baseline.create_estimator("learned")``). It is a
``PoseEstimator`` with per-session state (the tracker history ring buffer), reset on
``reset_estimator`` and when message timestamps jump backwards. Torch is not imported: the graph
is run by onnxruntime (``validation/sprint/runtime.py``). Export one with
``python -m validation.sprint.export_onnx``; see ``validation/sprint/RUNTIME.md``.

Frames. ``frame="godot"`` (default) adapts live Godot samples to the training convention: the
head transform already carries the +Z-forward fix (``_estimator_head_transform`` in
live_overlay_debug.gd); the hand transforms are Godot grip poses (fingers along local -Z), which
are re-expressed in the training hand frame (forearm axis +X on the left hand, -X on the right:
``rest.forearm_probe``) by a fixed +-90 degree yaw. That hand-frame mapping is the one input
convention that has not been verified on a headset. ``frame="model"`` feeds the samples through
untouched (replay / parity tests).

Outputs are the 14 canonical joints (``conventions.JOINTS``) with world positions and the world
rotation of each joint in the rest.py canonical rest pose (not avatar-bone rotations), plus the
debug-overlay names Godot expects (``neck``).
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from validation.sprint import runtime as RT
from validation.vr_pose_validation.geometry import Transform
from validation.vr_pose_validation.types import (
    AvatarCalibration,
    GroundingContext,
    HumanoidPoseFrame,
    JointPose,
    TrackedPoseSample,
)

# population ratios behind train.NORM (height 1.7 m, arm 0.58 m, shoulder width 0.35 m)
ARM_PER_HEIGHT = 0.58 / 1.7
SHOULDER_PER_HEIGHT = 0.35 / 1.7

CONFIG: dict = dict(
    model_path=os.environ.get("VRHP_LEARNED_MODEL", ".cache/r4a/model.onnx"),
    threads=int(os.environ.get("VRHP_LEARNED_THREADS", "1")),
    skeleton=os.environ.get("VRHP_LEARNED_SKELETON", "xsens"),
    frame=os.environ.get("VRHP_LEARNED_FRAME", "godot"),
    height_m=float(os.environ["VRHP_LEARNED_HEIGHT_M"])
    if "VRHP_LEARNED_HEIGHT_M" in os.environ
    else None,
    stats_every=int(os.environ.get("VRHP_LEARNED_STATS_EVERY", "300")),
)


def configure(**kw) -> None:
    CONFIG.update({k: v for k, v in kw.items() if v is not None})


def _ry(deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


# R_model = R_sample @ M : sample local -Z (grip forward) -> model fingers axis (+X left, -X right)
HAND_M = {"left": _ry(90.0), "right": _ry(-90.0)}


def _quat_to_mat(q) -> np.ndarray:
    x, y, z, w = np.asarray(q, np.float64) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


class LearnedEstimator:
    name = "learned"

    def __init__(
        self,
        model_path: str | Path | None = None,
        threads: int | None = None,
        skeleton: str | None = None,
        frame: str | None = None,
        height_m: float | None = None,
        stats_every: int | None = None,
    ):
        cfg = {**CONFIG}
        self.rt = RT.LearnedRuntime(
            model_path or cfg["model_path"], threads or cfg["threads"], skeleton or cfg["skeleton"]
        )
        self.height_override = height_m if height_m is not None else cfg["height_m"]
        self.stats_every = cfg["stats_every"] if stats_every is None else stats_every
        self._times: list[dict] = []
        self._last_stats = time.perf_counter()
        self.frame = frame or cfg["frame"]
        if self.frame not in ("godot", "model"):
            raise ValueError(f"frame must be godot|model, got {self.frame!r}")
        self._cal_key: tuple | None = None
        self._lengths = np.zeros(self.rt.nj)
        self.last_timing: dict = {}
        self.last_contact: np.ndarray | None = None
        self.skeleton = skeleton or cfg["skeleton"]

    def reset(self) -> None:
        self.rt.reset()

    # --- calibration -> (static scalars, bone lengths) ---------------------------------------
    def _calibrate(self, cal: AvatarCalibration) -> None:
        bl = dict(cal.bone_lengths_m)
        key = (self.height_override or cal.human_height_m, tuple(sorted(bl.items())))
        if key == self._cal_key:
            return
        h = float(self.height_override or cal.human_height_m)
        arm = float(bl.get("arm_length", ARM_PER_HEIGHT * h))
        shoulder = float(bl.get("shoulder_width", SHOULDER_PER_HEIGHT * h))
        self.rt.set_calibration((h, arm, shoulder))
        from validation.sprint import conventions as C

        off = C.template_offsets("cmu" if self.skeleton == "cmu" else "xsens", h, arm)
        lengths = np.linalg.norm(off, axis=-1)
        for i, name in enumerate(self.rt.joints):
            if name in bl:
                lengths[i] = float(bl[name])
        self._lengths, self._cal_key = lengths, key

    def _log_stats(self, timing: dict) -> None:
        """Every ``stats_every`` estimates print compute time and the rate requests arrive at."""
        if not self.stats_every:
            return
        self._times.append(timing)
        if len(self._times) < self.stats_every:
            return
        now = time.perf_counter()
        tot = np.array([t["total"] for t in self._times]) * 1e3
        ort = np.array([t["ort"] for t in self._times]) * 1e3
        print(
            f"[learned] {len(tot)} frames @ {len(tot) / (now - self._last_stats):.1f} Hz | "
            f"compute p50 {np.percentile(tot, 50):.2f} p99 {np.percentile(tot, 99):.2f} ms "
            f"(ort p50 {np.percentile(ort, 50):.2f})",
            flush=True,
        )
        self._times.clear()
        self._last_stats = now

    # --- sample -> raw24 -------------------------------------------------------------------
    def _raw(self, s: TrackedPoseSample) -> np.ndarray:
        raw = np.zeros(RT.RAW_DIM)
        for k, (side, tr) in enumerate(
            (("head", s.hmd), ("left", s.left_hand), ("right", s.right_hand))
        ):
            ps, qs = RT.TRACKERS[k]
            raw[ps] = tr.transform.position
            q = np.asarray(tr.transform.rotation, np.float64)
            if self.frame == "godot" and side != "head":
                m = _quat_to_mat(q) @ HAND_M[side]
                q = RT.rotmat_to_quat(m)
            raw[qs] = q / np.linalg.norm(q)
            raw[21 + k] = 1.0 if (tr.valid and tr.confidence > 0.5) else 0.0
        return raw

    def estimate(
        self, sample: TrackedPoseSample, calibration: AvatarCalibration, grounding: GroundingContext
    ) -> HumanoidPoseFrame:
        self._calibrate(calibration)
        res = self.rt.step(sample.timestamp_seconds, self._raw(sample), self._lengths)
        self.last_timing, self.last_contact = res.timing, res.contact
        self._log_stats(res.timing)
        quats = RT.rotmat_to_quat(res.global_rot)
        joints = {
            name: JointPose(
                Transform(tuple(map(float, res.positions[i])), tuple(map(float, quats[i]))),
                1.0,
                True,
            )
            for i, name in enumerate(self.rt.joints)
        }
        h, c = joints["head"].transform, joints["chest"].transform
        joints["neck"] = JointPose(
            Transform(tuple(0.5 * (np.asarray(h.position) + np.asarray(c.position))), h.rotation),
            1.0,
            True,
        )
        return HumanoidPoseFrame(
            timestamp_seconds=sample.timestamp_seconds,
            root=joints["hips"].transform,
            joints=joints,
        )


__all__ = ["CONFIG", "HAND_M", "LearnedEstimator", "configure"]
