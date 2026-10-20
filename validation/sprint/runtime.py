# ruff: noqa: E501
"""Torch-free runtime for ``export_onnx.py`` graphs: tracker stream in, posed skeleton out.

Everything the Godot sidecar needs, with only numpy + onnxruntime (no torch, no polars):

* ``TimedWindow``   rate conversion. The model runs on a 60 Hz grid with a ``W+1``-frame window, but a
                    headset delivers frames at 72-120 Hz with jittery timestamps. The window for an
                    arriving frame at time ``t`` is sampled at ``t - k/60`` (k = W..0) from the received
                    history (linear in position, nlerp in rotation). k = 0 is the newest sample itself and
                    every other point is interpolated between two *past* samples, so the model sees the
                    newest tracker frame with zero lookahead. On an exact 60 Hz stream the weights are 0/1,
                    so the window is bit-identical to the one training / ``predict_clip_meta`` builds.
* ``LearnedRuntime`` ring buffer + ORT session + FK; ``step(t, raw)`` -> positions / rotations.
* ``static_feats_np`` / ``fk`` / ``rotmat_to_quat`` numpy twins of the torch code (tests compare them).

Left padding: before ``W`` frames have arrived the window is padded with the first received frame,
exactly like ``ModelPredictor.predict_clip_meta`` ("causal left padding with the first frame").
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

RAW_DIM = 24
# (pos slice, quat slice) of head, left hand, right hand inside the 24-vector
TRACKERS = (
    (slice(0, 3), slice(3, 7)),
    (slice(7, 10), slice(10, 14)),
    (slice(14, 17), slice(17, 21)),
)
SNAP = (
    1e-6  # interpolation weights closer than this to 0/1 are snapped (keeps on-grid streams exact)
)
MAX_GAP_RESET_S = 1.0  # a timestamp jump of more than this (either way) starts a new session


def static_feats_np(calib, skel_id: int | None, norm: dict) -> np.ndarray:
    """numpy twin of ``train.static_feats`` for one subject; ``calib`` = (height, arm, shoulder) in m."""
    mu, sd = np.asarray(norm["mu"], np.float32), np.asarray(norm["sd"], np.float32)
    oh = np.zeros(2, np.float32)
    if skel_id is not None:
        oh[int(skel_id)] = 1.0
    return np.concatenate([(np.asarray(calib, np.float32) - mu) / sd, oh])[None].astype(np.float32)


def rotmat_to_quat(m: np.ndarray) -> np.ndarray:
    """[...,3,3] -> [...,4] xyzw, Shepperd's method (w >= 0)."""
    m = np.asarray(m, np.float64)
    out = np.empty(m.shape[:-2] + (4,))
    flat, o = m.reshape(-1, 3, 3), out.reshape(-1, 4)
    for i, r in enumerate(flat):
        tr = r[0, 0] + r[1, 1] + r[2, 2]
        if tr > 0:
            s = 2.0 * np.sqrt(tr + 1.0)
            q = [
                (r[2, 1] - r[1, 2]) / s,
                (r[0, 2] - r[2, 0]) / s,
                (r[1, 0] - r[0, 1]) / s,
                0.25 * s,
            ]
        elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
            s = 2.0 * np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2])
            q = [
                0.25 * s,
                (r[0, 1] + r[1, 0]) / s,
                (r[0, 2] + r[2, 0]) / s,
                (r[2, 1] - r[1, 2]) / s,
            ]
        elif r[1, 1] > r[2, 2]:
            s = 2.0 * np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2])
            q = [
                (r[0, 1] + r[1, 0]) / s,
                0.25 * s,
                (r[1, 2] + r[2, 1]) / s,
                (r[0, 2] - r[2, 0]) / s,
            ]
        else:
            s = 2.0 * np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1])
            q = [
                (r[0, 2] + r[2, 0]) / s,
                (r[1, 2] + r[2, 1]) / s,
                0.25 * s,
                (r[1, 0] - r[0, 1]) / s,
            ]
        q = np.asarray(q)
        o[i] = q / np.linalg.norm(q) * (1.0 if q[3] >= 0 else -1.0)
    return out


def fk(
    rot: np.ndarray, hips: np.ndarray, lengths: np.ndarray, rest_dirs: np.ndarray, parents
) -> tuple[np.ndarray, np.ndarray]:
    """Local rotation matrices [NJ,3,3] (root = world), hips [3], bone lengths [NJ] -> (positions
    [NJ,3], global rotation matrices [NJ,3,3]). Same recurrence as ``train.fk_torch``."""
    nj = len(parents)
    g = np.empty((nj, 3, 3))
    p = np.empty((nj, 3))
    g[0], p[0] = rot[0], hips
    off = rest_dirs * lengths[:, None]
    for j in range(1, nj):
        a = parents[j]
        g[j] = g[a] @ rot[j]
        p[j] = p[a] + g[j] @ off[j]
    return p, g


def _nlerp_quat(q0: np.ndarray, q1: np.ndarray, w: np.ndarray) -> np.ndarray:
    s = np.where(np.sum(q0 * q1, -1, keepdims=True) < 0, -1.0, 1.0)
    q = q0 * (1.0 - w) + (q1 * s) * w
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


class TimedWindow:
    """Timestamped tracker history -> the model's ``(W+1)``-frame, ``hz`` grid window."""

    def __init__(self, window: int, hz: float = 60.0):
        self.W, self.hz = int(window), float(hz)
        self.span = self.W / self.hz
        self.t: list[float] = []
        self.x: list[np.ndarray] = []

    def reset(self) -> None:
        self.t.clear()
        self.x.clear()

    def push(self, t: float, raw: np.ndarray) -> bool:
        """Add a sample; returns True if the history was reset (time jumped backwards)."""
        reset = False
        if self.t and abs(t - self.t[-1]) > MAX_GAP_RESET_S:
            self.reset()  # clock restarted (new client / clip) or a stall longer than the window
            reset = True
        elif self.t and t < self.t[-1]:
            return False  # slightly out of order: drop
        if self.t and t == self.t[-1]:
            self.x[-1] = np.asarray(raw, np.float64)
            return reset
        self.t.append(float(t))
        self.x.append(np.asarray(raw, np.float64))
        # keep every sample from the last one at-or-before (t_new - span) onward
        cut = t - self.span - 1.0 / self.hz
        k = 0
        while k + 1 < len(self.t) and self.t[k + 1] <= cut:
            k += 1
        if k:
            del self.t[:k], self.x[:k]
        return reset

    def build(self) -> np.ndarray:
        ts, xs = np.asarray(self.t), np.stack(self.x)
        n = len(ts)
        tq = ts[-1] - np.arange(self.W, -1, -1) / self.hz  # oldest -> newest
        i0 = np.clip(np.searchsorted(ts, tq, side="right") - 1, 0, n - 1)
        i1 = np.minimum(i0 + 1, n - 1)
        dt = ts[i1] - ts[i0]
        w = np.where(dt > 1e-9, (tq - ts[i0]) / np.where(dt > 1e-9, dt, 1.0), 0.0)
        w = np.clip(w, 0.0, 1.0)
        w[w < SNAP] = 0.0
        w[w > 1.0 - SNAP] = 1.0
        w = w[:, None]
        out = np.where(w == 1.0, xs[i1], xs[i0])  # flags: sample at-or-before the query time
        for ps, qs in TRACKERS:
            out[:, ps] = xs[i0][:, ps] * (1.0 - w) + xs[i1][:, ps] * w
            out[:, qs] = np.where(
                w == 0.0,
                xs[i0][:, qs],
                np.where(w == 1.0, xs[i1][:, qs], _nlerp_quat(xs[i0][:, qs], xs[i1][:, qs], w)),
            )
        return out.astype(np.float32)[None]


@dataclass
class StepResult:
    positions: np.ndarray  # [NJ,3] world
    global_rot: np.ndarray  # [NJ,3,3] world
    local_rot: np.ndarray  # [NJ,3,3]
    hips: np.ndarray  # [3]
    contact: np.ndarray | None  # [2] P(contact) or None
    timing: dict = field(default_factory=dict)  # seconds: window, ort, fk, total


class LearnedRuntime:
    def __init__(self, onnx_path: str | Path, threads: int = 1, skel: str | None = "xsens"):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = int(threads)
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
        raw_meta = self.sess.get_modelmeta().custom_metadata_map.get("vrhp_meta")
        if raw_meta is None:  # re-written graphs (e.g. quantised) may drop metadata_props
            raw_meta = Path(onnx_path).with_suffix(".json").read_text()
        self.meta = json.loads(raw_meta)
        m = self.meta
        self.W, self.hz, self.nj = m["window"], m["frame_hz"], m["nj"]
        self.parents = m["parents"]
        self.joints = m["joints"]
        self.rest_dirs = np.asarray(m["rest_dirs"], np.float64)
        self.has_contact = m["has_contact"]
        self.skel_id = None if not m["skel"] else {"cmu": 0, "xsens": 1}[skel or "xsens"]
        self.win = TimedWindow(self.W, self.hz)
        self._static: np.ndarray | None = None
        self._outs = [o.name for o in self.sess.get_outputs()]

    def set_calibration(self, calib) -> None:
        self._static = static_feats_np(calib, self.skel_id, self.meta["norm"])

    def reset(self) -> None:
        self.win.reset()

    def step(self, t: float, raw: np.ndarray, lengths: np.ndarray) -> StepResult:
        assert self._static is not None, "call set_calibration() first"
        t0 = time.perf_counter()
        if self.win.push(t, raw):
            pass  # history was reset by a backwards jump; the new window starts from this frame
        window = self.win.build()
        t1 = time.perf_counter()
        outs = self.sess.run(self._outs, {"raw": window, "static": self._static})
        t2 = time.perf_counter()
        rot, hips = outs[0][0].astype(np.float64), outs[1][0].astype(np.float64)
        pos, g = fk(rot, hips, np.asarray(lengths, np.float64), self.rest_dirs, self.parents)
        t3 = time.perf_counter()
        return StepResult(
            pos,
            g,
            rot,
            hips,
            outs[2][0].astype(np.float64) if self.has_contact else None,
            dict(window=t1 - t0, ort=t2 - t1, fk=t3 - t2, total=t3 - t0),
        )
