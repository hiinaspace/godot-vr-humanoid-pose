# ruff: noqa: E501
"""Causal inference-time post-processes for canonical-rest rotation predictions (Round 3).

Both operate on a ``CanonPrediction``-style (local rotations in the rest-v1 canonical pose, root
position) and the clip's bone lengths, and return new local rotations. They are per-frame (hand fix)
or causal-scan (foot lock): output at frame t depends only on inputs at frames <= t.

* ``hand_postfix``: hard analytic two-bone fix. The hand joint is placed exactly at the tracker-implied
  wrist position (the hand tracker is mounted at the hand joint), the elbow is re-solved with the
  predicted elbow as pole / swivel hint, so both bone lengths are preserved. Unreachable targets are
  clamped to the arm's reach along the same direction. Frames where the tracker flag is invalid are
  left as predicted.
* ``foot_lock``: while p(contact) > 0.5 the foot's horizontal position is pinned to its contact-onset
  location (taken from the already-corrected foot track, so there is no pop at onset); when contact
  ends the correction fades out linearly over ``BLEND`` frames. Height is left as predicted. The knee
  is re-solved with two-bone IK (predicted knee as pole).
"""

from __future__ import annotations

import numpy as np

from validation.sprint import conventions as C
from validation.sprint import rest as R

BLEND = 5
J = C.J
ARMS = (
    ("left_shoulder", "left_elbow", "left_hand"),
    ("right_shoulder", "right_elbow", "right_hand"),
)
LEGS = (("hips", "left_knee", "left_foot"), ("hips", "right_knee", "right_foot"))


def _unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def global_and_pos(local: np.ndarray, root: np.ndarray, lengths: np.ndarray):
    g = C.global_rotations(local)
    return g, R.fk(local, root, lengths)


def two_bone(root, mid, target, l1, l2, eps=1e-4):
    """Per-frame two-bone solve. root/mid/target [T,3]; returns new mid and end positions [T,3]
    with |mid-root| = l1, |end-mid| = l2, end = target (clamped to reach), mid on the predicted
    elbow's side of the root->target axis."""
    d = target - root
    dist = np.linalg.norm(d, axis=-1, keepdims=True)
    a = d / np.maximum(dist, 1e-9)
    dist_c = np.clip(dist, abs(l1 - l2) + eps, l1 + l2 - eps)
    x = (l1 * l1 - l2 * l2 + dist_c**2) / (2.0 * dist_c)  # along the axis
    h = np.sqrt(np.maximum(l1 * l1 - x * x, 0.0))
    pole = (mid - root) - np.sum((mid - root) * a, -1, keepdims=True) * a
    pn = np.linalg.norm(pole, axis=-1, keepdims=True)
    # degenerate pole (arm exactly straight): any perpendicular vector
    alt = np.cross(a, np.array([0.0, 0.0, 1.0]))
    alt = np.where(
        np.linalg.norm(alt, axis=-1, keepdims=True) < 1e-6, np.cross(a, [1.0, 0, 0]), alt
    )
    pole = np.where(pn > 1e-6, pole / np.maximum(pn, 1e-9), _unit(alt))
    new_mid = root + a * x + pole * h
    new_end = root + a * dist_c
    return new_mid, new_end


def _apply_chain(glob, pos, chain, target, valid, lengths):
    """Re-solve one 2-bone chain (root, mid, end joint names) toward ``target`` [T,3] on ``valid``."""
    r, m, e = (J[n] for n in chain)
    nm, ne = two_bone(pos[:, r], pos[:, m], target, lengths[m], lengths[e])
    cur1 = C.qrotate(glob[:, m], R.REST_DIRS[m])
    cur2 = C.qrotate(glob[:, e], R.REST_DIRS[e])
    f1 = R.min_rot(cur1, _unit(nm - pos[:, r]))
    f2 = R.min_rot(cur2, _unit(ne - nm))
    ident = np.array([0.0, 0.0, 0.0, 1.0])
    f1 = np.where(valid[:, None], f1, ident)
    f2 = np.where(valid[:, None], f2, ident)
    glob[:, m] = C.qnorm(C.qmul(f1, glob[:, m]))
    glob[:, e] = C.qnorm(C.qmul(f2, glob[:, e]))


def hand_postfix(local, root, lengths, hand_pos, hand_valid):
    """hand_pos [T,2,3] (left, right) tracker-implied wrist positions; hand_valid [T,2] bool."""
    glob, pos = global_and_pos(local, root, lengths)
    for k, chain in enumerate(ARMS):
        _apply_chain(glob, pos, chain, hand_pos[:, k], hand_valid[:, k], lengths)
    return C.local_rotations(glob)


def lock_track(contact: np.ndarray, foot_xz: np.ndarray, blend: int = BLEND):
    """Causal scan for one foot. contact [T] bool, foot_xz [T,2] predicted horizontal position.
    Returns corrected horizontal position [T,2] and weights [T]."""
    T = len(contact)
    out = foot_xz.copy()
    w_arr = np.zeros(T)
    anchor = np.zeros(2)
    w, prev = 0.0, False
    for t in range(T):
        if contact[t]:
            if not prev:  # onset: anchor at the currently displayed position
                anchor = foot_xz[t] + w * (anchor - foot_xz[t]) if w > 0 else foot_xz[t].copy()
            w = 1.0
        else:
            w = max(0.0, w - 1.0 / blend)
        prev = bool(contact[t])
        if w > 0:
            out[t] = foot_xz[t] + w * (anchor - foot_xz[t])
        w_arr[t] = w
    return out, w_arr


def foot_lock(local, root, lengths, contact_prob, blend: int = BLEND):
    """contact_prob [T,2] (left, right) from the model's contact head."""
    glob, pos = global_and_pos(local, root, lengths)
    for k, chain in enumerate(LEGS):
        f = J[chain[2]]
        foot = pos[:, f]
        xz, w = lock_track(contact_prob[:, k] > 0.5, foot[:, [0, 2]], blend)
        target = foot.copy()
        target[:, [0, 2]] = xz
        _apply_chain(glob, pos, chain, target, w > 0, lengths)
    return C.local_rotations(glob)
