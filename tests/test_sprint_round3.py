"""Round 3: post-processes (hand two-bone fix, causal foot lock) and contact labels."""

from __future__ import annotations

import numpy as np

from validation.sprint import conventions as C
from validation.sprint import post as P
from validation.sprint import rest as R

LEN = np.array([0, 0.1, 0.2, 0.15, 0.2, 0.3, 0.25, 0.2, 0.3, 0.25, 0.1, 0.4, 0.1, 0.4])


def _pose(t=60, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 0.6, (t, C.NJ, 3))
    ang = np.linalg.norm(a, axis=-1, keepdims=True)
    q = np.concatenate([a / np.maximum(ang, 1e-9) * np.sin(ang / 2), np.cos(ang / 2)], -1)
    root = np.cumsum(rng.normal(0, 0.01, (t, 3)), 0) + np.array([0, 1.0, 0])
    return q, root


def test_hand_postfix_hits_target_and_keeps_lengths():
    q, root = _pose()
    pos = R.fk(q, root, LEN)
    rng = np.random.default_rng(1)
    sh = np.stack([pos[:, C.J["left_shoulder"]], pos[:, C.J["right_shoulder"]]], 1)
    hd = np.stack([pos[:, C.J["left_hand"]], pos[:, C.J["right_hand"]]], 1)
    tgt = sh + 0.8 * (hd - sh) + rng.normal(0, 0.01, sh.shape)  # reachable
    out = P.hand_postfix(q, root, LEN, tgt, np.ones((len(q), 2), bool))
    new = R.fk(out, root, LEN)
    for k, side in enumerate(("left", "right")):
        assert np.abs(new[:, C.J[f"{side}_hand"]] - tgt[:, k]).max() < 1e-6
    bl = np.linalg.norm(new[:, 1:] - new[:, C.PARENT[1:]], axis=-1)
    assert np.abs(bl - LEN[1:]).max() < 1e-9
    keep = [C.J[n] for n in ("hips", "spine", "chest", "head", "left_knee", "right_foot")]
    assert np.abs(new[:, keep] - pos[:, keep]).max() < 1e-9
    # elbows stay near the predicted ones (pole hint)
    e0, e1 = pos[:, C.J["left_elbow"]], new[:, C.J["left_elbow"]]
    assert np.linalg.norm(e0 - e1, axis=-1).mean() < 0.1


def test_hand_postfix_invalid_frames_untouched():
    q, root = _pose()
    tgt = np.zeros((len(q), 2, 3))
    valid = np.zeros((len(q), 2), bool)
    out = P.hand_postfix(q, root, LEN, tgt, valid)
    assert np.abs(R.fk(out, root, LEN) - R.fk(q, root, LEN)).max() < 1e-9


def test_footlock_pins_during_contact_and_is_causal():
    q0, _ = _pose(t=1, seed=3)
    q = np.repeat(q0, 80, axis=0)  # constant (bent-leg) pose, so the anchor stays reachable
    root = np.zeros((80, 3)) + np.array([0, 1.0, 0])
    root[:, 0] += np.linspace(0, 0.03, 80)
    prob = np.zeros((80, 2))
    prob[20:50, 0] = 0.9
    out = P.foot_lock(q, root, LEN, prob)
    pos = R.fk(out, root, LEN)
    f = pos[:, C.J["left_foot"]]
    assert np.ptp(f[20:50, [0, 2]], axis=0).max() < 1e-6
    bl = np.linalg.norm(pos[:, 1:] - pos[:, C.PARENT[1:]], axis=-1)
    assert np.abs(bl - LEN[1:]).max() < 1e-9
    # blend-out: the correction fades over BLEND frames, then the track equals the prediction
    raw = R.fk(q, root, LEN)[:, C.J["left_foot"]]
    assert np.abs(f[50 + P.BLEND + 1 :] - raw[50 + P.BLEND + 1 :]).max() < 1e-9
    # causality: changing the future leaves the past unchanged
    q2, prob2 = q.copy(), prob.copy()
    q2[60:] = C.qnorm(q2[60:] + 0.05)
    prob2[60:, 0] = 1.0
    out2 = P.foot_lock(q2, root, LEN, prob2)
    assert np.abs(out2[:60] - out[:60]).max() < 1e-9


def test_contact_labels_match_oracle_rule():
    from validation.sprint import oracle_v2 as O2
    from validation.sprint import train as T

    rng = np.random.default_rng(0)
    n = 300
    pos = np.zeros((n, C.NJ, 3), np.float32)
    pos[:, :, 1] = 1.0
    for f in (C.J["left_foot"], C.J["right_foot"]):
        pos[:, f, 1] = 0.02 + 0.2 * (np.sin(np.arange(n) / 20.0) > 0.5)
        pos[:, f, 0] = np.cumsum(rng.normal(0, 0.002, n))
    lab = T.contact_labels(pos, [0], [n])
    gf = pos[:, O2.FEET].astype(np.float64)
    floor = np.percentile(gf[..., 1].min(1), 2)
    ref = (gf[1:-1, :, 1] < floor + 0.05) & (O2._hspeed(gf) < 0.10)
    assert (lab[1:-1] == ref).all()
