"""Round 4b: review-packet exporter helpers (no data caches needed)."""

from __future__ import annotations

import base64

import numpy as np

from validation.sprint import conventions as C
from validation.sprint import export_review as E


def test_quantisation_roundtrip_mm():
    rng = np.random.default_rng(0)
    pos = rng.uniform(-3, 3, size=(50, C.NJ, 3)).astype(np.float32)
    origin = np.array([1.0, 0.0, -2.0], dtype=np.float32)
    raw = base64.b64decode(E.b64(E.quant_pos(pos, origin)))
    back = np.frombuffer(raw, dtype="<i2").reshape(pos.shape) / 1000.0 + origin
    assert np.abs(back - pos).max() <= 0.0005 + 1e-6


def test_best_window_and_separated_top():
    idx = np.arange(0, 3000, 6)
    score = np.zeros(len(idx))
    score[(idx >= 1500) & (idx < 1980)] = 5.0
    f0, f1 = E.best_window(score, idx, 3000, 480)
    assert f1 - f0 == 480 and 1500 <= f0 + 240 <= 1980
    assert E.best_window(score, idx, 300, 480) == (0, 300)
    tops = E.separated_top(np.array([0, 9, 8, 0, 0, 0, 7, 0.0]), 3, 3)
    assert tops == [1, 6]  # frame 2 is within the gap of frame 1


def test_gt_contact_standing_still_is_contact_and_walking_foot_is_not():
    n = 120
    gt = np.zeros((n, C.NJ, 3))
    lf, rf = C.J["left_foot"], C.J["right_foot"]
    gt[:, lf] = [0.1, 0.03, 0.0]
    gt[:, rf] = [-0.1, 0.03, 0.0]
    gt[60:, rf, 1] = 0.30  # right foot lifted for the second half
    gt[60:, rf, 0] += np.linspace(0, 0.5, n - 60)
    c, floor = E.gt_contact(gt)
    assert abs(floor - 0.03) < 1e-9
    assert c[:, 0].all() and c[:55, 1].all() and not c[65:, 1].any()


def test_pop_frames_flags_single_frame_flip_only_where_gt_is_smooth():
    n = 40
    base = np.zeros((n, C.NJ, 3))
    # a crude standing pose: every bone along rest direction is irrelevant; use random-free offsets
    for j in range(1, C.NJ):
        base[:, j] = base[:, C.PARENT[j]] + np.array([0.0, 0.2, 0.0])
    flipped = base.copy()
    el = C.J["left_elbow"]
    flipped[20:, el] += np.array([0.3, 0.0, 0.0])  # elbow jumps sideways at frame 20 and stays
    assert 20 in E.pop_frames(base, flipped)
    assert E.pop_frames(flipped, flipped) == []
