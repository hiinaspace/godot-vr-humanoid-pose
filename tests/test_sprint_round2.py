"""Round 2 checks: canonical retarget exactness, mirror, rotation helpers, model shapes (CPU)."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from validation.sprint import conventions as C
from validation.sprint import rest as R
from validation.sprint import splits as S
from validation.sprint import train as T


def _clips(ds: str, n: int):
    ids = S.load_splits()[ds]["clips"]["train"]
    random.Random(0).shuffle(ids)
    for cid in ids[:n]:
        yield C.load_clip(S.clip_path(cid))


@pytest.mark.parametrize("ds", ["cmu", "xsens"])
def test_canonical_fk_reproduces_gt_with_per_frame_lengths(ds):
    for c in _clips(ds, 50):
        loc = R.canonical_local(c)
        g = C.global_rotations(loc)
        ln = np.linalg.norm(c.positions[:, 1:] - c.positions[:, C.PARENT[1:]], axis=-1)
        pos = np.zeros_like(c.positions)
        pos[:, 0] = c.positions[:, 0]
        for j in range(1, C.NJ):
            off = C.qrotate(g[:, j], R.REST_DIRS[j]) * ln[:, j - 1 : j]
            pos[:, j] = pos[:, C.PARENT[j]] + off
        assert np.abs(pos - c.positions).max() < 1e-3


def test_mirror_preserves_forearm_probe():
    c = next(_clips("xsens", 1))
    assert abs(R.forearm_probe(c) - R.forearm_probe(R.mirror_clip(c))) < 1e-6


def test_mat2quat_roundtrip_and_sixd():
    q = torch.nn.functional.normalize(torch.randn(500, 4, dtype=torch.float64), dim=-1)
    m = T.qmat(q)
    q2 = T.mat2quat(m)
    assert (torch.minimum((q - q2).abs().sum(-1), (q + q2).abs().sum(-1)) < 1e-9).all()
    assert torch.allclose(T.sixd2mat(T.sixd(m)), m, atol=1e-9)


@pytest.mark.parametrize("name", ["hmdposer", "avatarposer"])
def test_models_forward_shapes_and_causality(name):
    model = T.MODELS[name]().eval()
    f, st = torch.randn(2, T.L, T.NT, 18), torch.randn(2, 5)
    out = model(f, st)
    assert out.shape == (2, T.L, T.OUT)
    f2 = f.clone()
    f2[:, -1] += 1.0  # changing the newest frame must not change earlier outputs
    assert torch.allclose(model(f2, st)[:, :-1], out[:, :-1], atol=1e-5)
