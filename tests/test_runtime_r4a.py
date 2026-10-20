"""Round 4a: runtime pieces (numpy twins, rate conversion, ONNX export parity, random-init nets)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnxruntime")

from validation.sprint import conventions as C  # noqa: E402
from validation.sprint import oracle as O  # noqa: E402
from validation.sprint import rest as R  # noqa: E402
from validation.sprint import runtime as RT  # noqa: E402
from validation.sprint import train as T  # noqa: E402


def random_stream(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.zeros((n, 24))
    base = np.array(
        [0.3, 1.65, -0.2, 0, 0, 0, 1, 0.5, 1.2, 0.2, 0, 0, 0, 1, 0.1, 1.2, 0.3, 0, 0, 0, 1, 1, 1, 1]
    )
    x[:] = base + np.cumsum(rng.normal(0, 0.004, (n, 24)), 0)
    for _, qs in RT.TRACKERS:
        x[:, qs] /= np.linalg.norm(x[:, qs], axis=-1, keepdims=True)
    return x.astype(np.float32).astype(np.float64)


def quat_to_mat(q):
    return T.qmat(torch.tensor(q)).numpy()


def test_static_feats_match_torch():
    calib = np.array([1.74, 0.61, 0.37])
    for skel in (None, 0, 1):
        got = RT.static_feats_np(calib, skel, T.NORM)
        want = T.static_feats(
            torch.tensor(calib[None], dtype=torch.float32),
            torch.tensor([skel or 0]),
            skel is not None,
        ).numpy()
        np.testing.assert_allclose(got, want, atol=1e-6)


def test_on_grid_window_equals_training_window():
    x = random_stream(130).astype(np.float32)
    w = RT.TimedWindow(40, 60.0)
    xp = np.concatenate([np.repeat(x[:1], 40, 0), x])  # predict_clip_meta's causal left padding
    for k in range(len(x)):
        w.push(k / 60.0, x[k])
        np.testing.assert_array_equal(w.build()[0], xp[k : k + 41])


def test_window_90hz_matches_bruteforce():
    rng = np.random.default_rng(3)
    ts = np.cumsum(rng.uniform(0.009, 0.013, 400))  # jittery ~90 Hz
    x = random_stream(400, 4)
    w = RT.TimedWindow(40, 60.0)
    for k in range(400):
        w.push(float(ts[k]), x[k])
        if k % 37:
            continue
        got = w.build()[0].astype(np.float64)
        for j in range(41):
            tq = ts[k] - (40 - j) / 60.0
            if tq <= ts[0]:
                want = x[0]
            else:
                i = int(np.searchsorted(ts[: k + 1], tq, side="right") - 1)
                i1 = min(i + 1, k)
                a = (tq - ts[i]) / (ts[i1] - ts[i]) if i1 > i else 0.0
                want = x[i] * (1 - a) + x[i1] * a
                for _, qs in RT.TRACKERS:
                    q = want[qs]
                    want[qs] = q / np.linalg.norm(q)
            np.testing.assert_allclose(got[j, :21], want[:21], atol=2e-6)
        assert got[-1, :21].tolist() == pytest.approx(
            x[k, :21].astype(np.float32).tolist()
        )  # newest = exact


def test_timestamp_restart_resets():
    x = random_stream(20)
    w = RT.TimedWindow(40)
    for k in range(10):
        w.push(100.0 + k / 60, x[k])
    assert w.push(0.0, x[0]) is True
    assert len(w.t) == 1


def test_fk_matches_rest_fk():
    rng = np.random.default_rng(1)
    q = rng.normal(size=(1, C.NJ, 4))
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    hips = rng.normal(size=(1, 3))
    lengths = R.bone_lengths(rng.normal(size=(5, C.NJ, 3)) * 0.3)
    want = R.fk(q, hips, lengths)[0]
    got, g = RT.fk(quat_to_mat(q[0]), hips[0], lengths, R.REST_DIRS, C.PARENT)
    np.testing.assert_allclose(got, want, atol=1e-5)
    np.testing.assert_allclose(
        RT.rotmat_to_quat(g),
        C.global_rotations(q)[0] * np.sign(C.global_rotations(q)[0][:, 3:4]),
        atol=1e-5,
    )


def test_rotmat_to_quat_roundtrip():
    rng = np.random.default_rng(2)
    q = rng.normal(size=(200, 4))
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    back = RT.rotmat_to_quat(quat_to_mat(q))
    assert np.all(C.qgeodesic_deg(back, q) < 1e-3)


@pytest.mark.parametrize("arch", ["hmdposer", "avatarposer"])
def test_onnx_matches_torch_predictor(tmp_path, arch):
    from validation.sprint import export_onnx as X

    torch.manual_seed(0)
    T.L = 40
    model = T.MODELS[arch]()
    cfg = dict(model=arch, skel=True)
    ck = tmp_path / "ck.pt"
    torch.save(dict(model=model.state_dict(), cfg=cfg, step=0, best=(0, 0), hist=[]), ck)
    onnx = tmp_path / "m.onnx"
    meta = X.export(ck, onnx)
    assert meta["verify_max_abs"]["onnx"] < 1e-4
    rt = RT.LearnedRuntime(onnx)
    pred = T.ModelPredictor.load(ck, dev="cpu")
    inputs = random_stream(70, 5)
    calib = O.Calib(1.72, 0.6, 0.36)
    ref = pred.predict_clip_meta(inputs, calib, {"dataset": "xsens"})
    lengths = np.linalg.norm(
        C.template_offsets("xsens", calib.height_m, calib.arm_length_m), axis=-1
    )
    want = R.fk(ref.rotations, ref.root_pos, lengths)
    rt.set_calibration((calib.height_m, calib.arm_length_m, calib.shoulder_width_m))
    got = np.stack([rt.step(k / 60.0, inputs[k], lengths).positions for k in range(len(inputs))])
    assert np.abs(got - want).max() < 1e-4
    # reset reproducibility: same stream again after reset() gives identical output
    rt.reset()
    again = np.stack([rt.step(k / 60.0, inputs[k], lengths).positions for k in range(len(inputs))])
    np.testing.assert_array_equal(got, again)
