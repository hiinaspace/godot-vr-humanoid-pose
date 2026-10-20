"""Tests for the frozen sprint oracle (validation/sprint)."""

from __future__ import annotations

import numpy as np
import pytest

from validation.sprint import conventions as C
from validation.sprint import oracle as O
from validation.sprint import splits as S

HAVE_DATA = S.CMU_DIR.exists() and S.XSENS_DIR.exists()
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="cached CMU/Xsens parquet not present")


def synthetic_clip(n: int = 120, seed: int = 0) -> C.ClipData:
    rng = np.random.default_rng(seed)
    pos = np.cumsum(rng.normal(0, 0.01, (n, C.NJ, 3)), axis=0) + rng.normal(0, 1, (1, C.NJ, 3))
    rot = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (n, C.NJ, 1))
    return C.ClipData(
        "cmu", "cmu/00/syn", "00", "syn", pos, rot, "unknown", "unknown", 120.0, pos[:10].copy()
    )


def test_splits_disjoint_and_rule():
    doc = S.load_splits()
    S.assert_disjoint(doc)
    for s in doc["cmu"]["subjects"]["test"]:
        assert S.cmu_bucket(s) == 0
    for s in doc["cmu"]["subjects"]["val"]:
        assert S.cmu_bucket(s) == 1
    assert not set(doc["cmu"]["subjects"]["test"]) & set(doc["cmu"]["subjects"]["train"])
    x = doc["xsens"]
    assert (
        x["val_subject"] not in x["eval_subjects"] and x["val_subject"] not in x["train_subjects"]
    )
    assert all(c.split("/")[1] != x["val_subject"] for c in x["clips"]["train"])
    assert x["eval_subject_overlap_with_train"]  # recorded, not hidden
    pins = doc["source_pins"]["xsens_data_zip"]
    assert pins["md5"] == pins["md5_expected_zenodo"]


def test_split_assertion_catches_leak():
    doc = S.load_splits()
    bad = {**doc, "cmu": {**doc["cmu"], "subjects": {**doc["cmu"]["subjects"]}}}
    bad["cmu"]["subjects"]["train"] = (
        doc["cmu"]["subjects"]["train"] + doc["cmu"]["subjects"]["test"][:1]
    )
    with pytest.raises(AssertionError):
        S.assert_disjoint(bad)


def test_gt_scores_zero_synthetic():
    clip = synthetic_clip()
    m = O.clip_metrics(clip, clip.positions.copy(), C.local_rotations(clip.canonical_rotations()))
    for k in ("full", "upper", "lower", "rootrel", "hand", "legacy_upper"):
        assert np.all(m[k] == 0.0), k
    for k in ("bonedir", "mpjre"):  # degrees; float round-off only
        assert np.all(m[k] < 1e-9), k
    assert m["jit_pred_sum"] == m["jit_gt_sum"]


def test_known_offset_gives_known_mpjpe():
    clip = synthetic_clip()
    pred = clip.positions.copy()
    pred[:, C.NONINPUT_IDX, 0] += 0.01  # 1 cm on every non-input joint
    m = O.clip_metrics(clip, pred, None)
    assert (
        np.allclose(m["full"], 0.01)
        and np.allclose(m["upper"], 0.01)
        and np.allclose(m["lower"], 0.01)
    )
    assert np.allclose(m["hand"], 0.0)
    # root-relative: a rigid shift of the whole body (hips included) cancels
    shifted = clip.positions + np.array([0.3, 0.0, -0.2])
    m2 = O.clip_metrics(clip, shifted, None)
    assert np.allclose(m2["rootrel"], 0.0) and m2["full"].mean() > 0.3


def test_jitter_formula_on_sine():
    fps, amp, freq = C.FPS, 0.05, 1.0
    t = np.arange(int(10 * fps)) / fps
    pos = np.zeros((len(t), 1, 3))
    pos[:, 0, 0] = amp * np.sin(2 * np.pi * freq * t)
    w = 2 * np.pi * freq / fps
    expected = (
        amp * (2 * np.sin(w / 2)) ** 3 * fps**3 / 100.0 * (2 / np.pi)
    )  # mean |.| of a sinusoid
    assert O.jerk_norms(pos).mean() == pytest.approx(expected, rel=0.01)
    # constant acceleration (quadratic) has zero third difference
    quad = np.zeros((50, 1, 3))
    quad[:, 0, 1] = 0.5 * 9.8 * (np.arange(50) / fps) ** 2
    assert O.jerk_norms(quad).max() < 1e-6


def test_conventions_basics():
    assert np.array_equal(np.cross(C.UP, C.FORWARD_REST), C.LEFT_REST)
    assert (
        C.NJ == 14 and len(C.NONINPUT_IDX) == 11 and len(C.UPPER_IDX) == 6 and len(C.LOWER_IDX) == 5
    )
    assert set(C.NONINPUT_JOINTS) | set(C.INPUT_JOINTS) == set(C.JOINTS)
    assert np.array_equal(C.MIRROR_PERM[C.MIRROR_PERM], np.arange(C.NJ))  # involution
    assert C.JOINTS[C.MIRROR_PERM[C.J["left_hand"]]] == "right_hand"
    p = np.random.default_rng(1).normal(size=(5, C.NJ, 3))
    assert np.allclose(C.mirror_positions(C.mirror_positions(p)), p)
    q = C.qnorm(np.random.default_rng(2).normal(size=(4, C.NJ, 4)))
    assert np.allclose(C.global_rotations(C.local_rotations(q)), q)
    assert np.allclose(C.qgeodesic_deg(q, q), 0.0, atol=1e-4)


def test_fk_roundtrip_template():
    off = C.template_offsets("xsens", 1.7, 0.6)
    rng = np.random.default_rng(3)
    loc = C.qnorm(rng.normal(size=(6, C.NJ, 4)))
    pos = C.forward_kinematics(loc, np.zeros((6, 3)), off)
    # bone lengths are preserved by FK whatever the rotations
    lengths = np.linalg.norm(pos[:, 1:] - pos[:, C.PARENT[1:]], axis=-1)
    assert np.allclose(lengths, np.linalg.norm(off[1:], axis=-1))


def test_calib_noise_bounded_and_deterministic():
    clip = synthetic_clip()
    a, b = O.make_calib(clip), O.make_calib(clip)
    assert a == b
    assert a.floor_y_m == 0.0


@needs_data
def test_inputs_deterministic_and_variants_differ():
    clip = C.load_clip(S.clip_path(S.clip_ids("cmu_test")[0]))
    c1, c2 = O.make_inputs(clip, "clean"), O.make_inputs(clip, "clean")
    k1, k2 = O.make_inputs(clip, "corrupted"), O.make_inputs(clip, "corrupted")
    assert np.array_equal(c1, c2) and np.array_equal(k1, k2)
    assert not np.array_equal(c1, k1)
    # clean head tracker = head joint + fixed mounting offset, rotation untouched
    head = clip.positions[:, C.J["head"]] + np.array(C.TRACKER_LOCAL_OFFSET_M)
    assert np.allclose(c1[:, :3], head) and np.allclose(c1[:, 3:7], clip.rotations[:, C.J["head"]])
    hand = clip.positions[:, C.J["left_hand"]]
    assert np.allclose(c1[:, 7:10], hand)
    assert np.all(c1[:, 21:] == 1.0)


@needs_data
def test_gt_method_scores_zero_on_real_data():
    r = O.evaluate_set("gt", "xsens_eval", "corrupted", workers=1, quick=True)
    for k in ("full_m", "upper_m", "lower_m", "rootrel_m", "hand_m", "p95_full_m"):
        assert r[k] == 0.0, k
    assert r["mpjre_m"] < 1e-9 and r["bonedir_m"] < 1e-9
    assert r["jitter"] == r["jitter_gt"] > 0


@needs_data
def test_conventions_probe_passes():
    from validation.sprint import probe_conventions

    assert probe_conventions.main(["--quick"]) == 0
