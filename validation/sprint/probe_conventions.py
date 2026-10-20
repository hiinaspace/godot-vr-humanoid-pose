"""Assert the canonical conventions on real CMU and Xsens clips and print a short report.

    uv run python -m validation.sprint.probe_conventions [--quick]

Facing is inferred from knee flexion (knees only bend forward), which does not use any left/right
label, so the left/right checks are independent of the dataset's own naming. Exit code 1 on failure.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from validation.sprint import conventions as C
from validation.sprint import oracle as O
from validation.sprint import splits as S

J = C.J
UP = C.UP
failures: list[str] = []


def check(ok: bool, label: str, detail: str) -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}: {detail}")
    if not ok:
        failures.append(label)


def facing(P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame horizontal facing from knee flexion; returns (unit vectors [T,3], valid [T])."""
    f = np.zeros((len(P), 3))
    for s in ("left", "right"):
        mid = 0.5 * (P[:, J["hips"]] + P[:, J[f"{s}_foot"]])
        d = P[:, J[f"{s}_knee"]] - mid
        d[:, 1] = 0.0
        f += d
    n = np.linalg.norm(f, axis=1)
    return f / np.maximum(n, 1e-9)[:, None], n > 0.06


def probe_clips(clips: list[C.ClipData], name: str) -> None:
    print(
        f"\n== {name}: {len(clips)} clips, {sum(c.n_frames for c in clips)} frames @ {C.FPS:.0f} Hz"
    )
    lat = {k: [0, 0] for k in ("shoulder", "elbow", "hand", "knee", "foot")}
    walk_cos, head_cos, up_ok, up_n, foot_min, cv = [], [], 0, 0, [], {}
    for c in clips:
        P = c.positions
        f, valid = facing(P)
        left = np.cross(UP, f)
        for k in lat:
            d = np.sum((P[:, J[f"left_{k}"]] - P[:, J[f"right_{k}"]]) * left, axis=1)[valid]
            lat[k][0] += int((d > 0).sum())
            lat[k][1] += int(valid.sum())
        # walking direction vs forward implied by the labelled shoulders (independent of knees)
        sh = P[:, J["left_shoulder"]] - P[:, J["right_shoulder"]]
        f_lab = np.cross(sh, UP)
        f_lab[:, 1] = 0
        f_lab /= np.maximum(np.linalg.norm(f_lab, axis=1), 1e-9)[:, None]
        v = np.gradient(P[:, J["hips"]], axis=0) * C.FPS
        v[:, 1] = 0
        k = 15
        v = np.stack([np.convolve(v[:, i], np.ones(k) / k, mode="same") for i in range(3)], axis=1)
        sp = np.linalg.norm(v, axis=1)
        w = sp > 0.8
        if w.any():
            walk_cos.append(np.sum(v[w] * f_lab[w], axis=1) / sp[w])
        hf = C.qrotate(c.rotations[:, J["head"]], C.FORWARD_REST)
        hf[:, 1] = 0
        hf /= np.maximum(np.linalg.norm(hf, axis=1), 1e-9)[:, None]
        head_cos.append(np.sum(hf * f_lab, axis=1))
        if c.activity_class not in ("floor_pose", "swimming"):
            up_ok += int(((P[:, J["head"], 1] - P[:, J["hips"], 1]) > 0.3).sum())
            up_n += c.n_frames
        foot_min.append(np.min(P[:, [J["left_foot"], J["right_foot"]], 1]))
        for a, b in [
            ("head", "chest"),
            ("chest", "spine"),
            ("spine", "hips"),
            ("left_shoulder", "left_elbow"),
            ("left_elbow", "left_hand"),
            ("right_shoulder", "right_elbow"),
            ("right_elbow", "right_hand"),
            ("left_shoulder", "right_shoulder"),
            ("left_knee", "left_foot"),
            ("right_knee", "right_foot"),
        ]:
            L = np.linalg.norm(P[:, J[a]] - P[:, J[b]], axis=1)
            cv.setdefault(f"{a}-{b}", []).append(L.std() / L.mean())
    for k, (pos, n) in lat.items():
        thr = (
            0.85 if k in ("hand", "foot") else 0.93
        )  # hands/feet cross the midline; a label swap scores ~0.05
        check(
            n > 0 and pos / n >= thr,
            f"left {k} on +up x forward side (>= {thr})",
            f"{pos / max(n, 1):.4f} of {n} frames",
        )
    wc = np.concatenate(walk_cos) if walk_cos else np.array([0.0])
    if len(wc) >= 2000:  # backwards/side steps exist, so require the bulk, not all
        check(
            float(np.median(wc)) > 0.9 and float(np.mean(wc > 0)) >= 0.85,
            "walking direction == label-implied forward",
            f"median cos {np.median(wc):.2f}, {np.mean(wc > 0):.3f} of {len(wc)} "
            "walking frames forward",
        )
    else:
        print(f"  [skip] walking direction: only {len(wc)} walking frames")
    hc = np.concatenate(head_cos)
    check(
        float(np.median(hc)) > 0.8,
        "head rotation +Z == body forward",
        f"median cos {np.median(hc):.3f}",
    )
    check(
        up_n > 0 and up_ok / up_n >= 0.95,
        "head above hips (+Y up)",
        f"{up_ok / max(up_n, 1):.4f} of frames",
    )
    fm = np.array(foot_min)
    check(
        bool(0.0 <= np.median(fm) <= 0.2 and np.mean(fm > -0.1) >= 0.8),
        "floor at Y~0 (per-clip min ankle height)",
        f"median {np.median(fm):.3f} m, {np.mean(fm < -0.1):.2f} of clips dip >10 cm below the "
        "floor "
        "(Xsens A09 clips go up to 0.8 m below: stairs/drift, kept)",
    )
    worst = max(cv, key=lambda k: np.median(cv[k]))
    # CMU 'foot' is the end of the foot bone (mid-foot), so knee->foot flexes by a few percent
    lim = lambda k: 0.06 if k.endswith("foot") else 0.03  # noqa: E731
    bad = [k for k, v in cv.items() if np.median(v) >= lim(k)]
    check(
        not bad,
        "bone lengths stable (median CV per bone < 3%, foot bones < 6%)",
        f"worst {worst} {np.median(cv[worst]):.4f}" + (f"; over: {bad}" if bad else ""),
    )
    heights = np.array([O.make_calib(c).height_m for c in clips])
    check(
        1.35 < np.median(heights) < 2.0,
        "units are metres (calibrated height)",
        f"median {np.median(heights):.3f} m",
    )

    # incoming-bone rotation convention: offset in the bone's own frame must be ~constant
    spread = {}
    for c in clips[:: max(1, len(clips) // 20)]:
        R = c.canonical_rotations()
        for j in range(1, C.NJ):
            vloc = C.qrotate(C.qconj(R[:, j]), c.positions[:, j] - c.positions[:, C.PARENT[j]])
            d = vloc / np.maximum(np.linalg.norm(vloc, axis=1), 1e-9)[:, None]
            m = d.mean(0)
            m /= np.linalg.norm(m)
            spread.setdefault(C.JOINTS[j], []).append(
                np.degrees(np.arccos(np.clip(d @ m, -1, 1))).mean()
            )
    worst = max(spread, key=lambda k: np.median(spread[k]))
    torso_arm = ("left_elbow", "left_hand", "chest", "head", "left_shoulder")
    check(
        max(np.median(v) for v in spread.values()) < 25.0,
        "incoming-bone rotation offsets ~constant (mean angle to mean dir)",
        f"worst {worst} {np.median(spread[worst]):.1f} deg; arms/torso "
        f"{max(np.median(spread[k]) for k in torso_arm):.1f} deg",
    )
    errs = []
    for c in clips[:: max(1, len(clips) // 20)]:
        calib = O.make_calib(c)
        from validation.sprint.baselines import GroundTruthRotations

        pr = GroundTruthRotations().predict_clip_with_gt(None, calib, c)
        pos = O.predicted_positions(pr, c, calib)
        errs.append(np.linalg.norm(pos - c.positions, axis=-1)[:, C.NONINPUT_IDX].mean())
    print(f"  [info] template-FK floor with GT rotations: {100 * np.mean(errs):.2f} cm MPJPE-full")


def probe_mirror() -> None:
    print("\n== Xsens mirror pairs (orig vs *_mirror)")
    ids = [i for i in S.clip_ids("xsens_eval") + S.clip_ids("xsens_val") if i.endswith("_mirror")]
    for mid in ids:
        a = C.load_clip(S.clip_path(mid[: -len("_mirror")]))
        b = C.load_clip(S.clip_path(mid))
        n = min(a.n_frames, b.n_frames) - 12
        best = (np.inf, 0)
        for lag in range(-6, 7):
            e = np.linalg.norm(
                C.mirror_positions(a.positions[6 : 6 + n]) - b.positions[6 + lag : 6 + lag + n],
                axis=-1,
            ).mean()
            best = (e, lag) if e < best[0] else best
        e, lag = best
        unswapped = a.positions[6 : 6 + n] * np.array([-1.0, 1.0, 1.0])
        e_unsw = np.linalg.norm(unswapped - b.positions[6 + lag : 6 + lag + n], axis=-1).mean()
        check(
            e < 0.03 and e_unsw > 3 * e,
            f"{mid.split('/')[-1]} == x-reflected orig with L/R labels swapped",
            f"lag {lag:+d} fr, mean joint err {100 * e:.2f} cm "
            f"(without label swap {100 * e_unsw:.1f} cm)",
        )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)
    step = 40 if args.quick else 12
    cmu = [C.load_clip(S.clip_path(i)) for i in S.clip_ids("cmu_all")[::step]]
    xs_ids = S.clip_ids("xsens_train") + S.clip_ids("xsens_val") + S.clip_ids("xsens_eval")
    xs = [
        C.load_clip(S.clip_path(i))
        for i in xs_ids[:: 8 if args.quick else 2]
        if not i.endswith("_mirror")
    ]
    probe_clips(cmu, "CMU")
    probe_clips(xs, "Xsens (non-mirror)")
    xm = [
        C.load_clip(S.clip_path(i))
        for i in xs_ids[:: 8 if args.quick else 2]
        if i.endswith("_mirror")
    ]
    if xm:
        probe_clips(xm, "Xsens mirror clips (labels must also be left=+up x forward)")
    probe_mirror()
    print(
        "\nRESULT:", "PASS" if not failures else f"FAIL ({len(failures)}): " + "; ".join(failures)
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
