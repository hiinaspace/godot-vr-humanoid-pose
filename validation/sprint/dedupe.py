# ruff: noqa: E501
"""Content-level duplicate finder across splits (Round 2 audit finding), writes splits_v2.json.

Method: every clip (CMU + Xsens, 60 Hz canonical joints) -> root-relative 13-joint positions (hips
removed). Frames at stride 10 of each clip go into one KD-tree (39-d). A hit is a pair of frames of two
different clips within L2 0.018 m (mean joint distance <= 5 mm is implied-ish; verified later). For every
clip pair with >= 5 hits, the dominant frame offset is taken and the overlap is verified: >= 60 frames of
overlap where the per-frame mean joint distance (root-relative) is < 5 mm for at least 90 % of the
overlap, and the matched window is not static (pose std over the overlap > 2 mm). Query also includes
left/right mirrored copies of val / test / eval clips against everything (mirror = a data-augmentation copy).
A pair is *cross-split* if its clips are in different splits. Removal rule: a val / test / eval clip that
matches any clip in a different split is dropped from val / test / eval (train copies stay); val-vs-test
duplicates drop the val copy.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.spatial import cKDTree

from validation.sprint import conventions as C
from validation.sprint import splits as S

STRIDE, RADIUS, MIN_HITS = 10, 0.018, 5
JOINT_TOL, MIN_OVERLAP, FRAC_OK, MOTION_STD = 0.005, 60, 0.9, 0.002


def feat(pos: np.ndarray) -> np.ndarray:
    rel = (pos - pos[:, :1])[:, 1:]
    return rel.reshape(len(pos), -1).astype(np.float32)  # [T, 39]


def load(cid: str):
    clip = C.load_clip(S.clip_path(cid))
    return cid, feat(clip.positions), feat(C.mirror_positions(clip.positions))


def main() -> int:
    doc = S.load_splits()
    split_of: dict[str, str] = {}
    for k, v in doc["cmu"]["clips"].items():
        for c in v:
            split_of[c] = f"cmu_{k}"
    for k, v in doc["xsens"]["clips"].items():
        for c in v:
            split_of[c] = f"xsens_{k}"
    ids = sorted(split_of)
    with ProcessPoolExecutor(14) as ex:
        res = list(ex.map(load, ids, chunksize=8))
    F = {cid: f for cid, f, _ in res}
    M = {cid: m for cid, _, m in res}
    print("loaded", len(F), "clips", sum(len(f) for f in F.values()), "frames", flush=True)
    pts, owner, fidx = [], [], []
    for i, cid in enumerate(ids):
        f = F[cid][::STRIDE]
        pts.append(f)
        owner += [i] * len(f)
        fidx += list(range(0, len(F[cid]), STRIDE))
    pts = np.concatenate(pts)
    owner, fidx = np.array(owner), np.array(fidx)
    tree = cKDTree(pts)
    print("tree", pts.shape, flush=True)
    held = [c for c in ids if not split_of[c].endswith("_train")]
    pairs: dict[tuple, Counter] = defaultdict(
        Counter
    )  # (query, other, mirrored) -> Counter(offset)
    for mirrored in (False, True):
        for cid in held:
            q = (M if mirrored else F)[cid][::STRIDE]
            qi = np.arange(0, len(M[cid] if mirrored else F[cid]), STRIDE)
            hits = tree.query_ball_point(q, RADIUS, workers=8)
            for k, hs in enumerate(hits):
                for h in hs:
                    o = ids[owner[h]]
                    if o == cid or (mirrored and False):
                        continue
                    # offset: other_frame - query_frame (quantised to the stride)
                    pairs[(cid, o, mirrored)][int(fidx[h] - qi[k])] += 1
    print("candidate pairs", len(pairs), flush=True)
    found = []
    for (a, b, mir), cnt in pairs.items():
        if split_of[a] == split_of[b]:
            continue
        # dominant offset +- stride (the hit grids are not aligned)
        off, n = max(
            ((o, sum(cnt[o + d] for d in range(-STRIDE, STRIDE + 1))) for o in cnt),
            key=lambda t: t[1],
        )
        if n < MIN_HITS:
            continue
        fa = (M if mir else F)[a]
        fb = F[b]
        best = None
        for o in range(off - STRIDE, off + STRIDE + 1):
            lo, hi = max(0, -o), min(len(fa), len(fb) - o)  # query frame t <-> other frame t+o
            if hi - lo < MIN_OVERLAP:
                continue
            d = np.linalg.norm((fa[lo:hi] - fb[lo + o : hi + o]).reshape(-1, 13, 3), axis=-1).mean(
                1
            )
            ok = float((d < JOINT_TOL).mean())
            if best is None or ok > best[0]:
                best = (ok, o, hi - lo, float(d.mean()), float(fa[lo:hi].std(0).mean()))
        if (
            best
            and best[0] >= 0.2
            and not (best[0] >= FRAC_OK and best[2] >= MIN_OVERLAP and best[4] > MOTION_STD)
        ):
            print(
                "NEAR-MISS",
                a,
                b,
                mir,
                f"ok-frac {best[0]:.2f} overlap {best[2]} meanmm {best[3] * 1000:.1f}",
                flush=True,
            )
        if best and best[0] >= FRAC_OK and best[2] >= MIN_OVERLAP and best[4] > MOTION_STD:
            found.append(
                dict(
                    held=a,
                    other=b,
                    mirrored=mir,
                    held_split=split_of[a],
                    other_split=split_of[b],
                    offset=best[1],
                    overlap_frames=best[2],
                    frac_below_5mm=round(best[0], 4),
                    mean_joint_dist_mm=round(best[3] * 1000, 3),
                    frames_a=len(F[a]),
                    frames_b=len(F[b]),
                )
            )
    found.sort(key=lambda r: (r["held_split"], r["held"], r["other"]))
    json.dump(found, open(sys.argv[1], "w"), indent=1)
    print("duplicate pairs (cross-split):", len(found))
    rank = {"cmu_val": 1, "xsens_val_subject": 1, "cmu_test": 2, "xsens_eval": 2}
    drop = set()
    for r in found:
        a, b, sa, sb = r["held"], r["other"], r["held_split"], r["other_split"]
        if sb.endswith("_train"):
            drop.add(a)  # held-out copy of a train clip
        elif rank.get(sa, 0) < rank.get(sb, 0):
            drop.add(a)  # val copy of a test/eval clip
        elif rank.get(sa, 0) == rank.get(sb, 0) and a < b:
            drop.add(a)
    print(Counter(split_of[c] for c in drop), "dropped clips:", len(drop))
    json.dump(sorted(drop), open(sys.argv[2], "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
