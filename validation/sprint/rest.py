"""Canonical rest pose and cross-dataset rotation targets (rest-v1).

``conventions.py`` (locked) keeps one FK template per dataset because leg rest directions differ by
15-20 degrees between CMU and Xsens. For learning rotations we want one target space, so this module
defines a single canonical rest pose (unit bone directions, +Z forward, +X left, Y up) and re-expresses
each dataset's rotations in it:

    Rc[j] = swing_fix( R_ds[j] @ P_j ),   P_j = minimal rotation taking the canonical rest direction
                                          of bone parent(j)->j onto that dataset's rest direction.

``swing_fix`` removes the residual swing (template direction vs. the clip's real bone direction) so
that ``FK(canonical rest dirs x GT bone lengths, Rc)`` reproduces the GT positions exactly; twist about
the bone is inherited from the dataset rotation. The root keeps the dataset's orientation. The
"incoming-bone" convention of conventions.py is unchanged: ``pos[j] = pos[p] + Rc[j] @ (dir_j * len_j)``.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from validation.sprint import conventions as C

REST_VERSION = "rest-v1"


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _canonical_dirs() -> np.ndarray:
    """Mean of the two datasets' template directions, left/right symmetrised."""
    d = np.zeros((C.NJ, 3))
    for n, i in C.J.items():
        if n != "hips":
            d[i] = _unit(sum(_unit(C.TEMPLATES[ds][n][0]) for ds in ("cmu", "xsens")))
    for n, i in C.J.items():
        if n.startswith("left_"):
            r = C.J["right_" + n[5:]]
            m = _unit(d[i] + d[r] * np.array([-1.0, 1.0, 1.0]))
            d[i], d[r] = m, m * np.array([-1.0, 1.0, 1.0])
        elif n in ("spine", "chest", "head"):
            d[i] = _unit(d[i] * np.array([0.0, 1.0, 1.0]))
    return d


REST_DIRS = _canonical_dirs()  # [NJ, 3], root row zero


def dataset_dirs(dataset: str) -> np.ndarray:
    d = np.zeros((C.NJ, 3))
    for n, (v, _) in C.TEMPLATES[dataset].items():
        d[C.J[n]] = _unit(v)
    d[0] = REST_DIRS[0]
    return d


def min_rot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Minimal-rotation quaternions (xyzw) taking unit vectors a onto b (broadcasting)."""
    a, b = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float))
    dot = np.sum(a * b, -1)
    q = np.concatenate([np.cross(a, b), (1.0 + dot)[..., None]], -1)
    anti = dot < -1.0 + 1e-9
    if np.any(anti):  # 180 degrees: any axis perpendicular to a
        helper = np.where(np.abs(a[..., :1]) < 0.9, np.array([1.0, 0, 0]), np.array([0, 1.0, 0]))
        ax = _unit(np.cross(a, helper))
        q = np.where(anti[..., None], np.concatenate([ax, np.zeros_like(dot)[..., None]], -1), q)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    ident = np.array([0.0, 0, 0, 1.0])  # zero-vector inputs (the root row) -> identity
    return np.where(norm < 1e-12, ident, q / np.maximum(norm, 1e-12))


def bone_lengths(pos: np.ndarray) -> np.ndarray:
    """[NJ] median parent->child distance over the clip (root entry 0)."""
    out = np.zeros(C.NJ)
    out[1:] = np.median(np.linalg.norm(pos[:, 1:] - pos[:, C.PARENT[1:]], axis=-1), axis=0)
    return out


def rest_offsets(lengths: np.ndarray) -> np.ndarray:
    return REST_DIRS * lengths[:, None]


def canonical_global(dataset: str, incoming: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """[T,NJ,4] canonical global (incoming-bone) rotations from incoming-bone dataset rotations + GT pos."""
    p = min_rot(REST_DIRS, dataset_dirs(dataset))  # [NJ,4]
    out = C.qmul(incoming, p[None])
    vec = pos[:, 1:] - pos[:, C.PARENT[1:]]
    vec = vec / np.maximum(np.linalg.norm(vec, axis=-1, keepdims=True), 1e-9)
    cur = C.qrotate(out[:, 1:], REST_DIRS[1:][None])
    fix = min_rot(cur, vec)  # swing correction, applied in the world frame
    out[:, 1:] = C.qmul(fix, out[:, 1:])
    out[:, 0] = incoming[:, 0]
    return C.qnorm(out)


def canonical_local(clip: C.ClipData) -> np.ndarray:
    return C.local_rotations(
        canonical_global(clip.dataset, clip.canonical_rotations(), clip.positions)
    )


def fk(local_rot: np.ndarray, root_pos: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    return C.forward_kinematics(local_rot, root_pos, rest_offsets(lengths))


def swing_from_positions(pos: np.ndarray) -> np.ndarray:
    """Local swing-only rotations [T,NJ,4] derived from positions alone (twist unknown -> 0)."""
    vec = pos[:, 1:] - pos[:, C.PARENT[1:]]
    vec = vec / np.maximum(np.linalg.norm(vec, axis=-1, keepdims=True), 1e-9)
    g = np.zeros(pos.shape[:-1] + (4,))
    g[..., 3] = 1.0
    g[:, 1:] = min_rot(REST_DIRS[1:][None], vec)
    return C.local_rotations(g)


def mirror_clip(clip: C.ClipData) -> C.ClipData:
    """Left/right mirror of a clip (x -> -x). Quaternions (x,y,z,w) -> (x,-y,-z,w) with joints swapped;
    valid because left/right rest directions and local frames are mirror images (conventions.py)."""
    rot = clip.rotations[:, C.MIRROR_PERM] * np.array([1.0, -1.0, -1.0, 1.0])
    return dataclasses.replace(
        clip,
        clip_id=clip.clip_id + "_mirror",
        clip=clip.clip + "_mirror",
        positions=C.mirror_positions(clip.positions),
        rotations=np.ascontiguousarray(rot),
        calib_positions=C.mirror_positions(clip.calib_positions),
    )


def forearm_probe(clip: C.ClipData) -> float:
    """Median angle (deg) between the hand tracker's bone axis and the elbow->hand direction."""
    angs = []
    for side, sign in (("left", 1.0), ("right", -1.0)):
        h, e = C.J[f"{side}_hand"], C.J[f"{side}_elbow"]
        axis = C.qrotate(clip.rotations[:, h], np.array([sign, 0.0, 0.0]))
        v = clip.positions[:, h] - clip.positions[:, e]
        v = v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)
        angs.append(np.degrees(np.arccos(np.clip(np.sum(axis * v, -1), -1, 1))))
    return float(np.median(np.concatenate(angs)))
