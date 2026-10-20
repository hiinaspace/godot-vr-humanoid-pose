"""Canonical skeleton, units, axes and data loaders for the sprint oracle (conventions-v1).

Everything else in ``validation.sprint`` imports from here. LOCKED: see LOCK.md.

Canonical convention
--------------------
* Units: metres, seconds, degrees only in reports.
* World frame: right-handed, +Y up, floor at Y = 0. A body facing direction ``f`` (horizontal) has
  its left side at ``up x f`` (so the zero-rotation T-pose faces +Z with the left hand at +X).
  Clip heading is arbitrary (CMU walks anywhere); only handedness/up are fixed.
* Frame rate: 60 Hz (CMU is stride-2 downsampled from 120 Hz, Xsens is native).
* Quaternions: (x, y, z, w), unit, active (they rotate column vectors).
* Joints: the 14 joints that both caches provide. The memo's "17" assumed neck, pelvis and
  upper-leg joints that the cached canonical skeleton does not have, so the oracle has 14:
  hips, spine, chest, head, L/R shoulder, L/R elbow, L/R hand, L/R knee, L/R foot.
  ``shoulder`` is the glenohumeral joint (CMU clavicle end / Xsens LeftShoulder child offset),
  ``hand`` is the wrist, ``foot`` is the ankle, ``knee`` is the knee joint.
* Inputs: head, left_hand, right_hand (the three tracked joints). Everything else is predicted.
  MPJPE-full = the 11 non-input joints; upper = 6 (chest, spine, L/R shoulder, L/R elbow);
  lower = 5 (hips, L/R knee, L/R foot).
* Rotations ("incoming-bone" convention): ``R[j]`` is the global orientation of the frame that
  carries the offset parent(j) -> j, i.e. ``pos[j] = pos[parent(j)] + R[j] @ offset_j`` with
  ``offset_j`` constant. This is native for CMU ASF; for Xsens BVH it is ``R_bvh[parent(j)]``.
  Parent-relative rotation is ``inv(R[parent]) * R[j]``; the root entry stays global.
  Identity everywhere is the dataset rest pose (T-pose, +Z forward). Upper-body rest directions
  agree between the two datasets to within ~8 degrees; leg rest directions differ by 15-20
  degrees (CMU's femur starts at an offset hip joint), so FK templates are per dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

CONVENTIONS_VERSION = "conventions-v1"
FPS = 60.0
UP = np.array([0.0, 1.0, 0.0])
FORWARD_REST = np.array([0.0, 0.0, 1.0])
LEFT_REST = np.array([1.0, 0.0, 0.0])  # == cross(UP, FORWARD_REST)
FOOT_HEIGHT_M = 0.08
CMU_CALIB_FRAMES = (
    10  # CMU clips start in arbitrary poses; only pose-invariant lengths are measured
)
# standing head-joint height / (pose-invariant bone-chain length incl. foot height), per dataset;
# median over train clips (160 CMU, 94 Xsens; sd 0.09 / 0.01).
HEIGHT_FROM_CHAIN = {"cmu": 0.967, "xsens": 1.000}

JOINTS: tuple[str, ...] = (
    "hips",
    "spine",
    "chest",
    "head",
    "left_shoulder",
    "left_elbow",
    "left_hand",
    "right_shoulder",
    "right_elbow",
    "right_hand",
    "left_knee",
    "left_foot",
    "right_knee",
    "right_foot",
)
J = {name: i for i, name in enumerate(JOINTS)}
NJ = len(JOINTS)
PARENT_NAME: dict[str, str | None] = {
    "hips": None,
    "spine": "hips",
    "chest": "spine",
    "head": "chest",
    "left_shoulder": "chest",
    "left_elbow": "left_shoulder",
    "left_hand": "left_elbow",
    "right_shoulder": "chest",
    "right_elbow": "right_shoulder",
    "right_hand": "right_elbow",
    "left_knee": "hips",
    "left_foot": "left_knee",
    "right_knee": "hips",
    "right_foot": "right_knee",
}
PARENT = np.array([-1 if (p := PARENT_NAME[n]) is None else J[p] for n in JOINTS])
assert all(PARENT[i] < i for i in range(1, NJ))  # parents precede children

INPUT_JOINTS = ("head", "left_hand", "right_hand")
INPUT_IDX = np.array([J[n] for n in INPUT_JOINTS])
UPPER_JOINTS = ("chest", "spine", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow")
LOWER_JOINTS = ("hips", "left_knee", "right_knee", "left_foot", "right_foot")
NONINPUT_JOINTS = UPPER_JOINTS + LOWER_JOINTS
UPPER_IDX = np.array([J[n] for n in UPPER_JOINTS])
LOWER_IDX = np.array([J[n] for n in LOWER_JOINTS])
NONINPUT_IDX = np.array([J[n] for n in NONINPUT_JOINTS])
ROOTREL_IDX = np.array([J[n] for n in NONINPUT_JOINTS if n != "hips"])
# The pre-sprint reports' "upper_body" joint set (includes head and hands; head carries the
# fixed tracker offset). Kept only to compare against documented numbers.
LEGACY_UPPER_JOINTS = (
    "head",
    "chest",
    "spine",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_hand",
    "right_hand",
)
LEGACY_UPPER_IDX = np.array([J[n] for n in LEGACY_UPPER_JOINTS])
MIRROR_PERM = np.array(
    [
        J[n.replace("left_", "TMP_").replace("right_", "left_").replace("TMP_", "right_")]
        for n in JOINTS
    ]
)

# Input array layout, per frame: [pos(3) quat_xyzw(4)] for head, left_hand, right_hand, then
# one validity flag per tracker (1 = device reports tracking, incl. "hold-last" dropouts).
INPUT_DIM = 3 * 7 + 3
TRACKER_LOCAL_OFFSET_M = (0.0, 0.08, 0.02)  # head tracker sits above/forward of the head joint


def input_slice(tracker: str) -> slice:
    k = INPUT_JOINTS.index(tracker)
    return slice(7 * k, 7 * k + 7)


# --- FK template (per dataset): (unit rest direction in the incoming-bone frame, length / height)
# Measured as per-clip medians over random train clips (40 CMU, 30 Xsens); height is the
# calibration height defined in oracle.make_calib.
TEMPLATES: dict[str, dict[str, tuple[tuple[float, float, float], float]]] = {
    "cmu": {
        "spine": ((0.016, 0.999, -0.045), 0.1475),
        "chest": ((0.010, 0.999, 0.033), 0.0741),
        "head": ((0.083, 0.988, -0.129), 0.1815),
        "left_shoulder": ((0.947, 0.319, -0.045), 0.1268),
        "right_shoulder": ((-0.942, 0.328, -0.064), 0.1235),
        "left_elbow": ((1.000, 0.000, 0.000), 0.1810),
        "right_elbow": ((-1.000, 0.000, 0.000), 0.1857),
        "left_hand": ((1.000, 0.000, 0.000), 0.1796),
        "right_hand": ((-1.000, 0.000, 0.000), 0.1815),
        "left_knee": ((0.490, -0.872, 0.000), 0.3235),
        "right_knee": ((-0.493, -0.870, -0.007), 0.3255),
        "left_foot": ((0.341, -0.938, 0.056), 0.2799),
        "right_foot": ((-0.342, -0.939, 0.033), 0.2785),
    },
    "xsens": {
        "spine": ((-0.003, 1.000, 0.021), 0.2013),
        "chest": ((0.000, 1.000, 0.001), 0.0645),
        "head": ((0.000, 0.995, 0.098), 0.1427),
        "left_shoulder": ((0.931, 0.365, 0.015), 0.1108),
        "right_shoulder": ((-0.922, 0.386, 0.017), 0.1085),
        "left_elbow": ((1.000, 0.000, 0.000), 0.1832),
        "right_elbow": ((-1.000, 0.000, 0.000), 0.1832),
        "left_hand": ((1.000, 0.000, 0.000), 0.1540),
        "right_hand": ((-1.000, 0.000, 0.000), 0.1540),
        "left_knee": ((0.312, -0.934, 0.176), 0.2761),
        "right_knee": ((-0.296, -0.938, 0.181), 0.2793),
        "left_foot": ((0.000, -1.000, 0.001), 0.2736),
        "right_foot": ((0.000, -1.000, 0.001), 0.2736),
    },
}
ARM_JOINTS = ("left_elbow", "left_hand", "right_elbow", "right_hand")


# --- quaternion helpers (numpy, xyzw) -------------------------------------------------------


def qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = np.moveaxis(a, -1, 0)
    bx, by, bz, bw = np.moveaxis(b, -1, 0)
    return np.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        axis=-1,
    )


def qconj(q: np.ndarray) -> np.ndarray:
    return q * np.array([-1.0, -1.0, -1.0, 1.0])


def qnorm(q: np.ndarray) -> np.ndarray:
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def qrotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    u = q[..., :3]
    w = q[..., 3:4]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def qgeodesic_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Geodesic angle in degrees between unit quaternions (sign-invariant)."""
    dq = qmul(qconj(a), b)  # atan2 form: accurate near 0 (arccos of a dot product is not)
    return np.degrees(2.0 * np.arctan2(np.linalg.norm(dq[..., :3], axis=-1), np.abs(dq[..., 3])))


def local_rotations(global_rot: np.ndarray) -> np.ndarray:
    """[..., NJ, 4] global -> parent-relative (root stays global)."""
    out = global_rot.copy()
    out[..., 1:, :] = qmul(qconj(global_rot[..., PARENT[1:], :]), global_rot[..., 1:, :])
    return out


def global_rotations(local_rot: np.ndarray) -> np.ndarray:
    out = local_rot.copy()
    for j in range(1, NJ):
        out[..., j, :] = qmul(out[..., PARENT[j], :], local_rot[..., j, :])
    return out


def template_offsets(dataset: str, height_m: float, arm_length_m: float) -> np.ndarray:
    """[NJ, 3] rest offsets (incoming-bone frame) scaled to a calibration. Root row is zero."""
    tpl = TEMPLATES[dataset]
    arm_fracs = sum(tpl[n][1] for n in ARM_JOINTS) / 2.0  # elbow + hand, per arm
    arm_scale = arm_length_m / (arm_fracs * height_m)
    out = np.zeros((NJ, 3))
    for name, (direction, frac) in tpl.items():
        s = height_m * frac * (arm_scale if name in ARM_JOINTS else 1.0)
        out[J[name]] = np.array(direction) * s
    return out


def forward_kinematics(
    local_rot: np.ndarray, root_pos: np.ndarray, offsets: np.ndarray
) -> np.ndarray:
    """local_rot [T,NJ,4], root_pos [T,3], offsets [NJ,3] -> global positions [T,NJ,3]."""
    grot = global_rotations(local_rot)
    pos = np.zeros(local_rot.shape[:-1] + (3,))
    pos[:, 0] = root_pos
    for j in range(1, NJ):
        pos[:, j] = pos[:, PARENT[j]] + qrotate(grot[:, j], offsets[j])
    return pos


# --- clip loading ---------------------------------------------------------------------------

_XSENS_CMU_SOURCE_ORDER = None


@dataclass(slots=True)
class ClipData:
    dataset: str  # "cmu" | "xsens"
    clip_id: str  # "<dataset>/<subject>/<clip>"
    subject: str
    clip: str
    positions: np.ndarray  # [T, NJ, 3] metres, canonical frame, 60 Hz
    rotations: np.ndarray  # [T, NJ, 4] native global quaternions as cached (xyzw)
    activity_class: str
    environment_split: str
    source_frame_rate_hz: float
    calib_positions: (
        np.ndarray
    )  # [K, NJ, 3] calibration pose frames (not part of the evaluated clip)

    @property
    def n_frames(self) -> int:
        return self.positions.shape[0]

    def canonical_rotations(self) -> np.ndarray:
        """Global incoming-bone rotations (see module docstring)."""
        return incoming_bone_rotations(self.dataset, self.rotations)


def incoming_bone_rotations(dataset: str, native: np.ndarray) -> np.ndarray:
    if dataset == "cmu":
        return native
    if dataset == "xsens":
        out = native.copy()
        out[..., 1:, :] = native[..., PARENT[1:], :]
        return out
    raise ValueError(f"unknown dataset {dataset!r}")


def dataset_of_path(path: Path) -> str:
    return "xsens" if "sparseposer" in path.as_posix() else "cmu"


def load_clip(path: str | Path, dataset: str | None = None) -> ClipData:
    """Load one cached canonical clip and return it at 60 Hz in canonical joint order."""
    path = Path(path)
    dataset = dataset or dataset_of_path(path)
    frame = pl.read_parquet(path)
    ts = frame["timestamp_s"].to_numpy()
    src_hz = 1.0 / float(np.median(np.diff(ts))) if len(ts) > 1 else FPS
    if abs(src_hz - 120.0) < 3.0:
        stride = 2
    elif abs(src_hz - 60.0) < 2.0:
        stride = 1
    else:
        raise ValueError(f"{path}: unsupported frame rate {src_hz:.1f} Hz")
    pos = np.stack(
        [frame.select([f"{n}_px", f"{n}_py", f"{n}_pz"]).to_numpy() for n in JOINTS], axis=1
    )[::stride]
    rot = np.stack(
        [frame.select([f"{n}_qx", f"{n}_qy", f"{n}_qz", f"{n}_qw"]).to_numpy() for n in JOINTS],
        axis=1,
    )[::stride]
    rot = qnorm(rot)
    if dataset == "xsens":
        # Every Xsens BVH starts with a T-pose reference frame at the origin, followed by a 3-7 m
        # jump into the take. Use it as the calibration pose and drop it from the evaluated clip.
        calib, pos, rot = pos[:1].copy(), pos[1:], rot[1:]
    else:
        calib = pos[:CMU_CALIB_FRAMES].copy()
    subject = path.parent.name.split("=", 1)[-1]
    clip = path.stem
    return ClipData(
        dataset=dataset,
        clip_id=f"{dataset}/{subject}/{clip}",
        subject=subject,
        clip=clip,
        positions=np.ascontiguousarray(pos, dtype=np.float64),
        rotations=np.ascontiguousarray(rot, dtype=np.float64),
        activity_class=str(frame["activity_class"][0])
        if "activity_class" in frame.columns
        else "unknown",
        environment_split=str(frame["environment_split"][0])
        if "environment_split" in frame.columns
        else "unknown",
        source_frame_rate_hz=src_hz,
        calib_positions=np.ascontiguousarray(calib, dtype=np.float64),
    )


def mirror_positions(pos: np.ndarray) -> np.ndarray:
    """Reflect a pose sequence through the x = 0 plane and swap left/right labels."""
    out = pos[..., MIRROR_PERM, :].copy()
    out[..., 0] *= -1.0
    return out
