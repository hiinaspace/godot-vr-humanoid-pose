"""Sprint trainer: data -> model -> loss -> loop -> checkpoint -> oracle eval, one file.

Architecture after HMD-Poser (Dai et al., CVPR 2024, MIT): per-token feature embeddings, 2 blocks of
[per-token causal LSTM over time, 3-layer transformer over the tokens]. Ported, not copied; weights
are NOT used (AMASS-trained, non-commercial). Differences: HMD-only tokens (head, hands, hands in
head frame + a body token; no zero foot/pelvis tokens), canonical 14-joint rest pose (rest.py)
instead of SMPL, no shape head. ``avatarposer`` is an AvatarPoser-style 40-frame causal transformer.

Commands (all via ``uv run python -m validation.sprint.train <cmd>``; GPU jobs under flock):
  build-data            preprocess CMU/Xsens train+val once into .cache/sprint/data
  train --run NAME      resumable; stops at --time-min and resumes on the next call
  selftest              canonical<->world pipeline round trip + left/right probe
  overfit               one-clip overfit through the oracle path
  probe-wrist           CMU forearm-vs-hand-frame probe (one line)
Evaluate a checkpoint: ``python -m validation.sprint.eval --ckpt PATH --split val``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from validation.sprint import conventions as C
from validation.sprint import oracle as O
from validation.sprint import oracle_v2 as O2
from validation.sprint import rest as R
from validation.sprint import splits as S

DATA = S.REPO / ".cache/sprint/data"
RUNS = S.REPO / ".cache/sprint/runs"
MLFLOW_DB = S.REPO / ".cache/ml/mlflow.db"
REPORTS = S.REPO / "validation/reports/sprint-2026-10"
L = 40  # causal window (frames)
NT = 5  # tokens: head, lhand, rhand, lhand-in-head, rhand-in-head
OUT = C.NJ * 6 + 3
ID6 = torch.tensor([1.0, 0, 0, 0, 1.0, 0])
REST_T = torch.tensor(R.REST_DIRS, dtype=torch.float32)
PARENT = [int(p) for p in C.PARENT]
HANDS = [C.J["left_hand"], C.J["right_hand"]]
LOWER = [int(i) for i in C.LOWER_IDX]
UPPER = [int(i) for i in C.UPPER_IDX]
NONIN = [int(i) for i in C.NONINPUT_IDX]

# --- rotation helpers (torch) ----------------------------------------------------------------


def qmat(q: torch.Tensor) -> torch.Tensor:
    x, y, z, w = q.unbind(-1)
    r = [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)]
    r += [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)]
    r += [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]
    return torch.stack(r, -1).reshape(q.shape[:-1] + (3, 3))


def mat2quat(m: torch.Tensor) -> torch.Tensor:
    a = lambda i, j: m[..., i, j]  # noqa: E731
    d = torch.stack(
        [
            1 + a(0, 0) + a(1, 1) + a(2, 2),
            1 + a(0, 0) - a(1, 1) - a(2, 2),
            1 - a(0, 0) + a(1, 1) - a(2, 2),
            1 - a(0, 0) - a(1, 1) + a(2, 2),
        ],
        -1,
    )
    cand = torch.stack(  # wxyz, unnormalised, one row per branch
        [
            torch.stack([d[..., 0], a(2, 1) - a(1, 2), a(0, 2) - a(2, 0), a(1, 0) - a(0, 1)], -1),
            torch.stack([a(2, 1) - a(1, 2), d[..., 1], a(0, 1) + a(1, 0), a(0, 2) + a(2, 0)], -1),
            torch.stack([a(0, 2) - a(2, 0), a(0, 1) + a(1, 0), d[..., 2], a(1, 2) + a(2, 1)], -1),
            torch.stack([a(1, 0) - a(0, 1), a(0, 2) + a(2, 0), a(1, 2) + a(2, 1), d[..., 3]], -1),
        ],
        -2,
    )
    k = d.argmax(-1)
    q = cand.gather(-2, k[..., None, None].expand(*k.shape, 1, 4)).squeeze(-2)
    q = F.normalize(q, dim=-1)
    return torch.stack([q[..., 1], q[..., 2], q[..., 3], q[..., 0]], -1)


def sixd(m: torch.Tensor) -> torch.Tensor:
    return torch.cat([m[..., :, 0], m[..., :, 1]], -1)


def sixd2mat(x: torch.Tensor) -> torch.Tensor:
    b1 = F.normalize(x[..., :3], dim=-1)
    b2 = F.normalize(x[..., 3:] - (b1 * x[..., 3:]).sum(-1, keepdim=True) * b1, dim=-1)
    return torch.stack([b1, b2, torch.cross(b1, b2, dim=-1)], -1)


def euler_mats(ang: torch.Tensor) -> torch.Tensor:
    """ang [...,3] radians -> Rx(a0) Ry(a1) Rz(a2) as [...,3,3]."""
    c, s = torch.cos(ang), torch.sin(ang)
    o, z = torch.ones_like(c[..., 0]), torch.zeros_like(c[..., 0])
    rx = torch.stack([o, z, z, z, c[..., 0], -s[..., 0], z, s[..., 0], c[..., 0]], -1)
    ry = torch.stack([c[..., 1], z, s[..., 1], z, o, z, -s[..., 1], z, c[..., 1]], -1)
    rz = torch.stack([c[..., 2], -s[..., 2], z, s[..., 2], c[..., 2], z, z, z, o], -1)
    sh = ang.shape[:-1] + (3, 3)
    return rx.reshape(sh) @ ry.reshape(sh) @ rz.reshape(sh)


# --- data ------------------------------------------------------------------------------------


def _prep_clip(job):
    """One clip -> compact float32 arrays (clean inputs-v1 stream, canonical targets)."""
    torch.set_num_threads(1)
    cid, mirror, max_frames = job
    clip = C.load_clip(S.clip_path(cid))
    if mirror:
        clip = R.mirror_clip(clip)
    if max_frames:
        clip.positions, clip.rotations = clip.positions[:max_frames], clip.rotations[:max_frames]
    noise, O.CALIB_NOISE = O.CALIB_NOISE, 0.0  # true (un-noised) calibration scalars
    cal = O.make_calib(clip)
    O.CALIB_NOISE = noise
    loc = R.canonical_local(clip)
    r6 = sixd(qmat(torch.from_numpy(loc))).numpy().astype(np.float32)
    return dict(
        id=clip.clip_id,
        ds=0 if clip.dataset == "cmu" else 1,
        calib=np.array([cal.height_m, cal.arm_length_m, cal.shoulder_width_m], np.float32),
        lengths=R.bone_lengths(clip.positions).astype(np.float32),
        inp=O.make_inputs(clip, "clean").astype(np.float32),
        rot6=r6,
        pos=clip.positions.astype(np.float32),
    )


def build_set(name: str, jobs: list, workers: int = 14, out: Path = DATA) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_prep_clip, jobs, chunksize=4))
    res.sort(key=lambda r: (r["ds"], r["id"]))  # CMU first, then Xsens
    lens = np.array([len(r["inp"]) for r in res])
    np.savez(
        out / f"{name}_table.npz",
        ids=np.array([r["id"] for r in res]),
        ds=np.array([r["ds"] for r in res]),
        calib=np.stack([r["calib"] for r in res]),
        lengths=np.stack([r["lengths"] for r in res]),
        start=np.concatenate([[0], np.cumsum(lens)[:-1]]),
        length=lens,
    )
    for k in ("inp", "rot6", "pos"):
        np.save(out / f"{name}_{k}.npy", np.concatenate([r[k] for r in res]))
    print(name, len(res), "clips", int(lens.sum()), "frames", flush=True)


def build_data(args) -> None:
    doc = S.load_splits()
    cmu_tr = doc["cmu"]["clips"]["train"]
    xs_tr = doc["xsens"]["clips"]["train"]  # already minus S09, with the cached mirror clips
    mf = args.max_clips
    jobs = [(c, False, None) for c in cmu_tr[:mf]] + [(c, True, None) for c in cmu_tr[:mf]]
    jobs += [(c, False, None) for c in xs_tr[:mf]]
    build_set("train", jobs)
    val = doc["cmu"]["clips"]["val"][:mf] + doc["xsens"]["clips"]["val_subject"][:mf]
    build_set("val", [(c, False, None) for c in val])


class Store:
    """A preprocessed set resident on the GPU, plus window sampling."""

    def __init__(self, name: str, dev: str, data: Path = DATA):
        t = np.load(data / f"{name}_table.npz")
        self.dev, n = dev, int(t["length"].sum())
        self.ds, self.calib = torch.tensor(t["ds"]), torch.tensor(t["calib"])
        self.lengths = torch.tensor(t["lengths"]).to(dev)
        self.ids = list(t["ids"])
        start, length = t["start"], t["length"]
        cidx = np.repeat(np.arange(len(start)), length).astype(np.int64)
        self.cidx = torch.from_numpy(cidx).to(dev)
        self.cstart = torch.from_numpy(start[cidx]).to(dev)
        self.start, self.length = start, length
        self.inp, self.rot6, self.pos = (
            torch.from_numpy(np.load(data / f"{name}_{k}.npy")).to(dev)
            for k in ("inp", "rot6", "pos")
        )
        self.ds_dev, self.calib_dev = self.ds.to(dev), self.calib.to(dev)
        self.range = {}
        for d in (0, 1):
            m = np.where(t["ds"] == d)[0]
            if len(m):
                self.range[d] = (int(start[m[0]]), int(start[m[-1]] + length[m[-1]]))
        assert self.inp.shape[0] == n
        self.noise = (
            torch.rand(len(start), 3, generator=torch.Generator().manual_seed(7)) * 0.1 + 0.95
        ).to(dev)

    def sample(self, bs: int, xs_frac: float, gen: torch.Generator) -> torch.Tensor:
        """Uniform end frames; ``xs_frac`` of them from Xsens, the rest from CMU."""
        if len(self.range) == 1:
            xs_frac = float(1 in self.range)
        out = torch.zeros(bs, dtype=torch.long, device=self.dev)
        use_xs = torch.rand(bs, device=self.dev, generator=gen) < xs_frac
        for d, m in ((0, ~use_xs), (1, use_xs)):
            if d in self.range:
                out = torch.where(
                    m, torch.randint(*self.range[d], (bs,), device=self.dev, generator=gen), out
                )
        return out


NORM = {"mu": [1.7, 0.58, 0.35], "sd": [0.1, 0.05, 0.04]}  # static-scalar normalisation (fixed)


def static_feats(calib: torch.Tensor, ds: torch.Tensor, skel: bool) -> torch.Tensor:
    mu, sd = (
        torch.tensor(NORM["mu"], device=calib.device),
        torch.tensor(NORM["sd"], device=calib.device),
    )
    oh = F.one_hot(ds.long(), 2).float() * float(skel)
    return torch.cat([(calib - mu) / sd, oh], -1)


def wrist_aug(bs: int, w: int, dev, gen: torch.Generator) -> torch.Tensor:
    """Random wrist rotation for hand trackers: swing +-35 deg (y,z), twist +-20 deg (x); drifts
    linearly between two draws over the window. Returns [B,W,2,3,3] (applied R @ Rw)."""
    lim = torch.tensor([20.0, 35.0, 35.0], device=dev) * math.pi / 180
    a = (torch.rand(bs, 2, 2, 3, device=dev, generator=gen) * 2 - 1) * lim
    t = torch.linspace(0, 1, w, device=dev)[None, :, None, None]
    ang = a[:, 0:1] * (1 - t) + a[:, 1:2] * t  # [B,W,2,3]
    return euler_mats(ang)


def make_feats(raw: torch.Tensor, wrist: torch.Tensor | None = None):
    """raw [B,W,24] tracker stream -> feats [B,W-1,NT,18], Q [B,3,3], origin [B,3], head_pc [B,W-1,3].

    Head-relative and heading-canonical: x/z origin and yaw come from the head at the window's last
    frame (causal), height stays absolute. Token feature = [rot6, angvel6, pos, linvel]."""
    P = torch.stack([raw[..., 0:3], raw[..., 7:10], raw[..., 14:17]], 2)
    Rm = qmat(torch.stack([raw[..., 3:7], raw[..., 10:14], raw[..., 17:21]], 2))
    if wrist is not None:
        Rm = torch.cat([Rm[:, :, :1], Rm[:, :, 1:] @ wrist], 2)
    f, u = Rm[:, -1, 0, :, 2], Rm[:, -1, 0, :, 1]
    fxz, uxz = f[:, [0, 2]], u[:, [0, 2]]
    wgt = (1 - fxz.norm(dim=-1, keepdim=True) / 0.5).clamp(0, 1)  # looking steeply up/down
    d = fxz - wgt * torch.sign(f[:, 1:2]) * uxz
    th = torch.atan2(d[:, 0], d[:, 1])
    c, s, o, z = torch.cos(th), torch.sin(th), torch.ones_like(th), torch.zeros_like(th)
    Q = torch.stack([c, z, -s, z, o, z, s, z, c], -1).reshape(-1, 3, 3)  # Ry(-yaw)
    origin = P[:, -1, 0].clone()
    origin[:, 1] = 0
    Pc = torch.einsum("bij,btkj->btki", Q, P - origin[:, None, None])
    Rc = Q[:, None, None] @ Rm
    Rh = Rc[:, :, 0]
    ph = torch.einsum("btji,btkj->btki", Rh, Pc[:, :, 1:] - Pc[:, :, :1])
    rh = Rh[:, :, None].transpose(-1, -2) @ Rc[:, :, 1:]
    Pt = torch.cat([Pc, ph], 2)
    Rt = torch.cat([Rc, rh], 2)
    vel = (Pt[:, 1:] - Pt[:, :-1]) * 30.0
    ang = (sixd(Rt[:, 1:] @ Rt[:, :-1].transpose(-1, -2)) - ID6.to(raw.device)) * 10.0
    feats = torch.cat([sixd(Rt[:, 1:]), ang, Pt[:, 1:], vel], -1)
    return feats, Q, origin, Pc[:, 1:, 0]


def window(store: Store, g: torch.Tensor):
    idx = g[:, None] + torch.arange(-L, 1, device=g.device)[None]
    return torch.maximum(idx, store.cstart[g][:, None])  # [B, L+1], clamped at the clip start


def make_targets(store: Store, idx: torch.Tensor, Q, origin, head_pc):
    """GT rot6 [B,L,NJ,6] (root rotated into the canonical frame), hips-from-head [B,L,3], pos."""
    r6 = store.rot6[idx].clone()
    r6[:, :, 0] = sixd(Q[:, None] @ sixd2mat(r6[:, :, 0]))
    pc = torch.einsum("bij,btkj->btki", Q, store.pos[idx] - origin[:, None, None])
    return r6, pc[:, :, 0] - head_pc, pc


# --- models ----------------------------------------------------------------------------------


class HMDPoser(nn.Module):
    def __init__(self, d=256, nstatic=5, layers=3, heads=8, blocks=2, ff=1024):
        super().__init__()
        mk = lambda n: nn.Sequential(nn.Linear(n, d // 4), nn.LeakyReLU())  # noqa: E731
        self.emb = nn.ModuleList([nn.ModuleList([mk(6), mk(6), mk(3), mk(3)]) for _ in range(NT)])
        self.static = nn.Sequential(nn.Linear(nstatic, d), nn.LeakyReLU())
        self.norm = nn.LayerNorm(d, eps=1e-6)
        self.lstm = nn.ModuleList(
            [
                nn.ModuleList([nn.LSTM(d, d, batch_first=True) for _ in range(NT + 1)])
                for _ in range(blocks)
            ]
        )
        enc = lambda: nn.TransformerEncoder(  # noqa: E731
            nn.TransformerEncoderLayer(d, heads, ff, 0.05, batch_first=True),
            layers,
            enable_nested_tensor=False,
        )
        self.spatial = nn.ModuleList([enc() for _ in range(blocks)])
        self.head = nn.Sequential(nn.Linear((NT + 1) * d, 256), nn.LeakyReLU(), nn.Linear(256, OUT))
        for lstm in self.lstm.modules():
            if isinstance(lstm, nn.LSTM):
                for n, p in lstm.named_parameters():
                    if n.startswith("weight"):
                        nn.init.orthogonal_(p)

    def forward(self, feats: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        b, t = feats.shape[:2]
        sl = [slice(0, 6), slice(6, 12), slice(12, 15), slice(15, 18)]
        toks = [
            self.norm(torch.cat([e(feats[:, :, k, s]) for e, s in zip(self.emb[k], sl)], -1))
            for k in range(NT)
        ]
        toks.append(self.norm(self.static(static))[:, None].expand(b, t, -1))
        x = torch.stack(toks, 2)
        for lstm, sp in zip(self.lstm, self.spatial):
            x = torch.stack([lstm[k](x[:, :, k])[0] for k in range(NT + 1)], 2)
            x = sp(x.reshape(b * t, NT + 1, -1)).reshape(b, t, NT + 1, -1)
        return self.head(x.reshape(b, t, -1))


class AvatarPoser(nn.Module):
    """AvatarPoser-style: 54-d tracker features -> 3-layer causal transformer (d=256) over the window."""

    def __init__(self, d=256, nstatic=5, layers=3, heads=8):
        super().__init__()
        self.inp = nn.Linear(3 * 18 + nstatic, d)
        self.pos = nn.Parameter(torch.zeros(1, L, d))
        self.enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, heads, 4 * d, 0.05, batch_first=True),
            layers,
            enable_nested_tensor=False,
        )
        self.head = nn.Sequential(nn.Linear(d, 256), nn.LeakyReLU(), nn.Linear(256, OUT))
        self.register_buffer("mask", nn.Transformer.generate_square_subsequent_mask(L))

    def forward(self, feats, static):
        b, t = feats.shape[:2]
        x = torch.cat([feats[:, :, :3].reshape(b, t, -1), static[:, None].expand(b, t, -1)], -1)
        return self.head(self.enc(self.inp(x) + self.pos[:, :t], mask=self.mask[:t, :t]))


MODELS = {"hmdposer": HMDPoser, "avatarposer": AvatarPoser}


def decode(out: torch.Tensor):
    return out[..., : C.NJ * 6].reshape(*out.shape[:-1], C.NJ, 6) + ID6.to(out.device), out[
        ..., -3:
    ]


def fk_torch(rot6: torch.Tensor, hips: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """rot6 [B,T,NJ,6], hips [B,T,3] (canonical frame), lengths [B,NJ] -> positions [B,T,NJ,3]."""
    m = sixd2mat(rot6)
    off = REST_T.to(rot6.device)[None] * lengths[..., None]
    G, P = [m[:, :, 0]], [hips]
    for j in range(1, C.NJ):
        p = PARENT[j]
        G.append(G[p] @ m[:, :, j])
        P.append(P[p] + (G[j] @ off[:, None, j, :, None]).squeeze(-1))
    return torch.stack(P, 2)


# --- loss ------------------------------------------------------------------------------------

W = dict(rot=2.0, pos=10.0, hand=10.0, vel=10.0)


def loss_fn(out, r6, hips_rel, pc, head_pc, lengths):
    rot6, hr = decode(out.float())
    pos = fk_torch(rot6, head_pc + hr, lengths)
    terms = dict(
        rot=(rot6 - r6).abs().mean(),
        pos=(pos - pc).abs().mean(),
        hand=(pos[:, :, HANDS] - pc[:, :, HANDS]).abs().mean(),
        vel=sum(
            (pos[:, k:] - pos[:, :-k] - (pc[:, k:] - pc[:, :-k])).abs().mean() for k in (1, 3, 5)
        )
        / 3,
    )
    return sum(W[k] * v for k, v in terms.items()), terms, pos


def batch_inputs(store: Store, g: torch.Tensor, skel: bool, aug: bool, gen=None, noise=True):
    idx = window(store, g)
    wr = None
    if aug:  # CMU hand-tracker rotations come from the forearm frame: randomise the wrist
        wr = wrist_aug(len(g), L + 1, g.device, gen)
        cmu = (store.ds_dev[store.cidx[g]] == 0)[:, None, None, None, None]
        wr = torch.where(cmu, wr, torch.eye(3, device=g.device).expand_as(wr))
    feats, Q, origin, head_pc = make_feats(store.inp[idx], wr)
    cal = store.calib_dev[store.cidx[g]]
    if noise and gen is not None:
        cal = cal * (torch.rand(len(g), 3, device=g.device, generator=gen) * 0.1 + 0.95)
    elif noise:
        cal = cal * store.noise[store.cidx[g]]
    st = static_feats(cal, store.ds_dev[store.cidx[g]], skel)
    return idx, feats, st, Q, origin, head_pc


@torch.no_grad()
def quick_val(model, store: Store, skel: bool, stride=8, bs=512, max_frames=40000):
    """Subsampled val metrics (cm) per dataset: pooled end-frames every ``stride`` frames."""
    model.eval()
    res = {}
    for d, name in ((0, "cmu"), (1, "xsens")):
        if d not in store.range:
            continue
        lo, hi = store.range[d]
        gs = torch.arange(lo, hi, stride, device=store.dev)
        gs = gs[:: max(1, len(gs) // max_frames)]
        acc = dict(full=0.0, upper=0.0, lower=0.0, loss=0.0, n=0)
        for i in range(0, len(gs), bs):
            g = gs[i : i + bs]
            idx, feats, st, Q, origin, head_pc = batch_inputs(store, g, skel, False, noise=True)
            r6, hips_rel, pc = make_targets(store, idx[:, 1:], Q, origin, head_pc)
            out = model(feats, st)
            loss, _, pos = loss_fn(out, r6, hips_rel, pc, head_pc, store.lengths[store.cidx[g]])
            e = (pos[:, -1] - pc[:, -1]).norm(dim=-1)
            for k, ix in (("full", NONIN), ("upper", UPPER), ("lower", LOWER)):
                acc[k] += e[:, ix].mean(1).sum().item()
            acc["loss"] += loss.item() * len(g)
            acc["n"] += len(g)
        res[name] = {
            k: (v / acc["n"]) * (100 if k not in ("loss", "n") else 1)
            for k, v in acc.items()
            if k != "n"
        }
    model.train()
    return res


# --- oracle-facing predictor -----------------------------------------------------------------


class ModelPredictor:
    needs_meta = True
    parallel_safe = False

    def __init__(self, model, cfg: dict, name: str, dev: str = "cuda"):
        self.model, self.cfg, self.name, self.dev = model.to(dev).eval(), cfg, name, dev

    @classmethod
    def load(cls, path: Path, dev: str = "cuda"):
        ck = torch.load(path, map_location=dev, weights_only=False)
        model = MODELS[ck["cfg"]["model"]]()
        model.load_state_dict(ck["model"])
        return cls(
            model,
            ck["cfg"],
            f"{ck['cfg']['model']}_{Path(path).parent.name}_{Path(path).stem}",
            dev,
        )

    @torch.no_grad()
    def predict_clip_meta(self, inputs: np.ndarray, calib: O.Calib, meta: dict, bs=1024):
        x = torch.tensor(inputs, dtype=torch.float32, device=self.dev)
        x = torch.cat([x[:1].expand(L, -1), x])  # causal left padding with the first frame
        cal = torch.tensor(
            [[calib.height_m, calib.arm_length_m, calib.shoulder_width_m]], device=self.dev
        )
        ds = torch.tensor([0 if meta["dataset"] == "cmu" else 1], device=self.dev)
        st = static_feats(cal, ds, self.cfg.get("skel", True))
        rots, roots = [], []
        for i in range(0, len(inputs), bs):
            t = torch.arange(i, min(i + bs, len(inputs)), device=self.dev)
            raw = x[t[:, None] + torch.arange(L + 1, device=self.dev)[None]]
            feats, Q, origin, head_pc = make_feats(raw)
            rot6, hr = decode(self.model(feats, st.expand(len(t), -1))[:, -1:].float())
            m = sixd2mat(rot6[:, 0])  # [n,NJ,3,3] canonical frame
            m = torch.cat([(Q.transpose(-1, -2) @ m[:, 0])[:, None], m[:, 1:]], 1)  # root -> world
            hips = (Q.transpose(-1, -2) @ (head_pc[:, -1] + hr[:, 0])[..., None]).squeeze(
                -1
            ) + origin
            rots.append(mat2quat(m).double().cpu().numpy())
            roots.append(hips.double().cpu().numpy())
        return O2.CanonPrediction(rotations=np.concatenate(rots), root_pos=np.concatenate(roots))


# --- training --------------------------------------------------------------------------------


def train(args) -> int:
    dev = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    run = RUNS / args.run
    run.mkdir(parents=True, exist_ok=True)
    tr, va = Store("train", dev), Store("val", dev)
    model = MODELS[args.model]().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    step, best, hist = 0, (1e9, -1), []
    cfg = dict(vars(args), skel=not args.no_skel)
    if (run / "last.pt").exists():
        ck = torch.load(run / "last.pt", map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step, best, hist = ck["step"], tuple(ck["best"]), ck["hist"]
        print(f"resumed {args.run} at step {step}", flush=True)
    import mlflow

    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
    mlflow.set_experiment("sprint-2026-10")
    rid = (run / "mlflow_run").read_text() if (run / "mlflow_run").exists() else None
    mlflow.start_run(run_id=rid, run_name=args.run)
    (run / "mlflow_run").write_text(mlflow.active_run().info.run_id)
    if rid is None:
        mlflow.log_params({k: str(v) for k, v in cfg.items()})
    gen = torch.Generator(device=dev).manual_seed(args.seed + step)
    sched = lambda s: (
        min(1.0, (s + 1) / 500)
        * (0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps))) * 0.97 + 0.03)
    )  # noqa: E731
    t0, last_log = time.time(), time.time()
    model.train()

    def save(name):
        torch.save(
            dict(
                model=model.state_dict(),
                opt=opt.state_dict(),
                step=step,
                best=best,
                hist=hist,
                cfg=cfg,
            ),
            run / name,
        )

    while step < args.steps:
        for gparam in opt.param_groups:
            gparam["lr"] = args.lr * sched(step)
        g = tr.sample(args.bs, args.xs_frac, gen)
        idx, feats, st, Q, origin, head_pc = batch_inputs(
            tr, g, cfg["skel"], not args.no_wrist_aug, gen
        )
        r6, hips_rel, pc = make_targets(tr, idx[:, 1:], Q, origin, head_pc)
        with torch.autocast("cuda", torch.bfloat16, enabled=args.bf16):
            out = model(feats, st)
        loss, terms, _ = loss_fn(out, r6, hips_rel, pc, head_pc, tr.lengths[tr.cidx[g]])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        if step % 100 == 0:
            mlflow.log_metrics(
                {"train/loss": loss.item(), **{f"train/{k}": v.item() for k, v in terms.items()}},
                step=step,
            )
        if time.time() - last_log > 60:
            last_log = time.time()
            print(
                f"step {step} loss {loss.item():.4f} {step / (time.time() - t0 + 1e-9):.1f}it/s(run)",
                flush=True,
            )
        done = step >= args.steps
        timeout = (time.time() - t0) > args.time_min * 60
        if step % args.val_every == 0 or done or timeout:
            v = quick_val(model, va, cfg["skel"])
            t = quick_val(
                model, tr, cfg["skel"], stride=64, max_frames=20000
            )  # train error, same metric
            sel = float(np.mean([v[d]["full"] for d in v]))
            hist.append(dict(step=step, val=v, train=t, sel=sel))
            mlflow.log_metrics({f"val/{d}_{k}": x for d in v for k, x in v[d].items()}, step=step)
            mlflow.log_metrics(
                {f"trainerr/{d}_{k}": x for d in t for k, x in t[d].items()}, step=step
            )
            print(
                f"VAL step {step} sel {sel:.2f} cm | "
                + " | ".join(
                    f"{d} full {v[d]['full']:.2f} up {v[d]['upper']:.2f} lo {v[d]['lower']:.2f} loss {v[d]['loss']:.3f}"
                    for d in v
                )
                + " || train "
                + " ".join(f"{d} up {t[d]['upper']:.2f} lo {t[d]['lower']:.2f}" for d in t),
                flush=True,
            )
            if sel < best[0]:
                best = (sel, step)
                save("best.pt")
            save("last.pt")
        if timeout and not done:
            save("last.pt")
            print(f"PAUSED at step {step}; rerun to resume", flush=True)
            mlflow.end_run()
            return 0
    save("last.pt")
    (run / "elapsed_gpu_s").write_text(str(time.time() - t0))
    mlflow.end_run()
    print(f"DONE step {step} best {best}", flush=True)
    return 0


# --- gates -----------------------------------------------------------------------------------


def selftest(args) -> int:
    """GT targets -> decode -> canonical->world -> quats -> oracle-v2 FK must give GT positions; and
    the left elbow must lie on the body's left (up x forward) for GT and for the pipeline output."""
    dev = "cuda"
    va = Store("val", dev)
    worst = {}
    for d in (0, 1):
        lo, hi = va.range[d]
        g = torch.linspace(lo + 200, hi - 1, 64, device=dev).long()
        idx, feats, st, Q, origin, head_pc = batch_inputs(va, g, True, False, noise=False)
        r6, hips_rel, pc = make_targets(va, idx[:, 1:], Q, origin, head_pc)
        m = sixd2mat(r6[:, -1])
        m0 = Q.transpose(-1, -2) @ m[:, 0]
        m = torch.cat([m0[:, None], m[:, 1:]], 1)
        hips = (Q.transpose(-1, -2) @ (head_pc[:, -1] + hips_rel[:, -1])[..., None]).squeeze(
            -1
        ) + origin
        errs, sides = [], []
        q = mat2quat(m).double().cpu().numpy()
        for i in range(len(g)):
            c = int(va.cidx[g[i]])
            ln = va.lengths[c].double().cpu().numpy()
            loc = q[i][None]
            pos = R.fk(loc, hips[i].double().cpu().numpy()[None], ln)[0]
            gt = va.pos[g[i]].double().cpu().numpy()
            errs.append(np.linalg.norm(pos - gt, axis=-1)[C.NONINPUT_IDX].mean())
            hq = va.inp[g[i], 3:7].double().cpu().numpy()  # head tracker rotation (input stream)
            fwd = C.qrotate(hq, np.array([0.0, 0, 1.0]))
            fwd[1] = 0.0
            left = np.cross([0, 1.0, 0], fwd)  # conventions: left side at up x forward
            sides.append(float(np.dot(pos[C.J["left_elbow"]] - pos[C.J["right_elbow"]], left) > 0))
        worst[d] = (np.mean(errs) * 1000, float(np.mean(sides)))
        print(
            f"selftest ds={'cmu' if d == 0 else 'xsens'} mean (11 non-input joints, 64 frames; median-length FK floor ~4-5 mm) pipeline error {worst[d][0]:.2f} mm, left elbow on the head-forward left side {worst[d][1]:.2f}"
        )
    return 0


def probe_wrist(args) -> int:
    doc = S.load_splits()
    for ds, ids in (
        ("cmu", doc["cmu"]["clips"]["train"]),
        ("xsens", doc["xsens"]["clips"]["train"]),
    ):
        a = [
            R.forearm_probe(C.load_clip(S.clip_path(c)))
            for c in ids[:: max(1, len(ids) // 40)][:40]
        ]
        print(
            f"forearm probe {ds}: median angle(hand-tracker bone axis, elbow->hand) = {np.median(a):.1f} deg (n={len(a)})"
        )
    return 0


def overfit(args) -> int:
    dev = "cuda"
    cid, nfr = args.clip, args.frames
    tmp = Path(args.data)
    tmp.mkdir(parents=True, exist_ok=True)
    build_set("one", [(cid, False, nfr)], workers=1, out=tmp)
    st = Store("one", dev, data=tmp)
    model = MODELS[args.model]().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    gen = torch.Generator(device=dev).manual_seed(0)
    for step in range(1, args.steps + 1):
        for gp in opt.param_groups:
            gp["lr"] = args.lr * (0.5 * (1 + math.cos(math.pi * step / args.steps)) * 0.98 + 0.02)
        g = st.sample(args.bs, 1.0, gen)
        idx, feats, sf, Q, origin, head_pc = batch_inputs(st, g, True, False, gen, noise=False)
        r6, hips_rel, pc = make_targets(st, idx[:, 1:], Q, origin, head_pc)
        loss, terms, _ = loss_fn(model(feats, sf), r6, hips_rel, pc, head_pc, st.lengths[st.cidx[g]])
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 200 == 0:
            print(f"overfit step {step} loss {loss.item():.4f}", {k: round(v.item(), 4) for k, v in terms.items()}, flush=True)
    pred = ModelPredictor(model, dict(model=args.model, skel=True), "overfit", dev)
    # oracle path with un-noised calibration: the model was overfit on true calibration scalars
    noise, O.CALIB_NOISE = O.CALIB_NOISE, 0.0
    try:
        m = O2.evaluate_clip((cid, "clean", pred, nfr))
    finally:
        O.CALIB_NOISE = noise
    print(
        f"OVERFIT {cid} {nfr} frames: oracle-v2 MPJPE-full {m['full'].mean() * 100:.3f} cm upper {m['upper'].mean() * 100:.3f} lower {m['lower'].mean() * 100:.3f}"
    )
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build-data")
    b.add_argument("--max-clips", type=int, default=None)
    t = sub.add_parser("train")
    t.add_argument("--run", required=True)
    t.add_argument("--model", choices=sorted(MODELS), default="hmdposer")
    t.add_argument("--steps", type=int, default=60000)
    t.add_argument("--bs", type=int, default=256)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--xs-frac", type=float, default=0.5)
    t.add_argument("--val-every", type=int, default=2000)
    t.add_argument("--time-min", type=float, default=110.0)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--bf16", action="store_true")
    t.add_argument("--no-wrist-aug", action="store_true")
    t.add_argument("--no-skel", action="store_true", help="drop the skeleton-id input")
    sub.add_parser("selftest")
    sub.add_parser("probe-wrist")
    o = sub.add_parser("overfit")
    o.add_argument("--clip", default="xsens/S01/train_S01_A01")
    o.add_argument("--frames", type=int, default=1500)
    o.add_argument("--steps", type=int, default=2500)
    o.add_argument("--bs", type=int, default=128)
    o.add_argument("--lr", type=float, default=3e-4)
    o.add_argument("--model", choices=sorted(MODELS), default="hmdposer")
    o.add_argument("--data", default=str(S.REPO / ".cache/sprint/overfit"))
    a = ap.parse_args(argv)
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")
    return {
        "build-data": build_data,
        "train": train,
        "selftest": selftest,
        "probe-wrist": probe_wrist,
        "overfit": overfit,
    }[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
