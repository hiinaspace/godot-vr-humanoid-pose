# ruff: noqa: E501
"""Export a trained sprint checkpoint as a single-step ONNX runtime graph (Round 4a).

    uv run python -m validation.sprint.export_onnx --ckpt .../best.pt --out .cache/r4a/e1.onnx

What the model is, for the runtime: it is **not** a free-running recurrent stream. Every output frame
is a pure function of the newest ``W+1`` tracker frames (W = ``cfg.window``, default 40; 60 Hz data):
the window is canonicalised at its *last* frame (head yaw / x,z origin, see ``train.make_feats``),
the LSTMs start from a zero state at the window start, and the newest output is read. So the "state"
of the runtime is a ring buffer of raw tracker frames (``runtime.py``), and the graph consumes the
whole window. Feature construction (heading canonicalisation, velocities, head-relative hands), the
network, the 6D -> rotation-matrix decode and the canonical -> world transform are all in the graph.

Graph I/O (``<out>.json`` carries the same plus config; ONNX ``metadata_props`` carry the JSON too):
  in   raw    [1, W+1, 24]  tracker stream, oldest -> newest (inputs-v1 layout: per tracker pos3 quat4(xyzw),
                            then 3 validity flags, which the network ignores)
  in   static [1, 5]        (calib - mu) / sd  (height, arm, shoulder) ++ skeleton one-hot (``static_feats_np``)
  out  rot    [1, NJ, 3, 3] local rotation matrices in the canonical rest pose (rest.py); root = world
  out  hips   [1, 3]        world hips position
  out  contact[1, 2]        P(contact) left/right foot (only when the checkpoint has the head)
Positions come from FK(rot, hips, bone lengths) in ``runtime.py`` (13 joints; tests below).

Exactness-preserving trim for hmdposer: the second spatial transformer is per-timestep, so only the
newest timestep is evaluated (bit-identical outputs to ``model(feats, static)[:, -1]``, checked by
``--verify``).
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn

from validation.sprint import conventions as C
from validation.sprint import train as T

FORMAT = "vrhp-step-onnx-v1"
OPSET = 17
NC = 2


def static_feats_np(calib: np.ndarray, skel_id: int | None) -> np.ndarray:
    """numpy twin of ``train.static_feats`` for one subject: calib [3] (m), skel_id 0=cmu 1=xsens
    (None -> no skeleton input, as ``--no-skel`` checkpoints)."""
    mu, sd = np.asarray(T.NORM["mu"], np.float32), np.asarray(T.NORM["sd"], np.float32)
    oh = np.zeros(2, np.float32)
    if skel_id is not None:
        oh[int(skel_id)] = 1.0
    return np.concatenate([(np.asarray(calib, np.float32) - mu) / sd, oh])[None].astype(np.float32)


def load_checkpoint(path: Path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = dict(ck["cfg"])
    T.L = int(cfg.get("window", 40))  # train.py's module-global window (set_window() in Round 3)
    cls = T.MODELS[cfg["model"]]
    sd = ck["model"]
    nout = int(sd["head.2.weight"].shape[0])
    kwargs = {}
    if "nout" in inspect.signature(cls.__init__).parameters:
        kwargs["nout"] = nout
    elif nout != T.OUT:
        raise ValueError(f"checkpoint has {nout} outputs but this train.py has no `nout` argument")
    model = cls(**kwargs)
    model.load_state_dict(sd)
    return model.eval(), cfg, ck, nout


def hmd_last_step(model, feats: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
    """HMDPoser.forward with the last spatial block evaluated for the newest timestep only."""
    b, t = feats.shape[:2]
    sl = [slice(0, 6), slice(6, 12), slice(12, 15), slice(15, 18)]
    toks = [
        model.norm(
            torch.cat([e(feats[:, :, k, s]) for e, s in zip(model.emb[k], sl, strict=True)], -1)
        )
        for k in range(T.NT)
    ]
    toks.append(model.norm(model.static(static))[:, None].expand(b, t, -1))
    x = torch.stack(toks, 2)
    n = len(model.lstm)
    for i, (lstm, sp) in enumerate(zip(model.lstm, model.spatial, strict=True)):
        x = torch.stack([lstm[k](x[:, :, k])[0] for k in range(T.NT + 1)], 2)
        if i == n - 1:
            x = sp(x[:, -1])  # [b, NT+1, d]: only the newest timestep is read by the head
            return model.head(x.reshape(b, -1))
        x = sp(x.reshape(b * t, T.NT + 1, -1)).reshape(b, t, T.NT + 1, -1)
    raise AssertionError


class StepGraph(nn.Module):
    def __init__(self, model: nn.Module, cfg: dict, nout: int):
        super().__init__()
        self.model, self.nout = model, nout
        self.hmd = cfg["model"] == "hmdposer"
        mode = cfg.get("inputs", "full")
        self.register_buffer(
            "mask",
            T.feat_mask(mode, "cpu")
            if hasattr(T, "feat_mask") and mode != "full"
            else torch.ones(18),
        )

    def forward(self, raw: torch.Tensor, static: torch.Tensor):
        feats, Q, origin, head_pc = T.make_feats(raw)
        feats = feats * self.mask
        if self.hmd:
            out = hmd_last_step(self.model, feats, static)
        else:
            out = self.model(feats, static)[:, -1]
        nj = C.NJ
        rot6 = out[:, : nj * 6].reshape(-1, nj, 6) + T.ID6
        hr = out[:, nj * 6 : nj * 6 + 3]
        m = T.sixd2mat(rot6)  # [1, NJ, 3, 3]
        root = Q.transpose(-1, -2) @ m[:, 0]
        rot = torch.cat([root[:, None], m[:, 1:]], 1)
        hips = (Q.transpose(-1, -2) @ (head_pc[:, -1] + hr)[..., None]).squeeze(-1) + origin
        outs = [rot, hips]
        if self.nout > T.OUT:
            outs.append(torch.sigmoid(out[:, T.OUT : T.OUT + NC]))
        return tuple(outs)


def export(ckpt: Path, out: Path, verify: bool = True, opset: int = OPSET) -> dict:
    import onnx

    torch.backends.mha.set_fastpath_enabled(
        False
    )  # fused encoder-layer kernel has no ONNX symbolic
    model, cfg, ck, nout = load_checkpoint(ckpt)
    g = StepGraph(model, cfg, nout).eval()
    w = T.L
    raw = torch.randn(1, w + 1, 24)
    static = torch.randn(1, 5)
    names = ["rot", "hips"] + (["contact"] if nout > T.OUT else [])
    out.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            g,
            (raw, static),
            str(out),
            input_names=["raw", "static"],
            output_names=names,
            opset_version=opset,
            dynamo=False,
        )
    sha = hashlib.sha256(Path(ckpt).read_bytes()).hexdigest()
    meta = dict(
        format=FORMAT,
        window=w,
        frame_hz=C.FPS,
        raw_dim=24,
        nj=C.NJ,
        joints=list(C.JOINTS),
        parents=[int(p) for p in C.PARENT],
        model=cfg["model"],
        inputs=cfg.get("inputs", "full"),
        skel=bool(cfg.get("skel", True)),
        has_contact=nout > T.OUT,
        outputs=names,
        opset=opset,
        norm=T.NORM,
        rest_dirs=np.asarray(T.R.REST_DIRS).tolist(),
        rest_version=T.R.REST_VERSION,
        ckpt=str(ckpt),
        ckpt_sha256=sha,
        ckpt_step=int(ck.get("step", -1)),
        cfg={
            k: (v if isinstance(v, int | float | str | bool | type(None)) else str(v))
            for k, v in cfg.items()
        },
    )
    m = onnx.load(str(out))
    onnx.checker.check_model(m)
    for k, v in (("vrhp_meta", json.dumps(meta)),):
        e = m.metadata_props.add()
        e.key, e.value = k, v
    onnx.save(m, str(out))
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    meta["ops"] = dict(Counter(n.op_type for n in m.graph.node))
    meta["params"] = int(sum(p.numel() for p in model.parameters()))
    meta["bytes"] = out.stat().st_size
    if verify:
        meta["verify_max_abs"] = verify_against_torch(model, g, out, nout)
    return meta


@torch.no_grad()
def verify_against_torch(model, g: StepGraph, onnx_path: Path, nout: int, n: int = 8) -> dict:
    """Random-window check of (a) the last-step trim vs the full model and (b) ONNX vs StepGraph."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    gen = torch.Generator().manual_seed(1)
    worst = dict(trim=0.0, onnx=0.0)
    for _ in range(n):
        # a plausible window: smooth head/hands near standing height, random yaw
        raw = torch.zeros(1, T.L + 1, 24)
        base = torch.tensor(
            [
                0.3,
                1.65,
                -0.2,
                0.0,
                0.0,
                0.0,
                1.0,
                0.5,
                1.2,
                0.2,
                0,
                0,
                0,
                1,
                0.1,
                1.2,
                0.3,
                0,
                0,
                0,
                1,
                1,
                1,
                1,
            ]
        )
        raw[:] = base + 0.02 * torch.randn(1, T.L + 1, 24, generator=gen).cumsum(1) * 0.1
        raw[..., 3:7] = torch.nn.functional.normalize(raw[..., 3:7], dim=-1)
        raw[..., 10:14] = torch.nn.functional.normalize(raw[..., 10:14], dim=-1)
        raw[..., 17:21] = torch.nn.functional.normalize(raw[..., 17:21], dim=-1)
        st = torch.randn(1, 5, generator=gen) * 0.5
        feats, *_ = T.make_feats(raw)
        feats = feats * g.mask
        full = model(feats, st)[:, -1]
        part = hmd_last_step(model, feats, st) if g.hmd else full
        worst["trim"] = max(worst["trim"], float((full - part).abs().max()))
        ref = g(raw, st)
        got = sess.run(None, {"raw": raw.numpy(), "static": st.numpy()})
        for r, o in zip(ref, got, strict=True):
            worst["onnx"] = max(worst["onnx"], float(np.abs(r.numpy() - o).max()))
    return worst


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--opset", type=int, default=OPSET)
    ap.add_argument("--no-verify", action="store_true")
    a = ap.parse_args(argv)
    meta = export(a.ckpt, a.out, verify=not a.no_verify, opset=a.opset)
    print(
        json.dumps(
            {
                k: v
                for k, v in meta.items()
                if k not in ("rest_dirs", "cfg", "joints", "parents", "norm")
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
