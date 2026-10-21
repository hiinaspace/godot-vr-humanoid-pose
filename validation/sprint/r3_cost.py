# ruff: noqa: E501
"""Per-frame inference cost of the streaming HMD-Poser port: parameters and multiply-accumulates.

``python -m validation.sprint.r3_cost [CKPT]``. Counted analytically from module shapes for one new frame
with carried LSTM state: embeddings (one Linear per feature group per token), LSTM cells (one per token per
block), spatial transformer layers over the NT+1 tokens (qkv/out projections, 2-layer FFN, attention scores),
head. The windowed model re-runs the same network over 40 frames (zero state) per output frame.
"""

from __future__ import annotations

import sys

import torch
from torch import nn

from validation.sprint import train as T


def macs_per_frame(model: T.HMDPoser, n_tokens: int = T.NT + 1) -> dict:
    out = {}
    emb = sum(
        m.in_features * m.out_features for m in model.emb.modules() if isinstance(m, nn.Linear)
    )
    emb += sum(
        m.in_features * m.out_features for m in model.static.modules() if isinstance(m, nn.Linear)
    )
    out["embeddings"] = emb
    out["lstm"] = sum(
        4 * (m.input_size + m.hidden_size) * m.hidden_size
        for m in model.lstm.modules()
        if isinstance(m, nn.LSTM)
    )
    tr = 0
    for enc in model.spatial:
        for layer in enc.layers:
            d, ff = layer.self_attn.embed_dim, layer.linear1.out_features
            tr += n_tokens * (4 * d * d + 2 * d * ff) + 2 * n_tokens * n_tokens * d
    out["transformer"] = tr
    out["head"] = sum(
        m.in_features * m.out_features for m in model.head.modules() if isinstance(m, nn.Linear)
    )
    out["total"] = sum(out.values())
    return out


if __name__ == "__main__":
    model = T.HMDPoser()
    if len(sys.argv) > 1:
        ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
        model = T.build_model(ck["cfg"])
        model.load_state_dict(ck["model"])
    c = macs_per_frame(model)
    p = sum(x.numel() for x in model.parameters())
    print({k: f"{v / 1e6:.2f} M" for k, v in c.items()}, f"params {p / 1e6:.2f} M")
    print(
        f"streaming: {c['total'] / 1e9:.4f} GMAC/frame; windowed (40 frames re-run): {40 * c['total'] / 1e9:.3f} GMAC/frame"
    )
