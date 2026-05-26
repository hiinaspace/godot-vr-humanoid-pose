"""Torch model factories for learned upper-body latent predictors."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class SparseWindowLayout:
    window_size: int
    per_frame_dim: int
    static_dim: int

    @property
    def frame_dim(self) -> int:
        return self.window_size * self.per_frame_dim


def make_torch_upper_body_model(
    torch: Any,
    *,
    model_name: str,
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    feature_names: Sequence[str],
) -> Any:
    if model_name == "torch_linear":
        return torch.nn.Linear(input_dim, output_dim)
    if model_name == "torch_mlp":
        return torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, output_dim),
        )
    if model_name == "torch_tcn":
        layout = infer_sparse_window_layout(feature_names, input_dim=input_dim)
        return _make_tcn(
            torch,
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            layout=layout,
        )
    raise ValueError(f"Unsupported Torch upper-body model: {model_name}")


def infer_sparse_window_layout(
    feature_names: Sequence[str],
    *,
    input_dim: int,
) -> SparseWindowLayout:
    frame_feature_count = 0
    window_prefixes: dict[str, None] = {}
    for name in feature_names:
        prefix = name.split("_", 1)[0]
        if not _is_window_prefix(prefix):
            continue
        frame_feature_count += 1
        window_prefixes.setdefault(prefix, None)
    window_size = len(window_prefixes)
    if window_size <= 0:
        raise ValueError("TCN model requires sparse-window feature names")
    if frame_feature_count % window_size != 0:
        raise ValueError(
            f"Cannot infer per-frame feature count from {frame_feature_count} "
            f"features across {window_size} windows"
        )
    static_dim = input_dim - frame_feature_count
    if static_dim < 0:
        raise ValueError(
            f"Feature schema has {frame_feature_count} temporal features, "
            f"but input_dim is {input_dim}"
        )
    return SparseWindowLayout(
        window_size=window_size,
        per_frame_dim=frame_feature_count // window_size,
        static_dim=static_dim,
    )


def _make_tcn(
    torch: Any,
    *,
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    layout: SparseWindowLayout,
) -> Any:
    class CausalResidualBlock(torch.nn.Module):  # type: ignore[misc]
        def __init__(self, channels: int, dilation: int) -> None:
            super().__init__()
            kernel_size = 3
            padding = dilation * (kernel_size - 1)
            self.net = torch.nn.Sequential(
                torch.nn.ConstantPad1d((padding, 0), 0.0),
                torch.nn.Conv1d(
                    channels,
                    channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                ),
                torch.nn.GELU(),
                torch.nn.ConstantPad1d((padding, 0), 0.0),
                torch.nn.Conv1d(
                    channels,
                    channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                ),
            )
            self.activation = torch.nn.GELU()

        def forward(self, values: Any) -> Any:
            return self.activation(values + self.net(values))

    class SparseWindowTCN(torch.nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.window_size = layout.window_size
            self.per_frame_dim = layout.per_frame_dim
            self.static_dim = layout.static_dim
            self.frame_dim = layout.frame_dim
            if self.frame_dim + self.static_dim != input_dim:
                raise ValueError("TCN layout does not match input_dim")
            self.input_projection = torch.nn.Conv1d(layout.per_frame_dim, hidden_dim, kernel_size=1)
            self.blocks = torch.nn.ModuleList(
                CausalResidualBlock(hidden_dim, dilation) for dilation in (1, 2, 4, 8)
            )
            self.head = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim + layout.static_dim, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_dim, output_dim),
            )

        def forward(self, values: Any) -> Any:
            temporal = values[:, : self.frame_dim].reshape(
                values.shape[0],
                self.window_size,
                self.per_frame_dim,
            )
            temporal = temporal.transpose(1, 2)
            encoded = self.input_projection(temporal)
            for block in self.blocks:
                encoded = block(encoded)
            latest = encoded[:, :, -1]
            if self.static_dim > 0:
                latest = torch.cat((latest, values[:, self.frame_dim :]), dim=1)
            return self.head(latest)

    return SparseWindowTCN()


def _is_window_prefix(value: str) -> bool:
    return len(value) >= 3 and value[0] == "t" and value[1] in ("+", "-") and value[2:].isdigit()
