"""Train GPU-oriented upper-body latent predictors from sparse-window Parquet data."""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from validation.vr_pose_validation.torch_upper_body_decoder import (
    RuntimeDecoderFeatureIndices,
    decode_upper_body_runtime_points,
    runtime_decoder_feature_indices,
)
from validation.vr_pose_validation.upper_body_ml import (
    DECODED_REFERENCE_JOINTS,
    RUNTIME_DECODER_CONTEXT_JOINTS,
    TARGET_GROUPS,
    TARGET_NAMES,
    TARGET_SCHEMA,
    read_sparse_window_dataframe,
)

DEFAULT_DATASET_DIR = Path(".cache/ml/upper_body_windows")
DEFAULT_OUTPUT_DIR = Path(".cache/ml/upper_body_models")
TorchModelName = Literal[
    "constant",
    "torch_linear",
    "torch_mlp",
    "xgboost",
]


@dataclass(frozen=True, slots=True)
class LoadedDataset:
    features: np.ndarray
    targets: np.ndarray
    masks: np.ndarray
    heuristics: np.ndarray | None
    heuristic_masks: np.ndarray | None
    decoded_reference_points: np.ndarray | None
    decoded_reference_masks: np.ndarray | None
    runtime_context_points: np.ndarray | None
    runtime_context_masks: np.ndarray | None
    splits: np.ndarray
    subjects: np.ndarray
    clips: np.ndarray
    center_row_indices: np.ndarray
    timestamps_s: np.ndarray
    feature_names: list[str]
    target_names: list[str]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--models",
        nargs="*",
        choices=(
            "constant",
            "torch_linear",
            "torch_mlp",
            "xgboost",
        ),
        default=("constant", "torch_linear", "torch_mlp", "xgboost"),
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--xgboost-estimators", type=int, default=256)
    parser.add_argument("--latent-loss-weight", type=float, default=1.0)
    parser.add_argument("--decoded-pose-weight", type=float, default=0.0)
    parser.add_argument("--decoded-torso-weight", type=float, default=0.0)
    parser.add_argument("--decoded-shoulder-weight", type=float, default=1.0)
    parser.add_argument("--decoded-elbow-weight", type=float, default=1.0)
    parser.add_argument("--decoded-arm-direction-weight", type=float, default=0.25)
    parser.add_argument("--decoded-stretch-weight", type=float, default=0.25)
    parser.add_argument("--decoded-reach-hinge-weight", type=float, default=0.0)
    parser.add_argument("--reach-ratio-limit", type=float, default=0.97)
    parser.add_argument("--surrogate-position-weight", type=float, default=0.0)
    parser.add_argument("--temporal-velocity-weight", type=float, default=0.0)
    parser.add_argument("--temporal-acceleration-weight", type=float, default=0.0)
    parser.add_argument("--temporal-jerk-weight", type=float, default=0.0)
    parser.add_argument("--temporal-excess-jerk-weight", type=float, default=0.0)
    parser.add_argument("--pole-velocity-weight", type=float, default=0.0)
    parser.add_argument("--pole-excess-jerk-weight", type=float, default=0.0)
    parser.add_argument(
        "--temporal-excess-ratio",
        type=float,
        default=1.25,
        help=(
            "Allow predicted temporal magnitude up to this multiple of reference "
            "before excess loss."
        ),
    )
    parser.add_argument("--temporal-excess-margin", type=float, default=0.0)
    parser.add_argument(
        "--disabled-target-groups",
        nargs="*",
        choices=tuple(TARGET_GROUPS),
        default=(),
        help="Target groups to leave out of Torch latent training.",
    )
    parser.add_argument(
        "--validation-subjects",
        nargs="*",
        help="Override dataset splits by holding out these subject IDs.",
    )
    args = parser.parse_args(argv)

    report = train_upper_body_latents(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        model_names=cast(tuple[TorchModelName, ...], tuple(args.models)),
        epochs=max(1, args.epochs),
        batch_size=max(1, args.batch_size),
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        hidden_dim=max(8, args.hidden_dim),
        seed=args.seed,
        allow_cpu=args.allow_cpu,
        xgboost_estimators=max(1, args.xgboost_estimators),
        validation_subjects=set(args.validation_subjects or []),
        latent_loss_weight=max(0.0, args.latent_loss_weight),
        decoded_pose_weight=max(0.0, args.decoded_pose_weight),
        decoded_torso_weight=max(0.0, args.decoded_torso_weight),
        decoded_shoulder_weight=max(0.0, args.decoded_shoulder_weight),
        decoded_elbow_weight=max(0.0, args.decoded_elbow_weight),
        decoded_arm_direction_weight=max(0.0, args.decoded_arm_direction_weight),
        decoded_stretch_weight=max(0.0, args.decoded_stretch_weight),
        decoded_reach_hinge_weight=max(0.0, args.decoded_reach_hinge_weight),
        reach_ratio_limit=max(0.0, args.reach_ratio_limit),
        surrogate_position_weight=max(0.0, args.surrogate_position_weight),
        temporal_velocity_weight=max(0.0, args.temporal_velocity_weight),
        temporal_acceleration_weight=max(0.0, args.temporal_acceleration_weight),
        temporal_jerk_weight=max(0.0, args.temporal_jerk_weight),
        temporal_excess_jerk_weight=max(0.0, args.temporal_excess_jerk_weight),
        pole_velocity_weight=max(0.0, args.pole_velocity_weight),
        pole_excess_jerk_weight=max(0.0, args.pole_excess_jerk_weight),
        temporal_excess_ratio=max(0.0, args.temporal_excess_ratio),
        temporal_excess_margin=max(0.0, args.temporal_excess_margin),
        disabled_target_groups=set(args.disabled_target_groups),
    )
    print(json.dumps(report["leaderboard"], indent=2, sort_keys=True))
    return 0


def train_upper_body_latents(
    *,
    dataset_dir: Path,
    output_dir: Path,
    model_names: Sequence[TorchModelName],
    epochs: int = 80,
    batch_size: int = 4096,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    hidden_dim: int = 256,
    seed: int = 0,
    allow_cpu: bool = False,
    xgboost_estimators: int = 256,
    validation_subjects: set[str] | None = None,
    residual_regularization_weight: float = 1e-3,
    residual_scale: float = 1.0,
    residual_smoothing_alpha: float = 1.0,
    residual_gate_search: bool = False,
    latent_loss_weight: float = 1.0,
    decoded_pose_weight: float = 0.0,
    decoded_torso_weight: float = 0.0,
    decoded_shoulder_weight: float = 1.0,
    decoded_elbow_weight: float = 1.0,
    decoded_arm_direction_weight: float = 0.25,
    decoded_stretch_weight: float = 0.25,
    decoded_reach_hinge_weight: float = 0.0,
    reach_ratio_limit: float = 0.97,
    surrogate_position_weight: float = 0.0,
    temporal_velocity_weight: float = 0.0,
    temporal_acceleration_weight: float = 0.0,
    temporal_jerk_weight: float = 0.0,
    temporal_excess_jerk_weight: float = 0.0,
    pole_velocity_weight: float = 0.0,
    pole_excess_jerk_weight: float = 0.0,
    temporal_excess_ratio: float = 1.25,
    temporal_excess_margin: float = 0.0,
    disabled_target_groups: set[str] | None = None,
) -> dict[str, object]:
    dataset = load_dataset(dataset_dir)
    torch = _import_torch()
    device = _torch_device(torch, allow_cpu=allow_cpu)
    output_dir.mkdir(parents=True, exist_ok=True)
    disabled_target_groups = _validate_target_groups(disabled_target_groups or set())

    train_mask, validation_mask = split_masks(dataset, validation_subjects=validation_subjects)

    stats = _normalization_stats(dataset.features[train_mask])
    normalized_features = _normalize(dataset.features, stats)
    report: dict[str, object] = {
        "dataset_dir": dataset_dir.as_posix(),
        "output_dir": output_dir.as_posix(),
        "rows": int(dataset.features.shape[0]),
        "features": int(dataset.features.shape[1]),
        "targets": dataset.target_names,
        "train_rows": int(train_mask.sum()),
        "validation_rows": int(validation_mask.sum()),
        "torch_version": str(torch.__version__),
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_device": str(device),
        "validation_subjects": sorted(validation_subjects or []),
        "latent_loss_weight": latent_loss_weight,
        "decoded_pose_weight": decoded_pose_weight,
        "decoded_torso_weight": decoded_torso_weight,
        "decoded_shoulder_weight": decoded_shoulder_weight,
        "decoded_elbow_weight": decoded_elbow_weight,
        "decoded_arm_direction_weight": decoded_arm_direction_weight,
        "decoded_stretch_weight": decoded_stretch_weight,
        "decoded_reach_hinge_weight": decoded_reach_hinge_weight,
        "reach_ratio_limit": reach_ratio_limit,
        "surrogate_position_weight": surrogate_position_weight,
        "temporal_velocity_weight": temporal_velocity_weight,
        "temporal_acceleration_weight": temporal_acceleration_weight,
        "temporal_jerk_weight": temporal_jerk_weight,
        "temporal_excess_jerk_weight": temporal_excess_jerk_weight,
        "pole_velocity_weight": pole_velocity_weight,
        "pole_excess_jerk_weight": pole_excess_jerk_weight,
        "temporal_excess_ratio": temporal_excess_ratio,
        "temporal_excess_margin": temporal_excess_margin,
        "disabled_target_groups": sorted(disabled_target_groups),
        "leaderboard": [],
    }
    (output_dir / "normalization.json").write_text(
        json.dumps(
            {
                "feature_names": dataset.feature_names,
                "mean": stats["mean"].tolist(),
                "scale": stats["scale"].tolist(),
                "target_names": dataset.target_names,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    if dataset.heuristics is not None and dataset.heuristic_masks is not None:
        cast(list[dict[str, object]], report["leaderboard"]).append(
            {
                "model": "heuristic_estimator",
                "train_seconds": 0.0,
                "inference_seconds_per_sample": 0.0,
                **_metric_dict(
                    dataset.targets[validation_mask],
                    dataset.heuristics[validation_mask],
                    dataset.masks[validation_mask] * dataset.heuristic_masks[validation_mask],
                    dataset.target_names,
                ),
            }
        )

    for model_name in model_names:
        if model_name == "constant":
            result = _train_constant(dataset, train_mask, validation_mask)
        elif model_name == "xgboost":
            result = _train_xgboost(
                dataset,
                normalized_features,
                train_mask,
                validation_mask,
                output_dir=output_dir / "xgboost",
                device="cuda" if str(device).startswith("cuda") else "cpu",
                estimators=xgboost_estimators,
                seed=seed,
            )
        else:
            result = _train_torch_model(
                torch,
                model_name=model_name,
                features=normalized_features,
                raw_features=dataset.features,
                targets=dataset.targets,
                masks=dataset.masks,
                heuristics=dataset.heuristics,
                heuristic_masks=dataset.heuristic_masks,
                dataset=dataset,
                train_mask=train_mask,
                validation_mask=validation_mask,
                output_path=output_dir / f"{model_name}.pt",
                device=device,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                hidden_dim=hidden_dim,
                seed=seed,
                feature_names=dataset.feature_names,
                target_names=dataset.target_names,
                residual_regularization_weight=residual_regularization_weight,
                residual_scale=residual_scale,
                residual_smoothing_alpha=residual_smoothing_alpha,
                residual_gate_search=residual_gate_search,
                latent_loss_weight=latent_loss_weight,
                decoded_pose_weight=decoded_pose_weight,
                decoded_torso_weight=decoded_torso_weight,
                decoded_shoulder_weight=decoded_shoulder_weight,
                decoded_elbow_weight=decoded_elbow_weight,
                decoded_arm_direction_weight=decoded_arm_direction_weight,
                decoded_stretch_weight=decoded_stretch_weight,
                decoded_reach_hinge_weight=decoded_reach_hinge_weight,
                reach_ratio_limit=reach_ratio_limit,
                surrogate_position_weight=surrogate_position_weight,
                temporal_velocity_weight=temporal_velocity_weight,
                temporal_acceleration_weight=temporal_acceleration_weight,
                temporal_jerk_weight=temporal_jerk_weight,
                temporal_excess_jerk_weight=temporal_excess_jerk_weight,
                pole_velocity_weight=pole_velocity_weight,
                pole_excess_jerk_weight=pole_excess_jerk_weight,
                temporal_excess_ratio=temporal_excess_ratio,
                temporal_excess_margin=temporal_excess_margin,
                disabled_target_groups=disabled_target_groups,
            )
        cast(list[dict[str, object]], report["leaderboard"]).append(result)

    report_path = output_dir / "leaderboard.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def load_dataset(dataset_dir: Path) -> LoadedDataset:
    schema_path = dataset_dir / "schema.json"
    if not schema_path.exists():
        raise FileNotFoundError(
            f"Expected sparse-window dataset at {dataset_dir}; run build_sparse_window_dataset"
        )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    target_schema = str(schema.get("target_schema", ""))
    if target_schema != TARGET_SCHEMA:
        raise ValueError(
            f"Expected target_schema={TARGET_SCHEMA!r} in {schema_path}; "
            "regenerate the sparse-window cache."
        )
    feature_names = [str(name) for name in schema["feature_names"]]
    target_names = [str(name) for name in schema.get("target_names", TARGET_NAMES)]
    dataframe = read_sparse_window_dataframe(dataset_dir)
    if dataframe.is_empty():
        raise ValueError(f"No training rows found in {dataset_dir}")
    features = _feature_matrix(dataframe["features"].to_numpy())
    targets = np.asarray(
        dataframe.select(target_names).to_numpy(),
        dtype=np.float32,
    )
    mask_names = [f"{name}_mask" for name in target_names]
    masks = np.asarray(
        dataframe.select(mask_names).to_numpy(),
        dtype=np.float32,
    )
    heuristic_names = [f"heuristic_{name}" for name in target_names]
    heuristic_mask_names = [f"heuristic_{name}_mask" for name in target_names]
    if all(name in dataframe.columns for name in (*heuristic_names, *heuristic_mask_names)):
        heuristics = np.asarray(dataframe.select(heuristic_names).to_numpy(), dtype=np.float32)
        heuristic_masks = np.asarray(
            dataframe.select(heuristic_mask_names).to_numpy(),
            dtype=np.float32,
        )
    else:
        heuristics = None
        heuristic_masks = None
    decoded_reference_points, decoded_reference_masks = _load_decoded_reference_points(dataframe)
    runtime_context_points, runtime_context_masks = _load_runtime_decoder_context_points(dataframe)
    splits = np.asarray(dataframe["split"].to_list(), dtype=str)
    subjects = np.asarray(dataframe["subject"].to_list(), dtype=str)
    clips = np.asarray(dataframe["clip"].to_list(), dtype=str)
    center_row_indices = np.asarray(dataframe["center_row_index"].to_list(), dtype=np.int64)
    timestamps_s = np.asarray(dataframe["timestamp_s"].to_list(), dtype=np.float32)
    return LoadedDataset(
        features=features,
        targets=targets,
        masks=masks,
        heuristics=heuristics,
        heuristic_masks=heuristic_masks,
        decoded_reference_points=decoded_reference_points,
        decoded_reference_masks=decoded_reference_masks,
        runtime_context_points=runtime_context_points,
        runtime_context_masks=runtime_context_masks,
        splits=splits,
        subjects=subjects,
        clips=clips,
        center_row_indices=center_row_indices,
        timestamps_s=timestamps_s,
        feature_names=feature_names,
        target_names=target_names,
    )


def _feature_matrix(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.empty((0, 0), dtype=np.float32)
    return np.vstack(cast(Sequence[Any], values)).astype(np.float32, copy=False)


def _load_decoded_reference_points(dataframe: Any) -> tuple[np.ndarray | None, np.ndarray | None]:
    point_columns = [
        f"reference_{joint_name}_position_{component}"
        for joint_name in DECODED_REFERENCE_JOINTS
        for component in ("x", "y", "z")
    ]
    mask_columns = [f"reference_{joint_name}_valid" for joint_name in DECODED_REFERENCE_JOINTS]
    if not all(name in dataframe.columns for name in (*point_columns, *mask_columns)):
        return None, None
    points = np.asarray(dataframe.select(point_columns).to_numpy(), dtype=np.float32).reshape(
        dataframe.height,
        len(DECODED_REFERENCE_JOINTS),
        3,
    )
    masks = np.asarray(dataframe.select(mask_columns).to_numpy(), dtype=np.float32)
    return points, masks


def _load_runtime_decoder_context_points(
    dataframe: Any,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    point_columns = [
        f"heuristic_context_{joint_name}_position_{component}"
        for joint_name in RUNTIME_DECODER_CONTEXT_JOINTS
        for component in ("x", "y", "z")
    ]
    mask_columns = [
        f"heuristic_context_{joint_name}_valid" for joint_name in RUNTIME_DECODER_CONTEXT_JOINTS
    ]
    if not all(name in dataframe.columns for name in (*point_columns, *mask_columns)):
        return None, None
    points = np.asarray(dataframe.select(point_columns).to_numpy(), dtype=np.float32).reshape(
        dataframe.height,
        len(RUNTIME_DECODER_CONTEXT_JOINTS),
        3,
    )
    masks = np.asarray(dataframe.select(mask_columns).to_numpy(), dtype=np.float32)
    return points, masks


def split_masks(
    dataset: LoadedDataset,
    *,
    validation_subjects: set[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if validation_subjects:
        validation_mask = np.isin(dataset.subjects, list(validation_subjects))
        train_mask = ~validation_mask
    else:
        train_mask = dataset.splits == "train"
        validation_mask = dataset.splits == "validation"

    if not train_mask.any() or not validation_mask.any():
        split_index = max(1, int(dataset.features.shape[0] * 0.8))
        train_mask = np.zeros(dataset.features.shape[0], dtype=bool)
        validation_mask = np.zeros(dataset.features.shape[0], dtype=bool)
        train_mask[:split_index] = True
        validation_mask[split_index:] = True

    return train_mask, validation_mask


def _train_constant(
    dataset: LoadedDataset,
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
) -> dict[str, object]:
    start = time.perf_counter()
    train_targets = dataset.targets[train_mask]
    train_target_masks = dataset.masks[train_mask]
    numerator = (train_targets * train_target_masks).sum(axis=0)
    denominator = np.maximum(train_target_masks.sum(axis=0), 1.0)
    prediction = numerator / denominator
    validation_predictions = np.broadcast_to(
        prediction,
        dataset.targets[validation_mask].shape,
    )
    return {
        "model": "constant",
        "train_seconds": time.perf_counter() - start,
        "inference_seconds_per_sample": 0.0,
        **_metric_dict(
            dataset.targets[validation_mask],
            validation_predictions,
            dataset.masks[validation_mask],
            dataset.target_names,
        ),
    }


def _train_torch_model(
    torch: Any,
    *,
    model_name: TorchModelName,
    features: np.ndarray,
    raw_features: np.ndarray,
    targets: np.ndarray,
    masks: np.ndarray,
    heuristics: np.ndarray | None,
    heuristic_masks: np.ndarray | None,
    dataset: LoadedDataset,
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    output_path: Path,
    device: Any,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    hidden_dim: int,
    seed: int,
    feature_names: Sequence[str],
    target_names: Sequence[str],
    residual_regularization_weight: float,
    residual_scale: float,
    residual_smoothing_alpha: float,
    residual_gate_search: bool,
    latent_loss_weight: float,
    decoded_pose_weight: float,
    decoded_torso_weight: float,
    decoded_shoulder_weight: float,
    decoded_elbow_weight: float,
    decoded_arm_direction_weight: float,
    decoded_stretch_weight: float,
    decoded_reach_hinge_weight: float,
    reach_ratio_limit: float,
    surrogate_position_weight: float,
    temporal_velocity_weight: float,
    temporal_acceleration_weight: float,
    temporal_jerk_weight: float,
    temporal_excess_jerk_weight: float,
    pole_velocity_weight: float,
    pole_excess_jerk_weight: float,
    temporal_excess_ratio: float,
    temporal_excess_margin: float,
    disabled_target_groups: set[str],
) -> dict[str, object]:
    torch.manual_seed(seed)
    prediction_mode = "residual" if _is_residual_model(model_name) else "absolute"
    if prediction_mode == "residual" and (heuristics is None or heuristic_masks is None):
        raise RuntimeError("Residual Torch models require heuristic latent columns in the dataset")
    active_disabled_target_groups = (
        disabled_target_groups if prediction_mode == "residual" else set()
    )

    model = _make_torch_model(
        torch,
        model_name,
        input_dim=features.shape[1],
        output_dim=targets.shape[1],
        hidden_dim=hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    train_indices = np.flatnonzero(train_mask)
    x_train = torch.as_tensor(features[train_mask], dtype=torch.float32, device=device)
    raw_x_train = torch.as_tensor(raw_features[train_mask], dtype=torch.float32)
    y_train = torch.as_tensor(targets[train_mask], dtype=torch.float32, device=device)
    m_train = torch.as_tensor(masks[train_mask], dtype=torch.float32, device=device)
    x_val = torch.as_tensor(features[validation_mask], dtype=torch.float32, device=device)
    raw_x_val = torch.as_tensor(raw_features[validation_mask], dtype=torch.float32, device=device)
    y_val = torch.as_tensor(targets[validation_mask], dtype=torch.float32, device=device)
    m_val = torch.as_tensor(masks[validation_mask], dtype=torch.float32, device=device)
    if heuristics is not None and heuristic_masks is not None:
        h_train = torch.as_tensor(heuristics[train_mask], dtype=torch.float32, device=device)
        hm_train = torch.as_tensor(heuristic_masks[train_mask], dtype=torch.float32, device=device)
        h_val = torch.as_tensor(heuristics[validation_mask], dtype=torch.float32, device=device)
        hm_val = torch.as_tensor(
            heuristic_masks[validation_mask],
            dtype=torch.float32,
            device=device,
        )
    else:
        h_train = torch.zeros_like(y_train)
        hm_train = torch.ones_like(m_train)
        h_val = torch.zeros_like(y_val)
        hm_val = torch.ones_like(m_val)
    if dataset.decoded_reference_points is not None and dataset.decoded_reference_masks is not None:
        decoded_points_train = torch.as_tensor(
            dataset.decoded_reference_points[train_mask],
            dtype=torch.float32,
            device=device,
        )
        decoded_masks_train = torch.as_tensor(
            dataset.decoded_reference_masks[train_mask],
            dtype=torch.float32,
            device=device,
        )
        decoded_points_val = torch.as_tensor(
            dataset.decoded_reference_points[validation_mask],
            dtype=torch.float32,
            device=device,
        )
        decoded_masks_val = torch.as_tensor(
            dataset.decoded_reference_masks[validation_mask],
            dtype=torch.float32,
            device=device,
        )
    else:
        decoded_points_train = None
        decoded_masks_train = None
        decoded_points_val = None
        decoded_masks_val = None
    if dataset.runtime_context_points is not None and dataset.runtime_context_masks is not None:
        runtime_context_points_train = torch.as_tensor(
            dataset.runtime_context_points[train_mask],
            dtype=torch.float32,
            device=device,
        )
        runtime_context_masks_train = torch.as_tensor(
            dataset.runtime_context_masks[train_mask],
            dtype=torch.float32,
            device=device,
        )
        runtime_context_points_val = torch.as_tensor(
            dataset.runtime_context_points[validation_mask],
            dtype=torch.float32,
            device=device,
        )
        runtime_context_masks_val = torch.as_tensor(
            dataset.runtime_context_masks[validation_mask],
            dtype=torch.float32,
            device=device,
        )
    else:
        runtime_context_points_train = None
        runtime_context_masks_train = None
        runtime_context_points_val = None
        runtime_context_masks_val = None
    residual_group_gates = _residual_gates_for_disabled_groups(
        target_names,
        active_disabled_target_groups,
    )
    group_mask = _target_group_mask(
        torch,
        target_names,
        active_disabled_target_groups,
        dtype=m_train.dtype,
        device=device,
    )
    effective_m_train = m_train * hm_train if prediction_mode == "residual" else m_train
    effective_m_val = m_val * hm_val if prediction_mode == "residual" else m_val
    effective_m_train = effective_m_train * group_mask[None, :]
    effective_m_val = effective_m_val * group_mask[None, :]
    surrogate_indices = _surrogate_feature_indices(feature_names)
    runtime_decoder_indices = runtime_decoder_feature_indices(feature_names)
    decoded_decoder = (
        "runtime"
        if runtime_decoder_indices is not None
        and runtime_context_points_train is not None
        and runtime_context_masks_train is not None
        else "surrogate"
    )
    train_pairs = torch.as_tensor(
        _temporal_pairs(dataset, train_mask),
        dtype=torch.long,
        device=device,
    )
    train_triples = torch.as_tensor(
        _temporal_triples(dataset, train_mask),
        dtype=torch.long,
        device=device,
    )
    train_quads = torch.as_tensor(
        _temporal_quads(dataset, train_mask),
        dtype=torch.long,
        device=device,
    )
    validation_pairs = torch.as_tensor(
        _temporal_pairs(dataset, validation_mask),
        dtype=torch.long,
        device=device,
    )
    validation_triples = torch.as_tensor(
        _temporal_triples(dataset, validation_mask),
        dtype=torch.long,
        device=device,
    )
    validation_quads = torch.as_tensor(
        _temporal_quads(dataset, validation_mask),
        dtype=torch.long,
        device=device,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    history_path = output_path.with_name(f"{output_path.stem}_history.jsonl")
    if history_path.exists():
        history_path.unlink()

    start = time.perf_counter()

    def history_row(
        *,
        epoch: int,
        train_loss: float | None,
        train_latent_mse: float | None,
        epoch_seconds: float | None,
    ) -> tuple[Any, dict[str, object]]:
        if device.type == "cuda":
            torch.cuda.synchronize()
        with torch.no_grad():
            validation_raw = _apply_residual_gates(
                torch,
                model(x_val),
                residual_group_gates,
                target_names,
            )
            validation_prediction = _compose_torch_latents(
                torch,
                validation_raw,
                h_val,
                prediction_mode=prediction_mode,
                residual_scale=residual_scale,
                target_names=target_names,
            )
            validation_mse = float(
                _masked_mse(torch, validation_prediction, y_val, effective_m_val).detach().cpu()
            )
            row: dict[str, object] = {
                "epoch": epoch,
                "elapsed_seconds": time.perf_counter() - start,
                "epoch_seconds": epoch_seconds,
                "train_loss": train_loss,
                "train_latent_mse": train_latent_mse,
                "validation_mse": validation_mse,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
            row.update(
                _surrogate_validation_metrics(
                    torch,
                    validation_prediction,
                    y_val,
                    effective_m_val,
                    raw_x_val,
                    decoded_reference_points=decoded_points_val,
                    decoded_reference_masks=decoded_masks_val,
                    runtime_context_points=runtime_context_points_val,
                    runtime_context_masks=runtime_context_masks_val,
                    runtime_decoder_indices=runtime_decoder_indices,
                    feature_names=feature_names,
                    target_names=target_names,
                    temporal_pairs=validation_pairs,
                    temporal_triples=validation_triples,
                    temporal_quads=validation_quads,
                    temporal_excess_ratio=temporal_excess_ratio,
                    temporal_excess_margin=temporal_excess_margin,
                    reach_ratio_limit=reach_ratio_limit,
                )
            )
            row.update(_cuda_memory_metrics(torch, device))
        return validation_prediction, row

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _initial_prediction, initial_history_row = history_row(
        epoch=0,
        train_loss=None,
        train_latent_mse=None,
        epoch_seconds=None,
    )
    _write_jsonl_row(history_path, initial_history_row)
    initial_loss_value = initial_history_row["validation_mse"]
    if not isinstance(initial_loss_value, int | float):
        raise TypeError("Expected validation_mse in initial training history")
    initial_loss = float(initial_loss_value)
    for epoch_index in range(epochs):
        epoch_start = time.perf_counter()
        train_loss_sum = 0.0
        train_latent_sum = 0.0
        train_row_count = 0
        permutation = torch.randperm(len(train_indices), device=device)
        for start_index in range(0, len(train_indices), batch_size):
            batch_indices = permutation[start_index : start_index + batch_size]
            current_batch_size = int(batch_indices.numel())
            raw_prediction = _apply_residual_gates(
                torch,
                model(x_train[batch_indices]),
                residual_group_gates,
                target_names,
            )
            prediction = _compose_torch_latents(
                torch,
                raw_prediction,
                h_train[batch_indices],
                prediction_mode=prediction_mode,
                residual_scale=residual_scale,
                target_names=target_names,
            )
            latent_loss = _masked_mse(
                torch,
                prediction,
                y_train[batch_indices],
                effective_m_train[batch_indices],
            )
            loss = latent_loss_weight * latent_loss
            if prediction_mode == "residual" and residual_regularization_weight > 0.0:
                loss = loss + residual_regularization_weight * _masked_mse(
                    torch,
                    raw_prediction,
                    torch.zeros_like(raw_prediction),
                    effective_m_train[batch_indices],
                )
            effective_decoded_weight = decoded_pose_weight + surrogate_position_weight
            if effective_decoded_weight > 0.0 and surrogate_indices is not None:
                batch_raw_features = _tensor_rows(raw_x_train, batch_indices, device=device)
                loss = loss + effective_decoded_weight * _decoded_pose_loss(
                    torch,
                    prediction,
                    y_train[batch_indices],
                    effective_m_train[batch_indices],
                    batch_raw_features,
                    decoded_reference_points=(
                        decoded_points_train[batch_indices]
                        if decoded_points_train is not None
                        else None
                    ),
                    decoded_reference_masks=(
                        decoded_masks_train[batch_indices]
                        if decoded_masks_train is not None
                        else None
                    ),
                    runtime_context_points=(
                        runtime_context_points_train[batch_indices]
                        if runtime_context_points_train is not None
                        else None
                    ),
                    runtime_context_masks=(
                        runtime_context_masks_train[batch_indices]
                        if runtime_context_masks_train is not None
                        else None
                    ),
                    runtime_decoder_indices=runtime_decoder_indices,
                    surrogate_indices=surrogate_indices,
                    target_names=target_names,
                    torso_weight=decoded_torso_weight,
                    shoulder_weight=decoded_shoulder_weight,
                    elbow_weight=decoded_elbow_weight,
                    arm_direction_weight=decoded_arm_direction_weight,
                    stretch_weight=decoded_stretch_weight,
                    reach_hinge_weight=decoded_reach_hinge_weight,
                    reach_ratio_limit=reach_ratio_limit,
                )
            if temporal_velocity_weight > 0.0 and train_pairs.numel() > 0:
                loss = loss + temporal_velocity_weight * _decoded_temporal_motion_loss(
                    torch,
                    model,
                    x_train,
                    raw_x_train,
                    y_train,
                    effective_m_train,
                    h_train,
                    train_pairs,
                    decoded_reference_points=decoded_points_train,
                    decoded_reference_masks=decoded_masks_train,
                    runtime_context_points=runtime_context_points_train,
                    runtime_context_masks=runtime_context_masks_train,
                    runtime_decoder_indices=runtime_decoder_indices,
                    order=1,
                    prediction_mode=prediction_mode,
                    residual_scale=residual_scale,
                    residual_smoothing_alpha=residual_smoothing_alpha,
                    target_names=target_names,
                    feature_names=feature_names,
                    residual_gates=residual_group_gates,
                    max_windows=batch_size,
                )
            if pole_velocity_weight > 0.0 and train_pairs.numel() > 0:
                loss = loss + pole_velocity_weight * _pole_temporal_motion_loss(
                    torch,
                    model,
                    x_train,
                    targets=y_train,
                    target_masks=effective_m_train,
                    heuristics=h_train,
                    windows=train_pairs,
                    order=1,
                    prediction_mode=prediction_mode,
                    residual_scale=residual_scale,
                    residual_smoothing_alpha=residual_smoothing_alpha,
                    target_names=target_names,
                    residual_gates=residual_group_gates,
                    max_windows=batch_size,
                )
            if pole_excess_jerk_weight > 0.0 and train_quads.numel() > 0:
                loss = loss + pole_excess_jerk_weight * _pole_temporal_motion_loss(
                    torch,
                    model,
                    x_train,
                    targets=y_train,
                    target_masks=effective_m_train,
                    heuristics=h_train,
                    windows=train_quads,
                    order=3,
                    prediction_mode=prediction_mode,
                    residual_scale=residual_scale,
                    residual_smoothing_alpha=residual_smoothing_alpha,
                    target_names=target_names,
                    residual_gates=residual_group_gates,
                    max_windows=batch_size,
                    loss_mode="excess",
                    excess_ratio=temporal_excess_ratio,
                    excess_margin=temporal_excess_margin,
                )
            if temporal_acceleration_weight > 0.0 and train_triples.numel() > 0:
                loss = loss + temporal_acceleration_weight * _decoded_temporal_motion_loss(
                    torch,
                    model,
                    x_train,
                    raw_x_train,
                    y_train,
                    effective_m_train,
                    h_train,
                    train_triples,
                    decoded_reference_points=decoded_points_train,
                    decoded_reference_masks=decoded_masks_train,
                    runtime_context_points=runtime_context_points_train,
                    runtime_context_masks=runtime_context_masks_train,
                    runtime_decoder_indices=runtime_decoder_indices,
                    order=2,
                    prediction_mode=prediction_mode,
                    residual_scale=residual_scale,
                    residual_smoothing_alpha=residual_smoothing_alpha,
                    target_names=target_names,
                    feature_names=feature_names,
                    residual_gates=residual_group_gates,
                    max_windows=batch_size,
                )
            if temporal_jerk_weight > 0.0 and train_quads.numel() > 0:
                loss = loss + temporal_jerk_weight * _decoded_temporal_motion_loss(
                    torch,
                    model,
                    x_train,
                    raw_x_train,
                    y_train,
                    effective_m_train,
                    h_train,
                    train_quads,
                    decoded_reference_points=decoded_points_train,
                    decoded_reference_masks=decoded_masks_train,
                    runtime_context_points=runtime_context_points_train,
                    runtime_context_masks=runtime_context_masks_train,
                    runtime_decoder_indices=runtime_decoder_indices,
                    order=3,
                    prediction_mode=prediction_mode,
                    residual_scale=residual_scale,
                    residual_smoothing_alpha=residual_smoothing_alpha,
                    target_names=target_names,
                    feature_names=feature_names,
                    residual_gates=residual_group_gates,
                    max_windows=batch_size,
                )
            if temporal_excess_jerk_weight > 0.0 and train_quads.numel() > 0:
                loss = loss + temporal_excess_jerk_weight * _decoded_temporal_motion_loss(
                    torch,
                    model,
                    x_train,
                    raw_x_train,
                    y_train,
                    effective_m_train,
                    h_train,
                    train_quads,
                    decoded_reference_points=decoded_points_train,
                    decoded_reference_masks=decoded_masks_train,
                    runtime_context_points=runtime_context_points_train,
                    runtime_context_masks=runtime_context_masks_train,
                    runtime_decoder_indices=runtime_decoder_indices,
                    order=3,
                    prediction_mode=prediction_mode,
                    residual_scale=residual_scale,
                    residual_smoothing_alpha=residual_smoothing_alpha,
                    target_names=target_names,
                    feature_names=feature_names,
                    residual_gates=residual_group_gates,
                    max_windows=batch_size,
                    loss_mode="excess",
                    excess_ratio=temporal_excess_ratio,
                    excess_margin=temporal_excess_margin,
                )
            train_loss_sum += float(loss.detach().cpu()) * current_batch_size
            train_latent_sum += float(latent_loss.detach().cpu()) * current_batch_size
            train_row_count += current_batch_size
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        _epoch_prediction, epoch_history_row = history_row(
            epoch=epoch_index + 1,
            train_loss=train_loss_sum / max(1, train_row_count),
            train_latent_mse=train_latent_sum / max(1, train_row_count),
            epoch_seconds=time.perf_counter() - epoch_start,
        )
        _write_jsonl_row(history_path, epoch_history_row)
    if device.type == "cuda":
        torch.cuda.synchronize()
    train_seconds = time.perf_counter() - start

    inference_start = time.perf_counter()
    with torch.no_grad():
        validation_raw_tensor = _apply_residual_gates(
            torch,
            model(x_val),
            residual_group_gates,
            target_names,
        )
        residual_gates = (
            _fit_residual_gates(
                torch,
                validation_raw_tensor,
                h_val,
                y_val,
                effective_m_val,
                raw_x_val,
                decoded_reference_points=decoded_points_val,
                decoded_reference_masks=decoded_masks_val,
                runtime_context_points=runtime_context_points_val,
                runtime_context_masks=runtime_context_masks_val,
                runtime_decoder_indices=runtime_decoder_indices,
                prediction_mode=prediction_mode,
                residual_scale=residual_scale,
                target_names=target_names,
                feature_names=feature_names,
                surrogate_position_weight=surrogate_position_weight,
                decoded_pose_weight=decoded_pose_weight,
                decoded_torso_weight=decoded_torso_weight,
                decoded_shoulder_weight=decoded_shoulder_weight,
                decoded_elbow_weight=decoded_elbow_weight,
                decoded_arm_direction_weight=decoded_arm_direction_weight,
                decoded_stretch_weight=decoded_stretch_weight,
                decoded_reach_hinge_weight=decoded_reach_hinge_weight,
                reach_ratio_limit=reach_ratio_limit,
                base_residual_gates=residual_group_gates,
            )
            if residual_gate_search and prediction_mode == "residual"
            else dict(residual_group_gates)
        )
        gated_validation_raw_tensor = _apply_residual_gates(
            torch,
            validation_raw_tensor,
            residual_gates,
            target_names,
        )
        validation_predictions_tensor = _compose_torch_latents(
            torch,
            gated_validation_raw_tensor,
            h_val,
            prediction_mode=prediction_mode,
            residual_scale=residual_scale,
            target_names=target_names,
        )
    if device.type == "cuda":
        torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - inference_start
    validation_predictions = validation_predictions_tensor.detach().cpu().numpy()
    final_loss = float(
        _masked_mse(torch, validation_predictions_tensor, y_val, effective_m_val).detach().cpu()
    )
    extra_metrics = _surrogate_validation_metrics(
        torch,
        validation_predictions_tensor,
        y_val,
        effective_m_val,
        raw_x_val,
        decoded_reference_points=decoded_points_val,
        decoded_reference_masks=decoded_masks_val,
        runtime_context_points=runtime_context_points_val,
        runtime_context_masks=runtime_context_masks_val,
        runtime_decoder_indices=runtime_decoder_indices,
        feature_names=feature_names,
        target_names=target_names,
        temporal_pairs=validation_pairs,
        temporal_triples=validation_triples,
        temporal_quads=validation_quads,
        temporal_excess_ratio=temporal_excess_ratio,
        temporal_excess_margin=temporal_excess_margin,
        reach_ratio_limit=reach_ratio_limit,
    )
    torch.save(
        {
            "model_name": model_name,
            "prediction_mode": prediction_mode,
            "latent_loss_weight": latent_loss_weight,
            "decoded_pose_weight": decoded_pose_weight,
            "decoded_torso_weight": decoded_torso_weight,
            "decoded_shoulder_weight": decoded_shoulder_weight,
            "decoded_elbow_weight": decoded_elbow_weight,
            "decoded_arm_direction_weight": decoded_arm_direction_weight,
            "decoded_stretch_weight": decoded_stretch_weight,
            "decoded_reach_hinge_weight": decoded_reach_hinge_weight,
            "reach_ratio_limit": reach_ratio_limit,
            "temporal_velocity_weight": temporal_velocity_weight,
            "temporal_acceleration_weight": temporal_acceleration_weight,
            "temporal_jerk_weight": temporal_jerk_weight,
            "temporal_excess_jerk_weight": temporal_excess_jerk_weight,
            "pole_velocity_weight": pole_velocity_weight,
            "pole_excess_jerk_weight": pole_excess_jerk_weight,
            "temporal_excess_ratio": temporal_excess_ratio,
            "temporal_excess_margin": temporal_excess_margin,
            "disabled_target_groups": sorted(active_disabled_target_groups),
            "decoded_decoder": decoded_decoder,
            "history_path": history_path.as_posix(),
            "model_state_dict": model.state_dict(),
            "feature_names": list(feature_names),
            "target_names": list(target_names),
            "hidden_dim": hidden_dim,
        },
        output_path,
    )
    return {
        "model": model_name,
        "device": str(device),
        "epochs": epochs,
        "train_seconds": train_seconds,
        "initial_validation_mse": initial_loss,
        "final_validation_mse": final_loss,
        "inference_seconds_per_sample": inference_seconds / max(1, int(validation_mask.sum())),
        "artifact": output_path.as_posix(),
        "history_path": history_path.as_posix(),
        "prediction_mode": prediction_mode,
        "latent_loss_weight": latent_loss_weight,
        "decoded_pose_weight": decoded_pose_weight,
        "decoded_torso_weight": decoded_torso_weight,
        "decoded_shoulder_weight": decoded_shoulder_weight,
        "decoded_elbow_weight": decoded_elbow_weight,
        "decoded_arm_direction_weight": decoded_arm_direction_weight,
        "decoded_stretch_weight": decoded_stretch_weight,
        "decoded_reach_hinge_weight": decoded_reach_hinge_weight,
        "reach_ratio_limit": reach_ratio_limit,
        "temporal_jerk_weight": temporal_jerk_weight,
        "temporal_excess_jerk_weight": temporal_excess_jerk_weight,
        "pole_velocity_weight": pole_velocity_weight,
        "pole_excess_jerk_weight": pole_excess_jerk_weight,
        "temporal_excess_ratio": temporal_excess_ratio,
        "temporal_excess_margin": temporal_excess_margin,
        "disabled_target_groups": sorted(active_disabled_target_groups),
        "decoded_decoder": decoded_decoder,
        **extra_metrics,
        **_metric_dict(
            targets[validation_mask],
            validation_predictions,
            masks[validation_mask],
            target_names,
        ),
    }


def _train_xgboost(
    dataset: LoadedDataset,
    features: np.ndarray,
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    *,
    output_dir: Path,
    device: Literal["cuda", "cpu"],
    estimators: int,
    seed: int,
) -> dict[str, object]:
    if importlib.util.find_spec("xgboost") is None:
        raise RuntimeError("xgboost is not installed; run through `uv run --group ml ...`")
    import xgboost as xgb

    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    validation_predictions = np.zeros_like(dataset.targets[validation_mask])
    validation_matrix = xgb.DMatrix(features[validation_mask])
    params = {
        "objective": "reg:squarederror",
        "max_depth": 4,
        "eta": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "device": device,
        "seed": seed,
    }
    for target_index, target_name in enumerate(dataset.target_names):
        train_rows = train_mask & (dataset.masks[:, target_index] > 0.5)
        if not train_rows.any():
            continue
        train_matrix = xgb.DMatrix(
            features[train_rows],
            label=dataset.targets[train_rows, target_index],
        )
        model = xgb.train(params, train_matrix, num_boost_round=estimators, verbose_eval=False)
        validation_predictions[:, target_index] = model.predict(validation_matrix)
        model.save_model(str(output_dir / f"{target_name}.json"))
    train_seconds = time.perf_counter() - start
    return {
        "model": "xgboost",
        "device": device,
        "estimators": estimators,
        "train_seconds": train_seconds,
        "inference_seconds_per_sample": 0.0,
        "artifact": output_dir.as_posix(),
        **_metric_dict(
            dataset.targets[validation_mask],
            validation_predictions,
            dataset.masks[validation_mask],
            dataset.target_names,
        ),
    }


def _make_torch_model(
    torch: Any,
    model_name: TorchModelName,
    *,
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
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
    raise ValueError(f"Unsupported Torch model: {model_name}")


@dataclass(frozen=True, slots=True)
class SurrogateFeatureIndices:
    head_y: int
    left_hand: tuple[int, int, int]
    right_hand: tuple[int, int, int]
    human_height: int
    shoulder_width: int
    upper_arm: int
    lower_arm: int
    arm_length: int


def _is_residual_model(model_name: TorchModelName) -> bool:
    del model_name
    return False


def _compose_torch_latents(
    torch: Any,
    raw_prediction: Any,
    heuristic: Any,
    *,
    prediction_mode: Literal["absolute", "residual"],
    residual_scale: float,
    target_names: Sequence[str],
) -> Any:
    if prediction_mode == "absolute":
        return _normalize_torch_poles(torch, raw_prediction, target_names)
    prediction = heuristic + residual_scale * raw_prediction
    return _normalize_torch_poles(torch, prediction, target_names)


def _normalize_torch_poles(
    torch: Any,
    latents: Any,
    target_names: Sequence[str],
) -> Any:
    columns = [latents[:, index] for index in range(latents.shape[1])]
    for side in ("left", "right"):
        indices = [
            _target_index(target_names, f"{side}_elbow_pole_{component}")
            for component in ("x", "y", "z")
        ]
        magnitude_squared = columns[indices[0]] * columns[indices[0]]
        for index in indices[1:]:
            magnitude_squared = magnitude_squared + columns[index] * columns[index]
        magnitude = torch.sqrt(magnitude_squared).clamp_min(1e-6)
        for index in indices:
            columns[index] = columns[index] / magnitude
    return torch.stack(columns, dim=1)


def _fit_residual_gates(
    torch: Any,
    raw_prediction: Any,
    heuristic: Any,
    target: Any,
    mask: Any,
    raw_features: Any,
    *,
    decoded_reference_points: Any | None,
    decoded_reference_masks: Any | None,
    runtime_context_points: Any | None,
    runtime_context_masks: Any | None,
    runtime_decoder_indices: RuntimeDecoderFeatureIndices | None,
    prediction_mode: Literal["absolute", "residual"],
    residual_scale: float,
    target_names: Sequence[str],
    feature_names: Sequence[str],
    surrogate_position_weight: float,
    decoded_pose_weight: float,
    decoded_torso_weight: float,
    decoded_shoulder_weight: float,
    decoded_elbow_weight: float,
    decoded_arm_direction_weight: float,
    decoded_stretch_weight: float,
    decoded_reach_hinge_weight: float,
    reach_ratio_limit: float,
    base_residual_gates: Mapping[str, float],
) -> dict[str, float]:
    if prediction_mode != "residual":
        return {name: 1.0 for name in target_names}

    gates = {name: float(base_residual_gates.get(name, 1.0)) for name in target_names}
    locked_targets = {name for name, gate in gates.items() if gate <= 0.0}
    grid = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
    for _pass in range(2):
        for group_targets in TARGET_GROUPS.values():
            active_targets = [name for name in group_targets if name not in locked_targets]
            if not active_targets:
                continue
            best_gate = 1.0
            best_loss = float("inf")
            for candidate in grid:
                candidate_gates = dict(gates)
                for name in active_targets:
                    if name in candidate_gates:
                        candidate_gates[name] = candidate
                gated_raw = _apply_residual_gates(
                    torch,
                    raw_prediction,
                    candidate_gates,
                    target_names,
                )
                prediction = _compose_torch_latents(
                    torch,
                    gated_raw,
                    heuristic,
                    prediction_mode=prediction_mode,
                    residual_scale=residual_scale,
                    target_names=target_names,
                )
                loss = _residual_gate_loss(
                    torch,
                    prediction,
                    target,
                    mask,
                    raw_features,
                    decoded_reference_points=decoded_reference_points,
                    decoded_reference_masks=decoded_reference_masks,
                    runtime_context_points=runtime_context_points,
                    runtime_context_masks=runtime_context_masks,
                    runtime_decoder_indices=runtime_decoder_indices,
                    feature_names=feature_names,
                    target_names=target_names,
                    surrogate_position_weight=surrogate_position_weight,
                    decoded_pose_weight=decoded_pose_weight,
                    decoded_torso_weight=decoded_torso_weight,
                    decoded_shoulder_weight=decoded_shoulder_weight,
                    decoded_elbow_weight=decoded_elbow_weight,
                    decoded_arm_direction_weight=decoded_arm_direction_weight,
                    decoded_stretch_weight=decoded_stretch_weight,
                    decoded_reach_hinge_weight=decoded_reach_hinge_weight,
                    reach_ratio_limit=reach_ratio_limit,
                )
                if loss < best_loss:
                    best_loss = loss
                    best_gate = candidate
            for name in active_targets:
                if name in gates:
                    gates[name] = best_gate
    return gates


def _residual_gate_loss(
    torch: Any,
    prediction: Any,
    target: Any,
    mask: Any,
    raw_features: Any,
    *,
    decoded_reference_points: Any | None,
    decoded_reference_masks: Any | None,
    runtime_context_points: Any | None,
    runtime_context_masks: Any | None,
    runtime_decoder_indices: RuntimeDecoderFeatureIndices | None,
    feature_names: Sequence[str],
    target_names: Sequence[str],
    surrogate_position_weight: float,
    decoded_pose_weight: float,
    decoded_torso_weight: float,
    decoded_shoulder_weight: float,
    decoded_elbow_weight: float,
    decoded_arm_direction_weight: float,
    decoded_stretch_weight: float,
    decoded_reach_hinge_weight: float,
    reach_ratio_limit: float,
) -> float:
    loss = _masked_mse(torch, prediction, target, mask)
    surrogate_indices = _surrogate_feature_indices(feature_names)
    effective_decoded_weight = decoded_pose_weight + surrogate_position_weight
    if surrogate_indices is not None and effective_decoded_weight > 0.0:
        loss = loss + effective_decoded_weight * _decoded_pose_loss(
            torch,
            prediction,
            target,
            mask,
            raw_features,
            decoded_reference_points=decoded_reference_points,
            decoded_reference_masks=decoded_reference_masks,
            runtime_context_points=runtime_context_points,
            runtime_context_masks=runtime_context_masks,
            runtime_decoder_indices=runtime_decoder_indices,
            surrogate_indices=surrogate_indices,
            target_names=target_names,
            torso_weight=decoded_torso_weight,
            shoulder_weight=decoded_shoulder_weight,
            elbow_weight=decoded_elbow_weight,
            arm_direction_weight=decoded_arm_direction_weight,
            stretch_weight=decoded_stretch_weight,
            reach_hinge_weight=decoded_reach_hinge_weight,
            reach_ratio_limit=reach_ratio_limit,
        )
    return float(loss.detach().cpu())


def _apply_residual_gates(
    torch: Any,
    raw_prediction: Any,
    residual_gates: Mapping[str, float],
    target_names: Sequence[str],
) -> Any:
    gate_vector = torch.tensor(
        [float(residual_gates.get(name, 1.0)) for name in target_names],
        dtype=raw_prediction.dtype,
        device=raw_prediction.device,
    )
    return raw_prediction * gate_vector[None, :]


def _validate_target_groups(groups: set[str]) -> set[str]:
    invalid_groups = sorted(groups - set(TARGET_GROUPS))
    if invalid_groups:
        raise ValueError(f"Unknown target groups: {', '.join(invalid_groups)}")
    return set(groups)


def _residual_gates_for_disabled_groups(
    target_names: Sequence[str],
    disabled_target_groups: set[str],
) -> dict[str, float]:
    disabled_targets = {
        target
        for group_name in disabled_target_groups
        for target in TARGET_GROUPS.get(group_name, ())
    }
    return {name: 0.0 if name in disabled_targets else 1.0 for name in target_names}


def _target_group_mask(
    torch: Any,
    target_names: Sequence[str],
    disabled_target_groups: set[str],
    *,
    dtype: Any,
    device: Any,
) -> Any:
    gates = _residual_gates_for_disabled_groups(target_names, disabled_target_groups)
    return torch.tensor(
        [float(gates[name]) for name in target_names],
        dtype=dtype,
        device=device,
    )


def _decoded_pose_loss(
    torch: Any,
    prediction: Any,
    target: Any,
    target_mask: Any,
    raw_features: Any,
    *,
    decoded_reference_points: Any | None,
    decoded_reference_masks: Any | None,
    surrogate_indices: SurrogateFeatureIndices,
    target_names: Sequence[str],
    torso_weight: float,
    shoulder_weight: float,
    elbow_weight: float,
    arm_direction_weight: float,
    stretch_weight: float,
    reach_hinge_weight: float,
    reach_ratio_limit: float,
    runtime_context_points: Any | None = None,
    runtime_context_masks: Any | None = None,
    runtime_decoder_indices: RuntimeDecoderFeatureIndices | None = None,
) -> Any:
    terms = _decoded_pose_terms(
        torch,
        prediction,
        target,
        target_mask,
        raw_features,
        decoded_reference_points=decoded_reference_points,
        decoded_reference_masks=decoded_reference_masks,
        runtime_context_points=runtime_context_points,
        runtime_context_masks=runtime_context_masks,
        runtime_decoder_indices=runtime_decoder_indices,
        surrogate_indices=surrogate_indices,
        target_names=target_names,
        reach_ratio_limit=reach_ratio_limit,
    )
    total = torch.zeros((), dtype=prediction.dtype, device=prediction.device)
    total_weight = 0.0
    for term_name, term_weight in (
        ("torso_mse", torso_weight),
        ("shoulder_mse", shoulder_weight),
        ("elbow_mse", elbow_weight),
        ("arm_direction_mse", arm_direction_weight),
        ("stretch_mse", stretch_weight),
        ("reach_hinge_mse", reach_hinge_weight),
    ):
        if term_weight <= 0.0:
            continue
        total = total + term_weight * terms[term_name]
        total_weight += term_weight
    if total_weight <= 0.0:
        return total
    return total / total_weight


def _decoded_pose_terms(
    torch: Any,
    prediction: Any,
    target: Any,
    target_mask: Any,
    raw_features: Any,
    *,
    decoded_reference_points: Any | None,
    decoded_reference_masks: Any | None,
    surrogate_indices: SurrogateFeatureIndices,
    target_names: Sequence[str],
    runtime_context_points: Any | None = None,
    runtime_context_masks: Any | None = None,
    runtime_decoder_indices: RuntimeDecoderFeatureIndices | None = None,
    reach_ratio_limit: float = 0.97,
) -> dict[str, Any]:
    prediction_points, prediction_point_masks = _decoded_prediction_points_and_masks(
        torch,
        prediction,
        raw_features,
        runtime_context_points=runtime_context_points,
        runtime_context_masks=runtime_context_masks,
        runtime_decoder_indices=runtime_decoder_indices,
        surrogate_indices=surrogate_indices,
        target_names=target_names,
    )
    target_points, target_point_masks = _decoded_target_points_and_masks(
        torch,
        target,
        target_mask,
        raw_features,
        decoded_reference_points=decoded_reference_points,
        decoded_reference_masks=decoded_reference_masks,
        surrogate_indices=surrogate_indices,
        target_names=target_names,
    )
    point_masks = target_point_masks * prediction_point_masks
    left_hand = _runtime_context_joint_point(
        runtime_context_points,
        runtime_context_masks,
        "left_hand",
        fallback=_feature_position(raw_features, surrogate_indices.left_hand),
    )
    right_hand = _runtime_context_joint_point(
        runtime_context_points,
        runtime_context_masks,
        "right_hand",
        fallback=_feature_position(raw_features, surrogate_indices.right_hand),
    )
    upper_arm = raw_features[:, surrogate_indices.upper_arm]
    lower_arm = raw_features[:, surrogate_indices.lower_arm]
    human_height = raw_features[:, surrogate_indices.human_height]
    shoulder_width = raw_features[:, surrogate_indices.shoulder_width]
    head = _runtime_context_joint_point(
        runtime_context_points,
        runtime_context_masks,
        "head",
        fallback=torch.stack(
            (
                torch.zeros_like(raw_features[:, surrogate_indices.head_y]),
                raw_features[:, surrogate_indices.head_y],
                torch.zeros_like(raw_features[:, surrogate_indices.head_y]),
            ),
            dim=1,
        ),
    )

    torso_mask = point_masks[:, 0] * point_masks[:, 1]
    left_shoulder_mask = point_masks[:, 2]
    right_shoulder_mask = point_masks[:, 3]
    left_elbow_mask = point_masks[:, 4]
    right_elbow_mask = point_masks[:, 5]

    torso_mse = _average_losses(
        torch,
        (
            _masked_point_mse(prediction_points[:, 0], target_points[:, 0], torso_mask),
            _masked_point_mse(prediction_points[:, 1], target_points[:, 1], torso_mask),
        ),
    )
    shoulder_mse = _average_losses(
        torch,
        (
            _masked_point_mse(prediction_points[:, 2], target_points[:, 2], left_shoulder_mask),
            _masked_point_mse(prediction_points[:, 3], target_points[:, 3], right_shoulder_mask),
        ),
    )
    elbow_mse = _average_losses(
        torch,
        (
            _masked_point_mse(prediction_points[:, 4], target_points[:, 4], left_elbow_mask),
            _masked_point_mse(prediction_points[:, 5], target_points[:, 5], right_elbow_mask),
        ),
    )
    arm_direction_mse = _average_losses(
        torch,
        (
            _masked_direction_mse(
                torch,
                prediction_points[:, 4] - prediction_points[:, 2],
                target_points[:, 4] - target_points[:, 2],
                left_elbow_mask,
            ),
            _masked_direction_mse(
                torch,
                left_hand - prediction_points[:, 4],
                left_hand - target_points[:, 4],
                left_elbow_mask,
            ),
            _masked_direction_mse(
                torch,
                prediction_points[:, 5] - prediction_points[:, 3],
                target_points[:, 5] - target_points[:, 3],
                right_elbow_mask,
            ),
            _masked_direction_mse(
                torch,
                right_hand - prediction_points[:, 5],
                right_hand - target_points[:, 5],
                right_elbow_mask,
            ),
        ),
    )
    stretch_mse = _average_losses(
        torch,
        (
            _decoded_stretch_loss(
                torch,
                shoulder=prediction_points[:, 2],
                elbow=prediction_points[:, 4],
                hand=left_hand,
                upper_arm=upper_arm,
                lower_arm=lower_arm,
                mask=left_elbow_mask,
            ),
            _decoded_stretch_loss(
                torch,
                shoulder=prediction_points[:, 3],
                elbow=prediction_points[:, 5],
                hand=right_hand,
                upper_arm=upper_arm,
                lower_arm=lower_arm,
                mask=right_elbow_mask,
            ),
            _decoded_anthropometric_loss(
                torch,
                head=head,
                chest=prediction_points[:, 0],
                neck=prediction_points[:, 1],
                left_shoulder=prediction_points[:, 2],
                right_shoulder=prediction_points[:, 3],
                human_height=human_height,
                shoulder_width=shoulder_width,
                torso_mask=torso_mask,
                shoulder_mask=left_shoulder_mask * right_shoulder_mask,
            ),
        ),
    )
    reach_hinge_mse = _average_losses(
        torch,
        (
            _decoded_reach_hinge_loss(
                torch,
                shoulder=prediction_points[:, 2],
                hand=left_hand,
                arm_length=upper_arm + lower_arm,
                mask=left_elbow_mask,
                reach_ratio_limit=reach_ratio_limit,
            ),
            _decoded_reach_hinge_loss(
                torch,
                shoulder=prediction_points[:, 3],
                hand=right_hand,
                arm_length=upper_arm + lower_arm,
                mask=right_elbow_mask,
                reach_ratio_limit=reach_ratio_limit,
            ),
        ),
    )
    return {
        "torso_mse": torso_mse,
        "shoulder_mse": shoulder_mse,
        "elbow_mse": elbow_mse,
        "arm_direction_mse": arm_direction_mse,
        "stretch_mse": stretch_mse,
        "reach_hinge_mse": reach_hinge_mse,
        "position_mse": _average_losses(torch, (torso_mse, shoulder_mse, elbow_mse)),
    }


def _decoded_torso_mask(target_mask: Any, target_names: Sequence[str]) -> Any:
    mask = target_mask[:, _target_index(target_names, "chest_rel_x")]
    for name in (
        "chest_rel_y",
        "chest_rel_z",
        "neck_rel_x",
        "neck_rel_y",
        "neck_rel_z",
    ):
        mask = mask * target_mask[:, _target_index(target_names, name)]
    return mask


def _decoded_joint_mask(
    target_mask: Any,
    target_names: Sequence[str],
    side: Literal["left", "right"],
    joint: Literal["shoulder", "elbow"],
) -> Any:
    if joint == "shoulder":
        columns = (
            f"{side}_shoulder_rel_x",
            f"{side}_shoulder_rel_y",
            f"{side}_shoulder_rel_z",
        )
    else:
        columns = (
            f"{side}_elbow_pole_x",
            f"{side}_elbow_pole_y",
            f"{side}_elbow_pole_z",
        )
    joint_mask = target_mask[:, _target_index(target_names, columns[0])]
    for name in columns[1:]:
        joint_mask = joint_mask * target_mask[:, _target_index(target_names, name)]
    if joint == "elbow":
        joint_mask = joint_mask * _decoded_joint_mask(target_mask, target_names, side, "shoulder")
    return joint_mask


def _average_losses(torch: Any, losses: Sequence[Any]) -> Any:
    if not losses:
        return torch.zeros(())
    total = losses[0]
    for loss in losses[1:]:
        total = total + loss
    return total / max(1, len(losses))


def _masked_point_mse(prediction: Any, target: Any, mask: Any) -> Any:
    distance_squared = ((prediction - target) ** 2).sum(dim=1)
    return (distance_squared * mask).sum() / mask.sum().clamp_min(1.0)


def _masked_direction_mse(torch: Any, prediction: Any, target: Any, mask: Any) -> Any:
    prediction_direction = _torch_normalize(torch, prediction, fallback=(0.0, 1.0, 0.0))
    target_direction = _torch_normalize(torch, target, fallback=(0.0, 1.0, 0.0))
    direction_error = ((prediction_direction - target_direction) ** 2).sum(dim=1)
    return (direction_error * mask).sum() / mask.sum().clamp_min(1.0)


def _decoded_stretch_loss(
    torch: Any,
    *,
    shoulder: Any,
    elbow: Any,
    hand: Any,
    upper_arm: Any,
    lower_arm: Any,
    mask: Any,
) -> Any:
    upper_error = torch.linalg.vector_norm(elbow - shoulder, dim=1) - upper_arm
    lower_error = torch.linalg.vector_norm(hand - elbow, dim=1) - lower_arm
    stretch_error = upper_error * upper_error + lower_error * lower_error
    return (stretch_error * mask).sum() / mask.sum().clamp_min(1.0)


def _decoded_reach_hinge_loss(
    torch: Any,
    *,
    shoulder: Any,
    hand: Any,
    arm_length: Any,
    mask: Any,
    reach_ratio_limit: float,
) -> Any:
    reach_ratio = torch.linalg.vector_norm(hand - shoulder, dim=1) / arm_length.clamp_min(1e-4)
    excess = torch.relu(reach_ratio - reach_ratio_limit)
    return (excess * excess * mask).sum() / mask.sum().clamp_min(1.0)


def _decoded_anthropometric_loss(
    torch: Any,
    *,
    head: Any,
    chest: Any,
    neck: Any,
    left_shoulder: Any,
    right_shoulder: Any,
    human_height: Any,
    shoulder_width: Any,
    torso_mask: Any,
    shoulder_mask: Any,
) -> Any:
    neck_error = torch.linalg.vector_norm(head - neck, dim=1) - human_height * 0.04
    chest_error = torch.linalg.vector_norm(neck - chest, dim=1) - human_height * 0.09
    width_error = torch.linalg.vector_norm(left_shoulder - right_shoulder, dim=1) - shoulder_width
    torso_loss = ((neck_error * neck_error + chest_error * chest_error) * torso_mask).sum()
    torso_loss = torso_loss / torso_mask.sum().clamp_min(1.0)
    shoulder_loss = (width_error * width_error * shoulder_mask).sum()
    shoulder_loss = shoulder_loss / shoulder_mask.sum().clamp_min(1.0)
    return (torso_loss + shoulder_loss) * 0.5


def _pole_temporal_motion_loss(
    torch: Any,
    model: Any,
    features: Any,
    *,
    targets: Any,
    target_masks: Any,
    heuristics: Any,
    windows: Any,
    order: Literal[1, 2, 3],
    prediction_mode: Literal["absolute", "residual"],
    residual_scale: float,
    residual_smoothing_alpha: float,
    target_names: Sequence[str],
    residual_gates: Mapping[str, float],
    max_windows: int,
    loss_mode: Literal["match", "excess"] = "match",
    excess_ratio: float = 1.25,
    excess_margin: float = 0.0,
) -> Any:
    if windows.shape[1] != order + 1:
        raise ValueError("Temporal windows must be one frame wider than the derivative order")
    selected_windows = windows[
        torch.randint(
            windows.shape[0],
            (min(max_windows, windows.shape[0]),),
            device=features.device,
        )
    ]
    flat_indices = selected_windows.flatten()
    raw_prediction = _apply_residual_gates(
        torch,
        model(features[flat_indices]),
        residual_gates,
        target_names,
    )
    if prediction_mode == "residual":
        raw_prediction = _smooth_temporal_residuals(
            torch,
            raw_prediction,
            window_count=selected_windows.shape[0],
            window_width=selected_windows.shape[1],
            alpha=residual_smoothing_alpha,
        )
    prediction = _compose_torch_latents(
        torch,
        raw_prediction,
        heuristics[flat_indices],
        prediction_mode=prediction_mode,
        residual_scale=residual_scale,
        target_names=target_names,
    )
    prediction_poles = _pole_vectors(torch, prediction, target_names)
    target_poles = _pole_vectors(torch, targets[flat_indices], target_names)
    pole_masks = _pole_masks(target_masks[flat_indices], target_names)
    window_count = selected_windows.shape[0]
    window_width = selected_windows.shape[1]
    prediction_delta = _temporal_difference(
        prediction_poles.reshape(window_count, window_width, -1, 3),
        order=order,
    )
    target_delta = _temporal_difference(
        target_poles.reshape(window_count, window_width, -1, 3),
        order=order,
    )
    temporal_mask = pole_masks.reshape(window_count, window_width, -1).prod(dim=1)
    if loss_mode == "excess":
        return _masked_temporal_excess_motion_mse(
            torch,
            prediction_delta,
            target_delta,
            temporal_mask,
            excess_ratio=excess_ratio,
            excess_margin=excess_margin,
        )
    return _masked_temporal_point_mse(prediction_delta, target_delta, temporal_mask)


def _decoded_temporal_motion_loss(
    torch: Any,
    model: Any,
    features: Any,
    raw_features: Any,
    targets: Any,
    target_masks: Any,
    heuristics: Any,
    windows: Any,
    *,
    decoded_reference_points: Any | None,
    decoded_reference_masks: Any | None,
    runtime_context_points: Any | None,
    runtime_context_masks: Any | None,
    runtime_decoder_indices: RuntimeDecoderFeatureIndices | None,
    order: Literal[1, 2, 3],
    prediction_mode: Literal["absolute", "residual"],
    residual_scale: float,
    residual_smoothing_alpha: float,
    target_names: Sequence[str],
    feature_names: Sequence[str],
    residual_gates: Mapping[str, float],
    max_windows: int,
    loss_mode: Literal["match", "excess"] = "match",
    excess_ratio: float = 1.25,
    excess_margin: float = 0.0,
) -> Any:
    surrogate_indices = _surrogate_feature_indices(feature_names)
    if surrogate_indices is None:
        return torch.zeros((), dtype=features.dtype, device=features.device)
    if windows.shape[1] != order + 1:
        raise ValueError("Temporal windows must be one frame wider than the derivative order")
    selected_windows = windows[
        torch.randint(
            windows.shape[0],
            (min(max_windows, windows.shape[0]),),
            device=features.device,
        )
    ]
    flat_indices = selected_windows.flatten()
    flat_raw_features = _tensor_rows(raw_features, flat_indices, device=features.device)
    raw_prediction = _apply_residual_gates(
        torch,
        model(features[flat_indices]),
        residual_gates,
        target_names,
    )
    prediction_points, target_points, point_masks = _decoded_temporal_points(
        torch,
        raw_prediction,
        targets[flat_indices],
        target_masks[flat_indices],
        heuristics[flat_indices],
        flat_raw_features,
        decoded_reference_points=(
            decoded_reference_points[flat_indices] if decoded_reference_points is not None else None
        ),
        decoded_reference_masks=(
            decoded_reference_masks[flat_indices] if decoded_reference_masks is not None else None
        ),
        runtime_context_points=(
            runtime_context_points[flat_indices] if runtime_context_points is not None else None
        ),
        runtime_context_masks=(
            runtime_context_masks[flat_indices] if runtime_context_masks is not None else None
        ),
        window_count=selected_windows.shape[0],
        window_width=selected_windows.shape[1],
        prediction_mode=prediction_mode,
        residual_scale=residual_scale,
        residual_smoothing_alpha=residual_smoothing_alpha,
        runtime_decoder_indices=runtime_decoder_indices,
        surrogate_indices=surrogate_indices,
        target_names=target_names,
    )
    prediction_delta = _temporal_difference(prediction_points, order=order)
    target_delta = _temporal_difference(target_points, order=order)
    temporal_mask = point_masks.prod(dim=1)
    if loss_mode == "excess":
        return _masked_temporal_excess_motion_mse(
            torch,
            prediction_delta,
            target_delta,
            temporal_mask,
            excess_ratio=excess_ratio,
            excess_margin=excess_margin,
        )
    return _masked_temporal_point_mse(prediction_delta, target_delta, temporal_mask)


def _decoded_temporal_points(
    torch: Any,
    raw_prediction: Any,
    targets: Any,
    target_masks: Any,
    heuristics: Any,
    raw_features: Any,
    *,
    decoded_reference_points: Any | None,
    decoded_reference_masks: Any | None,
    runtime_context_points: Any | None,
    runtime_context_masks: Any | None,
    window_count: int,
    window_width: int,
    prediction_mode: Literal["absolute", "residual"],
    residual_scale: float,
    residual_smoothing_alpha: float,
    runtime_decoder_indices: RuntimeDecoderFeatureIndices | None,
    surrogate_indices: SurrogateFeatureIndices,
    target_names: Sequence[str],
) -> tuple[Any, Any, Any]:
    if prediction_mode == "residual":
        raw_prediction = _smooth_temporal_residuals(
            torch,
            raw_prediction,
            window_count=window_count,
            window_width=window_width,
            alpha=residual_smoothing_alpha,
        )
    prediction = _compose_torch_latents(
        torch,
        raw_prediction,
        heuristics,
        prediction_mode=prediction_mode,
        residual_scale=residual_scale,
        target_names=target_names,
    )
    prediction_points, prediction_point_masks = _decoded_prediction_points_and_masks(
        torch,
        prediction,
        raw_features,
        runtime_context_points=runtime_context_points,
        runtime_context_masks=runtime_context_masks,
        runtime_decoder_indices=runtime_decoder_indices,
        surrogate_indices=surrogate_indices,
        target_names=target_names,
    )
    target_points, point_masks = _decoded_target_points_and_masks(
        torch,
        targets,
        target_masks,
        raw_features,
        decoded_reference_points=decoded_reference_points,
        decoded_reference_masks=decoded_reference_masks,
        surrogate_indices=surrogate_indices,
        target_names=target_names,
    )
    point_masks = point_masks * prediction_point_masks
    return (
        prediction_points.reshape(window_count, window_width, -1, 3),
        target_points.reshape(window_count, window_width, -1, 3),
        point_masks.reshape(window_count, window_width, -1),
    )


def _smooth_temporal_residuals(
    torch: Any,
    raw_prediction: Any,
    *,
    window_count: int,
    window_width: int,
    alpha: float,
) -> Any:
    if window_width <= 1 or alpha >= 1.0:
        return raw_prediction
    clamped_alpha = max(0.0, alpha)
    windowed = raw_prediction.reshape(window_count, window_width, -1)
    previous = windowed[:, 0]
    smoothed = [previous]
    for frame_index in range(1, window_width):
        previous = previous + clamped_alpha * (windowed[:, frame_index] - previous)
        smoothed.append(previous)
    return torch.stack(smoothed, dim=1).reshape_as(raw_prediction)


def _decoded_prediction_points_and_masks(
    torch: Any,
    prediction: Any,
    raw_features: Any,
    *,
    runtime_context_points: Any | None,
    runtime_context_masks: Any | None,
    runtime_decoder_indices: RuntimeDecoderFeatureIndices | None,
    surrogate_indices: SurrogateFeatureIndices,
    target_names: Sequence[str],
) -> tuple[Any, Any]:
    if (
        runtime_decoder_indices is not None
        and runtime_context_points is not None
        and runtime_context_masks is not None
    ):
        return decode_upper_body_runtime_points(
            torch,
            prediction,
            raw_features,
            runtime_context_points,
            runtime_context_masks,
            feature_indices=runtime_decoder_indices,
            target_names=target_names,
        )
    points = _upper_body_surrogate_points(
        torch,
        prediction,
        raw_features,
        surrogate_indices=surrogate_indices,
        target_names=target_names,
    )
    return points, torch.ones(
        points.shape[:2],
        dtype=points.dtype,
        device=points.device,
    )


def _runtime_context_joint_point(
    runtime_context_points: Any | None,
    runtime_context_masks: Any | None,
    joint_name: str,
    *,
    fallback: Any,
) -> Any:
    if runtime_context_points is None or runtime_context_masks is None:
        return fallback
    joint_index = RUNTIME_DECODER_CONTEXT_JOINTS.index(joint_name)
    mask = runtime_context_masks[:, joint_index].to(dtype=fallback.dtype)[:, None]
    return fallback * (1.0 - mask) + runtime_context_points[:, joint_index] * mask


def _decoded_target_points_and_masks(
    torch: Any,
    target: Any,
    target_mask: Any,
    raw_features: Any,
    *,
    decoded_reference_points: Any | None,
    decoded_reference_masks: Any | None,
    surrogate_indices: SurrogateFeatureIndices,
    target_names: Sequence[str],
) -> tuple[Any, Any]:
    if decoded_reference_points is not None and decoded_reference_masks is not None:
        return (
            decoded_reference_points,
            decoded_reference_masks * _decoded_point_masks(torch, target_mask, target_names),
        )
    return (
        _upper_body_surrogate_points(
            torch,
            target,
            raw_features,
            surrogate_indices=surrogate_indices,
            target_names=target_names,
        ),
        _decoded_point_masks(torch, target_mask, target_names),
    )


def _decoded_point_masks(torch: Any, target_mask: Any, target_names: Sequence[str]) -> Any:
    torso_mask = _decoded_torso_mask(target_mask, target_names)
    return torch.stack(
        (
            torso_mask,
            torso_mask,
            _decoded_joint_mask(target_mask, target_names, "left", "shoulder"),
            _decoded_joint_mask(target_mask, target_names, "right", "shoulder"),
            _decoded_joint_mask(target_mask, target_names, "left", "elbow"),
            _decoded_joint_mask(target_mask, target_names, "right", "elbow"),
        ),
        dim=1,
    )


def _pole_vectors(torch: Any, latents: Any, target_names: Sequence[str]) -> Any:
    poles = []
    for side in ("left", "right"):
        pole = latents[
            :,
            [
                _target_index(target_names, f"{side}_elbow_pole_x"),
                _target_index(target_names, f"{side}_elbow_pole_y"),
                _target_index(target_names, f"{side}_elbow_pole_z"),
            ],
        ]
        poles.append(_torch_normalize(torch, pole, fallback=(0.0, -1.0, 0.0)))
    return torch.stack(poles, dim=1)


def _pole_masks(target_mask: Any, target_names: Sequence[str]) -> Any:
    return target_mask[
        :,
        [
            _target_index(target_names, "left_elbow_pole_x"),
            _target_index(target_names, "right_elbow_pole_x"),
        ],
    ]


def _temporal_difference(points: Any, *, order: Literal[1, 2, 3]) -> Any:
    if order == 1:
        return points[:, 1] - points[:, 0]
    if order == 2:
        return points[:, 2] - 2.0 * points[:, 1] + points[:, 0]
    return points[:, 3] - 3.0 * points[:, 2] + 3.0 * points[:, 1] - points[:, 0]


def _masked_temporal_point_mse(prediction: Any, target: Any, mask: Any) -> Any:
    distance_squared = ((prediction - target) ** 2).sum(dim=2)
    return (distance_squared * mask).sum() / mask.sum().clamp_min(1.0)


def _masked_temporal_excess_motion_mse(
    torch: Any,
    prediction: Any,
    target: Any,
    mask: Any,
    *,
    excess_ratio: float,
    excess_margin: float,
) -> Any:
    prediction_magnitude = torch.linalg.vector_norm(prediction, dim=2)
    target_magnitude = torch.linalg.vector_norm(target, dim=2)
    allowed_magnitude = target_magnitude * excess_ratio + excess_margin
    excess = torch.relu(prediction_magnitude - allowed_magnitude)
    return (excess * excess * mask).sum() / mask.sum().clamp_min(1.0)


def _surrogate_validation_metrics(
    torch: Any,
    prediction: Any,
    target: Any,
    target_mask: Any,
    raw_features: Any,
    *,
    decoded_reference_points: Any | None,
    decoded_reference_masks: Any | None,
    runtime_context_points: Any | None,
    runtime_context_masks: Any | None,
    runtime_decoder_indices: RuntimeDecoderFeatureIndices | None,
    feature_names: Sequence[str],
    target_names: Sequence[str],
    temporal_pairs: Any,
    temporal_triples: Any,
    temporal_quads: Any,
    temporal_excess_ratio: float,
    temporal_excess_margin: float,
    reach_ratio_limit: float,
) -> dict[str, float]:
    surrogate_indices = _surrogate_feature_indices(feature_names)
    if surrogate_indices is None:
        return {}
    with torch.no_grad():
        prediction_points, prediction_point_masks = _decoded_prediction_points_and_masks(
            torch,
            prediction,
            raw_features,
            runtime_context_points=runtime_context_points,
            runtime_context_masks=runtime_context_masks,
            runtime_decoder_indices=runtime_decoder_indices,
            surrogate_indices=surrogate_indices,
            target_names=target_names,
        )
        target_points, point_masks = _decoded_target_points_and_masks(
            torch,
            target,
            target_mask,
            raw_features,
            decoded_reference_points=decoded_reference_points,
            decoded_reference_masks=decoded_reference_masks,
            surrogate_indices=surrogate_indices,
            target_names=target_names,
        )
        point_masks = point_masks * prediction_point_masks
        terms = _decoded_pose_terms(
            torch,
            prediction,
            target,
            target_mask,
            raw_features,
            decoded_reference_points=decoded_reference_points,
            decoded_reference_masks=decoded_reference_masks,
            runtime_context_points=runtime_context_points,
            runtime_context_masks=runtime_context_masks,
            runtime_decoder_indices=runtime_decoder_indices,
            surrogate_indices=surrogate_indices,
            target_names=target_names,
            reach_ratio_limit=reach_ratio_limit,
        )
        position_mse = float(
            _masked_temporal_point_mse(prediction_points, target_points, point_masks).detach().cpu()
        )
        velocity_mse = _decoded_temporal_validation_metric(
            torch,
            prediction_points,
            target_points,
            point_masks,
            temporal_pairs,
            order=1,
        )
        acceleration_mse = _decoded_temporal_validation_metric(
            torch,
            prediction_points,
            target_points,
            point_masks,
            temporal_triples,
            order=2,
        )
        jerk_mse = _decoded_temporal_validation_metric(
            torch,
            prediction_points,
            target_points,
            point_masks,
            temporal_quads,
            order=3,
        )
        prediction_poles = _pole_vectors(torch, prediction, target_names)
        target_poles = _pole_vectors(torch, target, target_names)
        pole_masks = _pole_masks(target_mask, target_names)
        pole_velocity_mse = _decoded_temporal_validation_metric(
            torch,
            prediction_poles,
            target_poles,
            pole_masks,
            temporal_pairs,
            order=1,
        )
        pole_jerk_mse = _decoded_temporal_validation_metric(
            torch,
            prediction_poles,
            target_poles,
            pole_masks,
            temporal_quads,
            order=3,
        )
        pole_excess_jerk_mse = _temporal_excess_validation_metric(
            torch,
            prediction_poles,
            target_poles,
            pole_masks,
            temporal_quads,
            order=3,
            excess_ratio=temporal_excess_ratio,
            excess_margin=temporal_excess_margin,
        )
    return {
        "validation_surrogate_position_mse": position_mse,
        "validation_surrogate_velocity_mse": velocity_mse,
        "validation_decoded_velocity_mse": velocity_mse,
        "validation_decoded_acceleration_mse": acceleration_mse,
        "validation_decoded_jerk_mse": jerk_mse,
        "validation_decoded_position_mse": float(terms["position_mse"].detach().cpu()),
        "validation_decoded_torso_mse": float(terms["torso_mse"].detach().cpu()),
        "validation_decoded_shoulder_mse": float(terms["shoulder_mse"].detach().cpu()),
        "validation_decoded_elbow_mse": float(terms["elbow_mse"].detach().cpu()),
        "validation_decoded_arm_direction_mse": float(terms["arm_direction_mse"].detach().cpu()),
        "validation_decoded_stretch_mse": float(terms["stretch_mse"].detach().cpu()),
        "validation_decoded_reach_hinge_mse": float(terms["reach_hinge_mse"].detach().cpu()),
        "validation_pole_velocity_mse": pole_velocity_mse,
        "validation_pole_jerk_mse": pole_jerk_mse,
        "validation_pole_excess_jerk_mse": pole_excess_jerk_mse,
    }


def _decoded_temporal_validation_metric(
    torch: Any,
    prediction_points: Any,
    target_points: Any,
    point_masks: Any,
    windows: Any,
    *,
    order: Literal[1, 2, 3],
) -> float:
    if windows.numel() == 0:
        return 0.0
    del torch
    prediction_delta = _temporal_difference(prediction_points[windows], order=order)
    target_delta = _temporal_difference(target_points[windows], order=order)
    temporal_mask = point_masks[windows].prod(dim=1)
    return float(
        _masked_temporal_point_mse(prediction_delta, target_delta, temporal_mask).detach().cpu()
    )


def _temporal_excess_validation_metric(
    torch: Any,
    prediction_points: Any,
    target_points: Any,
    point_masks: Any,
    windows: Any,
    *,
    order: Literal[1, 2, 3],
    excess_ratio: float,
    excess_margin: float,
) -> float:
    if windows.numel() == 0:
        return 0.0
    prediction_delta = _temporal_difference(prediction_points[windows], order=order)
    target_delta = _temporal_difference(target_points[windows], order=order)
    temporal_mask = point_masks[windows].prod(dim=1)
    return float(
        _masked_temporal_excess_motion_mse(
            torch,
            prediction_delta,
            target_delta,
            temporal_mask,
            excess_ratio=excess_ratio,
            excess_margin=excess_margin,
        )
        .detach()
        .cpu()
    )


def _upper_body_surrogate_points(
    torch: Any,
    latents: Any,
    raw_features: Any,
    *,
    surrogate_indices: SurrogateFeatureIndices,
    target_names: Sequence[str],
) -> Any:
    zeros = torch.zeros_like(latents[:, 0])
    head_y = raw_features[:, surrogate_indices.head_y]
    human_height = raw_features[:, surrogate_indices.human_height].clamp_min(0.1)
    head_position = torch.stack((zeros, head_y, zeros), dim=1)
    chest_position = _surrogate_relative_point(
        torch,
        latents,
        target_names=target_names,
        joint_name="chest",
        head_position=head_position,
        human_height=human_height,
    )
    neck_position = _surrogate_relative_point(
        torch,
        latents,
        target_names=target_names,
        joint_name="neck",
        head_position=head_position,
        human_height=human_height,
    )
    upper_arm = raw_features[:, surrogate_indices.upper_arm]
    lower_arm = raw_features[:, surrogate_indices.lower_arm]
    left_shoulder = _surrogate_relative_point(
        torch,
        latents,
        target_names=target_names,
        joint_name="left_shoulder",
        head_position=head_position,
        human_height=human_height,
    )
    right_shoulder = _surrogate_relative_point(
        torch,
        latents,
        target_names=target_names,
        joint_name="right_shoulder",
        head_position=head_position,
        human_height=human_height,
    )
    left_hand = _feature_position(raw_features, surrogate_indices.left_hand)
    right_hand = _feature_position(raw_features, surrogate_indices.right_hand)
    left_elbow = _surrogate_elbow(
        torch,
        latents,
        target_names=target_names,
        side="left",
        shoulder=left_shoulder,
        hand=left_hand,
        upper_arm=upper_arm,
        lower_arm=lower_arm,
    )
    right_elbow = _surrogate_elbow(
        torch,
        latents,
        target_names=target_names,
        side="right",
        shoulder=right_shoulder,
        hand=right_hand,
        upper_arm=upper_arm,
        lower_arm=lower_arm,
    )
    return torch.stack(
        (
            chest_position,
            neck_position,
            left_shoulder,
            right_shoulder,
            left_elbow,
            right_elbow,
        ),
        dim=1,
    )


def _surrogate_relative_point(
    torch: Any,
    latents: Any,
    *,
    target_names: Sequence[str],
    joint_name: str,
    head_position: Any,
    human_height: Any,
) -> Any:
    rel = torch.stack(
        (
            latents[:, _target_index(target_names, f"{joint_name}_rel_x")],
            latents[:, _target_index(target_names, f"{joint_name}_rel_y")],
            latents[:, _target_index(target_names, f"{joint_name}_rel_z")],
        ),
        dim=1,
    )
    return head_position + rel * human_height[:, None]


def _surrogate_elbow(
    torch: Any,
    latents: Any,
    *,
    target_names: Sequence[str],
    side: Literal["left", "right"],
    shoulder: Any,
    hand: Any,
    upper_arm: Any,
    lower_arm: Any,
) -> Any:
    bend_hint = torch.stack(
        (
            latents[:, _target_index(target_names, f"{side}_elbow_pole_x")],
            latents[:, _target_index(target_names, f"{side}_elbow_pole_y")],
            latents[:, _target_index(target_names, f"{side}_elbow_pole_z")],
        ),
        dim=1,
    )
    return _torch_two_bone_midpoint(
        torch,
        root=shoulder,
        tip=hand,
        root_to_mid_length=upper_arm,
        mid_to_tip_length=lower_arm,
        bend_hint=bend_hint,
    )


def _torch_two_bone_midpoint(
    torch: Any,
    *,
    root: Any,
    tip: Any,
    root_to_mid_length: Any,
    mid_to_tip_length: Any,
    bend_hint: Any,
) -> Any:
    root_to_tip = tip - root
    root_tip_distance = torch.linalg.vector_norm(root_to_tip, dim=1).clamp_min(1e-4)
    max_reach = (root_to_mid_length + mid_to_tip_length - 1e-4).clamp_min(1e-4)
    min_reach = torch.minimum(
        torch.abs(root_to_mid_length - mid_to_tip_length) + 1e-4,
        max_reach - 1e-6,
    )
    clamped_distance = torch.minimum(torch.maximum(root_tip_distance, min_reach), max_reach)
    direction = _torch_normalize(torch, root_to_tip, fallback=(0.0, 1.0, 0.0))
    along = (
        root_to_mid_length * root_to_mid_length
        - mid_to_tip_length * mid_to_tip_length
        + clamped_distance * clamped_distance
    ) / (2.0 * clamped_distance)
    bend_height_squared = root_to_mid_length * root_to_mid_length - along * along
    bend_height = torch.sqrt(bend_height_squared.clamp_min(0.0))
    bend_direction = _torch_normalize(
        torch,
        _torch_project_on_plane(bend_hint, direction),
        fallback=(0.0, -1.0, 0.0),
    )
    return root + direction * along[:, None] + bend_direction * bend_height[:, None]


def _torch_project_on_plane(vector: Any, normal: Any) -> Any:
    return vector - normal * (vector * normal).sum(dim=1, keepdim=True)


def _torch_normalize(
    torch: Any,
    vector: Any,
    *,
    fallback: tuple[float, float, float],
) -> Any:
    magnitude = torch.linalg.vector_norm(vector, dim=1, keepdim=True)
    fallback_tensor = torch.tensor(fallback, dtype=vector.dtype, device=vector.device)
    normalized = vector / magnitude.clamp_min(1e-6)
    return torch.where(magnitude > 1e-6, normalized, fallback_tensor[None, :])


def _feature_position(raw_features: Any, indices: tuple[int, int, int]) -> Any:
    return raw_features[:, [indices[0], indices[1], indices[2]]]


def _surrogate_feature_indices(
    feature_names: Sequence[str],
) -> SurrogateFeatureIndices | None:
    center_prefix = "t+00" if "t+00_head_position_y" in feature_names else "t+000"
    required = (
        f"{center_prefix}_head_position_y",
        f"{center_prefix}_left_hand_position_x",
        f"{center_prefix}_left_hand_position_y",
        f"{center_prefix}_left_hand_position_z",
        f"{center_prefix}_right_hand_position_x",
        f"{center_prefix}_right_hand_position_y",
        f"{center_prefix}_right_hand_position_z",
        "calibration_human_height_m",
        "calibration_shoulder_width_m",
        "calibration_upper_arm_m",
        "calibration_lower_arm_m",
        "calibration_arm_length_m",
    )
    if not all(name in feature_names for name in required):
        return None
    return SurrogateFeatureIndices(
        head_y=feature_names.index(f"{center_prefix}_head_position_y"),
        left_hand=(
            feature_names.index(f"{center_prefix}_left_hand_position_x"),
            feature_names.index(f"{center_prefix}_left_hand_position_y"),
            feature_names.index(f"{center_prefix}_left_hand_position_z"),
        ),
        right_hand=(
            feature_names.index(f"{center_prefix}_right_hand_position_x"),
            feature_names.index(f"{center_prefix}_right_hand_position_y"),
            feature_names.index(f"{center_prefix}_right_hand_position_z"),
        ),
        human_height=feature_names.index("calibration_human_height_m"),
        shoulder_width=feature_names.index("calibration_shoulder_width_m"),
        upper_arm=feature_names.index("calibration_upper_arm_m"),
        lower_arm=feature_names.index("calibration_lower_arm_m"),
        arm_length=feature_names.index("calibration_arm_length_m"),
    )


def _temporal_pairs(dataset: LoadedDataset, mask: np.ndarray) -> np.ndarray:
    return _temporal_windows(dataset, mask, width=2)


def _temporal_triples(dataset: LoadedDataset, mask: np.ndarray) -> np.ndarray:
    return _temporal_windows(dataset, mask, width=3)


def _temporal_quads(dataset: LoadedDataset, mask: np.ndarray) -> np.ndarray:
    return _temporal_windows(dataset, mask, width=4)


def _temporal_windows(dataset: LoadedDataset, mask: np.ndarray, *, width: int) -> np.ndarray:
    global_indices = np.flatnonzero(mask)
    local_for_global = {
        int(global_index): index for index, global_index in enumerate(global_indices)
    }
    sorted_indices = sorted(
        global_indices,
        key=lambda index: (
            str(dataset.subjects[index]),
            str(dataset.clips[index]),
            int(dataset.center_row_indices[index]),
        ),
    )
    windows: list[list[int]] = []
    current_key: tuple[str, str] | None = None
    current_window: list[int] = []
    for global_index in sorted_indices:
        key = (str(dataset.subjects[global_index]), str(dataset.clips[global_index]))
        if key != current_key:
            current_key = key
            current_window = []
        current_window.append(int(global_index))
        if len(current_window) > width:
            current_window = current_window[-width:]
        if len(current_window) == width:
            windows.append([local_for_global[index] for index in current_window])
    if not windows:
        return np.empty((0, width), dtype=np.int64)
    return np.asarray(windows, dtype=np.int64)


def _target_index(target_names: Sequence[str], name: str) -> int:
    return target_names.index(name)


def _masked_mse(torch: Any, prediction: Any, target: Any, mask: Any) -> Any:
    del torch
    return (((prediction - target) ** 2) * mask).sum() / mask.sum().clamp_min(1.0)


def _tensor_rows(values: Any, indices: Any, *, device: Any) -> Any:
    row_indices = indices if values.device == indices.device else indices.detach().cpu()
    selected = values[row_indices]
    return selected if selected.device == device else selected.to(device)


def _write_jsonl_row(path: Path, row: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")


def _cuda_memory_metrics(torch: Any, device: Any) -> dict[str, float]:
    if getattr(device, "type", None) != "cuda":
        return {}
    bytes_per_mib = 1024.0 * 1024.0
    return {
        "cuda_memory_allocated_mib": float(torch.cuda.memory_allocated(device) / bytes_per_mib),
        "cuda_memory_reserved_mib": float(torch.cuda.memory_reserved(device) / bytes_per_mib),
        "cuda_max_memory_allocated_mib": float(
            torch.cuda.max_memory_allocated(device) / bytes_per_mib
        ),
    }


def _metric_dict(
    targets: np.ndarray,
    predictions: np.ndarray,
    masks: np.ndarray,
    target_names: Sequence[str],
) -> dict[str, object]:
    residual = np.abs(predictions - targets) * masks
    denominator = np.maximum(masks.sum(axis=0), 1.0)
    per_target_mae = {
        target_name: float(residual[:, index].sum() / denominator[index])
        for index, target_name in enumerate(target_names)
    }
    mse = float((((predictions - targets) ** 2) * masks).sum() / max(1.0, masks.sum()))
    mae = float(residual.sum() / max(1.0, masks.sum()))
    groups: dict[str, float] = {}
    for group_name, group_targets in TARGET_GROUPS.items():
        indices = [target_names.index(name) for name in group_targets if name in target_names]
        if indices:
            group_masks = masks[:, indices]
            group_residual = residual[:, indices]
            groups[f"{group_name}_mae"] = float(group_residual.sum() / max(1.0, group_masks.sum()))
    return {
        "validation_mse": mse,
        "validation_mae": mae,
        "per_target_mae": per_target_mae,
        **groups,
    }


def _normalization_stats(features: np.ndarray) -> dict[str, np.ndarray]:
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    return {"mean": mean.astype(np.float32), "scale": scale.astype(np.float32)}


def _normalize(features: np.ndarray, stats: Mapping[str, np.ndarray]) -> np.ndarray:
    return ((features - stats["mean"]) / stats["scale"]).astype(np.float32)


def _import_torch() -> Any:
    if importlib.util.find_spec("torch") is None:
        raise RuntimeError("torch is not installed; run through `uv run --group ml ...`")
    import torch

    return torch


def _torch_device(torch: Any, *, allow_cpu: bool) -> Any:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if allow_cpu:
        return torch.device("cpu")
    raise RuntimeError(
        "Torch CUDA is unavailable. Install the ml group with the CUDA torch wheel, "
        "or pass --allow-cpu only for local smoke tests."
    )


if __name__ == "__main__":
    raise SystemExit(main())
