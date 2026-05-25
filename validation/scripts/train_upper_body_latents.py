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

from validation.vr_pose_validation.upper_body_ml import (
    DECODED_REFERENCE_JOINTS,
    TARGET_GROUPS,
    TARGET_NAMES,
    read_sparse_window_dataframe,
)

DEFAULT_DATASET_DIR = Path(".cache/ml/upper_body_windows")
DEFAULT_OUTPUT_DIR = Path(".cache/ml/upper_body_models")
TorchModelName = Literal[
    "constant",
    "torch_linear",
    "torch_mlp",
    "torch_residual_linear",
    "torch_residual_mlp",
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
            "torch_residual_linear",
            "torch_residual_mlp",
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
    parser.add_argument("--residual-regularization-weight", type=float, default=1e-3)
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--residual-smoothing-alpha", type=float, default=1.0)
    parser.add_argument("--residual-gate-search", action="store_true")
    parser.add_argument("--latent-loss-weight", type=float, default=1.0)
    parser.add_argument("--decoded-pose-weight", type=float, default=0.0)
    parser.add_argument("--decoded-shoulder-weight", type=float, default=1.0)
    parser.add_argument("--decoded-elbow-weight", type=float, default=1.0)
    parser.add_argument("--decoded-arm-direction-weight", type=float, default=0.25)
    parser.add_argument("--decoded-stretch-weight", type=float, default=0.25)
    parser.add_argument("--surrogate-position-weight", type=float, default=0.0)
    parser.add_argument("--temporal-velocity-weight", type=float, default=0.0)
    parser.add_argument("--temporal-acceleration-weight", type=float, default=0.0)
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
        residual_regularization_weight=max(0.0, args.residual_regularization_weight),
        residual_scale=max(0.0, args.residual_scale),
        residual_smoothing_alpha=min(max(0.0, args.residual_smoothing_alpha), 1.0),
        residual_gate_search=args.residual_gate_search,
        latent_loss_weight=max(0.0, args.latent_loss_weight),
        decoded_pose_weight=max(0.0, args.decoded_pose_weight),
        decoded_shoulder_weight=max(0.0, args.decoded_shoulder_weight),
        decoded_elbow_weight=max(0.0, args.decoded_elbow_weight),
        decoded_arm_direction_weight=max(0.0, args.decoded_arm_direction_weight),
        decoded_stretch_weight=max(0.0, args.decoded_stretch_weight),
        surrogate_position_weight=max(0.0, args.surrogate_position_weight),
        temporal_velocity_weight=max(0.0, args.temporal_velocity_weight),
        temporal_acceleration_weight=max(0.0, args.temporal_acceleration_weight),
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
    decoded_shoulder_weight: float = 1.0,
    decoded_elbow_weight: float = 1.0,
    decoded_arm_direction_weight: float = 0.25,
    decoded_stretch_weight: float = 0.25,
    surrogate_position_weight: float = 0.0,
    temporal_velocity_weight: float = 0.0,
    temporal_acceleration_weight: float = 0.0,
) -> dict[str, object]:
    dataset = load_dataset(dataset_dir)
    torch = _import_torch()
    device = _torch_device(torch, allow_cpu=allow_cpu)
    output_dir.mkdir(parents=True, exist_ok=True)

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
        "residual_regularization_weight": residual_regularization_weight,
        "residual_scale": residual_scale,
        "residual_smoothing_alpha": residual_smoothing_alpha,
        "residual_gate_search": residual_gate_search,
        "latent_loss_weight": latent_loss_weight,
        "decoded_pose_weight": decoded_pose_weight,
        "decoded_shoulder_weight": decoded_shoulder_weight,
        "decoded_elbow_weight": decoded_elbow_weight,
        "decoded_arm_direction_weight": decoded_arm_direction_weight,
        "decoded_stretch_weight": decoded_stretch_weight,
        "surrogate_position_weight": surrogate_position_weight,
        "temporal_velocity_weight": temporal_velocity_weight,
        "temporal_acceleration_weight": temporal_acceleration_weight,
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
                decoded_shoulder_weight=decoded_shoulder_weight,
                decoded_elbow_weight=decoded_elbow_weight,
                decoded_arm_direction_weight=decoded_arm_direction_weight,
                decoded_stretch_weight=decoded_stretch_weight,
                surrogate_position_weight=surrogate_position_weight,
                temporal_velocity_weight=temporal_velocity_weight,
                temporal_acceleration_weight=temporal_acceleration_weight,
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
    decoded_shoulder_weight: float,
    decoded_elbow_weight: float,
    decoded_arm_direction_weight: float,
    decoded_stretch_weight: float,
    surrogate_position_weight: float,
    temporal_velocity_weight: float,
    temporal_acceleration_weight: float,
) -> dict[str, object]:
    torch.manual_seed(seed)
    prediction_mode = "residual" if _is_residual_model(model_name) else "absolute"
    if prediction_mode == "residual" and (heuristics is None or heuristic_masks is None):
        raise RuntimeError("Residual Torch models require heuristic latent columns in the dataset")

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
    raw_x_train = torch.as_tensor(raw_features[train_mask], dtype=torch.float32, device=device)
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
    effective_m_train = m_train * hm_train if prediction_mode == "residual" else m_train
    effective_m_val = m_val * hm_val if prediction_mode == "residual" else m_val
    surrogate_indices = _surrogate_feature_indices(feature_names)
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

    start = time.perf_counter()
    initial_raw = model(x_val)
    initial_prediction = _compose_torch_latents(
        torch,
        initial_raw,
        h_val,
        prediction_mode=prediction_mode,
        residual_scale=residual_scale,
        target_names=target_names,
    )
    initial_loss = float(
        _masked_mse(torch, initial_prediction, y_val, effective_m_val).detach().cpu()
    )
    for _epoch in range(epochs):
        permutation = torch.randperm(len(train_indices), device=device)
        for start_index in range(0, len(train_indices), batch_size):
            batch_indices = permutation[start_index : start_index + batch_size]
            raw_prediction = model(x_train[batch_indices])
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
                loss = loss + effective_decoded_weight * _decoded_pose_loss(
                    torch,
                    prediction,
                    y_train[batch_indices],
                    effective_m_train[batch_indices],
                    raw_x_train[batch_indices],
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
                    surrogate_indices=surrogate_indices,
                    target_names=target_names,
                    shoulder_weight=decoded_shoulder_weight,
                    elbow_weight=decoded_elbow_weight,
                    arm_direction_weight=decoded_arm_direction_weight,
                    stretch_weight=decoded_stretch_weight,
                )
            if temporal_velocity_weight > 0.0 and train_pairs.numel() > 0:
                loss = loss + temporal_velocity_weight * _surrogate_velocity_loss(
                    torch,
                    model,
                    x_train,
                    raw_x_train,
                    y_train,
                    h_train,
                    train_pairs,
                    prediction_mode=prediction_mode,
                    residual_scale=residual_scale,
                    target_names=target_names,
                    feature_names=feature_names,
                    max_edges=batch_size,
                )
            if temporal_acceleration_weight > 0.0 and train_triples.numel() > 0:
                loss = loss + temporal_acceleration_weight * _surrogate_acceleration_loss(
                    torch,
                    model,
                    x_train,
                    raw_x_train,
                    y_train,
                    h_train,
                    train_triples,
                    prediction_mode=prediction_mode,
                    residual_scale=residual_scale,
                    target_names=target_names,
                    feature_names=feature_names,
                    max_triples=batch_size,
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    train_seconds = time.perf_counter() - start

    inference_start = time.perf_counter()
    with torch.no_grad():
        validation_raw_tensor = model(x_val)
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
                prediction_mode=prediction_mode,
                residual_scale=residual_scale,
                target_names=target_names,
                feature_names=feature_names,
                surrogate_position_weight=surrogate_position_weight,
                decoded_pose_weight=decoded_pose_weight,
                decoded_shoulder_weight=decoded_shoulder_weight,
                decoded_elbow_weight=decoded_elbow_weight,
                decoded_arm_direction_weight=decoded_arm_direction_weight,
                decoded_stretch_weight=decoded_stretch_weight,
            )
            if residual_gate_search and prediction_mode == "residual"
            else {name: 1.0 for name in target_names}
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
        feature_names=feature_names,
        target_names=target_names,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": model_name,
            "prediction_mode": prediction_mode,
            "residual_scale": residual_scale,
            "residual_smoothing_alpha": residual_smoothing_alpha,
            "residual_gates": residual_gates,
            "latent_loss_weight": latent_loss_weight,
            "decoded_pose_weight": decoded_pose_weight,
            "decoded_shoulder_weight": decoded_shoulder_weight,
            "decoded_elbow_weight": decoded_elbow_weight,
            "decoded_arm_direction_weight": decoded_arm_direction_weight,
            "decoded_stretch_weight": decoded_stretch_weight,
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
        "prediction_mode": prediction_mode,
        "residual_scale": residual_scale,
        "residual_smoothing_alpha": residual_smoothing_alpha,
        "residual_gates": residual_gates,
        "latent_loss_weight": latent_loss_weight,
        "decoded_pose_weight": decoded_pose_weight,
        "decoded_shoulder_weight": decoded_shoulder_weight,
        "decoded_elbow_weight": decoded_elbow_weight,
        "decoded_arm_direction_weight": decoded_arm_direction_weight,
        "decoded_stretch_weight": decoded_stretch_weight,
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
    if model_name in ("torch_linear", "torch_residual_linear"):
        return torch.nn.Linear(input_dim, output_dim)
    if model_name in ("torch_mlp", "torch_residual_mlp"):
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
    shoulder_width: int
    upper_arm: int
    lower_arm: int
    arm_length: int


def _is_residual_model(model_name: TorchModelName) -> bool:
    return model_name in ("torch_residual_linear", "torch_residual_mlp")


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
        return raw_prediction
    prediction = heuristic + residual_scale * raw_prediction
    return _normalize_torch_latent_pairs(torch, prediction, target_names)


def _normalize_torch_latent_pairs(
    torch: Any,
    latents: Any,
    target_names: Sequence[str],
) -> Any:
    columns = [latents[:, index] for index in range(latents.shape[1])]
    for prefix in ("torso_heading", "left_elbow_swivel", "right_elbow_swivel"):
        sin_index = _target_index(target_names, f"{prefix}_sin")
        cos_index = _target_index(target_names, f"{prefix}_cos")
        magnitude = torch.sqrt(
            columns[sin_index] * columns[sin_index] + columns[cos_index] * columns[cos_index]
        ).clamp_min(1e-6)
        columns[sin_index] = columns[sin_index] / magnitude
        columns[cos_index] = columns[cos_index] / magnitude
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
    prediction_mode: Literal["absolute", "residual"],
    residual_scale: float,
    target_names: Sequence[str],
    feature_names: Sequence[str],
    surrogate_position_weight: float,
    decoded_pose_weight: float,
    decoded_shoulder_weight: float,
    decoded_elbow_weight: float,
    decoded_arm_direction_weight: float,
    decoded_stretch_weight: float,
) -> dict[str, float]:
    if prediction_mode != "residual":
        return {name: 1.0 for name in target_names}

    gates = {name: 1.0 for name in target_names}
    grid = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
    for _pass in range(2):
        for group_targets in TARGET_GROUPS.values():
            best_gate = 1.0
            best_loss = float("inf")
            for candidate in grid:
                candidate_gates = dict(gates)
                for name in group_targets:
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
                    feature_names=feature_names,
                    target_names=target_names,
                    surrogate_position_weight=surrogate_position_weight,
                    decoded_pose_weight=decoded_pose_weight,
                    decoded_shoulder_weight=decoded_shoulder_weight,
                    decoded_elbow_weight=decoded_elbow_weight,
                    decoded_arm_direction_weight=decoded_arm_direction_weight,
                    decoded_stretch_weight=decoded_stretch_weight,
                )
                if loss < best_loss:
                    best_loss = loss
                    best_gate = candidate
            for name in group_targets:
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
    feature_names: Sequence[str],
    target_names: Sequence[str],
    surrogate_position_weight: float,
    decoded_pose_weight: float,
    decoded_shoulder_weight: float,
    decoded_elbow_weight: float,
    decoded_arm_direction_weight: float,
    decoded_stretch_weight: float,
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
            surrogate_indices=surrogate_indices,
            target_names=target_names,
            shoulder_weight=decoded_shoulder_weight,
            elbow_weight=decoded_elbow_weight,
            arm_direction_weight=decoded_arm_direction_weight,
            stretch_weight=decoded_stretch_weight,
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
    shoulder_weight: float,
    elbow_weight: float,
    arm_direction_weight: float,
    stretch_weight: float,
) -> Any:
    terms = _decoded_pose_terms(
        torch,
        prediction,
        target,
        target_mask,
        raw_features,
        decoded_reference_points=decoded_reference_points,
        decoded_reference_masks=decoded_reference_masks,
        surrogate_indices=surrogate_indices,
        target_names=target_names,
    )
    total = torch.zeros((), dtype=prediction.dtype, device=prediction.device)
    total_weight = 0.0
    for term_name, term_weight in (
        ("shoulder_mse", shoulder_weight),
        ("elbow_mse", elbow_weight),
        ("arm_direction_mse", arm_direction_weight),
        ("stretch_mse", stretch_weight),
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
    decoded_reference_points: Any | None = None,
    decoded_reference_masks: Any | None = None,
    surrogate_indices: SurrogateFeatureIndices,
    target_names: Sequence[str],
) -> dict[str, Any]:
    prediction_points = _upper_body_surrogate_points(
        torch,
        prediction,
        raw_features,
        surrogate_indices=surrogate_indices,
        target_names=target_names,
    )
    target_points = (
        decoded_reference_points
        if decoded_reference_points is not None
        else _upper_body_surrogate_points(
            torch,
            target,
            raw_features,
            surrogate_indices=surrogate_indices,
            target_names=target_names,
        )
    )
    left_hand = _feature_position(raw_features, surrogate_indices.left_hand)
    right_hand = _feature_position(raw_features, surrogate_indices.right_hand)
    upper_arm = raw_features[:, surrogate_indices.upper_arm]
    lower_arm = raw_features[:, surrogate_indices.lower_arm]

    left_shoulder_mask = _decoded_joint_mask(target_mask, target_names, "left", "shoulder")
    right_shoulder_mask = _decoded_joint_mask(target_mask, target_names, "right", "shoulder")
    left_elbow_mask = _decoded_joint_mask(target_mask, target_names, "left", "elbow")
    right_elbow_mask = _decoded_joint_mask(target_mask, target_names, "right", "elbow")
    if decoded_reference_masks is not None:
        left_shoulder_mask = left_shoulder_mask * decoded_reference_masks[:, 1]
        right_shoulder_mask = right_shoulder_mask * decoded_reference_masks[:, 2]
        left_elbow_mask = left_elbow_mask * decoded_reference_masks[:, 3]
        right_elbow_mask = right_elbow_mask * decoded_reference_masks[:, 4]

    shoulder_mse = _average_losses(
        torch,
        (
            _masked_point_mse(prediction_points[:, 1], target_points[:, 1], left_shoulder_mask),
            _masked_point_mse(prediction_points[:, 2], target_points[:, 2], right_shoulder_mask),
        ),
    )
    elbow_mse = _average_losses(
        torch,
        (
            _masked_point_mse(prediction_points[:, 3], target_points[:, 3], left_elbow_mask),
            _masked_point_mse(prediction_points[:, 4], target_points[:, 4], right_elbow_mask),
        ),
    )
    arm_direction_mse = _average_losses(
        torch,
        (
            _masked_direction_mse(
                torch,
                prediction_points[:, 3] - prediction_points[:, 1],
                target_points[:, 3] - target_points[:, 1],
                left_elbow_mask,
            ),
            _masked_direction_mse(
                torch,
                left_hand - prediction_points[:, 3],
                left_hand - target_points[:, 3],
                left_elbow_mask,
            ),
            _masked_direction_mse(
                torch,
                prediction_points[:, 4] - prediction_points[:, 2],
                target_points[:, 4] - target_points[:, 2],
                right_elbow_mask,
            ),
            _masked_direction_mse(
                torch,
                right_hand - prediction_points[:, 4],
                right_hand - target_points[:, 4],
                right_elbow_mask,
            ),
        ),
    )
    stretch_mse = _average_losses(
        torch,
        (
            _decoded_stretch_loss(
                torch,
                shoulder=prediction_points[:, 1],
                elbow=prediction_points[:, 3],
                hand=left_hand,
                upper_arm=upper_arm,
                lower_arm=lower_arm,
                mask=left_elbow_mask,
            ),
            _decoded_stretch_loss(
                torch,
                shoulder=prediction_points[:, 2],
                elbow=prediction_points[:, 4],
                hand=right_hand,
                upper_arm=upper_arm,
                lower_arm=lower_arm,
                mask=right_elbow_mask,
            ),
        ),
    )
    return {
        "shoulder_mse": shoulder_mse,
        "elbow_mse": elbow_mse,
        "arm_direction_mse": arm_direction_mse,
        "stretch_mse": stretch_mse,
        "position_mse": _average_losses(torch, (shoulder_mse, elbow_mse)),
    }


def _decoded_joint_mask(
    target_mask: Any,
    target_names: Sequence[str],
    side: Literal["left", "right"],
    joint: Literal["shoulder", "elbow"],
) -> Any:
    if joint == "shoulder":
        columns = (
            f"{side}_shoulder_lateral_delta",
            f"{side}_shoulder_elevation",
            f"{side}_shoulder_protraction",
        )
    else:
        columns = (f"{side}_elbow_swivel_sin", f"{side}_elbow_swivel_cos")
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


def _surrogate_velocity_loss(
    torch: Any,
    model: Any,
    features: Any,
    raw_features: Any,
    targets: Any,
    heuristics: Any,
    pairs: Any,
    *,
    prediction_mode: Literal["absolute", "residual"],
    residual_scale: float,
    target_names: Sequence[str],
    feature_names: Sequence[str],
    max_edges: int,
) -> Any:
    surrogate_indices = _surrogate_feature_indices(feature_names)
    if surrogate_indices is None:
        return torch.zeros((), dtype=features.dtype, device=features.device)
    selected_pairs = pairs[
        torch.randint(pairs.shape[0], (min(max_edges, pairs.shape[0]),), device=features.device)
    ]
    flat_indices = selected_pairs.flatten()
    prediction = _compose_torch_latents(
        torch,
        model(features[flat_indices]),
        heuristics[flat_indices],
        prediction_mode=prediction_mode,
        residual_scale=residual_scale,
        target_names=target_names,
    )
    prediction_points = _upper_body_surrogate_points(
        torch,
        prediction,
        raw_features[flat_indices],
        surrogate_indices=surrogate_indices,
        target_names=target_names,
    ).reshape(selected_pairs.shape[0], 2, -1, 3)
    target_points = _upper_body_surrogate_points(
        torch,
        targets[flat_indices],
        raw_features[flat_indices],
        surrogate_indices=surrogate_indices,
        target_names=target_names,
    ).reshape(selected_pairs.shape[0], 2, -1, 3)
    prediction_delta = prediction_points[:, 1] - prediction_points[:, 0]
    target_delta = target_points[:, 1] - target_points[:, 0]
    return ((prediction_delta - target_delta) ** 2).mean()


def _surrogate_acceleration_loss(
    torch: Any,
    model: Any,
    features: Any,
    raw_features: Any,
    targets: Any,
    heuristics: Any,
    triples: Any,
    *,
    prediction_mode: Literal["absolute", "residual"],
    residual_scale: float,
    target_names: Sequence[str],
    feature_names: Sequence[str],
    max_triples: int,
) -> Any:
    surrogate_indices = _surrogate_feature_indices(feature_names)
    if surrogate_indices is None:
        return torch.zeros((), dtype=features.dtype, device=features.device)
    selected_triples = triples[
        torch.randint(
            triples.shape[0],
            (min(max_triples, triples.shape[0]),),
            device=features.device,
        )
    ]
    flat_indices = selected_triples.flatten()
    prediction = _compose_torch_latents(
        torch,
        model(features[flat_indices]),
        heuristics[flat_indices],
        prediction_mode=prediction_mode,
        residual_scale=residual_scale,
        target_names=target_names,
    )
    prediction_points = _upper_body_surrogate_points(
        torch,
        prediction,
        raw_features[flat_indices],
        surrogate_indices=surrogate_indices,
        target_names=target_names,
    ).reshape(selected_triples.shape[0], 3, -1, 3)
    target_points = _upper_body_surrogate_points(
        torch,
        targets[flat_indices],
        raw_features[flat_indices],
        surrogate_indices=surrogate_indices,
        target_names=target_names,
    ).reshape(selected_triples.shape[0], 3, -1, 3)
    prediction_acceleration = (
        prediction_points[:, 2] - 2.0 * prediction_points[:, 1] + prediction_points[:, 0]
    )
    target_acceleration = target_points[:, 2] - 2.0 * target_points[:, 1] + target_points[:, 0]
    return ((prediction_acceleration - target_acceleration) ** 2).mean()


def _surrogate_validation_metrics(
    torch: Any,
    prediction: Any,
    target: Any,
    target_mask: Any,
    raw_features: Any,
    *,
    decoded_reference_points: Any | None,
    decoded_reference_masks: Any | None,
    feature_names: Sequence[str],
    target_names: Sequence[str],
) -> dict[str, float]:
    surrogate_indices = _surrogate_feature_indices(feature_names)
    if surrogate_indices is None:
        return {}
    with torch.no_grad():
        prediction_points = _upper_body_surrogate_points(
            torch,
            prediction,
            raw_features,
            surrogate_indices=surrogate_indices,
            target_names=target_names,
        )
        target_points = _upper_body_surrogate_points(
            torch,
            target,
            raw_features,
            surrogate_indices=surrogate_indices,
            target_names=target_names,
        )
        terms = _decoded_pose_terms(
            torch,
            prediction,
            target,
            target_mask,
            raw_features,
            decoded_reference_points=decoded_reference_points,
            decoded_reference_masks=decoded_reference_masks,
            surrogate_indices=surrogate_indices,
            target_names=target_names,
        )
        position_mse = float(((prediction_points - target_points) ** 2).mean().detach().cpu())
        if prediction_points.shape[0] > 1:
            prediction_delta = prediction_points[1:] - prediction_points[:-1]
            target_delta = target_points[1:] - target_points[:-1]
            velocity_mse = float(((prediction_delta - target_delta) ** 2).mean().detach().cpu())
        else:
            velocity_mse = 0.0
    return {
        "validation_surrogate_position_mse": position_mse,
        "validation_surrogate_velocity_mse": velocity_mse,
        "validation_decoded_position_mse": float(terms["position_mse"].detach().cpu()),
        "validation_decoded_shoulder_mse": float(terms["shoulder_mse"].detach().cpu()),
        "validation_decoded_elbow_mse": float(terms["elbow_mse"].detach().cpu()),
        "validation_decoded_arm_direction_mse": float(terms["arm_direction_mse"].detach().cpu()),
        "validation_decoded_stretch_mse": float(terms["stretch_mse"].detach().cpu()),
    }


def _upper_body_surrogate_points(
    torch: Any,
    latents: Any,
    raw_features: Any,
    *,
    surrogate_indices: SurrogateFeatureIndices,
    target_names: Sequence[str],
) -> Any:
    zeros = torch.zeros_like(latents[:, 0])
    ones = torch.ones_like(latents[:, 0])
    head_y = raw_features[:, surrogate_indices.head_y]
    torso_x = latents[:, _target_index(target_names, "torso_lean_x")]
    torso_z = latents[:, _target_index(target_names, "torso_lean_z")]
    torso_y = torch.sqrt((1.0 - torso_x * torso_x - torso_z * torso_z).clamp_min(1e-4))
    torso_direction = torch.stack((torso_x, torso_y, torso_z), dim=1)
    head_position = torch.stack((zeros, head_y, zeros), dim=1)
    chest_position = head_position - 0.28 * torso_direction

    heading_sin = latents[:, _target_index(target_names, "torso_heading_sin")]
    heading_cos = latents[:, _target_index(target_names, "torso_heading_cos")]
    forward = torch.stack((heading_sin, zeros, heading_cos), dim=1)
    right = torch.stack((heading_cos, zeros, -heading_sin), dim=1)
    up = torch.stack((zeros, ones, zeros), dim=1)
    shoulder_width = raw_features[:, surrogate_indices.shoulder_width]
    upper_arm = raw_features[:, surrogate_indices.upper_arm]
    lower_arm = raw_features[:, surrogate_indices.lower_arm]
    arm_length = raw_features[:, surrogate_indices.arm_length].clamp_min(0.1)

    left_shoulder = _surrogate_shoulder(
        latents,
        target_names=target_names,
        side="left",
        chest_position=chest_position,
        right=right,
        up=up,
        forward=forward,
        shoulder_width=shoulder_width,
        arm_length=arm_length,
    )
    right_shoulder = _surrogate_shoulder(
        latents,
        target_names=target_names,
        side="right",
        chest_position=chest_position,
        right=right,
        up=up,
        forward=forward,
        shoulder_width=shoulder_width,
        arm_length=arm_length,
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
        (chest_position, left_shoulder, right_shoulder, left_elbow, right_elbow),
        dim=1,
    )


def _surrogate_shoulder(
    latents: Any,
    *,
    target_names: Sequence[str],
    side: Literal["left", "right"],
    chest_position: Any,
    right: Any,
    up: Any,
    forward: Any,
    shoulder_width: Any,
    arm_length: Any,
) -> Any:
    side_sign = 1.0 if side == "left" else -1.0
    offset_x = (
        side_sign * shoulder_width * 0.5
        + latents[:, _target_index(target_names, f"{side}_shoulder_lateral_delta")] * arm_length
    )
    offset_y = latents[:, _target_index(target_names, f"{side}_shoulder_elevation")] * arm_length
    offset_z = latents[:, _target_index(target_names, f"{side}_shoulder_protraction")] * arm_length
    return (
        chest_position
        + right * offset_x[:, None]
        + up * offset_y[:, None]
        + forward * offset_z[:, None]
    )


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
    arm_axis = _torch_normalize(torch, hand - shoulder, fallback=(0.0, 1.0, 0.0))
    world_down = torch.tensor((0.0, -1.0, 0.0), dtype=latents.dtype, device=latents.device)
    base = _torch_project_on_plane(world_down[None, :].expand_as(arm_axis), arm_axis)
    base = _torch_normalize(torch, base, fallback=(0.0, 0.0, 1.0))
    tangent = _torch_normalize(torch, torch.cross(arm_axis, base, dim=1), fallback=(1.0, 0.0, 0.0))
    swivel_sin = latents[:, _target_index(target_names, f"{side}_elbow_swivel_sin")]
    swivel_cos = latents[:, _target_index(target_names, f"{side}_elbow_swivel_cos")]
    bend_hint = _torch_normalize(
        torch,
        tangent * swivel_sin[:, None] + base * swivel_cos[:, None],
        fallback=(0.0, -1.0, 0.0),
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
        shoulder_width=feature_names.index("calibration_shoulder_width_m"),
        upper_arm=feature_names.index("calibration_upper_arm_m"),
        lower_arm=feature_names.index("calibration_lower_arm_m"),
        arm_length=feature_names.index("calibration_arm_length_m"),
    )


def _temporal_pairs(dataset: LoadedDataset, mask: np.ndarray) -> np.ndarray:
    return _temporal_windows(dataset, mask, width=2)


def _temporal_triples(dataset: LoadedDataset, mask: np.ndarray) -> np.ndarray:
    return _temporal_windows(dataset, mask, width=3)


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
