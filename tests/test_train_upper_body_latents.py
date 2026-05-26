from __future__ import annotations

import importlib.util
from collections.abc import Mapping
from typing import cast

import polars as pl
import pytest

from validation.vr_pose_validation.upper_body_ml import TARGET_NAMES, TARGET_SCHEMA


def test_torch_linear_smoke_reduces_synthetic_validation_loss(tmp_path) -> None:
    pytest.importorskip("numpy")
    pytest.importorskip("torch")

    from validation.scripts.train_upper_body_latents import train_upper_body_latents

    dataset_dir = _write_synthetic_dataset(tmp_path)
    report = train_upper_body_latents(
        dataset_dir=dataset_dir,
        output_dir=tmp_path / "models",
        model_names=("constant", "torch_linear"),
        epochs=80,
        batch_size=16,
        learning_rate=5e-2,
        weight_decay=0.0,
        hidden_dim=16,
        allow_cpu=True,
        seed=0,
    )
    leaderboard_rows = cast(list[dict[str, object]], report["leaderboard"])
    leaderboard = {row["model"]: row for row in leaderboard_rows}

    assert _metric(leaderboard["torch_linear"], "final_validation_mse") < _metric(
        leaderboard["torch_linear"],
        "initial_validation_mse",
    )
    assert _metric(leaderboard["torch_linear"], "validation_mae") < _metric(
        leaderboard["constant"],
        "validation_mae",
    )


def test_torch_mlp_saves_absolute_chain_artifact(tmp_path) -> None:
    pytest.importorskip("numpy")
    pytest.importorskip("torch")

    from validation.scripts.train_upper_body_latents import train_upper_body_latents

    dataset_dir = _write_synthetic_dataset(tmp_path)
    report = train_upper_body_latents(
        dataset_dir=dataset_dir,
        output_dir=tmp_path / "models",
        model_names=("torch_mlp",),
        epochs=2,
        batch_size=16,
        learning_rate=1e-2,
        weight_decay=0.0,
        hidden_dim=16,
        allow_cpu=True,
        seed=1,
    )
    leaderboard_rows = cast(list[dict[str, object]], report["leaderboard"])
    leaderboard = {row["model"]: row for row in leaderboard_rows}

    assert leaderboard["torch_mlp"]["prediction_mode"] == "absolute"
    assert (tmp_path / "models" / "torch_mlp.pt").exists()
    history_path = tmp_path / "models" / "torch_mlp_history.jsonl"
    assert leaderboard["torch_mlp"]["history_path"] == history_path.as_posix()
    assert history_path.exists()
    assert len(history_path.read_text(encoding="utf-8").splitlines()) == 3


def test_decoded_pose_loss_penalizes_shoulder_and_elbow_geometry() -> None:
    torch = pytest.importorskip("torch")

    from validation.scripts.train_upper_body_latents import (
        _decoded_pose_loss,
        _surrogate_feature_indices,
    )

    feature_names = [
        "t+00_head_position_y",
        "t+00_left_hand_position_x",
        "t+00_left_hand_position_y",
        "t+00_left_hand_position_z",
        "t+00_right_hand_position_x",
        "t+00_right_hand_position_y",
        "t+00_right_hand_position_z",
        "calibration_human_height_m",
        "calibration_shoulder_width_m",
        "calibration_upper_arm_m",
        "calibration_lower_arm_m",
        "calibration_arm_length_m",
    ]
    raw_features = torch.tensor(
        [[1.60, 0.42, 1.20, 0.12, -0.42, 1.20, 0.12, 1.80, 0.40, 0.30, 0.25, 0.55]],
        dtype=torch.float32,
    )
    target = torch.zeros((1, len(TARGET_NAMES)), dtype=torch.float32)
    target[:, TARGET_NAMES.index("chest_rel_y")] = -0.13
    target[:, TARGET_NAMES.index("neck_rel_y")] = -0.04
    target[:, TARGET_NAMES.index("left_shoulder_rel_x")] = 0.111111
    target[:, TARGET_NAMES.index("left_shoulder_rel_y")] = -0.12
    target[:, TARGET_NAMES.index("right_shoulder_rel_x")] = -0.111111
    target[:, TARGET_NAMES.index("right_shoulder_rel_y")] = -0.12
    target[:, TARGET_NAMES.index("left_elbow_pole_y")] = -1.0
    target[:, TARGET_NAMES.index("right_elbow_pole_y")] = -1.0
    prediction = target.clone()
    perturbed = target.clone()
    perturbed[:, TARGET_NAMES.index("left_shoulder_rel_y")] = 0.2
    masks = torch.ones_like(target)
    surrogate_indices = _surrogate_feature_indices(feature_names)
    assert surrogate_indices is not None

    matching_loss = _decoded_pose_loss(
        torch,
        prediction,
        target,
        masks,
        raw_features,
        decoded_reference_points=None,
        decoded_reference_masks=None,
        surrogate_indices=surrogate_indices,
        target_names=TARGET_NAMES,
        torso_weight=1.0,
        shoulder_weight=1.0,
        elbow_weight=1.0,
        arm_direction_weight=1.0,
        stretch_weight=1.0,
        reach_hinge_weight=0.0,
        reach_ratio_limit=0.97,
    )
    perturbed_loss = _decoded_pose_loss(
        torch,
        perturbed,
        target,
        masks,
        raw_features,
        decoded_reference_points=None,
        decoded_reference_masks=None,
        surrogate_indices=surrogate_indices,
        target_names=TARGET_NAMES,
        torso_weight=1.0,
        shoulder_weight=1.0,
        elbow_weight=1.0,
        arm_direction_weight=1.0,
        stretch_weight=1.0,
        reach_hinge_weight=0.0,
        reach_ratio_limit=0.97,
    )

    assert float(matching_loss) == pytest.approx(0.0, abs=1e-6)
    assert float(perturbed_loss) > 0.001


def test_decoded_reach_hinge_loss_penalizes_near_full_extension() -> None:
    torch = pytest.importorskip("torch")

    from validation.scripts.train_upper_body_latents import _decoded_reach_hinge_loss

    mask = torch.ones((2,), dtype=torch.float32)
    loss = _decoded_reach_hinge_loss(
        torch,
        shoulder=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float32),
        hand=torch.tensor([[0.95, 0.0, 0.0], [1.05, 0.0, 0.0]], dtype=torch.float32),
        arm_length=torch.ones((2,), dtype=torch.float32),
        mask=mask,
        reach_ratio_limit=0.97,
    )

    assert float(loss) == pytest.approx((0.08 * 0.08) / 2.0, rel=1e-5)


def test_pole_temporal_motion_loss_penalizes_extra_pole_velocity() -> None:
    torch = pytest.importorskip("torch")

    from validation.scripts.train_upper_body_latents import _pole_temporal_motion_loss

    class SequenceModel(torch.nn.Module):
        def forward(self, features):
            output = torch.zeros((features.shape[0], len(TARGET_NAMES)), dtype=features.dtype)
            output[:, TARGET_NAMES.index("left_elbow_pole_x")] = features[:, 0]
            output[:, TARGET_NAMES.index("left_elbow_pole_y")] = features[:, 1]
            output[:, TARGET_NAMES.index("right_elbow_pole_x")] = features[:, 0]
            output[:, TARGET_NAMES.index("right_elbow_pole_y")] = features[:, 1]
            return output

    features = torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=torch.float32)
    targets = torch.zeros((2, len(TARGET_NAMES)), dtype=torch.float32)
    targets[:, TARGET_NAMES.index("left_elbow_pole_y")] = -1.0
    targets[:, TARGET_NAMES.index("right_elbow_pole_y")] = -1.0
    masks = torch.ones_like(targets)
    windows = torch.tensor([[0, 1]], dtype=torch.long)

    loss = _pole_temporal_motion_loss(
        torch,
        SequenceModel(),
        features,
        targets=targets,
        target_masks=masks,
        heuristics=torch.zeros_like(targets),
        windows=windows,
        order=1,
        prediction_mode="absolute",
        residual_scale=1.0,
        residual_smoothing_alpha=1.0,
        target_names=TARGET_NAMES,
        residual_gates={name: 1.0 for name in TARGET_NAMES},
        max_windows=1,
    )

    assert float(loss) > 0.0


def test_decoded_temporal_validation_metric_penalizes_jerk() -> None:
    torch = pytest.importorskip("torch")

    from validation.scripts.train_upper_body_latents import _decoded_temporal_validation_metric

    target_points = torch.zeros((4, 5, 3), dtype=torch.float32)
    prediction_points = target_points.clone()
    prediction_points[3, 1, 0] = 1.0
    point_masks = torch.ones((4, 5), dtype=torch.float32)
    windows = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)

    matching = _decoded_temporal_validation_metric(
        torch,
        target_points,
        target_points,
        point_masks,
        windows,
        order=3,
    )
    perturbed = _decoded_temporal_validation_metric(
        torch,
        prediction_points,
        target_points,
        point_masks,
        windows,
        order=3,
    )

    assert matching == pytest.approx(0.0)
    assert perturbed > 0.0


def test_temporal_residual_smoothing_matches_runtime_ema() -> None:
    torch = pytest.importorskip("torch")

    from validation.scripts.train_upper_body_latents import _smooth_temporal_residuals

    raw = torch.tensor(
        [
            [0.0, 0.0],
            [10.0, -10.0],
            [0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    smoothed = _smooth_temporal_residuals(
        torch,
        raw,
        window_count=1,
        window_width=3,
        alpha=0.5,
    )

    assert torch.allclose(
        smoothed,
        torch.tensor(
            [
                [0.0, 0.0],
                [5.0, -5.0],
                [2.5, -2.5],
            ],
            dtype=torch.float32,
        ),
    )


def test_temporal_excess_motion_loss_only_penalizes_excess_magnitude() -> None:
    torch = pytest.importorskip("torch")

    from validation.scripts.train_upper_body_latents import _masked_temporal_excess_motion_mse

    target = torch.tensor([[[1.0, 0.0, 0.0]]], dtype=torch.float32)
    matching_prediction = torch.tensor([[[1.2, 0.0, 0.0]]], dtype=torch.float32)
    excess_prediction = torch.tensor([[[2.0, 0.0, 0.0]]], dtype=torch.float32)
    mask = torch.ones((1, 1), dtype=torch.float32)

    matching = _masked_temporal_excess_motion_mse(
        torch,
        matching_prediction,
        target,
        mask,
        excess_ratio=1.25,
        excess_margin=0.0,
    )
    excess = _masked_temporal_excess_motion_mse(
        torch,
        excess_prediction,
        target,
        mask,
        excess_ratio=1.25,
        excess_margin=0.0,
    )

    assert float(matching) == pytest.approx(0.0)
    assert float(excess) == pytest.approx(0.75 * 0.75)


def test_xgboost_cuda_smoke_when_available(tmp_path) -> None:
    if importlib.util.find_spec("xgboost") is None:
        pytest.skip("xgboost is not installed")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("Torch CUDA is not available")

    from validation.scripts.train_upper_body_latents import train_upper_body_latents

    dataset_dir = _write_synthetic_dataset(tmp_path, row_count=32)
    report = train_upper_body_latents(
        dataset_dir=dataset_dir,
        output_dir=tmp_path / "models",
        model_names=("xgboost",),
        xgboost_estimators=4,
        allow_cpu=False,
    )
    leaderboard = cast(list[dict[str, object]], report["leaderboard"])
    row = next(row for row in leaderboard if row["model"] == "xgboost")

    assert row["device"] == "cuda"


def test_validation_subject_override_prevents_subject_leakage(tmp_path) -> None:
    pytest.importorskip("numpy")

    from validation.scripts.train_upper_body_latents import load_dataset, split_masks

    dataset = load_dataset(_write_synthetic_dataset(tmp_path))
    train_mask, validation_mask = split_masks(dataset, validation_subjects={"00", "01"})
    train_subjects = set(dataset.subjects[train_mask].tolist())
    validation_subjects = set(dataset.subjects[validation_mask].tolist())

    assert validation_subjects == {"00", "01"}
    assert train_subjects.isdisjoint(validation_subjects)


def test_cross_validation_runner_writes_subject_folds(tmp_path) -> None:
    pytest.importorskip("numpy")
    pytest.importorskip("torch")

    from validation.scripts.run_upper_body_latent_cross_validation import (
        run_upper_body_latent_cross_validation,
    )

    report = run_upper_body_latent_cross_validation(
        dataset_dir=_write_synthetic_dataset(tmp_path),
        output_dir=tmp_path / "cross_validation",
        folds=3,
        model_names=("constant",),
        epochs=1,
        batch_size=16,
        allow_cpu=True,
    )

    summary = cast(Mapping[str, object], report["summary"])

    assert summary["folds"] == 3
    assert (tmp_path / "cross_validation" / "cross_validation.json").exists()
    fold_reports = cast(list[dict[str, object]], report["folds"])
    fold_subjects = [
        subject
        for fold_report in fold_reports
        for subject in cast(list[str], fold_report["validation_subjects"])
    ]
    assert sorted(fold_subjects) == [f"{index:02d}" for index in range(8)]


def _write_synthetic_dataset(tmp_path, *, row_count: int = 80):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    feature_names = ["f0", "f1", "f2", "f3"]
    rows = []
    for index in range(row_count):
        x0 = (index % 11) / 10.0
        x1 = ((index * 3) % 13) / 10.0
        x2 = ((index * 5) % 17) / 10.0
        x3 = 1.0 if index % 2 == 0 else -1.0
        base = 0.4 * x0 - 0.2 * x1 + 0.1 * x2 + 0.05 * x3
        row = {
            "subject": f"{index % 8:02d}",
            "clip": f"{index % 8:02d}_01",
            "frame_index": index,
            "timestamp_s": index / 120.0,
            "center_row_index": index,
            "split": "validation" if index % 5 == 0 else "train",
            "features": [x0, x1, x2, x3],
            "environment_split": "flat",
            "activity_class": "stationary_upper_body",
            "motion_description": "synthetic",
        }
        for target_index, target_name in enumerate(TARGET_NAMES):
            target_value = _synthetic_target_value(target_name, base, target_index)
            row[target_name] = target_value
            row[f"{target_name}_mask"] = 1.0
            row[f"heuristic_{target_name}"] = target_value - 0.15
            row[f"heuristic_{target_name}_mask"] = 1.0
        rows.append(row)
    pl.DataFrame(rows).write_parquet(dataset_dir / "windows.parquet")
    (dataset_dir / "schema.json").write_text(
        __import__("json").dumps(
            {
                "feature_names": feature_names,
                "target_names": list(TARGET_NAMES),
                "mask_names": [f"{name}_mask" for name in TARGET_NAMES],
                "target_schema": TARGET_SCHEMA,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return dataset_dir


def _metric(row: dict[str, object], name: str) -> float:
    value = row[name]
    assert isinstance(value, int | float)
    return float(value)


def _synthetic_target_value(target_name: str, base: float, target_index: int) -> float:
    del target_index
    if target_name.endswith("_sin"):
        return 0.4 * base
    if target_name.endswith("_cos"):
        sine = 0.4 * base
        return (1.0 - sine * sine) ** 0.5
    return base
