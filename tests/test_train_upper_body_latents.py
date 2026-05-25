from __future__ import annotations

import importlib.util
from collections.abc import Mapping
from typing import cast

import polars as pl
import pytest

from validation.vr_pose_validation.upper_body_ml import TARGET_NAMES


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


def test_residual_mlp_smoke_learns_correction_over_heuristic(tmp_path) -> None:
    pytest.importorskip("numpy")
    pytest.importorskip("torch")

    from validation.scripts.train_upper_body_latents import train_upper_body_latents

    dataset_dir = _write_synthetic_dataset(tmp_path)
    report = train_upper_body_latents(
        dataset_dir=dataset_dir,
        output_dir=tmp_path / "models",
        model_names=("torch_residual_mlp",),
        epochs=80,
        batch_size=16,
        learning_rate=5e-2,
        weight_decay=0.0,
        hidden_dim=16,
        allow_cpu=True,
        seed=1,
        residual_regularization_weight=1e-4,
        residual_gate_search=True,
    )
    leaderboard_rows = cast(list[dict[str, object]], report["leaderboard"])
    leaderboard = {row["model"]: row for row in leaderboard_rows}

    assert leaderboard["torch_residual_mlp"]["prediction_mode"] == "residual"
    assert "residual_gates" in leaderboard["torch_residual_mlp"]
    assert _metric(leaderboard["torch_residual_mlp"], "validation_mae") < _metric(
        leaderboard["heuristic_estimator"],
        "validation_mae",
    )


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
