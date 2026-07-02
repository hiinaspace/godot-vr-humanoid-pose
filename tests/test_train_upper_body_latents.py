from __future__ import annotations

import importlib.util
import json
from collections.abc import Mapping
from typing import cast

import polars as pl
import pytest

from validation.vr_pose_validation.upper_body_ml import (
    DECODED_REFERENCE_JOINTS,
    FK_TARGET_NAMES,
    FK_TARGET_SCHEMA,
    TARGET_NAMES,
    TARGET_SCHEMA,
    feature_schema,
)


def test_sparse_window_layout_infers_tcn_shape() -> None:
    from validation.vr_pose_validation.torch_upper_body_models import infer_sparse_window_layout

    names = feature_schema(3)
    layout = infer_sparse_window_layout(names, input_dim=len(names))

    assert layout.window_size == 3
    assert layout.per_frame_dim == 57
    assert layout.static_dim == 11


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
        tracking_backend="jsonl",
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
    torch = pytest.importorskip("torch")

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
        tracking_backend="jsonl",
    )
    leaderboard_rows = cast(list[dict[str, object]], report["leaderboard"])
    leaderboard = {row["model"]: row for row in leaderboard_rows}

    assert leaderboard["torch_mlp"]["prediction_mode"] == "absolute"
    assert (tmp_path / "models" / "torch_mlp.pt").exists()
    artifact = torch.load(tmp_path / "models" / "torch_mlp.pt", map_location="cpu")
    assert artifact["best_checkpoint_metric"] in (
        "validation_decoded_position_mse",
        "validation_mse",
    )
    assert artifact["best_epoch"] <= 2
    assert leaderboard["torch_mlp"]["best_epoch"] == artifact["best_epoch"]
    history_path = tmp_path / "models" / "torch_mlp_history.jsonl"
    assert leaderboard["torch_mlp"]["history_path"] == history_path.as_posix()
    assert history_path.exists()
    assert len(history_path.read_text(encoding="utf-8").splitlines()) == 3


def test_torch_tcn_saves_absolute_chain_artifact(tmp_path) -> None:
    pytest.importorskip("numpy")
    pytest.importorskip("torch")

    from validation.scripts.train_upper_body_latents import train_upper_body_latents

    dataset_dir = _write_synthetic_dataset(
        tmp_path,
        feature_names=feature_schema(3),
        clip_mod=1,
        row_count=32,
    )
    report = train_upper_body_latents(
        dataset_dir=dataset_dir,
        output_dir=tmp_path / "models",
        model_names=("torch_tcn",),
        epochs=2,
        batch_size=16,
        learning_rate=1e-2,
        weight_decay=0.0,
        hidden_dim=16,
        allow_cpu=True,
        seed=2,
        tracking_backend="jsonl",
    )
    leaderboard_rows = cast(list[dict[str, object]], report["leaderboard"])
    leaderboard = {row["model"]: row for row in leaderboard_rows}

    assert leaderboard["torch_tcn"]["prediction_mode"] == "absolute"
    assert (tmp_path / "models" / "torch_tcn.pt").exists()
    assert (tmp_path / "models" / "torch_tcn_history.jsonl").exists()


def test_torch_gru_saves_absolute_chain_artifact(tmp_path) -> None:
    pytest.importorskip("numpy")
    pytest.importorskip("torch")

    from validation.scripts.train_upper_body_latents import train_upper_body_latents

    dataset_dir = _write_synthetic_dataset(
        tmp_path,
        feature_names=feature_schema(3),
        clip_mod=1,
        row_count=32,
    )
    report = train_upper_body_latents(
        dataset_dir=dataset_dir,
        output_dir=tmp_path / "models",
        model_names=("torch_gru",),
        epochs=2,
        batch_size=16,
        learning_rate=1e-2,
        weight_decay=0.0,
        hidden_dim=16,
        allow_cpu=True,
        seed=6,
        tracking_backend="jsonl",
    )
    leaderboard_rows = cast(list[dict[str, object]], report["leaderboard"])
    leaderboard = {row["model"]: row for row in leaderboard_rows}

    assert leaderboard["torch_gru"]["prediction_mode"] == "absolute"
    assert (tmp_path / "models" / "torch_gru.pt").exists()
    assert (tmp_path / "models" / "torch_gru_history.jsonl").exists()


def test_streaming_torch_training_reads_parquet_batches(tmp_path) -> None:
    pytest.importorskip("numpy")
    pytest.importorskip("pyarrow")
    torch = pytest.importorskip("torch")

    from validation.scripts.train_upper_body_latents import train_upper_body_latents

    dataset_dir = _write_synthetic_dataset(
        tmp_path,
        feature_names=feature_schema(3),
        row_count=48,
    )
    report = train_upper_body_latents(
        dataset_dir=dataset_dir,
        output_dir=tmp_path / "models",
        model_names=("torch_mlp",),
        epochs=2,
        batch_size=12,
        learning_rate=1e-2,
        weight_decay=0.0,
        hidden_dim=16,
        allow_cpu=True,
        seed=4,
        stream_training=True,
        stream_validation_rows=16,
        tracking_backend="jsonl",
    )
    leaderboard_rows = cast(list[dict[str, object]], report["leaderboard"])
    leaderboard = {row["model"]: row for row in leaderboard_rows}
    artifact = torch.load(tmp_path / "models" / "torch_mlp.pt", map_location="cpu")

    assert report["stream_training"] is True
    assert leaderboard["torch_mlp"]["stream_training"] is True
    assert leaderboard["torch_mlp"]["epochs_completed"] == 2
    assert artifact["stream_training"] is True
    assert report["stream_temporal_losses"] is False


def test_stream_temporal_windows_stay_inside_contiguous_clip_runs() -> None:
    pytest.importorskip("numpy")

    from validation.scripts.train_upper_body_latents import _temporal_windows_from_arrays

    windows = _temporal_windows_from_arrays(
        subjects=pl.Series(["00", "00", "00", "00", "00", "00", "01"]).to_numpy(),
        clips=pl.Series(["a", "a", "a", "a", "b", "a", "a"]).to_numpy(),
        center_row_indices=pl.Series([0, 1, 4, 5, 2, 2, 3]).to_numpy(),
        width=2,
    )

    assert windows.tolist() == [[0, 1], [1, 5], [2, 3]]


def test_streaming_torch_training_applies_batch_local_temporal_losses(tmp_path) -> None:
    pytest.importorskip("numpy")
    pytest.importorskip("pyarrow")
    torch = pytest.importorskip("torch")

    from validation.scripts.train_upper_body_latents import train_upper_body_latents

    dataset_dir = _write_synthetic_dataset(
        tmp_path,
        feature_names=feature_schema(3),
        row_count=48,
        clip_mod=1,
        subject_mod=1,
    )
    report = train_upper_body_latents(
        dataset_dir=dataset_dir,
        output_dir=tmp_path / "models_temporal",
        model_names=("torch_mlp",),
        epochs=1,
        batch_size=24,
        learning_rate=1e-2,
        weight_decay=0.0,
        hidden_dim=16,
        allow_cpu=True,
        seed=5,
        stream_training=True,
        stream_validation_rows=16,
        temporal_velocity_weight=0.01,
        temporal_acceleration_weight=0.01,
        temporal_excess_jerk_weight=0.01,
        pole_velocity_weight=0.01,
        pole_excess_jerk_weight=0.01,
        temporal_loss_windows=4,
        tracking_backend="jsonl",
    )
    history_path = tmp_path / "models_temporal" / "torch_mlp_history.jsonl"
    history_rows = [
        json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines()
    ]
    artifact = torch.load(tmp_path / "models_temporal" / "torch_mlp.pt", map_location="cpu")

    assert report["stream_temporal_losses"] is True
    assert history_rows[1]["train_temporal_windows"] > 0
    assert artifact["stream_temporal_losses"] is True


def test_yaw_augmentation_rotates_training_tensors_coherently() -> None:
    torch = pytest.importorskip("torch")

    from validation.scripts.train_upper_body_latents import (
        apply_yaw_augmentation,
        yaw_augmentation_indices_for_schema,
    )

    feature_names = feature_schema(1)
    head_position_z = feature_names.index("t+00_head_position_z")
    head_forward_z = feature_names.index("t+00_head_rotation6d_2")
    head_up_y = feature_names.index("t+00_head_rotation6d_4")
    target_names = list(TARGET_NAMES)
    chest_z = target_names.index("chest_rel_z")
    pole_z = target_names.index("left_elbow_pole_z")
    raw_features = torch.zeros((1, len(feature_names)), dtype=torch.float32)
    raw_features[:, head_position_z] = 1.0
    raw_features[:, head_forward_z] = 1.0
    raw_features[:, head_up_y] = 1.0
    targets = torch.zeros((1, len(target_names)), dtype=torch.float32)
    targets[:, chest_z] = 1.0
    targets[:, pole_z] = 1.0
    decoded_points = torch.zeros((1, len(DECODED_REFERENCE_JOINTS), 3), dtype=torch.float32)
    decoded_points[:, 0, 2] = 1.0
    torch.manual_seed(4)

    augmented = apply_yaw_augmentation(
        torch,
        raw_features=raw_features,
        targets=targets,
        heuristics=targets.clone(),
        decoded_reference_points=decoded_points,
        runtime_context_points=decoded_points.clone(),
        feature_mean=torch.zeros(len(feature_names), dtype=torch.float32),
        feature_scale=torch.ones(len(feature_names), dtype=torch.float32),
        indices=yaw_augmentation_indices_for_schema(feature_names, target_names),
        max_degrees=180.0,
    )

    head_xz = augmented.raw_features[
        0,
        [
            feature_names.index("t+00_head_position_x"),
            feature_names.index("t+00_head_position_z"),
        ],
    ]
    assert torch.linalg.vector_norm(head_xz).item() == pytest.approx(1.0, abs=1e-6)
    assert float(augmented.targets[0, target_names.index("chest_rel_x")]) == pytest.approx(
        float(head_xz[0]),
    )
    assert float(augmented.targets[0, target_names.index("chest_rel_z")]) == pytest.approx(
        float(head_xz[1]),
    )
    assert float(augmented.targets[0, target_names.index("left_elbow_pole_x")]) == pytest.approx(
        float(head_xz[0]),
    )
    assert augmented.decoded_reference_points is not None
    assert float(augmented.decoded_reference_points[0, 0, 0]) == pytest.approx(float(head_xz[0]))
    assert float(augmented.decoded_reference_points[0, 0, 2]) == pytest.approx(float(head_xz[1]))
    assert float(augmented.raw_features[0, head_up_y]) == pytest.approx(1.0)


def test_torch_fk_mode_trains_against_fk_targets(tmp_path) -> None:
    pytest.importorskip("numpy")
    pytest.importorskip("torch")

    from validation.scripts.train_upper_body_latents import train_upper_body_latents

    dataset_dir = _write_synthetic_dataset(
        tmp_path,
        feature_names=feature_schema(3),
        row_count=32,
    )
    report = train_upper_body_latents(
        dataset_dir=dataset_dir,
        output_dir=tmp_path / "models",
        model_names=("torch_mlp",),
        target_mode="fk",
        epochs=2,
        batch_size=16,
        learning_rate=1e-2,
        weight_decay=0.0,
        hidden_dim=16,
        allow_cpu=True,
        seed=3,
        fk_decoded_pose_weight=0.1,
        temporal_velocity_weight=0.01,
        temporal_loss_windows=8,
        tracking_backend="jsonl",
    )
    leaderboard_rows = cast(list[dict[str, object]], report["leaderboard"])
    leaderboard = {row["model"]: row for row in leaderboard_rows}
    decoded_mse = leaderboard["torch_mlp"]["validation_fk_decoded_position_mse"]

    assert leaderboard["torch_mlp"]["decoded_decoder"] == "fk"
    assert isinstance(decoded_mse, int | float)
    assert decoded_mse >= 0.0
    assert "validation_decoded_velocity_mse" in leaderboard["torch_mlp"]
    assert (tmp_path / "models" / "torch_mlp.pt").exists()


def test_mlflow_tracking_backend_requires_importable_mlflow(tmp_path, monkeypatch) -> None:
    from validation.scripts import train_upper_body_latents

    original_find_spec = importlib.util.find_spec

    def fake_find_spec(name: str, package: str | None = None) -> object | None:
        if name == "mlflow":
            return None
        return original_find_spec(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

    with pytest.raises(RuntimeError, match="MLflow tracking was requested"):
        train_upper_body_latents._create_training_tracker(
            "mlflow",
            history_path=tmp_path / "unused.jsonl",
            mlflow_tracking_uri=f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
            mlflow_artifact_root=tmp_path / "mlflow_artifacts",
            mlflow_experiment="test",
            run_name="test-run",
            params={"model_name": "torch_mlp"},
        )


def test_mlflow_tracking_backend_logs_params_and_metrics(tmp_path) -> None:
    pytest.importorskip("mlflow")

    from validation.scripts import train_upper_body_latents

    tracker = train_upper_body_latents._create_training_tracker(
        "mlflow",
        history_path=tmp_path / "unused.jsonl",
        mlflow_tracking_uri=f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
        mlflow_artifact_root=tmp_path / "mlflow_artifacts",
        mlflow_experiment="test",
        run_name="test-run",
        params={"model_name": "torch_mlp", "disabled_target_groups": ["elbow"]},
    )
    tracker.log({"epoch": 1, "validation_mse": 0.25, "note": "ignored"})
    metadata = tracker.metadata()
    tracker.close()

    assert metadata["tracking_backend"] == "mlflow"
    assert isinstance(metadata["mlflow_run_id"], str)


def test_decoded_pose_loss_penalizes_shoulder_and_elbow_geometry() -> None:
    torch = pytest.importorskip("torch")

    from validation.scripts.train_upper_body_latents import (
        _decoded_pose_loss,
        _surrogate_feature_indices,
    )

    feature_names = [
        "t+00_head_position_x",
        "t+00_head_position_y",
        "t+00_head_position_z",
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
        [
            [
                0.0,
                1.60,
                0.0,
                0.42,
                1.20,
                0.12,
                -0.42,
                1.20,
                0.12,
                1.80,
                0.40,
                0.30,
                0.25,
                0.55,
            ]
        ],
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
        tracking_backend="jsonl",
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


def test_load_dataset_can_limit_rows_and_clips(tmp_path) -> None:
    pytest.importorskip("numpy")

    from validation.scripts.train_upper_body_latents import load_dataset

    dataset_dir = _write_synthetic_dataset(tmp_path, row_count=32, clip_mod=4)
    row_limited = load_dataset(dataset_dir, max_rows=5)
    clip_limited = load_dataset(dataset_dir, max_clips=2)

    assert row_limited.features.shape[0] == 5
    assert clip_limited.features.shape[0] == 16
    assert set(clip_limited.clips.tolist()) == {"00_01", "01_01"}


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


def _write_synthetic_dataset(
    tmp_path,
    *,
    row_count: int = 80,
    feature_names: list[str] | None = None,
    clip_mod: int = 8,
    subject_mod: int = 8,
):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    feature_names = feature_names or ["f0", "f1", "f2", "f3"]
    rows = []
    for index in range(row_count):
        features = _synthetic_features(feature_names, index)
        x0, x1, x2, x3 = features[:4]
        base = 0.4 * x0 - 0.2 * x1 + 0.1 * x2 + 0.05 * x3
        row = {
            "subject": f"{index % subject_mod:02d}",
            "clip": f"{index % clip_mod:02d}_01",
            "frame_index": index,
            "timestamp_s": index / 120.0,
            "center_row_index": index,
            "split": "validation" if index % 5 == 0 else "train",
            "features": features,
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
        for target_index, target_name in enumerate(FK_TARGET_NAMES):
            row[target_name] = _synthetic_fk_target_value(target_name, base, target_index)
            row[f"{target_name}_mask"] = 1.0
        for joint_name, position in _synthetic_decoded_reference_points(base).items():
            for component, value in zip(("x", "y", "z"), position, strict=True):
                row[f"reference_{joint_name}_position_{component}"] = value
            row[f"reference_{joint_name}_valid"] = 1.0
        rows.append(row)
    pl.DataFrame(rows).write_parquet(dataset_dir / "windows.parquet")
    (dataset_dir / "schema.json").write_text(
        __import__("json").dumps(
            {
                "feature_names": feature_names,
                "target_names": list(TARGET_NAMES),
                "mask_names": [f"{name}_mask" for name in TARGET_NAMES],
                "target_schema": TARGET_SCHEMA,
                "fk_target_names": list(FK_TARGET_NAMES),
                "fk_mask_names": [f"{name}_mask" for name in FK_TARGET_NAMES],
                "fk_target_schema": FK_TARGET_SCHEMA,
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


def _synthetic_features(feature_names: list[str], index: int) -> list[float]:
    if feature_names == ["f0", "f1", "f2", "f3"]:
        return [
            (index % 11) / 10.0,
            ((index * 3) % 13) / 10.0,
            ((index * 5) % 17) / 10.0,
            1.0 if index % 2 == 0 else -1.0,
        ]
    values: list[float] = []
    for feature_index, name in enumerate(feature_names):
        if name.endswith("_valid"):
            values.append(1.0)
        elif name.endswith("_head_position_x") or name.endswith("_head_position_z"):
            values.append(0.0)
        elif name.endswith("_head_position_y"):
            values.append(1.6 + 0.01 * (index % 3))
        elif name.endswith("_left_hand_position_x"):
            values.append(0.35 + 0.01 * (index % 5))
        elif name.endswith("_right_hand_position_x"):
            values.append(-0.35 - 0.01 * (index % 5))
        elif name.endswith("_left_hand_position_y") or name.endswith("_right_hand_position_y"):
            values.append(1.2 + 0.01 * (index % 7))
        elif name.endswith("_left_hand_position_z") or name.endswith("_right_hand_position_z"):
            values.append(0.15 + 0.01 * (index % 4))
        elif name == "calibration_human_height_m":
            values.append(1.8)
        elif name == "calibration_head_to_neck_m":
            values.append(0.10)
        elif name == "calibration_neck_to_chest_m":
            values.append(0.14)
        elif name == "calibration_chest_to_spine_m":
            values.append(0.20)
        elif name == "calibration_spine_to_hips_m":
            values.append(0.22)
        elif name == "calibration_shoulder_width_m":
            values.append(0.4)
        elif name == "calibration_upper_arm_m":
            values.append(0.31)
        elif name == "calibration_lower_arm_m":
            values.append(0.27)
        elif name == "calibration_arm_length_m":
            values.append(0.58)
        elif name == "calibration_shoulder_width_to_height":
            values.append(0.4 / 1.8)
        elif name == "calibration_arm_length_to_height":
            values.append(0.58 / 1.8)
        else:
            values.append((((index + 1) * (feature_index + 3)) % 17) / 20.0)
    return values


def _synthetic_target_value(target_name: str, base: float, target_index: int) -> float:
    del target_index
    if target_name.endswith("_sin"):
        return 0.4 * base
    if target_name.endswith("_cos"):
        sine = 0.4 * base
        return (1.0 - sine * sine) ** 0.5
    return base


def _synthetic_fk_target_value(target_name: str, base: float, target_index: int) -> float:
    del target_index
    if target_name.startswith("fk_hips_rel_"):
        if target_name.endswith("_x"):
            return 0.02 * base
        if target_name.endswith("_y"):
            return -0.35 + 0.01 * base
        return 0.04 * base
    rotation_component = int(target_name.rsplit("_", maxsplit=1)[1])
    identity_rotation6d = (0.0, 0.0, 1.0, 0.0, 1.0, 0.0)
    return identity_rotation6d[rotation_component] + 0.01 * base


def _synthetic_decoded_reference_points(base: float) -> dict[str, tuple[float, float, float]]:
    del base
    values = {
        "chest": (0.0, 1.32, 0.0),
        "neck": (0.0, 1.488, 0.0),
        "left_shoulder": (0.20, 1.32, 0.0),
        "right_shoulder": (-0.20, 1.32, 0.0),
        "left_elbow": (0.50, 1.32, 0.0),
        "right_elbow": (-0.50, 1.32, 0.0),
    }
    return {name: values[name] for name in DECODED_REFERENCE_JOINTS}
