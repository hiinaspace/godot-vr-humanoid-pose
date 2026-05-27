"""Run subject-fold upper-body latent training and optional downstream validation."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import cast

from validation.scripts.run_learned_upper_body_validation import (
    run_learned_upper_body_validation,
)
from validation.scripts.train_upper_body_latents import (
    DEFAULT_DATASET_DIR,
    DEFAULT_MLFLOW_ARTIFACT_ROOT,
    DEFAULT_MLFLOW_EXPERIMENT,
    DEFAULT_MLFLOW_TRACKING_URI,
    DEFAULT_OUTPUT_DIR,
    TargetMode,
    TorchModelName,
    TrackingBackend,
    load_dataset,
    train_upper_body_latents,
)
from validation.vr_pose_validation.cmu_metadata import ActivityClass, EnvironmentSplit
from validation.vr_pose_validation.upper_body_ml import (
    DEFAULT_CLEAN_ACTIVITY_CLASSES,
    TARGET_GROUPS,
)

DEFAULT_REPORT_NAME = "cross_validation.json"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "cross_validation")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--models",
        nargs="*",
        choices=(
            "constant",
            "torch_linear",
            "torch_mlp",
            "torch_tcn",
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
    parser.add_argument(
        "--target-mode",
        choices=("chain", "fk"),
        default="chain",
        help="Train existing point/pole chain latents or auxiliary rotation-FK targets.",
    )
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument(
        "--tracking-backend",
        choices=("jsonl", "mlflow", "none"),
        default="mlflow",
        help="Per-epoch tracking backend for Torch models trained in each fold.",
    )
    parser.add_argument("--mlflow-tracking-uri", default=DEFAULT_MLFLOW_TRACKING_URI)
    parser.add_argument("--mlflow-artifact-root", type=Path, default=DEFAULT_MLFLOW_ARTIFACT_ROOT)
    parser.add_argument("--mlflow-experiment", default=DEFAULT_MLFLOW_EXPERIMENT)
    parser.add_argument("--run-name", help="Optional run name prefix for tracked fold models.")
    parser.add_argument("--xgboost-estimators", type=int, default=256)
    parser.add_argument("--latent-loss-weight", type=float, default=1.0)
    parser.add_argument("--decoded-pose-weight", type=float, default=0.0)
    parser.add_argument("--decoded-torso-weight", type=float, default=0.0)
    parser.add_argument("--decoded-shoulder-weight", type=float, default=1.0)
    parser.add_argument("--decoded-elbow-weight", type=float, default=1.0)
    parser.add_argument("--decoded-arm-direction-weight", type=float, default=0.25)
    parser.add_argument("--decoded-stretch-weight", type=float, default=0.25)
    parser.add_argument("--decoded-reach-hinge-weight", type=float, default=0.0)
    parser.add_argument("--fk-decoded-pose-weight", type=float, default=0.0)
    parser.add_argument("--reach-ratio-limit", type=float, default=0.97)
    parser.add_argument("--surrogate-position-weight", type=float, default=0.0)
    parser.add_argument("--temporal-velocity-weight", type=float, default=0.0)
    parser.add_argument("--temporal-acceleration-weight", type=float, default=0.0)
    parser.add_argument("--temporal-jerk-weight", type=float, default=0.0)
    parser.add_argument("--temporal-excess-jerk-weight", type=float, default=0.0)
    parser.add_argument("--pole-velocity-weight", type=float, default=0.0)
    parser.add_argument("--pole-excess-jerk-weight", type=float, default=0.0)
    parser.add_argument("--temporal-loss-windows", type=int, default=0)
    parser.add_argument(
        "--disabled-target-groups",
        nargs="*",
        choices=tuple(TARGET_GROUPS),
        default=(),
        help="Target groups to leave out of Torch latent training.",
    )
    parser.add_argument(
        "--validate-cache-dir",
        type=Path,
        help="Run cached CMU downstream metrics for each fold using the validation subjects.",
    )
    parser.add_argument(
        "--validation-model",
        choices=("torch_linear", "torch_mlp", "torch_tcn"),
        default="torch_mlp",
        help="Torch artifact to use for optional downstream cached validation.",
    )
    parser.add_argument("--max-validation-clips", type=int)
    parser.add_argument(
        "--environment-split",
        choices=("flat", "nonflat", "unknown"),
        nargs="*",
        default=("flat",),
    )
    parser.add_argument(
        "--activity-class",
        choices=(
            "idle_standing",
            "sitting",
            "stationary_upper_body",
            "locomotion",
            "dance_exercise",
            "swimming",
            "floor_pose",
            "sports",
            "unknown",
        ),
        nargs="*",
        default=DEFAULT_CLEAN_ACTIVITY_CLASSES,
    )
    args = parser.parse_args(argv)

    report = run_upper_body_latent_cross_validation(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        folds=max(2, args.folds),
        model_names=cast(tuple[TorchModelName, ...], tuple(args.models)),
        epochs=max(1, args.epochs),
        batch_size=max(1, args.batch_size),
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        hidden_dim=max(8, args.hidden_dim),
        seed=args.seed,
        target_mode=cast(TargetMode, args.target_mode),
        allow_cpu=args.allow_cpu,
        tracking_backend=cast(TrackingBackend, args.tracking_backend),
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        mlflow_artifact_root=args.mlflow_artifact_root,
        mlflow_experiment=args.mlflow_experiment,
        run_name=args.run_name,
        xgboost_estimators=max(1, args.xgboost_estimators),
        latent_loss_weight=max(0.0, args.latent_loss_weight),
        decoded_pose_weight=max(0.0, args.decoded_pose_weight),
        decoded_torso_weight=max(0.0, args.decoded_torso_weight),
        decoded_shoulder_weight=max(0.0, args.decoded_shoulder_weight),
        decoded_elbow_weight=max(0.0, args.decoded_elbow_weight),
        decoded_arm_direction_weight=max(0.0, args.decoded_arm_direction_weight),
        decoded_stretch_weight=max(0.0, args.decoded_stretch_weight),
        decoded_reach_hinge_weight=max(0.0, args.decoded_reach_hinge_weight),
        fk_decoded_pose_weight=max(0.0, args.fk_decoded_pose_weight),
        reach_ratio_limit=max(0.0, args.reach_ratio_limit),
        surrogate_position_weight=max(0.0, args.surrogate_position_weight),
        temporal_velocity_weight=max(0.0, args.temporal_velocity_weight),
        temporal_acceleration_weight=max(0.0, args.temporal_acceleration_weight),
        temporal_jerk_weight=max(0.0, args.temporal_jerk_weight),
        temporal_excess_jerk_weight=max(0.0, args.temporal_excess_jerk_weight),
        pole_velocity_weight=max(0.0, args.pole_velocity_weight),
        pole_excess_jerk_weight=max(0.0, args.pole_excess_jerk_weight),
        temporal_loss_windows=max(0, args.temporal_loss_windows),
        disabled_target_groups=set(args.disabled_target_groups),
        validate_cache_dir=args.validate_cache_dir,
        validation_model=args.validation_model,
        max_validation_clips=args.max_validation_clips,
        environment_splits=cast(set[EnvironmentSplit], set(args.environment_split or [])),
        activity_classes=cast(set[ActivityClass], set(args.activity_class or [])),
    )
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0


def run_upper_body_latent_cross_validation(
    *,
    dataset_dir: Path,
    output_dir: Path,
    folds: int,
    model_names: Sequence[TorchModelName],
    epochs: int = 80,
    batch_size: int = 4096,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    hidden_dim: int = 256,
    seed: int = 0,
    target_mode: TargetMode = "chain",
    allow_cpu: bool = False,
    tracking_backend: TrackingBackend = "mlflow",
    mlflow_tracking_uri: str = DEFAULT_MLFLOW_TRACKING_URI,
    mlflow_artifact_root: Path = DEFAULT_MLFLOW_ARTIFACT_ROOT,
    mlflow_experiment: str = DEFAULT_MLFLOW_EXPERIMENT,
    run_name: str | None = None,
    xgboost_estimators: int = 256,
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
    fk_decoded_pose_weight: float = 0.0,
    reach_ratio_limit: float = 0.97,
    surrogate_position_weight: float = 0.0,
    temporal_velocity_weight: float = 0.0,
    temporal_acceleration_weight: float = 0.0,
    temporal_jerk_weight: float = 0.0,
    temporal_excess_jerk_weight: float = 0.0,
    pole_velocity_weight: float = 0.0,
    pole_excess_jerk_weight: float = 0.0,
    temporal_loss_windows: int = 0,
    disabled_target_groups: set[str] | None = None,
    validate_cache_dir: Path | None = None,
    validation_model: str = "torch_mlp",
    max_validation_clips: int | None = None,
    environment_splits: set[EnvironmentSplit] | None = None,
    activity_classes: set[ActivityClass] | None = None,
) -> dict[str, object]:
    start = time.perf_counter()
    dataset = load_dataset(dataset_dir)
    subjects = sorted({str(subject) for subject in dataset.subjects.tolist()})
    if len(subjects) < 2:
        raise ValueError("Cross-validation requires at least two subjects")
    fold_subjects = _fold_subjects(subjects, min(max(2, folds), len(subjects)))

    output_dir.mkdir(parents=True, exist_ok=True)
    fold_reports: list[dict[str, object]] = []
    for fold_index, validation_subjects in enumerate(fold_subjects):
        fold_dir = output_dir / f"fold_{fold_index:02d}"
        train_report = train_upper_body_latents(
            dataset_dir=dataset_dir,
            output_dir=fold_dir,
            model_names=model_names,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            hidden_dim=hidden_dim,
            seed=seed + fold_index,
            target_mode=target_mode,
            allow_cpu=allow_cpu,
            tracking_backend=tracking_backend,
            mlflow_tracking_uri=mlflow_tracking_uri,
            mlflow_artifact_root=mlflow_artifact_root,
            mlflow_experiment=mlflow_experiment,
            run_name=f"{run_name or output_dir.name}:fold_{fold_index:02d}",
            xgboost_estimators=xgboost_estimators,
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
            fk_decoded_pose_weight=fk_decoded_pose_weight,
            reach_ratio_limit=reach_ratio_limit,
            surrogate_position_weight=surrogate_position_weight,
            temporal_velocity_weight=temporal_velocity_weight,
            temporal_acceleration_weight=temporal_acceleration_weight,
            temporal_jerk_weight=temporal_jerk_weight,
            temporal_excess_jerk_weight=temporal_excess_jerk_weight,
            pole_velocity_weight=pole_velocity_weight,
            pole_excess_jerk_weight=pole_excess_jerk_weight,
            temporal_loss_windows=temporal_loss_windows,
            disabled_target_groups=disabled_target_groups,
            validation_subjects=set(validation_subjects),
        )
        fold_report: dict[str, object] = {
            "fold": fold_index,
            "validation_subjects": validation_subjects,
            "training": train_report,
        }
        if validate_cache_dir is not None:
            downstream = _run_fold_downstream_validation(
                cache_dir=validate_cache_dir,
                dataset_dir=dataset_dir,
                fold_dir=fold_dir,
                validation_model=validation_model,
                validation_subjects=set(validation_subjects),
                allow_cpu=allow_cpu,
                max_validation_clips=max_validation_clips,
                environment_splits=environment_splits or set(),
                activity_classes=activity_classes or set(),
            )
            fold_report["downstream_validation"] = downstream
        fold_reports.append(fold_report)

    report: dict[str, object] = {
        "summary": {
            "dataset_dir": dataset_dir.as_posix(),
            "output_dir": output_dir.as_posix(),
            "folds": len(fold_subjects),
            "subjects": subjects,
            "models": list(model_names),
            "total_seconds": time.perf_counter() - start,
            "target_mode": target_mode,
            "tracking_backend": tracking_backend,
            "mlflow_tracking_uri": mlflow_tracking_uri,
            "mlflow_artifact_root": mlflow_artifact_root.as_posix(),
            "mlflow_experiment": mlflow_experiment,
            "run_name": run_name,
            "latent_leaderboard": _aggregate_latent_leaderboard(fold_reports),
            "downstream": _aggregate_downstream(fold_reports),
        },
        "folds": fold_reports,
    }
    (output_dir / DEFAULT_REPORT_NAME).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _run_fold_downstream_validation(
    *,
    cache_dir: Path,
    dataset_dir: Path,
    fold_dir: Path,
    validation_model: str,
    validation_subjects: set[str],
    allow_cpu: bool,
    max_validation_clips: int | None,
    environment_splits: set[EnvironmentSplit],
    activity_classes: set[ActivityClass],
) -> dict[str, object]:
    artifact_path = fold_dir / f"{validation_model}.pt"
    if not artifact_path.exists():
        return {"skipped": f"{artifact_path.as_posix()} does not exist"}
    report_path = fold_dir / f"{validation_model}_cached_validation.json"
    result = run_learned_upper_body_validation(
        cache_dir=cache_dir,
        dataset_dir=dataset_dir,
        artifact_path=artifact_path,
        normalization_path=fold_dir / "normalization.json",
        report_path=report_path,
        window_size=_dataset_window_size(dataset_dir),
        max_clips=max_validation_clips,
        subjects=validation_subjects,
        allow_cpu=allow_cpu,
        compare_baseline=True,
        environment_splits=environment_splits,
        activity_classes=activity_classes,
        validation_split_only=False,
    )
    return cast(dict[str, object], result["summary"])


def _fold_subjects(subjects: Sequence[str], folds: int) -> list[list[str]]:
    fold_subjects: list[list[str]] = [[] for _ in range(folds)]
    for index, subject in enumerate(subjects):
        fold_subjects[index % folds].append(subject)
    return [fold for fold in fold_subjects if fold]


def _dataset_window_size(dataset_dir: Path) -> int:
    manifest_path = dataset_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return int(manifest.get("window_size", 40))
    schema = json.loads((dataset_dir / "schema.json").read_text(encoding="utf-8"))
    feature_names = schema["feature_names"]
    return max(2, int(len(feature_names) / 57))


def _aggregate_latent_leaderboard(
    fold_reports: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, float]]:
    by_model: dict[str, list[Mapping[str, object]]] = {}
    for fold_report in fold_reports:
        training = cast(Mapping[str, object], fold_report["training"])
        for row in cast(Sequence[Mapping[str, object]], training["leaderboard"]):
            by_model.setdefault(str(row["model"]), []).append(row)
    metrics = (
        "validation_mse",
        "validation_mae",
        "torso_mae",
        "shoulder_mae",
        "elbow_mae",
        "train_seconds",
        "inference_seconds_per_sample",
    )
    return {
        model: {
            f"{metric}_mean": _mean(_numeric(row, metric) for row in rows)
            for metric in metrics
            if any(metric in row for row in rows)
        }
        | {
            f"{metric}_std": _std(_numeric(row, metric) for row in rows)
            for metric in metrics
            if any(metric in row for row in rows)
        }
        for model, rows in sorted(by_model.items())
    }


def _aggregate_downstream(
    fold_reports: Sequence[Mapping[str, object]],
) -> dict[str, float]:
    summaries = [
        cast(Mapping[str, object], fold_report["downstream_validation"])
        for fold_report in fold_reports
        if "downstream_validation" in fold_report
        and "learned" in cast(Mapping[str, object], fold_report["downstream_validation"])
    ]
    if not summaries:
        return {}
    learned_rows = [cast(Mapping[str, float], summary["learned"]) for summary in summaries]
    baseline_rows = [
        cast(Mapping[str, float], summary["baseline"])
        for summary in summaries
        if "baseline" in summary
    ]
    aggregate: dict[str, float] = {
        f"learned_{metric}_mean": _mean(row[metric] for row in learned_rows)
        for metric in (
            "mean_upper_body_reference_error_m",
            "mean_shoulder_error_m",
            "mean_elbow_error_m",
            "mean_head_forward_error_deg",
        )
    }
    if baseline_rows:
        aggregate.update(
            {
                f"baseline_{metric}_mean": _mean(row[metric] for row in baseline_rows)
                for metric in (
                    "mean_upper_body_reference_error_m",
                    "mean_shoulder_error_m",
                    "mean_elbow_error_m",
                    "mean_head_forward_error_deg",
                )
            }
        )
    return aggregate


def _numeric(row: Mapping[str, object], metric: str) -> float:
    value = row.get(metric, 0.0)
    return float(value) if isinstance(value, int | float) else 0.0


def _mean(values: Iterable[float]) -> float:
    value_list = list(values)
    return sum(value_list) / len(value_list) if value_list else 0.0


def _std(values: Iterable[float]) -> float:
    value_list = list(values)
    if len(value_list) <= 1:
        return 0.0
    mean = _mean(value_list)
    variance = sum((value - mean) ** 2 for value in value_list) / len(value_list)
    return variance**0.5


if __name__ == "__main__":
    raise SystemExit(main())
