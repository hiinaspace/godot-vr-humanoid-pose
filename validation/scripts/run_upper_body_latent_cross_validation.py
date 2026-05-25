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
    DEFAULT_OUTPUT_DIR,
    TorchModelName,
    load_dataset,
    train_upper_body_latents,
)
from validation.vr_pose_validation.cmu_metadata import ActivityClass, EnvironmentSplit
from validation.vr_pose_validation.upper_body_ml import DEFAULT_CLEAN_ACTIVITY_CLASSES

DEFAULT_REPORT_NAME = "cross_validation.json"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "cross_validation")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--models",
        nargs="*",
        choices=("constant", "torch_linear", "torch_mlp", "xgboost"),
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
    parser.add_argument(
        "--validate-cache-dir",
        type=Path,
        help="Run cached CMU downstream metrics for each fold using the validation subjects.",
    )
    parser.add_argument(
        "--validation-model",
        choices=("torch_linear", "torch_mlp"),
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
        allow_cpu=args.allow_cpu,
        xgboost_estimators=max(1, args.xgboost_estimators),
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
    allow_cpu: bool = False,
    xgboost_estimators: int = 256,
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
            allow_cpu=allow_cpu,
            xgboost_estimators=xgboost_estimators,
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
