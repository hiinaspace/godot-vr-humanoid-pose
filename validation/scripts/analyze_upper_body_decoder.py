"""Analyze upper-body latent decoder fidelity and oracle group ablations."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import sqrt
from pathlib import Path
from typing import Any, cast

from validation.vr_pose_validation.baseline import EstimatorName, create_estimator
from validation.vr_pose_validation.calibration import calibration_from_reference_frames
from validation.vr_pose_validation.cmu_metadata import ActivityClass, EnvironmentSplit
from validation.vr_pose_validation.geometry import (
    Transform,
    Vector3,
    distance,
    lerp,
)
from validation.vr_pose_validation.metrics import infer_reference_floor_height
from validation.vr_pose_validation.parquet_cache import (
    filter_clip_cache_files_by_environment,
    iter_cached_pose_frames,
    list_clip_cache_files,
)
from validation.vr_pose_validation.types import (
    AvatarCalibration,
    GroundingContext,
    HumanoidPoseFrame,
    TrackedPoseSample,
)
from validation.vr_pose_validation.upper_body_ml import (
    DECODED_REFERENCE_JOINTS,
    DEFAULT_CLEAN_ACTIVITY_CLASSES,
    TARGET_GROUPS,
    TARGET_NAMES,
    UpperBodyLatentPrediction,
    UpperBodyLatents,
    _tracked_sample_from_cached_frame,
    apply_upper_body_latents,
    extract_upper_body_latents,
    feature_schema,
    local_position,
    pose_joints,
    read_sparse_window_dataframe,
    sparse_window_features,
    yaw_basis_from_rotation,
)

DEFAULT_CACHE_DIR = Path(".cache/cmu/parquet/canonical_full")
DEFAULT_DATASET_DIR = Path(".cache/ml/upper_body_windows")
DEFAULT_REPORT_PATH = Path("validation/reports/upper_body_decoder_analysis.json")
UPPER_BODY_ERROR_JOINTS = (
    "chest",
    "neck",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
)
ORACLE_VARIANTS: Mapping[str, tuple[str, ...]] = {
    "baseline": (),
    "runtime_reapply_heuristic": (),
    "oracle_torso": ("torso",),
    "oracle_shoulder": ("shoulder",),
    "oracle_elbow": ("elbow",),
    "oracle_shoulder_elbow": ("shoulder", "elbow"),
    "oracle_all": ("torso", "shoulder", "elbow"),
}


@dataclass(slots=True)
class MetricAccumulator:
    values: list[float] = field(default_factory=list)

    def add(self, value: float | None) -> None:
        if value is not None:
            self.values.append(float(value))

    def summary(self) -> dict[str, float | int]:
        if not self.values:
            return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0}
        ordered = sorted(self.values)
        return {
            "count": len(ordered),
            "mean": sum(ordered) / len(ordered),
            "p50": _percentile(ordered, 0.50),
            "p95": _percentile(ordered, 0.95),
        }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--window-size", type=int, default=40)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--max-clips", type=int)
    parser.add_argument("--subjects", nargs="*")
    parser.add_argument(
        "--base-estimator",
        default="basis_research",
        choices=(
            "trivial",
            "planted",
            "basis",
            "basis_knee_hint",
            "basis_footstep",
            "basis_arm_lookup",
            "basis_arm_swivel",
            "basis_shrug",
            "basis_research",
        ),
    )
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
    parser.add_argument(
        "--validation-split-only",
        action="store_true",
        default=True,
        help="Restrict to subject/clip pairs marked validation in the sparse-window dataset.",
    )
    parser.add_argument(
        "--all-splits",
        action="store_false",
        dest="validation_split_only",
        help="Evaluate all filtered clips instead of only the sparse-window validation split.",
    )
    args = parser.parse_args(argv)

    report = analyze_upper_body_decoder(
        cache_dir=args.cache_dir,
        dataset_dir=args.dataset_dir,
        report_path=args.report,
        window_size=max(2, args.window_size),
        stride=max(1, args.stride),
        max_clips=args.max_clips,
        subjects=set(args.subjects or []),
        base_estimator_name=cast(EstimatorName, args.base_estimator),
        environment_splits=cast(set[EnvironmentSplit], set(args.environment_split or [])),
        activity_classes=cast(set[ActivityClass], set(args.activity_class or [])),
        validation_split_only=args.validation_split_only,
    )
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0


def analyze_upper_body_decoder(
    *,
    cache_dir: Path,
    dataset_dir: Path,
    report_path: Path,
    window_size: int = 40,
    stride: int = 4,
    max_clips: int | None = None,
    subjects: set[str] | None = None,
    base_estimator_name: EstimatorName = "basis_research",
    environment_splits: set[EnvironmentSplit] | None = None,
    activity_classes: set[ActivityClass] | None = None,
    validation_split_only: bool = True,
) -> dict[str, object]:
    start = time.perf_counter()
    paths = _filtered_paths(
        cache_dir,
        dataset_dir=dataset_dir,
        subjects=subjects or set(),
        environment_splits=environment_splits or set(),
        activity_classes=activity_classes or set(),
        validation_split_only=validation_split_only,
    )
    if max_clips is not None:
        paths = paths[:max_clips]

    state = _AnalysisState(window_size=window_size)
    feature_names = feature_schema(window_size)
    surrogate_decoder = _SurrogateDecoder(feature_names)
    for path in paths:
        _analyze_clip(
            path,
            state=state,
            feature_names=feature_names,
            surrogate_decoder=surrogate_decoder,
            window_size=window_size,
            stride=stride,
            base_estimator_name=base_estimator_name,
        )

    summary = {
        "cache_dir": cache_dir.as_posix(),
        "dataset_dir": dataset_dir.as_posix(),
        "report": report_path.as_posix(),
        "base_estimator": base_estimator_name,
        "validation_split_only": validation_split_only,
        "environment_splits": sorted(environment_splits or []),
        "activity_classes": sorted(activity_classes or []),
        "window_size": window_size,
        "stride": stride,
        "clips": state.clips,
        "frames": state.frames,
        "total_seconds": time.perf_counter() - start,
        "fidelity": _summarize_nested(state.fidelity),
        "oracle_ablation": _summarize_oracle(state.oracle),
        "reach_buckets": _summarize_reach_buckets(state.reach_buckets),
    }
    report: dict[str, object] = {"summary": summary}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


@dataclass(slots=True)
class _AnalysisState:
    window_size: int
    clips: int = 0
    frames: int = 0
    fidelity: dict[str, dict[str, MetricAccumulator]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(MetricAccumulator))
    )
    oracle: dict[str, dict[str, MetricAccumulator]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(MetricAccumulator))
    )
    reach_buckets: dict[str, dict[str, dict[str, MetricAccumulator]]] = field(
        default_factory=lambda: defaultdict(
            lambda: defaultdict(lambda: defaultdict(MetricAccumulator))
        )
    )


def _analyze_clip(
    path: Path,
    *,
    state: _AnalysisState,
    feature_names: Sequence[str],
    surrogate_decoder: _SurrogateDecoder,
    window_size: int,
    stride: int,
    base_estimator_name: EstimatorName,
) -> None:
    cached_frames = list(iter_cached_pose_frames(path))
    if len(cached_frames) < window_size:
        return
    reference_frames = [frame.joints for frame in cached_frames]
    floor_height = infer_reference_floor_height(reference_frames)
    calibration = calibration_from_reference_frames(reference_frames, floor_height_m=floor_height)
    grounding = GroundingContext(floor_height_m=floor_height)
    samples = [_tracked_sample_from_cached_frame(frame) for frame in cached_frames]
    estimator = create_estimator(base_estimator_name)
    base_poses = [estimator.estimate(sample, calibration, grounding) for sample in samples]
    state.clips += 1

    for center_index in range(window_size - 1, len(cached_frames), stride):
        frame = cached_frames[center_index]
        sample = samples[center_index]
        base_pose = base_poses[center_index]
        reference_points = _reference_points(frame.joints)
        base_points = _pose_points(base_pose)
        target_latents = extract_upper_body_latents(frame.joints, calibration)
        heuristic_latents = extract_upper_body_latents(pose_joints(base_pose), calibration)
        origin = sample.hmd.transform.position
        basis = yaw_basis_from_rotation(sample.hmd.transform.rotation)
        feature_vector = sparse_window_features(
            samples,
            center_index,
            window_size,
            calibration=calibration,
        )
        surrogate_heuristic = surrogate_decoder.decode(feature_vector, heuristic_latents)
        surrogate_target = surrogate_decoder.decode(feature_vector, target_latents)
        base_local = _local_points(base_points, origin=origin, basis=basis)
        reference_local = _local_points(reference_points, origin=origin, basis=basis)

        _add_point_errors(
            state.fidelity,
            "runtime_baseline_to_reference",
            base_points,
            reference_points,
        )
        _add_point_errors(
            state.fidelity,
            "surrogate_heuristic_to_reference",
            surrogate_heuristic,
            reference_local,
        )
        _add_point_errors(
            state.fidelity,
            "surrogate_heuristic_to_runtime",
            surrogate_heuristic,
            base_local,
        )
        _add_point_errors(
            state.fidelity,
            "surrogate_target_to_reference",
            surrogate_target,
            reference_local,
        )

        runtime_reapply = apply_upper_body_latents(
            base_pose,
            _prediction_from_latents(heuristic_latents),
            calibration=calibration,
        )
        _add_point_errors(
            state.fidelity,
            "runtime_reapply_heuristic_to_runtime",
            _pose_points(runtime_reapply),
            base_points,
        )

        for variant_name, groups in ORACLE_VARIANTS.items():
            pose = (
                base_pose
                if variant_name == "baseline"
                else apply_upper_body_latents(
                    base_pose,
                    _oracle_prediction(
                        heuristic_latents=heuristic_latents,
                        target_latents=target_latents,
                        groups=set(groups),
                    ),
                    calibration=calibration,
                )
            )
            pose_points = _pose_points(pose)
            _add_point_errors(state.oracle, variant_name, pose_points, reference_points)
            _add_reach_bucket_errors(
                state.reach_buckets,
                variant_name,
                pose_points,
                reference_points,
                sample=sample,
                calibration=calibration,
            )
        state.frames += 1


@dataclass(slots=True)
class _SurrogateDecoder:
    feature_names: Sequence[str]
    torch: Any = field(default=None, init=False)
    surrogate_indices: Any = field(default=None, init=False)
    upper_body_surrogate_points: Any = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.torch = _import_torch()
        from validation.scripts.train_upper_body_latents import (  # noqa: PLC0415
            _surrogate_feature_indices,
            _upper_body_surrogate_points,
        )

        self.surrogate_indices = _surrogate_feature_indices(self.feature_names)
        self.upper_body_surrogate_points = _upper_body_surrogate_points

    def decode(
        self,
        feature_vector: Sequence[float],
        latents: UpperBodyLatents,
    ) -> dict[str, Vector3]:
        if self.surrogate_indices is None:
            return {}
        latent_row = [[latents.values[name] for name in TARGET_NAMES]]
        with self.torch.no_grad():
            points = self.upper_body_surrogate_points(
                self.torch,
                self.torch.tensor(latent_row, dtype=self.torch.float32),
                self.torch.tensor([list(feature_vector)], dtype=self.torch.float32),
                surrogate_indices=self.surrogate_indices,
                target_names=TARGET_NAMES,
            )[0].cpu()
        return {
            joint_name: _vector3_from_sequence(points[index].tolist())
            for index, joint_name in enumerate(DECODED_REFERENCE_JOINTS)
        }


def _reference_points(joints: Mapping[str, Transform]) -> dict[str, Vector3]:
    points: dict[str, Vector3] = {}
    for joint_name in UPPER_BODY_ERROR_JOINTS:
        transform = joints.get(joint_name)
        if transform is None and joint_name == "neck":
            transform = _synthetic_neck_transform(joints)
        if transform is not None:
            points[joint_name] = transform.position
    return points


def _pose_points(pose: HumanoidPoseFrame) -> dict[str, Vector3]:
    return {
        joint_name: joint.transform.position
        for joint_name, joint in pose.joints.items()
        if joint_name in UPPER_BODY_ERROR_JOINTS
    }


def _local_points(
    points: Mapping[str, Vector3],
    *,
    origin: Vector3,
    basis: Any,
) -> dict[str, Vector3]:
    return {
        joint_name: local_position(position, origin=origin, basis=basis)
        for joint_name, position in points.items()
    }


def _add_point_errors(
    metrics: dict[str, dict[str, MetricAccumulator]],
    variant: str,
    prediction: Mapping[str, Vector3],
    reference: Mapping[str, Vector3],
) -> None:
    joint_errors = []
    for joint_name in UPPER_BODY_ERROR_JOINTS:
        predicted = prediction.get(joint_name)
        target = reference.get(joint_name)
        if predicted is None or target is None:
            continue
        error = distance(predicted, target)
        metrics[variant][joint_name].add(error)
        joint_errors.append(error)
    for group_name, joint_names in (
        ("torso", ("chest", "neck")),
        ("shoulder", ("left_shoulder", "right_shoulder")),
        ("elbow", ("left_elbow", "right_elbow")),
    ):
        group_errors = [
            distance(prediction[joint_name], reference[joint_name])
            for joint_name in joint_names
            if joint_name in prediction and joint_name in reference
        ]
        if group_errors:
            metrics[variant][group_name].add(sum(group_errors) / len(group_errors))
    if joint_errors:
        metrics[variant]["upper_body_subset"].add(sum(joint_errors) / len(joint_errors))


def _add_reach_bucket_errors(
    metrics: dict[str, dict[str, dict[str, MetricAccumulator]]],
    variant: str,
    prediction: Mapping[str, Vector3],
    reference: Mapping[str, Vector3],
    *,
    sample: TrackedPoseSample,
    calibration: AvatarCalibration,
) -> None:
    for side in ("left", "right"):
        shoulder_name = f"{side}_shoulder"
        elbow_name = f"{side}_elbow"
        shoulder = reference.get(shoulder_name)
        elbow = prediction.get(elbow_name)
        reference_elbow = reference.get(elbow_name)
        hand = getattr(sample, f"{side}_hand").transform.position
        if shoulder is None or elbow is None or reference_elbow is None:
            continue
        reach_ratio, radius_ratio = _arm_geometry_ratios(shoulder, hand, calibration)
        bucket = _reach_bucket(reach_ratio, radius_ratio)
        metrics[variant][f"{side}:{bucket}"]["elbow"].add(distance(elbow, reference_elbow))
        predicted_shoulder = prediction.get(shoulder_name)
        if predicted_shoulder is not None:
            metrics[variant][f"{side}:{bucket}"]["shoulder"].add(
                distance(predicted_shoulder, shoulder)
            )


def _arm_geometry_ratios(
    shoulder: Vector3,
    hand: Vector3,
    calibration: AvatarCalibration,
) -> tuple[float, float]:
    upper_arm = _calibrated_arm_length(calibration, "upper_arm", calibration.human_height_m * 0.174)
    lower_arm = _calibrated_arm_length(calibration, "lower_arm", calibration.human_height_m * 0.151)
    arm_length = max(upper_arm + lower_arm, 1e-6)
    reach = distance(shoulder, hand)
    along = (upper_arm * upper_arm + reach * reach - lower_arm * lower_arm) / max(
        2.0 * reach,
        1e-6,
    )
    radius = sqrt(max(0.0, upper_arm * upper_arm - along * along))
    return reach / arm_length, radius / max(upper_arm, 1e-6)


def _reach_bucket(reach_ratio: float, radius_ratio: float) -> str:
    if radius_ratio < 0.15:
        return "near_straight"
    if reach_ratio > 0.95:
        return "high_reach"
    if reach_ratio < 0.55:
        return "low_reach"
    return "mid_reach"


def _oracle_prediction(
    *,
    heuristic_latents: UpperBodyLatents,
    target_latents: UpperBodyLatents,
    groups: set[str],
) -> UpperBodyLatentPrediction:
    values = dict(heuristic_latents.values)
    masks = dict(heuristic_latents.masks)
    for group_name in groups:
        for target_name in TARGET_GROUPS[group_name]:
            values[target_name] = target_latents.values[target_name]
            masks[target_name] = target_latents.masks[target_name]
    return UpperBodyLatentPrediction(values=values, masks=masks)


def _prediction_from_latents(latents: UpperBodyLatents) -> UpperBodyLatentPrediction:
    return UpperBodyLatentPrediction(values=dict(latents.values), masks=dict(latents.masks))


def _synthetic_neck_transform(joints: Mapping[str, Transform]) -> Transform | None:
    head = joints.get("head")
    chest = joints.get("chest")
    if head is None or chest is None:
        return None
    return Transform(position=lerp(chest.position, head.position, 0.7), rotation=head.rotation)


def _calibrated_arm_length(
    calibration: AvatarCalibration,
    key: str,
    default: float,
) -> float:
    for candidate in (key, "humerus" if key == "upper_arm" else "forearm"):
        value = calibration.bone_lengths_m.get(candidate)
        if value is not None and value > 0.0:
            return value
    return default


def _filtered_paths(
    cache_dir: Path,
    *,
    dataset_dir: Path,
    subjects: set[str],
    environment_splits: set[EnvironmentSplit],
    activity_classes: set[ActivityClass],
    validation_split_only: bool,
) -> list[Path]:
    paths = list_clip_cache_files(cache_dir, subjects or None)
    paths = filter_clip_cache_files_by_environment(
        paths,
        environment_splits=environment_splits,
        activity_classes=activity_classes,
    )
    if not validation_split_only:
        return paths
    validation_clips = _dataset_split_clips(dataset_dir, "validation")
    return [
        path
        for path in paths
        if (path.parent.name.split("=", 1)[-1], path.stem) in validation_clips
    ]


def _dataset_split_clips(dataset_dir: Path, split: str) -> set[tuple[str, str]]:
    if dataset_dir.exists():
        frame = read_sparse_window_dataframe(dataset_dir).select("subject", "clip", "split")
        return {
            (str(row["subject"]), str(row["clip"]))
            for row in frame.filter(frame["split"] == split)
            .select("subject", "clip")
            .unique()
            .to_dicts()
        }
    return set()


def _summarize_nested(
    metrics: Mapping[str, Mapping[str, MetricAccumulator]],
) -> dict[str, dict[str, dict[str, float | int]]]:
    return {
        variant: {joint: accumulator.summary() for joint, accumulator in joints.items()}
        for variant, joints in sorted(metrics.items())
    }


def _summarize_oracle(
    metrics: Mapping[str, Mapping[str, MetricAccumulator]],
) -> dict[str, dict[str, dict[str, float | int] | dict[str, float]]]:
    summary = _summarize_nested(metrics)
    baseline = summary.get("baseline", {})
    for variant, joint_metrics in summary.items():
        if variant == "baseline":
            continue
        deltas: dict[str, float] = {}
        for joint_name, metric_summary in joint_metrics.items():
            baseline_mean = float(baseline.get(joint_name, {}).get("mean", 0.0))
            deltas[f"{joint_name}_mean_delta_vs_baseline"] = (
                float(metric_summary["mean"]) - baseline_mean
            )
        joint_metrics["delta_vs_baseline"] = deltas
    return summary


def _summarize_reach_buckets(
    metrics: Mapping[str, Mapping[str, Mapping[str, MetricAccumulator]]],
) -> dict[str, dict[str, dict[str, dict[str, float | int]]]]:
    return {
        variant: {
            bucket: {joint: accumulator.summary() for joint, accumulator in joint_metrics.items()}
            for bucket, joint_metrics in sorted(buckets.items())
        }
        for variant, buckets in sorted(metrics.items())
    }


def _percentile(ordered_values: Sequence[float], quantile: float) -> float:
    if not ordered_values:
        return 0.0
    index = min(len(ordered_values) - 1, max(0, round((len(ordered_values) - 1) * quantile)))
    return float(ordered_values[index])


def _vector3_from_sequence(values: Sequence[float]) -> Vector3:
    return (float(values[0]), float(values[1]), float(values[2]))


def _import_torch() -> Any:
    import torch

    return torch


__all__ = [
    "MetricAccumulator",
    "analyze_upper_body_decoder",
]


if __name__ == "__main__":
    raise SystemExit(main())
