"""Sparse-window features and upper-body latent targets for ML experiments."""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass, field
from hashlib import sha1
from math import isfinite, sqrt
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, Protocol, cast

import polars as pl

from validation.vr_pose_validation.baseline import EstimatorName, create_estimator
from validation.vr_pose_validation.calibration import calibration_from_reference_frames
from validation.vr_pose_validation.cmu_metadata import ActivityClass, EnvironmentSplit
from validation.vr_pose_validation.geometry import (
    Quaternion,
    Transform,
    Vector3,
    add,
    axis_name_quaternion,
    conjugate_quaternion,
    cross,
    distance,
    dot,
    horizontal,
    lerp,
    look_rotation,
    multiply_quaternion,
    normalize,
    project_on_plane,
    rotate_vector,
    scale,
    subtract,
)
from validation.vr_pose_validation.metrics import infer_reference_floor_height
from validation.vr_pose_validation.parquet_cache import (
    CachedPoseFrame,
    filter_clip_cache_files_by_environment,
    iter_cached_pose_frames,
    list_clip_cache_files,
)
from validation.vr_pose_validation.synthetic_trackers import (
    SyntheticTrackerConfig,
    head_tracker_rotation_from_pose_frame,
)
from validation.vr_pose_validation.torch_upper_body_models import make_torch_upper_body_model
from validation.vr_pose_validation.types import (
    AvatarCalibration,
    GroundingContext,
    HumanoidPoseFrame,
    JointPose,
    PoseEstimator,
    TrackedPoseSample,
    TrackerRole,
    TrackerState,
)

WORLD_UP: Vector3 = (0.0, 1.0, 0.0)
WORLD_DOWN: Vector3 = (0.0, -1.0, 0.0)
WORLD_FORWARD: Vector3 = (0.0, 0.0, 1.0)
TRACKER_ROLES: tuple[TrackerRole, ...] = ("head", "left_hand", "right_hand")
DEFAULT_CLEAN_ACTIVITY_CLASSES: tuple[ActivityClass, ...] = (
    "idle_standing",
    "sitting",
    "stationary_upper_body",
    "locomotion",
)
EXCLUDED_TAGS = frozenset(("data_issue", "parse_issue", "inverted_segment"))
TARGET_SCHEMA = "upper_body_chain_v1"
FK_TARGET_SCHEMA = "upper_body_fk_v1"
type TrackingSpace = Literal["head_yaw", "calibration_yaw"]
TARGET_NAMES = (
    "chest_rel_x",
    "chest_rel_y",
    "chest_rel_z",
    "neck_rel_x",
    "neck_rel_y",
    "neck_rel_z",
    "left_shoulder_rel_x",
    "left_shoulder_rel_y",
    "left_shoulder_rel_z",
    "right_shoulder_rel_x",
    "right_shoulder_rel_y",
    "right_shoulder_rel_z",
    "left_elbow_pole_x",
    "left_elbow_pole_y",
    "left_elbow_pole_z",
    "right_elbow_pole_x",
    "right_elbow_pole_y",
    "right_elbow_pole_z",
)
TARGET_GROUPS = {
    "torso": (
        "chest_rel_x",
        "chest_rel_y",
        "chest_rel_z",
        "neck_rel_x",
        "neck_rel_y",
        "neck_rel_z",
    ),
    "shoulder": (
        "left_shoulder_rel_x",
        "left_shoulder_rel_y",
        "left_shoulder_rel_z",
        "right_shoulder_rel_x",
        "right_shoulder_rel_y",
        "right_shoulder_rel_z",
    ),
    "elbow": (
        "left_elbow_pole_x",
        "left_elbow_pole_y",
        "left_elbow_pole_z",
        "right_elbow_pole_x",
        "right_elbow_pole_y",
        "right_elbow_pole_z",
    ),
}
MASK_NAMES = tuple(f"{name}_mask" for name in TARGET_NAMES)
FK_CHAIN_PARENTS = {
    "spine": "hips",
    "chest": "spine",
    "head": "chest",
    "left_shoulder": "chest",
    "left_elbow": "left_shoulder",
    "left_hand": "left_elbow",
    "right_shoulder": "chest",
    "right_elbow": "right_shoulder",
    "right_hand": "right_elbow",
}
FK_TARGET_NAMES = (
    *(f"fk_hips_rel_{component}" for component in ("x", "y", "z")),
    *(f"fk_hips_rotation6d_{index}" for index in range(6)),
    *(
        f"fk_{joint_name}_local_rotation6d_{index}"
        for joint_name in FK_CHAIN_PARENTS
        for index in range(6)
    ),
)
FK_MASK_NAMES = tuple(f"{name}_mask" for name in FK_TARGET_NAMES)
FK_TARGET_GROUPS = {
    "global": (
        *(f"fk_hips_rel_{component}" for component in ("x", "y", "z")),
        *(f"fk_hips_rotation6d_{index}" for index in range(6)),
    ),
    "torso": tuple(
        f"fk_{joint_name}_local_rotation6d_{index}"
        for joint_name in ("spine", "chest", "head")
        for index in range(6)
    ),
    "arm": tuple(
        f"fk_{joint_name}_local_rotation6d_{index}"
        for joint_name in (
            "left_shoulder",
            "left_elbow",
            "left_hand",
            "right_shoulder",
            "right_elbow",
            "right_hand",
        )
        for index in range(6)
    ),
}
STATIC_CALIBRATION_FEATURE_NAMES = (
    "calibration_human_height_m",
    "calibration_head_to_neck_m",
    "calibration_neck_to_chest_m",
    "calibration_chest_to_spine_m",
    "calibration_spine_to_hips_m",
    "calibration_shoulder_width_m",
    "calibration_upper_arm_m",
    "calibration_lower_arm_m",
    "calibration_arm_length_m",
    "calibration_shoulder_width_to_height",
    "calibration_arm_length_to_height",
)
DECODED_REFERENCE_JOINTS = (
    "chest",
    "neck",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
)
RUNTIME_DECODER_CONTEXT_JOINTS = (
    "head",
    "hips",
    "spine",
    "chest",
    "neck",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_hand",
    "right_hand",
)


@dataclass(frozen=True, slots=True)
class Basis3:
    right: Vector3
    up: Vector3
    forward: Vector3


IDENTITY_BASIS = Basis3(
    right=(1.0, 0.0, 0.0),
    up=(0.0, 1.0, 0.0),
    forward=(0.0, 0.0, 1.0),
)


@dataclass(frozen=True, slots=True)
class TrackingFrame:
    """Clip-local tracking frame used to encode learned upper-body latents."""

    origin: Vector3
    basis: Basis3


@dataclass(frozen=True, slots=True)
class SparseWindowConfig:
    """Configuration for causal sparse-tracker ML windows."""

    window_size: int = 40
    stride: int = 4
    estimator_name: EstimatorName = "basis_research"
    holdout_folds: int = 5
    environment_splits: tuple[EnvironmentSplit, ...] = ("flat",)
    activity_classes: tuple[ActivityClass, ...] = DEFAULT_CLEAN_ACTIVITY_CLASSES
    tracking_space: TrackingSpace = "calibration_yaw"
    workers: int = 1
    shard_rows: int = 0


@dataclass(frozen=True, slots=True)
class SparseWindowExportSummary:
    cache_dir: str
    output_dir: str
    clips_seen: int
    clips_exported: int
    windows: int
    feature_count: int
    target_count: int
    window_size: int
    stride: int
    estimator_name: str
    tracking_space: str
    workers: int
    shards: int
    total_seconds: float
    phase_seconds: dict[str, float]


@dataclass(frozen=True, slots=True)
class UpperBodyLatents:
    values: dict[str, float]
    masks: dict[str, float]


@dataclass(frozen=True, slots=True)
class UpperBodyFkTargets:
    values: dict[str, float]
    masks: dict[str, float]


@dataclass(frozen=True, slots=True)
class UpperBodyLatentPrediction:
    values: Mapping[str, float]
    masks: Mapping[str, float]
    tracking_frame: TrackingFrame | None = None


class LatentUpperBodyProvider(Protocol):
    def predict(
        self,
        sample: TrackedPoseSample,
        calibration: AvatarCalibration,
        grounding: GroundingContext,
    ) -> UpperBodyLatentPrediction:
        """Predict upper-body control latents for the current sparse tracker sample."""


class ResettableLatentUpperBodyProvider(LatentUpperBodyProvider, Protocol):
    def reset(self) -> None:
        """Clear any clip-local provider state."""


@dataclass(frozen=True, slots=True)
class StaticUpperBodyLatentProvider:
    values: Mapping[str, float]
    masks: Mapping[str, float] | None = None

    def predict(
        self,
        sample: TrackedPoseSample,
        calibration: AvatarCalibration,
        grounding: GroundingContext,
    ) -> UpperBodyLatentPrediction:
        del sample, calibration, grounding
        return UpperBodyLatentPrediction(
            values=self.values,
            masks=self.masks or {name: 1.0 for name in self.values},
        )


@dataclass(slots=True)
class TorchUpperBodyLatentProvider:
    """Causal-window provider backed by a Torch latent predictor artifact."""

    artifact_path: Path
    normalization_path: Path
    window_size: int
    allow_cpu: bool = False
    _history: list[TrackedPoseSample] = field(default_factory=list, init=False)
    _torch: Any = field(default=None, init=False)
    _model: Any = field(default=None, init=False)
    _device: Any = field(default=None, init=False)
    _mean: list[float] = field(default_factory=list, init=False)
    _scale: list[float] = field(default_factory=list, init=False)
    _target_names: list[str] = field(default_factory=list, init=False)
    _prediction_mode: str = field(default="absolute", init=False)
    _tracking_space: TrackingSpace = field(default="head_yaw", init=False)
    _tracking_frame: TrackingFrame | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        torch = _import_torch()
        self._torch = torch
        self._device = _torch_device(torch, allow_cpu=self.allow_cpu)
        artifact = torch.load(self.artifact_path, map_location=self._device, weights_only=False)
        feature_names = [str(name) for name in artifact["feature_names"]]
        artifact_window_size = window_size_from_feature_names(feature_names)
        if artifact_window_size is not None:
            self.window_size = artifact_window_size
        self._target_names = [str(name) for name in artifact["target_names"]]
        self._prediction_mode = str(artifact.get("prediction_mode", "absolute"))
        self._tracking_space = _tracking_space_from_value(
            str(artifact.get("tracking_space", "head_yaw"))
        )
        self._model = _make_torch_provider_model(
            torch,
            model_name=str(artifact["model_name"]),
            input_dim=len(feature_names),
            output_dim=len(self._target_names),
            hidden_dim=int(artifact.get("hidden_dim", 256)),
            feature_names=feature_names,
        ).to(self._device)
        self._model.load_state_dict(artifact["model_state_dict"])
        self._model.eval()
        normalization = json.loads(self.normalization_path.read_text(encoding="utf-8"))
        self._mean = [float(value) for value in normalization["mean"]]
        self._scale = [max(1e-6, float(value)) for value in normalization["scale"]]

    def predict(
        self,
        sample: TrackedPoseSample,
        calibration: AvatarCalibration,
        grounding: GroundingContext,
    ) -> UpperBodyLatentPrediction:
        del grounding
        self._history.append(sample)
        if self._tracking_frame is None:
            self._tracking_frame = tracking_frame_for_sample(
                sample,
                tracking_space=self._tracking_space,
            )
        if len(self._history) > self.window_size:
            self._history = self._history[-self.window_size :]
        current_tracking_frame = tracking_frame_for_sample(
            sample,
            tracking_space=self._tracking_space,
            calibration_frame=self._tracking_frame,
        )
        if len(self._history) < self.window_size:
            return UpperBodyLatentPrediction(
                values={name: 0.0 for name in self._target_names},
                masks={name: 0.0 for name in self._target_names},
                tracking_frame=current_tracking_frame,
            )

        features = sparse_window_features(
            self._history,
            len(self._history) - 1,
            self.window_size,
            calibration=calibration,
            tracking_space=self._tracking_space,
            tracking_frame=current_tracking_frame,
        )
        normalized = [
            (value - mean) / scale
            for value, mean, scale in zip(features, self._mean, self._scale, strict=True)
        ]
        with self._torch.no_grad():
            tensor = self._torch.tensor(
                [normalized], dtype=self._torch.float32, device=self._device
            )
            prediction = self._model(tensor).detach().cpu().numpy()[0]
        values = {name: float(prediction[index]) for index, name in enumerate(self._target_names)}
        return UpperBodyLatentPrediction(
            values=values,
            masks={name: 1.0 for name in self._target_names},
            tracking_frame=current_tracking_frame,
        )

    def reset(self) -> None:
        self._history.clear()
        self._tracking_frame = None

    @property
    def prediction_mode(self) -> str:
        return self._prediction_mode


@dataclass(slots=True)
class EmaUpperBodyLatentProvider:
    """Exponential smoothing wrapper for runtime latent predictions."""

    provider: LatentUpperBodyProvider
    alpha: float = 1.0
    min_mask: float = 0.5
    _previous_values: dict[str, float] = field(default_factory=dict, init=False)

    def predict(
        self,
        sample: TrackedPoseSample,
        calibration: AvatarCalibration,
        grounding: GroundingContext,
    ) -> UpperBodyLatentPrediction:
        prediction = self.provider.predict(sample, calibration, grounding)
        alpha = min(1.0, max(0.0, self.alpha))
        if alpha >= 1.0:
            self._remember(prediction)
            return prediction
        values: dict[str, float] = {}
        for name, value in prediction.values.items():
            mask = float(prediction.masks.get(name, 1.0))
            previous = self._previous_values.get(name)
            if mask < self.min_mask or previous is None:
                smoothed = float(value)
            else:
                smoothed = previous + alpha * (float(value) - previous)
            values[name] = smoothed
            if mask >= self.min_mask:
                self._previous_values[name] = smoothed
        return UpperBodyLatentPrediction(
            values=values,
            masks=prediction.masks,
            tracking_frame=prediction.tracking_frame,
        )

    def reset(self) -> None:
        self._previous_values.clear()
        reset = getattr(self.provider, "reset", None)
        if callable(reset):
            reset()

    def _remember(self, prediction: UpperBodyLatentPrediction) -> None:
        for name, value in prediction.values.items():
            if float(prediction.masks.get(name, 1.0)) >= self.min_mask:
                self._previous_values[name] = float(value)


@dataclass(slots=True)
class LearnedUpperBodyEstimator:
    """Validation-only wrapper that lets learned latents drive analytic arm IK."""

    latent_provider: LatentUpperBodyProvider
    base_estimator: PoseEstimator
    min_mask: float = 0.5

    def estimate(
        self,
        sample: TrackedPoseSample,
        calibration: AvatarCalibration,
        grounding: GroundingContext,
    ) -> HumanoidPoseFrame:
        base_pose = self.base_estimator.estimate(sample, calibration, grounding)
        prediction = self.latent_provider.predict(sample, calibration, grounding)
        if _is_fk_prediction(prediction):
            return apply_upper_body_fk_latents(
                base_pose,
                prediction,
                calibration=calibration,
                min_mask=self.min_mask,
            )
        return apply_upper_body_latents(
            base_pose,
            prediction,
            calibration=calibration,
            min_mask=self.min_mask,
        )


def _is_fk_prediction(prediction: UpperBodyLatentPrediction) -> bool:
    return any(name.startswith("fk_") for name in prediction.values)


def build_sparse_window_dataset(
    *,
    cache_dir: Path,
    output_dir: Path,
    config: SparseWindowConfig | None = None,
    subjects: set[str] | None = None,
    max_clips: int | None = None,
) -> SparseWindowExportSummary:
    """Export causal sparse-tracker windows and GT upper-body latent targets."""

    effective_config = config or SparseWindowConfig()
    start = perf_counter()
    list_start = perf_counter()
    paths = list_clip_cache_files(cache_dir, subjects or None)
    paths = filter_clip_cache_files_by_environment(
        paths,
        environment_splits=set(effective_config.environment_splits),
        activity_classes=set(effective_config.activity_classes),
    )
    if max_clips is not None:
        paths = paths[:max_clips]
    phase_seconds: dict[str, float] = {"list_filter_seconds": perf_counter() - list_start}

    feature_names = feature_schema(effective_config.window_size)
    workers = max(1, effective_config.workers)
    shard_rows = max(0, effective_config.shard_rows)
    clips_exported = 0
    total_windows = 0
    shards = 0
    rows: list[dict[str, object]] = []
    pending_rows: list[dict[str, object]] = []
    shard_dir = output_dir / "windows_shards"

    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_file in (
        output_dir / "schema.json",
        output_dir / "manifest.json",
        output_dir / "windows.parquet",
    ):
        if stale_file.exists():
            stale_file.unlink()
    if shard_dir.exists():
        shutil.rmtree(shard_dir)

    for clip_rows, clip_phase_seconds in _iter_clip_window_rows(
        paths,
        effective_config,
        feature_names,
    ):
        _accumulate_phase_seconds(phase_seconds, clip_phase_seconds)
        if clip_rows:
            clips_exported += 1
            total_windows += len(clip_rows)
            if shard_rows > 0:
                pending_rows.extend(clip_rows)
                if len(pending_rows) >= shard_rows:
                    write_start = perf_counter()
                    shards += _write_window_shard(
                        shard_dir,
                        pending_rows,
                        shard_index=shards,
                    )
                    phase_seconds["write_seconds"] = (
                        phase_seconds.get("write_seconds", 0.0) + perf_counter() - write_start
                    )
                    pending_rows.clear()
            else:
                rows.extend(clip_rows)

    if shard_rows > 0 and pending_rows:
        write_start = perf_counter()
        shards += _write_window_shard(shard_dir, pending_rows, shard_index=shards)
        phase_seconds["write_seconds"] = (
            phase_seconds.get("write_seconds", 0.0) + perf_counter() - write_start
        )
        pending_rows.clear()

    windows_path = output_dir / "windows.parquet"
    if shard_rows <= 0 and rows:
        write_start = perf_counter()
        _window_rows_frame(rows).write_parquet(windows_path, compression="zstd")
        phase_seconds["write_seconds"] = (
            phase_seconds.get("write_seconds", 0.0) + perf_counter() - write_start
        )
    elif total_windows == 0:
        if shard_dir.exists():
            shutil.rmtree(shard_dir)
        shards = 0
        write_start = perf_counter()
        _empty_windows_frame(feature_names).write_parquet(windows_path, compression="zstd")
        phase_seconds["write_seconds"] = (
            phase_seconds.get("write_seconds", 0.0) + perf_counter() - write_start
        )
    else:
        if windows_path.exists():
            windows_path.unlink()

    wrote_shards = shard_rows > 0 and total_windows > 0
    schemas = {
        "feature_names": feature_names,
        "target_names": list(TARGET_NAMES),
        "mask_names": list(MASK_NAMES),
        "target_schema": TARGET_SCHEMA,
        "target_groups": {key: list(value) for key, value in TARGET_GROUPS.items()},
        "fk_target_names": list(FK_TARGET_NAMES),
        "fk_mask_names": list(FK_MASK_NAMES),
        "fk_target_schema": FK_TARGET_SCHEMA,
        "fk_target_groups": {key: list(value) for key, value in FK_TARGET_GROUPS.items()},
        "windows_path": "windows_shards/*.parquet" if wrote_shards else "windows.parquet",
        "config": asdict(effective_config),
    }
    (output_dir / "schema.json").write_text(
        json.dumps(schemas, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    summary = SparseWindowExportSummary(
        cache_dir=cache_dir.as_posix(),
        output_dir=output_dir.as_posix(),
        clips_seen=len(paths),
        clips_exported=clips_exported,
        windows=total_windows,
        feature_count=len(feature_names),
        target_count=len(TARGET_NAMES),
        window_size=effective_config.window_size,
        stride=effective_config.stride,
        estimator_name=effective_config.estimator_name,
        tracking_space=effective_config.tracking_space,
        workers=workers,
        shards=shards,
        total_seconds=perf_counter() - start,
        phase_seconds={key: round(value, 6) for key, value in sorted(phase_seconds.items())},
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _iter_clip_window_rows(
    paths: Sequence[Path],
    config: SparseWindowConfig,
    feature_names: Sequence[str],
) -> Iterable[tuple[list[dict[str, object]], dict[str, float]]]:
    workers = max(1, config.workers)
    if workers == 1:
        for path in paths:
            yield _export_clip_windows(path, config, feature_names)
        return

    feature_name_tuple = tuple(feature_names)
    jobs = [(path, config, feature_name_tuple) for path in paths]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures: dict[
            Future[tuple[list[dict[str, object]], dict[str, float]]],
            tuple[Path, SparseWindowConfig, tuple[str, ...]],
        ] = {}
        job_iter = iter(jobs)
        for _ in range(workers):
            try:
                job = next(job_iter)
            except StopIteration:
                break
            futures[executor.submit(_export_clip_windows_job, job)] = job

        while futures:
            completed, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in completed:
                futures.pop(future)
                yield future.result()
                try:
                    next_job = next(job_iter)
                except StopIteration:
                    continue
                futures[executor.submit(_export_clip_windows_job, next_job)] = next_job


def _export_clip_windows_job(
    job: tuple[Path, SparseWindowConfig, tuple[str, ...]],
) -> tuple[list[dict[str, object]], dict[str, float]]:
    path, config, feature_names = job
    return _export_clip_windows(path, config, feature_names)


def _accumulate_phase_seconds(
    totals: dict[str, float],
    seconds: Mapping[str, float],
) -> None:
    for name, value in seconds.items():
        totals[name] = totals.get(name, 0.0) + float(value)


def _write_window_shard(
    shard_dir: Path,
    rows: Sequence[dict[str, object]],
    *,
    shard_index: int,
) -> int:
    if not rows:
        return 0
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_path = shard_dir / f"part-{shard_index:05d}.parquet"
    _window_rows_frame(rows).write_parquet(shard_path, compression="zstd")
    return 1


def _window_rows_frame(rows: Sequence[dict[str, object]]) -> pl.DataFrame:
    return pl.DataFrame(rows).with_columns(
        pl.col("timestamp_s").cast(pl.Float32),
        pl.col("features").cast(pl.List(pl.Float32)),
        *(pl.col(name).cast(pl.Float32) for name in _float32_window_columns()),
    )


def _float32_window_columns() -> tuple[str, ...]:
    columns: list[str] = []
    for name in TARGET_NAMES:
        columns.extend(
            (
                name,
                f"{name}_mask",
                f"heuristic_{name}",
                f"heuristic_{name}_mask",
            )
        )
    for name in FK_TARGET_NAMES:
        columns.extend((name, f"{name}_mask"))
    for prefix in ("reference", "heuristic"):
        for joint_name in DECODED_REFERENCE_JOINTS:
            for component in ("x", "y", "z"):
                columns.append(f"{prefix}_{joint_name}_position_{component}")
            columns.append(f"{prefix}_{joint_name}_valid")
    for joint_name in RUNTIME_DECODER_CONTEXT_JOINTS:
        for component in ("x", "y", "z"):
            columns.append(f"heuristic_context_{joint_name}_position_{component}")
        columns.append(f"heuristic_context_{joint_name}_valid")
    return tuple(columns)


def read_sparse_window_dataframe(
    dataset_dir: Path,
    *,
    max_rows: int | None = None,
    max_clips: int | None = None,
    split: str | None = None,
    subjects: set[str] | None = None,
    exclude_subjects: set[str] | None = None,
    clip_ids: set[str] | None = None,
) -> pl.DataFrame:
    windows_path = dataset_dir / "windows.parquet"
    if windows_path.exists():
        scan = pl.scan_parquet(windows_path)
        scan = _filter_sparse_window_scan(
            scan,
            split=split,
            subjects=subjects,
            exclude_subjects=exclude_subjects,
            clip_ids=clip_ids,
        )
        return _collect_sparse_window_scan(scan, max_rows=max_rows, max_clips=max_clips)
    shard_dir = dataset_dir / "windows_shards"
    if shard_dir.exists():
        scan = pl.scan_parquet(str(shard_dir / "*.parquet"))
        scan = _filter_sparse_window_scan(
            scan,
            split=split,
            subjects=subjects,
            exclude_subjects=exclude_subjects,
            clip_ids=clip_ids,
        )
        return _collect_sparse_window_scan(scan, max_rows=max_rows, max_clips=max_clips)
    raise FileNotFoundError(
        f"Expected sparse-window dataset at {dataset_dir}; run build_sparse_window_dataset"
    )


def _filter_sparse_window_scan(
    scan: pl.LazyFrame,
    *,
    split: str | None,
    subjects: set[str] | None,
    exclude_subjects: set[str] | None,
    clip_ids: set[str] | None,
) -> pl.LazyFrame:
    if split is not None:
        scan = scan.filter(pl.col("split") == split)
    if subjects:
        scan = scan.filter(pl.col("subject").is_in(sorted(subjects)))
    if exclude_subjects:
        scan = scan.filter(~pl.col("subject").is_in(sorted(exclude_subjects)))
    if clip_ids:
        scan = scan.filter(pl.col("clip").is_in(sorted(clip_ids)))
    return scan


def _collect_sparse_window_scan(
    scan: pl.LazyFrame,
    *,
    max_rows: int | None,
    max_clips: int | None,
) -> pl.DataFrame:
    if max_clips is not None and max_clips > 0:
        clip_frame = cast(
            pl.DataFrame,
            scan.select("clip").unique(maintain_order=True).limit(max_clips).collect(),
        )
        clip_ids = clip_frame["clip"].to_list()
        scan = scan.filter(pl.col("clip").is_in(clip_ids))
    if max_rows is not None and max_rows > 0:
        scan = scan.head(max_rows)
    return cast(pl.DataFrame, scan.collect())


def feature_schema(window_size: int) -> list[str]:
    names: list[str] = []
    for relative_index in range(-(window_size - 1), 1):
        prefix = f"t{relative_index:+03d}"
        for role in TRACKER_ROLES:
            for component in ("x", "y", "z"):
                names.append(f"{prefix}_{role}_position_{component}")
            for component_index in range(6):
                names.append(f"{prefix}_{role}_rotation6d_{component_index}")
            for component in ("x", "y", "z"):
                names.append(f"{prefix}_{role}_linear_velocity_{component}")
            for component_index in range(6):
                names.append(f"{prefix}_{role}_rotation6d_velocity_{component_index}")
            names.append(f"{prefix}_{role}_valid")
    names.extend(STATIC_CALIBRATION_FEATURE_NAMES)
    return names


def window_size_from_feature_names(feature_names: Sequence[str]) -> int | None:
    prefixes = {
        name.split("_", maxsplit=1)[0]
        for name in feature_names
        if name.startswith("t") and "_position_" in name
    }
    return len(prefixes) if prefixes else None


def split_for_subject(subject: str, holdout_folds: int) -> Literal["train", "validation"]:
    folds = max(2, holdout_folds)
    digest = sha1(subject.encode("utf-8")).digest()
    fold = int.from_bytes(digest[:4], "big") % folds
    return "validation" if fold == 0 else "train"


def extract_upper_body_latents(
    joints: Mapping[str, Transform],
    calibration: AvatarCalibration | None = None,
    tracking_frame: TrackingFrame | None = None,
) -> UpperBodyLatents:
    values = {name: 0.0 for name in TARGET_NAMES}
    masks = {name: 0.0 for name in TARGET_NAMES}
    head = joints.get("head")
    if head is None:
        return UpperBodyLatents(values=values, masks=masks)

    effective_calibration = calibration or AvatarCalibration()
    height = max(0.1, effective_calibration.human_height_m)
    frame = tracking_frame or tracking_frame_for_joints(joints)

    chest = joints.get("chest")
    neck = joints.get("neck") or _synthetic_neck_transform(joints)
    for latent_prefix, transform in (("chest", chest), ("neck", neck)):
        if transform is None:
            continue
        rel = _head_relative_position(
            transform.position,
            origin=frame.origin,
            basis=frame.basis,
            height_m=height,
        )
        for component, value in zip(("x", "y", "z"), rel, strict=True):
            name = f"{latent_prefix}_rel_{component}"
            values[name] = value
            masks[name] = 1.0

    for side in ("left", "right"):
        shoulder_name = f"{side}_shoulder"
        shoulder = joints.get(shoulder_name)
        if shoulder is not None:
            rel = _head_relative_position(
                shoulder.position,
                origin=frame.origin,
                basis=frame.basis,
                height_m=height,
            )
            for component, value in zip(("x", "y", "z"), rel, strict=True):
                name = f"{side}_shoulder_rel_{component}"
                values[name] = value
                masks[name] = 1.0

        elbow = joints.get(f"{side}_elbow")
        hand = joints.get(f"{side}_hand")
        if shoulder is None or elbow is None or hand is None:
            continue
        pole, valid = elbow_pole_vector(
            shoulder=shoulder.position,
            elbow=elbow.position,
            hand=hand.position,
            basis=frame.basis,
            min_distance_m=max(0.025, 0.12 * _upper_arm_length(effective_calibration)),
        )
        for component, value in zip(("x", "y", "z"), pole, strict=True):
            name = f"{side}_elbow_pole_{component}"
            values[name] = value
            masks[name] = 1.0 if valid else 0.0

    return UpperBodyLatents(values=values, masks=masks)


def extract_upper_body_fk_targets(
    joints: Mapping[str, Transform],
    calibration: AvatarCalibration | None = None,
    tracking_frame: TrackingFrame | None = None,
) -> UpperBodyFkTargets:
    """Extract AvatarPoser-style global plus local rotation targets.

    The existing chain latents predict point anchors. These FK targets keep a
    separate global stabilizer target for the hips and local 6D rotations for
    the upper-body kinematic chain, so a later model can train through FK
    without using head-relative point anchors as its only representation.
    """

    values = {name: 0.0 for name in FK_TARGET_NAMES}
    masks = {name: 0.0 for name in FK_TARGET_NAMES}
    head = joints.get("head")
    hips = joints.get("hips")
    if head is None or hips is None:
        return UpperBodyFkTargets(values=values, masks=masks)

    effective_calibration = calibration or AvatarCalibration()
    height = max(0.1, effective_calibration.human_height_m)
    frame = tracking_frame or tracking_frame_for_joints(joints)
    hips_rel = _head_relative_position(
        hips.position,
        origin=frame.origin,
        basis=frame.basis,
        height_m=height,
    )
    _set_vector_target(values, masks, "fk_hips_rel", hips_rel)
    _set_sequence_target(
        values,
        masks,
        "fk_hips_rotation6d",
        rotation6d(hips.rotation, basis=frame.basis),
    )

    for joint_name, parent_name in FK_CHAIN_PARENTS.items():
        joint = joints.get(joint_name)
        parent = joints.get(parent_name)
        if joint is None or parent is None:
            continue
        local_rotation = _local_rotation(parent.rotation, joint.rotation)
        _set_sequence_target(
            values,
            masks,
            f"fk_{joint_name}_local_rotation6d",
            rotation6d(local_rotation, basis=IDENTITY_BASIS),
        )
    return UpperBodyFkTargets(values=values, masks=masks)


def yaw_basis_from_rotation(rotation: tuple[float, float, float, float]) -> Basis3:
    forward = normalize(horizontal(rotate_vector(rotation, WORLD_FORWARD)), WORLD_FORWARD)
    right = normalize(cross(WORLD_UP, forward), (1.0, 0.0, 0.0))
    return Basis3(right=right, up=WORLD_UP, forward=forward)


def basis_from_rotation(rotation: tuple[float, float, float, float]) -> Basis3:
    forward = normalize(rotate_vector(rotation, WORLD_FORWARD), WORLD_FORWARD)
    up = normalize(rotate_vector(rotation, WORLD_UP), WORLD_UP)
    right = normalize(cross(up, forward), (1.0, 0.0, 0.0))
    up = normalize(cross(forward, right), WORLD_UP)
    return Basis3(right=right, up=up, forward=forward)


def tracking_frame_for_sample(
    sample: TrackedPoseSample,
    *,
    tracking_space: TrackingSpace = "head_yaw",
    calibration_frame: TrackingFrame | None = None,
) -> TrackingFrame:
    basis = (
        calibration_frame.basis
        if tracking_space == "calibration_yaw" and calibration_frame is not None
        else yaw_basis_from_rotation(sample.hmd.transform.rotation)
    )
    return TrackingFrame(origin=sample.hmd.transform.position, basis=basis)


def tracking_frame_for_joints(
    joints: Mapping[str, Transform],
    *,
    tracking_space: TrackingSpace = "head_yaw",
    calibration_frame: TrackingFrame | None = None,
) -> TrackingFrame:
    head = joints.get("head")
    if head is None:
        return calibration_frame or TrackingFrame(origin=(0.0, 0.0, 0.0), basis=IDENTITY_BASIS)
    basis = (
        calibration_frame.basis
        if tracking_space == "calibration_yaw" and calibration_frame is not None
        else yaw_basis_from_rotation(head.rotation)
    )
    return TrackingFrame(origin=head.position, basis=basis)


def _tracking_space_from_value(value: str) -> TrackingSpace:
    if value == "calibration_yaw":
        return "calibration_yaw"
    return "head_yaw"


def _runtime_decode_tracking_frame(
    prediction: UpperBodyLatentPrediction,
    head: Transform,
) -> TrackingFrame:
    if prediction.tracking_frame is None:
        return TrackingFrame(
            origin=head.position,
            basis=yaw_basis_from_rotation(head.rotation),
        )
    return TrackingFrame(origin=head.position, basis=prediction.tracking_frame.basis)


def vector_to_basis(vector: Vector3, basis: Basis3) -> Vector3:
    return (dot(vector, basis.right), dot(vector, basis.up), dot(vector, basis.forward))


def _head_relative_position(
    position: Vector3,
    *,
    origin: Vector3,
    basis: Basis3,
    height_m: float,
) -> Vector3:
    local = vector_to_basis(subtract(position, origin), basis)
    scale_m = max(0.1, height_m)
    return (local[0] / scale_m, local[1] / scale_m, local[2] / scale_m)


def _world_from_head_relative(
    rel: Vector3,
    *,
    origin: Vector3,
    basis: Basis3,
    height_m: float,
) -> Vector3:
    scale_m = max(0.1, height_m)
    return add(
        origin,
        add(
            add(scale(basis.right, rel[0] * scale_m), scale(basis.up, rel[1] * scale_m)),
            scale(basis.forward, rel[2] * scale_m),
        ),
    )


def elbow_pole_vector(
    *,
    shoulder: Vector3,
    elbow: Vector3,
    hand: Vector3,
    basis: Basis3,
    min_distance_m: float = 0.03,
) -> tuple[Vector3, bool]:
    arm_axis = subtract(hand, shoulder)
    if _length(arm_axis) <= 1e-6:
        return (0.0, -1.0, 0.0), False
    axis = normalize(arm_axis)
    elbow_offset = subtract(elbow, shoulder)
    pole_world = project_on_plane(elbow_offset, axis)
    if _length(pole_world) < min_distance_m:
        return vector_to_basis(_fallback_bend(axis), basis), False
    return vector_to_basis(normalize(pole_world), basis), True


def _vector_from_basis(vector: Vector3, basis: Basis3) -> Vector3:
    return add(
        add(scale(basis.right, vector[0]), scale(basis.up, vector[1])),
        scale(basis.forward, vector[2]),
    )


def _local_rotation(parent_rotation: Quaternion, child_rotation: Quaternion) -> Quaternion:
    return _normalize_quaternion(
        multiply_quaternion(conjugate_quaternion(parent_rotation), child_rotation)
    )


def _normalize_quaternion(rotation: Quaternion) -> Quaternion:
    magnitude = sqrt(sum(component * component for component in rotation))
    if magnitude <= 1e-9:
        return (0.0, 0.0, 0.0, 1.0)
    return (
        rotation[0] / magnitude,
        rotation[1] / magnitude,
        rotation[2] / magnitude,
        rotation[3] / magnitude,
    )


def _set_vector_target(
    values: dict[str, float],
    masks: dict[str, float],
    prefix: str,
    vector: Vector3,
) -> None:
    for component, value in zip(("x", "y", "z"), vector, strict=True):
        name = f"{prefix}_{component}"
        values[name] = float(value)
        masks[name] = 1.0


def _set_sequence_target(
    values: dict[str, float],
    masks: dict[str, float],
    prefix: str,
    sequence: Sequence[float],
) -> None:
    for index, value in enumerate(sequence):
        name = f"{prefix}_{index}"
        values[name] = float(value)
        masks[name] = 1.0


def calibration_feature_values(calibration: AvatarCalibration | None = None) -> list[float]:
    effective_calibration = calibration or AvatarCalibration()
    human_height = max(0.1, effective_calibration.human_height_m)
    head_to_neck = _calibrated_length(
        effective_calibration,
        ("head_to_neck",),
        human_height * 0.055,
    )
    neck_to_chest = _calibrated_length(
        effective_calibration,
        ("neck_to_chest",),
        human_height * 0.085,
    )
    chest_to_spine = _calibrated_length(
        effective_calibration,
        ("chest_to_spine",),
        human_height * 0.13,
    )
    spine_to_hips = _calibrated_length(
        effective_calibration,
        ("spine_to_hips",),
        human_height * 0.14,
    )
    shoulder_width = _shoulder_width(effective_calibration)
    upper_arm = _upper_arm_length(effective_calibration)
    lower_arm = _lower_arm_length(effective_calibration)
    arm_length = max(0.1, upper_arm + lower_arm)
    return [
        human_height,
        head_to_neck,
        neck_to_chest,
        chest_to_spine,
        spine_to_hips,
        shoulder_width,
        upper_arm,
        lower_arm,
        arm_length,
        shoulder_width / human_height,
        arm_length / human_height,
    ]


def _shoulder_width(calibration: AvatarCalibration) -> float:
    return _calibrated_length(
        calibration,
        ("shoulder_width", "left_shoulder_to_right_shoulder"),
        calibration.human_height_m * 0.201,
    )


def _upper_arm_length(calibration: AvatarCalibration) -> float:
    return _calibrated_length(
        calibration,
        ("upper_arm", "humerus"),
        calibration.human_height_m * 0.174,
    )


def _lower_arm_length(calibration: AvatarCalibration) -> float:
    return _calibrated_length(
        calibration,
        ("lower_arm", "forearm", "radius"),
        calibration.human_height_m * 0.151,
    )


def local_position(position: Vector3, *, origin: Vector3, basis: Basis3) -> Vector3:
    delta = subtract(position, origin)
    return (dot(delta, basis.right), position[1], dot(delta, basis.forward))


def _basis_position(position: Vector3, basis: Basis3) -> Vector3:
    return (dot(position, basis.right), position[1], dot(position, basis.forward))


def _relative_cached_position(position: Vector3, origin: Vector3) -> Vector3:
    return (position[0] - origin[0], position[1], position[2] - origin[2])


def rotation6d(rotation: tuple[float, float, float, float], *, basis: Basis3) -> tuple[float, ...]:
    forward = vector_to_basis(rotate_vector(rotation, WORLD_FORWARD), basis)
    up = vector_to_basis(rotate_vector(rotation, WORLD_UP), basis)
    return (*forward, *up)


def yaw_rotated_transform(transform: Transform, degrees: float) -> Transform:
    yaw = axis_name_quaternion("y", degrees)
    return Transform(
        position=rotate_vector(yaw, transform.position),
        rotation=multiply_quaternion(yaw, transform.rotation),
    )


def yaw_rotated_pose_frame(
    joints: Mapping[str, Transform],
    degrees: float,
) -> dict[str, Transform]:
    return {name: yaw_rotated_transform(transform, degrees) for name, transform in joints.items()}


def yaw_rotated_sample(sample: TrackedPoseSample, degrees: float) -> TrackedPoseSample:
    yaw = axis_name_quaternion("y", degrees)
    return TrackedPoseSample(
        timestamp_seconds=sample.timestamp_seconds,
        hmd=_yaw_rotated_tracker(sample.hmd, degrees),
        left_hand=_yaw_rotated_tracker(sample.left_hand, degrees),
        right_hand=_yaw_rotated_tracker(sample.right_hand, degrees),
        optional_trackers={
            role: _yaw_rotated_tracker(state, degrees)
            for role, state in sample.optional_trackers.items()
        },
        character_velocity=rotate_vector(yaw, sample.character_velocity),
    )


def apply_upper_body_latents(
    pose: HumanoidPoseFrame,
    prediction: UpperBodyLatentPrediction,
    *,
    calibration: AvatarCalibration,
    min_mask: float = 0.5,
) -> HumanoidPoseFrame:
    joints = dict(pose.joints)
    values = prediction.values
    masks = prediction.masks
    head_joint = joints.get("head")
    if head_joint is None:
        return HumanoidPoseFrame(
            timestamp_seconds=pose.timestamp_seconds, root=pose.root, joints=joints
        )
    tracking_frame = _runtime_decode_tracking_frame(prediction, head_joint.transform)
    height_m = max(0.1, calibration.human_height_m)

    for joint_name in ("chest", "neck", "left_shoulder", "right_shoulder"):
        if _relative_position_enabled(values, masks, joint_name, min_mask):
            position = _world_from_head_relative(
                _relative_position_values(values, joint_name),
                origin=tracking_frame.origin,
                basis=tracking_frame.basis,
                height_m=height_m,
            )
            old_joint = joints.get(joint_name, JointPose(Transform()))
            joints[joint_name] = JointPose(
                transform=Transform(position=position, rotation=old_joint.transform.rotation),
                confidence=old_joint.confidence,
                estimated=True,
            )

    chest_rotation = _derived_chest_rotation(joints, fallback_basis=tracking_frame.basis)
    if chest_rotation is not None:
        for joint_name in ("spine", "chest", "neck"):
            if joint_name in joints:
                old_joint = joints[joint_name]
                joints[joint_name] = JointPose(
                    transform=Transform(
                        position=old_joint.transform.position,
                        rotation=chest_rotation,
                    ),
                    confidence=old_joint.confidence,
                    estimated=old_joint.estimated,
                )

    for side in ("left", "right"):
        shoulder_name = f"{side}_shoulder"
        elbow_name = f"{side}_elbow"
        hand_name = f"{side}_hand"
        if (
            shoulder_name in joints
            and hand_name in joints
            and _pole_enabled(values, masks, side, min_mask)
        ):
            bend_hint = _vector_from_basis(
                (
                    values.get(f"{side}_elbow_pole_x", 0.0),
                    values.get(f"{side}_elbow_pole_y", -1.0),
                    values.get(f"{side}_elbow_pole_z", 0.0),
                ),
                tracking_frame.basis,
            )
            elbow_position = solve_two_bone_midpoint(
                root=joints[shoulder_name].transform.position,
                tip=joints[hand_name].transform.position,
                root_to_mid_length=_upper_arm_length(calibration),
                mid_to_tip_length=_lower_arm_length(calibration),
                bend_hint=bend_hint,
            )
            old_joint = joints.get(elbow_name, JointPose(Transform()))
            joints[elbow_name] = JointPose(
                transform=Transform(position=elbow_position, rotation=old_joint.transform.rotation),
                confidence=old_joint.confidence,
                estimated=True,
            )
    return HumanoidPoseFrame(
        timestamp_seconds=pose.timestamp_seconds, root=pose.root, joints=joints
    )


def apply_upper_body_fk_latents(
    pose: HumanoidPoseFrame,
    prediction: UpperBodyLatentPrediction,
    *,
    calibration: AvatarCalibration,
    min_mask: float = 0.5,
) -> HumanoidPoseFrame:
    joints = dict(pose.joints)
    values = prediction.values
    masks = prediction.masks
    head_joint = joints.get("head")
    if head_joint is None:
        return HumanoidPoseFrame(
            timestamp_seconds=pose.timestamp_seconds, root=pose.root, joints=joints
        )
    if not (
        _fk_vector_enabled(values, masks, "fk_hips_rel", min_mask)
        and _fk_rotation6d_enabled(values, masks, "fk_hips_rotation6d", min_mask)
    ):
        return HumanoidPoseFrame(
            timestamp_seconds=pose.timestamp_seconds, root=pose.root, joints=joints
        )

    tracking_frame = _runtime_decode_tracking_frame(prediction, head_joint.transform)
    height_m = max(0.1, calibration.human_height_m)
    lengths = _fk_runtime_lengths(calibration)
    positions: dict[str, Vector3] = {
        "hips": _world_from_head_relative(
            _fk_vector_values(values, "fk_hips_rel"),
            origin=tracking_frame.origin,
            basis=tracking_frame.basis,
            height_m=height_m,
        )
    }
    rotations: dict[str, Quaternion] = {
        "hips": _rotation6d_quaternion(
            values,
            "fk_hips_rotation6d",
            basis=tracking_frame.basis,
        )
    }

    _set_fk_joint(joints, "hips", positions["hips"], rotations["hips"])

    _decode_fk_segment(
        joints,
        values,
        masks,
        positions=positions,
        rotations=rotations,
        joint_name="spine",
        parent_name="hips",
        local_prefix="fk_spine_local_rotation6d",
        offset=(0.0, lengths["spine_to_hips"], 0.0),
        min_mask=min_mask,
    )
    _decode_fk_segment(
        joints,
        values,
        masks,
        positions=positions,
        rotations=rotations,
        joint_name="chest",
        parent_name="spine",
        local_prefix="fk_chest_local_rotation6d",
        offset=(0.0, lengths["chest_to_spine"], 0.0),
        min_mask=min_mask,
    )
    decoded_head = _decode_fk_child_transform(
        values,
        masks,
        positions=positions,
        rotations=rotations,
        joint_name="head",
        parent_name="chest",
        local_prefix="fk_head_local_rotation6d",
        offset=(0.0, lengths["head_to_neck"] + lengths["neck_to_chest"], 0.0),
        min_mask=min_mask,
    )
    if decoded_head is not None and "chest" in positions:
        chest_position = positions["chest"]
        neck_position = add(
            chest_position,
            scale(subtract(decoded_head.position, chest_position), 0.7),
        )
        _set_fk_joint(joints, "neck", neck_position, decoded_head.rotation)

    half_shoulder_width = 0.5 * lengths["shoulder_width"]
    for side, sign in (("left", 1.0), ("right", -1.0)):
        shoulder_name = f"{side}_shoulder"
        elbow_name = f"{side}_elbow"
        if (
            _decode_fk_child_transform(
                values,
                masks,
                positions=positions,
                rotations=rotations,
                joint_name=shoulder_name,
                parent_name="chest",
                local_prefix=f"fk_{shoulder_name}_local_rotation6d",
                offset=(sign * half_shoulder_width, 0.0, 0.0),
                min_mask=min_mask,
            )
            is not None
        ):
            _set_fk_joint(joints, shoulder_name, positions[shoulder_name], rotations[shoulder_name])
        if (
            _decode_fk_child_transform(
                values,
                masks,
                positions=positions,
                rotations=rotations,
                joint_name=elbow_name,
                parent_name=shoulder_name,
                local_prefix=f"fk_{elbow_name}_local_rotation6d",
                offset=(sign * lengths["upper_arm"], 0.0, 0.0),
                min_mask=min_mask,
            )
            is not None
        ):
            _set_fk_joint(joints, elbow_name, positions[elbow_name], rotations[elbow_name])

    return HumanoidPoseFrame(
        timestamp_seconds=pose.timestamp_seconds, root=pose.root, joints=joints
    )


def _fk_runtime_lengths(calibration: AvatarCalibration) -> dict[str, float]:
    values = calibration_feature_values(calibration)
    return {
        "human_height": values[0],
        "head_to_neck": values[1],
        "neck_to_chest": values[2],
        "chest_to_spine": values[3],
        "spine_to_hips": values[4],
        "shoulder_width": values[5],
        "upper_arm": values[6],
        "lower_arm": values[7],
        "arm_length": values[8],
    }


def _decode_fk_segment(
    joints: dict[str, JointPose],
    values: Mapping[str, float],
    masks: Mapping[str, float],
    *,
    positions: dict[str, Vector3],
    rotations: dict[str, Quaternion],
    joint_name: str,
    parent_name: str,
    local_prefix: str,
    offset: Vector3,
    min_mask: float,
) -> None:
    decoded = _decode_fk_child_transform(
        values,
        masks,
        positions=positions,
        rotations=rotations,
        joint_name=joint_name,
        parent_name=parent_name,
        local_prefix=local_prefix,
        offset=offset,
        min_mask=min_mask,
    )
    if decoded is not None:
        _set_fk_joint(joints, joint_name, decoded.position, decoded.rotation)


def _decode_fk_child_transform(
    values: Mapping[str, float],
    masks: Mapping[str, float],
    *,
    positions: dict[str, Vector3],
    rotations: dict[str, Quaternion],
    joint_name: str,
    parent_name: str,
    local_prefix: str,
    offset: Vector3,
    min_mask: float,
) -> Transform | None:
    if parent_name not in positions or parent_name not in rotations:
        return None
    if not _fk_rotation6d_enabled(values, masks, local_prefix, min_mask):
        return None
    rotation = _normalize_quaternion(
        multiply_quaternion(
            rotations[parent_name],
            _rotation6d_quaternion(values, local_prefix, basis=IDENTITY_BASIS),
        )
    )
    position = add(positions[parent_name], rotate_vector(rotation, offset))
    positions[joint_name] = position
    rotations[joint_name] = rotation
    return Transform(position=position, rotation=rotation)


def _set_fk_joint(
    joints: dict[str, JointPose],
    joint_name: str,
    position: Vector3,
    rotation: Quaternion,
) -> None:
    old_joint = joints.get(joint_name, JointPose(Transform()))
    joints[joint_name] = JointPose(
        transform=Transform(position=position, rotation=rotation),
        confidence=old_joint.confidence,
        estimated=True,
    )


def _fk_vector_enabled(
    values: Mapping[str, float],
    masks: Mapping[str, float],
    prefix: str,
    min_mask: float,
) -> bool:
    return all(
        _enabled(values, masks, f"{prefix}_{component}", min_mask) for component in ("x", "y", "z")
    )


def _fk_rotation6d_enabled(
    values: Mapping[str, float],
    masks: Mapping[str, float],
    prefix: str,
    min_mask: float,
) -> bool:
    return all(_enabled(values, masks, f"{prefix}_{index}", min_mask) for index in range(6))


def _fk_vector_values(values: Mapping[str, float], prefix: str) -> Vector3:
    return (
        values.get(f"{prefix}_x", 0.0),
        values.get(f"{prefix}_y", 0.0),
        values.get(f"{prefix}_z", 0.0),
    )


def _rotation6d_quaternion(
    values: Mapping[str, float],
    prefix: str,
    *,
    basis: Basis3,
) -> Quaternion:
    forward = _vector_from_basis(
        (
            values.get(f"{prefix}_0", 0.0),
            values.get(f"{prefix}_1", 0.0),
            values.get(f"{prefix}_2", 1.0),
        ),
        basis,
    )
    raw_up = _vector_from_basis(
        (
            values.get(f"{prefix}_3", 0.0),
            values.get(f"{prefix}_4", 1.0),
            values.get(f"{prefix}_5", 0.0),
        ),
        basis,
    )
    forward = normalize(forward, WORLD_FORWARD)
    right = normalize(cross(raw_up, forward), basis.right)
    up = normalize(cross(forward, right), basis.up)
    return look_rotation(forward, up)


def _relative_position_enabled(
    values: Mapping[str, float],
    masks: Mapping[str, float],
    joint_name: str,
    min_mask: float,
) -> bool:
    return all(
        _enabled(values, masks, f"{joint_name}_rel_{component}", min_mask)
        for component in ("x", "y", "z")
    )


def _relative_position_values(values: Mapping[str, float], joint_name: str) -> Vector3:
    return (
        values.get(f"{joint_name}_rel_x", 0.0),
        values.get(f"{joint_name}_rel_y", 0.0),
        values.get(f"{joint_name}_rel_z", 0.0),
    )


def _pole_enabled(
    values: Mapping[str, float],
    masks: Mapping[str, float],
    side: str,
    min_mask: float,
) -> bool:
    return all(
        _enabled(values, masks, f"{side}_elbow_pole_{component}", min_mask)
        for component in ("x", "y", "z")
    )


def _derived_chest_rotation(
    joints: Mapping[str, JointPose],
    *,
    fallback_basis: Basis3,
) -> tuple[float, float, float, float] | None:
    chest = joints.get("chest")
    neck = joints.get("neck")
    left_shoulder = joints.get("left_shoulder")
    right_shoulder = joints.get("right_shoulder")
    if chest is None or neck is None or left_shoulder is None or right_shoulder is None:
        return None
    lateral = subtract(
        left_shoulder.transform.position,
        right_shoulder.transform.position,
    )
    up = subtract(neck.transform.position, chest.transform.position)
    if _length(lateral) <= 1e-5 or _length(up) <= 1e-5:
        return None
    lateral = normalize(lateral, fallback_basis.right)
    up = normalize(up, fallback_basis.up)
    forward = normalize(cross(lateral, up), fallback_basis.forward)
    if dot(forward, fallback_basis.forward) < 0.0:
        forward = scale(forward, -1.0)
    return look_rotation(forward, up)


def solve_two_bone_midpoint(
    *,
    root: Vector3,
    tip: Vector3,
    root_to_mid_length: float,
    mid_to_tip_length: float,
    bend_hint: Vector3,
) -> Vector3:
    root_to_tip = subtract(tip, root)
    root_tip_distance = max(0.0001, distance(root, tip))
    max_reach = max(0.0001, root_to_mid_length + mid_to_tip_length - 0.0001)
    min_reach = abs(root_to_mid_length - mid_to_tip_length) + 0.0001
    clamped_distance = min(max(root_tip_distance, min_reach), max_reach)
    direction = normalize(root_to_tip, WORLD_UP)
    along = (
        root_to_mid_length * root_to_mid_length
        - mid_to_tip_length * mid_to_tip_length
        + clamped_distance * clamped_distance
    ) / (2.0 * clamped_distance)
    bend_height_squared = root_to_mid_length * root_to_mid_length - along * along
    bend_height = sqrt(bend_height_squared) if bend_height_squared > 0.0 else 0.0
    bend_direction = normalize(project_on_plane(bend_hint, direction), _fallback_bend(direction))
    return add(add(root, scale(direction, along)), scale(bend_direction, bend_height))


def _export_clip_windows(
    path: Path,
    config: SparseWindowConfig,
    feature_names: Sequence[str],
) -> tuple[list[dict[str, object]], dict[str, float]]:
    timings: dict[str, float] = {}
    read_start = perf_counter()
    cached_frames = list(iter_cached_pose_frames(path))
    timings["read_cache_seconds"] = perf_counter() - read_start
    if len(cached_frames) < config.window_size:
        return [], timings
    first_frame = cached_frames[0]
    if not _clean_clip(first_frame):
        return [], timings

    setup_start = perf_counter()
    reference_frames = [frame.joints for frame in cached_frames]
    floor_height = infer_reference_floor_height(reference_frames)
    calibration = calibration_from_reference_frames(reference_frames, floor_height_m=floor_height)
    grounding = GroundingContext(floor_height_m=floor_height)
    samples = [_tracked_sample_from_cached_frame(frame) for frame in cached_frames]
    calibration_frame = tracking_frame_for_sample(
        samples[0],
        tracking_space=config.tracking_space,
    )
    feature_cache = (
        precompute_sparse_tracker_features(samples, calibration_frame.basis)
        if config.tracking_space == "calibration_yaw"
        else None
    )
    timings["setup_seconds"] = perf_counter() - setup_start

    estimate_start = perf_counter()
    estimated_poses = _estimated_poses(
        samples,
        calibration=calibration,
        grounding=grounding,
        estimator_name=config.estimator_name,
    )
    timings["estimate_seconds"] = perf_counter() - estimate_start

    target_start = perf_counter()
    target_latents = [
        extract_upper_body_latents(
            frame.joints,
            calibration,
            tracking_frame=tracking_frame_for_joints(
                frame.joints,
                tracking_space=config.tracking_space,
                calibration_frame=calibration_frame,
            ),
        )
        for frame in cached_frames
    ]
    fk_targets = [
        extract_upper_body_fk_targets(
            frame.joints,
            calibration,
            tracking_frame=tracking_frame_for_joints(
                frame.joints,
                tracking_space=config.tracking_space,
                calibration_frame=calibration_frame,
            ),
        )
        for frame in cached_frames
    ]
    heuristic_latents = [
        extract_upper_body_latents(
            pose_joints(pose),
            calibration,
            tracking_frame=tracking_frame_for_joints(
                pose_joints(pose),
                tracking_space=config.tracking_space,
                calibration_frame=calibration_frame,
            ),
        )
        for pose in estimated_poses
    ]
    timings["target_seconds"] = perf_counter() - target_start

    row_start = perf_counter()
    rows: list[dict[str, object]] = []
    for center_index in range(config.window_size - 1, len(cached_frames), config.stride):
        center_tracking_frame = tracking_frame_for_sample(
            samples[center_index],
            tracking_space=config.tracking_space,
            calibration_frame=calibration_frame,
        )
        if feature_cache is None:
            feature_vector = sparse_window_features(
                samples,
                center_index,
                config.window_size,
                calibration=calibration,
                tracking_space=config.tracking_space,
                tracking_frame=center_tracking_frame,
            )
        else:
            feature_vector = sparse_window_features_from_cache(
                feature_cache,
                center_index,
                config.window_size,
                calibration=calibration,
            )
        if len(feature_vector) != len(feature_names):
            raise ValueError("Feature vector length does not match feature schema")
        row: dict[str, object] = {
            "subject": first_frame.subject,
            "clip": first_frame.clip,
            "frame_index": cached_frames[center_index].frame_index,
            "timestamp_s": cached_frames[center_index].timestamp_s,
            "center_row_index": center_index,
            "split": split_for_subject(first_frame.subject, config.holdout_folds),
            "features": feature_vector,
            "environment_split": first_frame.environment_split,
            "activity_class": first_frame.activity_class,
            "motion_description": first_frame.motion_description,
        }
        target = target_latents[center_index]
        heuristic = heuristic_latents[center_index]
        _add_decoded_reference_points(
            row,
            prefix="reference",
            joints=cached_frames[center_index].joints,
            origin=center_tracking_frame.origin,
            basis=center_tracking_frame.basis,
        )
        _add_decoded_reference_points(
            row,
            prefix="heuristic",
            joints=pose_joints(estimated_poses[center_index]),
            origin=center_tracking_frame.origin,
            basis=center_tracking_frame.basis,
        )
        _add_runtime_decoder_context_points(
            row,
            prefix="heuristic_context",
            joints=pose_joints(estimated_poses[center_index]),
            origin=center_tracking_frame.origin,
            basis=center_tracking_frame.basis,
        )
        for name in TARGET_NAMES:
            row[name] = target.values[name]
            row[f"{name}_mask"] = target.masks[name]
            row[f"heuristic_{name}"] = heuristic.values[name]
            row[f"heuristic_{name}_mask"] = heuristic.masks[name]
        fk_target = fk_targets[center_index]
        for name in FK_TARGET_NAMES:
            row[name] = fk_target.values[name]
            row[f"{name}_mask"] = fk_target.masks[name]
        rows.append(row)
    timings["row_seconds"] = perf_counter() - row_start
    return rows, timings


def _add_decoded_reference_points(
    row: dict[str, object],
    *,
    prefix: str,
    joints: Mapping[str, Transform],
    origin: Vector3,
    basis: Basis3,
) -> None:
    for joint_name in DECODED_REFERENCE_JOINTS:
        transform = joints.get(joint_name)
        if transform is None and joint_name == "neck":
            transform = _synthetic_neck_transform(joints)
        valid = transform is not None
        position = (
            local_position(transform.position, origin=origin, basis=basis)
            if transform is not None
            else (0.0, 0.0, 0.0)
        )
        for component, value in zip(("x", "y", "z"), position, strict=True):
            row[f"{prefix}_{joint_name}_position_{component}"] = float(value)
        row[f"{prefix}_{joint_name}_valid"] = 1.0 if valid else 0.0


def _add_runtime_decoder_context_points(
    row: dict[str, object],
    *,
    prefix: str,
    joints: Mapping[str, Transform],
    origin: Vector3,
    basis: Basis3,
) -> None:
    for joint_name in RUNTIME_DECODER_CONTEXT_JOINTS:
        transform = joints.get(joint_name)
        if transform is None and joint_name == "neck":
            transform = _synthetic_neck_transform(joints)
        valid = transform is not None
        position = (
            local_position(transform.position, origin=origin, basis=basis)
            if transform is not None
            else (0.0, 0.0, 0.0)
        )
        for component, value in zip(("x", "y", "z"), position, strict=True):
            row[f"{prefix}_{joint_name}_position_{component}"] = float(value)
        row[f"{prefix}_{joint_name}_valid"] = 1.0 if valid else 0.0


def _synthetic_neck_transform(joints: Mapping[str, Transform]) -> Transform | None:
    head = joints.get("head")
    chest = joints.get("chest")
    if head is None or chest is None:
        return None
    return Transform(
        position=lerp(chest.position, head.position, 0.7),
        rotation=head.rotation,
    )


@dataclass(frozen=True, slots=True)
class SparseTrackerFeatureCache:
    timestamps: tuple[float, ...]
    center_origin_positions: tuple[Vector3, ...]
    positions: Mapping[TrackerRole, tuple[Vector3, ...]]
    rotations: Mapping[TrackerRole, tuple[tuple[float, ...], ...]]
    valid: Mapping[TrackerRole, tuple[float, ...]]


def precompute_sparse_tracker_features(
    samples: Sequence[TrackedPoseSample],
    basis: Basis3,
) -> SparseTrackerFeatureCache:
    positions: dict[TrackerRole, list[Vector3]] = {role: [] for role in TRACKER_ROLES}
    rotations: dict[TrackerRole, list[tuple[float, ...]]] = {role: [] for role in TRACKER_ROLES}
    valid: dict[TrackerRole, list[float]] = {role: [] for role in TRACKER_ROLES}
    timestamps: list[float] = []
    center_origin_positions: list[Vector3] = []
    for sample in samples:
        timestamps.append(sample.timestamp_seconds)
        center_origin_positions.append(_basis_position(sample.hmd.transform.position, basis))
        for role in TRACKER_ROLES:
            state = _state_for_role(sample, role)
            positions[role].append(_basis_position(state.transform.position, basis))
            rotations[role].append(rotation6d(state.transform.rotation, basis=basis))
            valid[role].append(1.0 if state.valid else 0.0)
    return SparseTrackerFeatureCache(
        timestamps=tuple(timestamps),
        center_origin_positions=tuple(center_origin_positions),
        positions={role: tuple(values) for role, values in positions.items()},
        rotations={role: tuple(values) for role, values in rotations.items()},
        valid={role: tuple(values) for role, values in valid.items()},
    )


def sparse_window_features_from_cache(
    cache: SparseTrackerFeatureCache,
    center_index: int,
    window_size: int,
    *,
    calibration: AvatarCalibration | None = None,
) -> list[float]:
    start_index = center_index - window_size + 1
    if start_index < 0:
        raise ValueError("Sparse windows are causal and require a complete history")
    center_origin = cache.center_origin_positions[center_index]
    features: list[float] = []
    previous_by_role: dict[TrackerRole, tuple[Vector3, tuple[float, ...], float]] = {}
    for sample_index in range(start_index, center_index + 1):
        timestamp = cache.timestamps[sample_index]
        for role in TRACKER_ROLES:
            position = _relative_cached_position(
                cache.positions[role][sample_index],
                center_origin,
            )
            rotation = cache.rotations[role][sample_index]
            previous = previous_by_role.get(role)
            if previous is None:
                linear_velocity = (0.0, 0.0, 0.0)
                rotation_velocity = (0.0,) * 6
            else:
                previous_position, previous_rotation, previous_timestamp = previous
                dt = max(1e-6, timestamp - previous_timestamp)
                linear_velocity = tuple(
                    (position[index] - previous_position[index]) / dt for index in range(3)
                )
                rotation_velocity = tuple(
                    (rotation[index] - previous_rotation[index]) / dt for index in range(6)
                )
            features.extend(position)
            features.extend(rotation)
            features.extend(linear_velocity)
            features.extend(rotation_velocity)
            features.append(cache.valid[role][sample_index])
            previous_by_role[role] = (position, rotation, timestamp)
    features.extend(calibration_feature_values(calibration))
    return [float(value) if isfinite(float(value)) else 0.0 for value in features]


def sparse_window_features(
    samples: Sequence[TrackedPoseSample],
    center_index: int,
    window_size: int,
    *,
    calibration: AvatarCalibration | None = None,
    tracking_space: TrackingSpace = "head_yaw",
    tracking_frame: TrackingFrame | None = None,
) -> list[float]:
    center_sample = samples[center_index]
    frame = tracking_frame or tracking_frame_for_sample(
        center_sample,
        tracking_space=tracking_space,
    )
    basis = frame.basis
    origin = frame.origin
    start_index = center_index - window_size + 1
    if start_index < 0:
        raise ValueError("Sparse windows are causal and require a complete history")
    features: list[float] = []
    previous_by_role: dict[TrackerRole, tuple[Vector3, tuple[float, ...], float]] = {}
    for sample_index in range(start_index, center_index + 1):
        sample = samples[sample_index]
        timestamp = sample.timestamp_seconds
        for role in TRACKER_ROLES:
            state = _state_for_role(sample, role)
            position = local_position(state.transform.position, origin=origin, basis=basis)
            rotation = rotation6d(state.transform.rotation, basis=basis)
            previous = previous_by_role.get(role)
            if previous is None:
                linear_velocity = (0.0, 0.0, 0.0)
                rotation_velocity = (0.0,) * 6
            else:
                previous_position, previous_rotation, previous_timestamp = previous
                dt = max(1e-6, timestamp - previous_timestamp)
                linear_velocity = tuple(
                    (position[index] - previous_position[index]) / dt for index in range(3)
                )
                rotation_velocity = tuple(
                    (rotation[index] - previous_rotation[index]) / dt for index in range(6)
                )
            features.extend(position)
            features.extend(rotation)
            features.extend(linear_velocity)
            features.extend(rotation_velocity)
            features.append(1.0 if state.valid else 0.0)
            previous_by_role[role] = (position, rotation, timestamp)
    features.extend(calibration_feature_values(calibration))
    return [float(value) if isfinite(float(value)) else 0.0 for value in features]


def pose_joints(pose: HumanoidPoseFrame) -> dict[str, Transform]:
    return {name: joint.transform for name, joint in pose.joints.items()}


def _estimated_poses(
    samples: Sequence[TrackedPoseSample],
    *,
    calibration: AvatarCalibration,
    grounding: GroundingContext,
    estimator_name: EstimatorName,
) -> list[HumanoidPoseFrame]:
    estimator = create_estimator(estimator_name)
    return [estimator.estimate(sample, calibration, grounding) for sample in samples]


def _tracked_sample_from_cached_frame(frame: CachedPoseFrame) -> TrackedPoseSample:
    return TrackedPoseSample(
        timestamp_seconds=frame.timestamp_s,
        hmd=_tracker_state("head", frame),
        left_hand=_tracker_state("left_hand", frame),
        right_hand=_tracker_state("right_hand", frame),
    )


def _tracker_state(role: TrackerRole, frame: CachedPoseFrame) -> TrackerState:
    attachment = SyntheticTrackerConfig().attachments[role]
    source = frame.joints[attachment.source_joint]
    rotation = (
        head_tracker_rotation_from_pose_frame(frame.joints) if role == "head" else source.rotation
    )
    return TrackerState(
        role=role,
        transform=Transform(
            position=add(source.position, attachment.local_transform.position),
            rotation=rotation,
        ),
        confidence=attachment.confidence,
        attachment=attachment,
    )


def _yaw_rotated_tracker(state: TrackerState, degrees: float) -> TrackerState:
    return TrackerState(
        role=state.role,
        transform=yaw_rotated_transform(state.transform, degrees),
        confidence=state.confidence,
        valid=state.valid,
        attachment=state.attachment,
    )


def _state_for_role(sample: TrackedPoseSample, role: TrackerRole) -> TrackerState:
    if role == "head":
        return sample.hmd
    if role == "left_hand":
        return sample.left_hand
    if role == "right_hand":
        return sample.right_hand
    return sample.optional_trackers[role]


def _clean_clip(frame: CachedPoseFrame) -> bool:
    tags = set(frame.environment_tags) | set(frame.activity_tags)
    return not (tags & EXCLUDED_TAGS)


def _empty_windows_frame(feature_names: Sequence[str]) -> pl.DataFrame:
    schema: dict[str, Any] = {
        "subject": pl.String,
        "clip": pl.String,
        "frame_index": pl.Int64,
        "timestamp_s": pl.Float32,
        "center_row_index": pl.Int64,
        "split": pl.String,
        "features": pl.List(pl.Float32),
        "environment_split": pl.String,
        "activity_class": pl.String,
        "motion_description": pl.String,
    }
    del feature_names
    for name in TARGET_NAMES:
        schema[name] = pl.Float32
        schema[f"{name}_mask"] = pl.Float32
        schema[f"heuristic_{name}"] = pl.Float32
        schema[f"heuristic_{name}_mask"] = pl.Float32
    for name in FK_TARGET_NAMES:
        schema[name] = pl.Float32
        schema[f"{name}_mask"] = pl.Float32
    for prefix in ("reference", "heuristic"):
        for joint_name in DECODED_REFERENCE_JOINTS:
            for component in ("x", "y", "z"):
                schema[f"{prefix}_{joint_name}_position_{component}"] = pl.Float32
            schema[f"{prefix}_{joint_name}_valid"] = pl.Float32
    for joint_name in RUNTIME_DECODER_CONTEXT_JOINTS:
        for component in ("x", "y", "z"):
            schema[f"heuristic_context_{joint_name}_position_{component}"] = pl.Float32
        schema[f"heuristic_context_{joint_name}_valid"] = pl.Float32
    return pl.DataFrame(schema=schema)


def _enabled(
    values: Mapping[str, float],
    masks: Mapping[str, float],
    name: str,
    min_mask: float,
) -> bool:
    return name in values and masks.get(name, 1.0) >= min_mask


def _fallback_bend(direction: Vector3) -> Vector3:
    horizontal_right = normalize(cross(direction, WORLD_UP), (1.0, 0.0, 0.0))
    return normalize(cross(horizontal_right, direction), WORLD_DOWN)


def _calibrated_length(
    calibration: AvatarCalibration,
    names: Iterable[str],
    fallback: float,
) -> float:
    for name in names:
        value = calibration.bone_lengths_m.get(name)
        if value is not None and value > 0.0:
            return value
    return fallback


def _length(vector: Vector3) -> float:
    return sqrt(dot(vector, vector))


def _import_torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("torch is not installed; run through `uv run --group ml ...`") from exc
    return torch


def _torch_device(torch: Any, *, allow_cpu: bool) -> Any:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if allow_cpu:
        return torch.device("cpu")
    raise RuntimeError(
        "Torch CUDA is unavailable. Install the ml group with the CUDA torch wheel, "
        "or pass allow_cpu=True only for local smoke tests."
    )


def _make_torch_provider_model(
    torch: Any,
    *,
    model_name: str,
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    feature_names: Sequence[str],
) -> Any:
    return make_torch_upper_body_model(
        torch,
        model_name=model_name,
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        feature_names=feature_names,
    )
