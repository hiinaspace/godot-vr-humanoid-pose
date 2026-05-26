"""Sparse-window features and upper-body latent targets for ML experiments."""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from hashlib import sha1
from math import isfinite, sqrt
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, Protocol

import polars as pl

from validation.vr_pose_validation.baseline import EstimatorName, create_estimator
from validation.vr_pose_validation.calibration import calibration_from_reference_frames
from validation.vr_pose_validation.cmu_metadata import ActivityClass, EnvironmentSplit
from validation.vr_pose_validation.geometry import (
    Transform,
    Vector3,
    add,
    axis_name_quaternion,
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
STATIC_CALIBRATION_FEATURE_NAMES = (
    "calibration_human_height_m",
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
class SparseWindowConfig:
    """Configuration for causal sparse-tracker ML windows."""

    window_size: int = 40
    stride: int = 4
    estimator_name: EstimatorName = "basis_research"
    holdout_folds: int = 5
    environment_splits: tuple[EnvironmentSplit, ...] = ("flat",)
    activity_classes: tuple[ActivityClass, ...] = DEFAULT_CLEAN_ACTIVITY_CLASSES
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
    workers: int
    shards: int
    total_seconds: float


@dataclass(frozen=True, slots=True)
class UpperBodyLatents:
    values: dict[str, float]
    masks: dict[str, float]


@dataclass(frozen=True, slots=True)
class UpperBodyLatentPrediction:
    values: Mapping[str, float]
    masks: Mapping[str, float]


class LatentUpperBodyProvider(Protocol):
    def predict(
        self,
        sample: TrackedPoseSample,
        calibration: AvatarCalibration,
        grounding: GroundingContext,
    ) -> UpperBodyLatentPrediction:
        """Predict upper-body control latents for the current sparse tracker sample."""


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

    def __post_init__(self) -> None:
        torch = _import_torch()
        self._torch = torch
        self._device = _torch_device(torch, allow_cpu=self.allow_cpu)
        artifact = torch.load(self.artifact_path, map_location=self._device, weights_only=False)
        self._target_names = [str(name) for name in artifact["target_names"]]
        self._prediction_mode = str(artifact.get("prediction_mode", "absolute"))
        self._model = _make_torch_provider_model(
            torch,
            model_name=str(artifact["model_name"]),
            input_dim=len(artifact["feature_names"]),
            output_dim=len(self._target_names),
            hidden_dim=int(artifact.get("hidden_dim", 256)),
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
        if len(self._history) > self.window_size:
            self._history = self._history[-self.window_size :]
        if len(self._history) < self.window_size:
            return UpperBodyLatentPrediction(
                values={name: 0.0 for name in self._target_names},
                masks={name: 0.0 for name in self._target_names},
            )

        features = sparse_window_features(
            self._history,
            len(self._history) - 1,
            self.window_size,
            calibration=calibration,
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
        )

    def reset(self) -> None:
        self._history.clear()

    @property
    def prediction_mode(self) -> str:
        return self._prediction_mode


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
        return apply_upper_body_latents(
            base_pose,
            prediction,
            calibration=calibration,
            min_mask=self.min_mask,
        )


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
    paths = list_clip_cache_files(cache_dir, subjects or None)
    paths = filter_clip_cache_files_by_environment(
        paths,
        environment_splits=set(effective_config.environment_splits),
        activity_classes=set(effective_config.activity_classes),
    )
    if max_clips is not None:
        paths = paths[:max_clips]

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
    if shard_rows > 0 and shard_dir.exists():
        shutil.rmtree(shard_dir)

    for clip_rows in _iter_clip_window_rows(paths, effective_config, feature_names):
        if clip_rows:
            clips_exported += 1
            total_windows += len(clip_rows)
            if shard_rows > 0:
                pending_rows.extend(clip_rows)
                if len(pending_rows) >= shard_rows:
                    shards += _write_window_shard(
                        shard_dir,
                        pending_rows,
                        shard_index=shards,
                    )
                    pending_rows.clear()
            else:
                rows.extend(clip_rows)

    if shard_rows > 0 and pending_rows:
        shards += _write_window_shard(shard_dir, pending_rows, shard_index=shards)
        pending_rows.clear()

    windows_path = output_dir / "windows.parquet"
    if shard_rows <= 0 and rows:
        pl.DataFrame(rows).write_parquet(windows_path, compression="zstd")
    elif total_windows == 0:
        if shard_dir.exists():
            shutil.rmtree(shard_dir)
        shards = 0
        _empty_windows_frame(feature_names).write_parquet(windows_path, compression="zstd")
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
        workers=workers,
        shards=shards,
        total_seconds=perf_counter() - start,
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
) -> Iterable[list[dict[str, object]]]:
    workers = max(1, config.workers)
    if workers == 1:
        for path in paths:
            yield _export_clip_windows(path, config, feature_names)
        return

    jobs = [(path, config, tuple(feature_names)) for path in paths]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        yield from executor.map(_export_clip_windows_job, jobs)


def _export_clip_windows_job(
    job: tuple[Path, SparseWindowConfig, tuple[str, ...]],
) -> list[dict[str, object]]:
    path, config, feature_names = job
    return _export_clip_windows(path, config, feature_names)


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
    pl.DataFrame(rows).write_parquet(shard_path, compression="zstd")
    return 1


def read_sparse_window_dataframe(dataset_dir: Path) -> pl.DataFrame:
    windows_path = dataset_dir / "windows.parquet"
    if windows_path.exists():
        return pl.read_parquet(windows_path)
    shard_dir = dataset_dir / "windows_shards"
    if shard_dir.exists():
        return pl.read_parquet(str(shard_dir / "*.parquet"))
    raise FileNotFoundError(
        f"Expected sparse-window dataset at {dataset_dir}; run build_sparse_window_dataset"
    )


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


def split_for_subject(subject: str, holdout_folds: int) -> Literal["train", "validation"]:
    folds = max(2, holdout_folds)
    digest = sha1(subject.encode("utf-8")).digest()
    fold = int.from_bytes(digest[:4], "big") % folds
    return "validation" if fold == 0 else "train"


def extract_upper_body_latents(
    joints: Mapping[str, Transform],
    calibration: AvatarCalibration | None = None,
) -> UpperBodyLatents:
    values = {name: 0.0 for name in TARGET_NAMES}
    masks = {name: 0.0 for name in TARGET_NAMES}
    head = joints.get("head")
    if head is None:
        return UpperBodyLatents(values=values, masks=masks)

    effective_calibration = calibration or AvatarCalibration()
    height = max(0.1, effective_calibration.human_height_m)
    head_basis = yaw_basis_from_rotation(head.rotation)
    head_position = head.position

    chest = joints.get("chest")
    neck = joints.get("neck") or _synthetic_neck_transform(joints)
    for latent_prefix, transform in (("chest", chest), ("neck", neck)):
        if transform is None:
            continue
        rel = _head_relative_position(
            transform.position,
            origin=head_position,
            basis=head_basis,
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
                origin=head_position,
                basis=head_basis,
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
            basis=head_basis,
            min_distance_m=max(0.025, 0.12 * _upper_arm_length(effective_calibration)),
        )
        for component, value in zip(("x", "y", "z"), pole, strict=True):
            name = f"{side}_elbow_pole_{component}"
            values[name] = value
            masks[name] = 1.0 if valid else 0.0

    return UpperBodyLatents(values=values, masks=masks)


@dataclass(frozen=True, slots=True)
class Basis3:
    right: Vector3
    up: Vector3
    forward: Vector3


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


def calibration_feature_values(calibration: AvatarCalibration | None = None) -> list[float]:
    effective_calibration = calibration or AvatarCalibration()
    human_height = max(0.1, effective_calibration.human_height_m)
    shoulder_width = _shoulder_width(effective_calibration)
    upper_arm = _upper_arm_length(effective_calibration)
    lower_arm = _lower_arm_length(effective_calibration)
    arm_length = max(0.1, upper_arm + lower_arm)
    return [
        human_height,
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
    head_basis = yaw_basis_from_rotation(head_joint.transform.rotation)
    head_position = head_joint.transform.position
    height_m = max(0.1, calibration.human_height_m)

    for joint_name in ("chest", "neck", "left_shoulder", "right_shoulder"):
        if _relative_position_enabled(values, masks, joint_name, min_mask):
            position = _world_from_head_relative(
                _relative_position_values(values, joint_name),
                origin=head_position,
                basis=head_basis,
                height_m=height_m,
            )
            old_joint = joints.get(joint_name, JointPose(Transform()))
            joints[joint_name] = JointPose(
                transform=Transform(position=position, rotation=old_joint.transform.rotation),
                confidence=old_joint.confidence,
                estimated=True,
            )

    chest_rotation = _derived_chest_rotation(joints, fallback_basis=head_basis)
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
                head_basis,
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
) -> list[dict[str, object]]:
    cached_frames = list(iter_cached_pose_frames(path))
    if len(cached_frames) < config.window_size:
        return []
    first_frame = cached_frames[0]
    if not _clean_clip(first_frame):
        return []

    reference_frames = [frame.joints for frame in cached_frames]
    floor_height = infer_reference_floor_height(reference_frames)
    calibration = calibration_from_reference_frames(reference_frames, floor_height_m=floor_height)
    grounding = GroundingContext(floor_height_m=floor_height)
    samples = [_tracked_sample_from_cached_frame(frame) for frame in cached_frames]
    estimated_poses = _estimated_poses(
        samples,
        calibration=calibration,
        grounding=grounding,
        estimator_name=config.estimator_name,
    )
    target_latents = [
        extract_upper_body_latents(frame.joints, calibration) for frame in cached_frames
    ]
    heuristic_latents = [
        extract_upper_body_latents(pose_joints(pose), calibration) for pose in estimated_poses
    ]

    rows: list[dict[str, object]] = []
    for center_index in range(config.window_size - 1, len(cached_frames), config.stride):
        feature_vector = sparse_window_features(
            samples,
            center_index,
            config.window_size,
            calibration=calibration,
        )
        if len(feature_vector) != len(feature_names):
            raise ValueError("Feature vector length does not match feature schema")
        center_sample = samples[center_index]
        center_basis = yaw_basis_from_rotation(center_sample.hmd.transform.rotation)
        center_origin = center_sample.hmd.transform.position
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
            origin=center_origin,
            basis=center_basis,
        )
        _add_decoded_reference_points(
            row,
            prefix="heuristic",
            joints=pose_joints(estimated_poses[center_index]),
            origin=center_origin,
            basis=center_basis,
        )
        _add_runtime_decoder_context_points(
            row,
            prefix="heuristic_context",
            joints=pose_joints(estimated_poses[center_index]),
            origin=center_origin,
            basis=center_basis,
        )
        for name in TARGET_NAMES:
            row[name] = target.values[name]
            row[f"{name}_mask"] = target.masks[name]
            row[f"heuristic_{name}"] = heuristic.values[name]
            row[f"heuristic_{name}_mask"] = heuristic.masks[name]
        rows.append(row)
    return rows


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


def sparse_window_features(
    samples: Sequence[TrackedPoseSample],
    center_index: int,
    window_size: int,
    *,
    calibration: AvatarCalibration | None = None,
) -> list[float]:
    center_sample = samples[center_index]
    basis = yaw_basis_from_rotation(center_sample.hmd.transform.rotation)
    origin = center_sample.hmd.transform.position
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
        "timestamp_s": pl.Float64,
        "center_row_index": pl.Int64,
        "split": pl.String,
        "features": pl.List(pl.Float64),
        "environment_split": pl.String,
        "activity_class": pl.String,
        "motion_description": pl.String,
    }
    del feature_names
    for name in TARGET_NAMES:
        schema[name] = pl.Float64
        schema[f"{name}_mask"] = pl.Float64
        schema[f"heuristic_{name}"] = pl.Float64
        schema[f"heuristic_{name}_mask"] = pl.Float64
    for prefix in ("reference", "heuristic"):
        for joint_name in DECODED_REFERENCE_JOINTS:
            for component in ("x", "y", "z"):
                schema[f"{prefix}_{joint_name}_position_{component}"] = pl.Float64
            schema[f"{prefix}_{joint_name}_valid"] = pl.Float64
    for joint_name in RUNTIME_DECODER_CONTEXT_JOINTS:
        for component in ("x", "y", "z"):
            schema[f"heuristic_context_{joint_name}_position_{component}"] = pl.Float64
        schema[f"heuristic_context_{joint_name}_valid"] = pl.Float64
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
    raise ValueError(f"Unsupported Torch model artifact: {model_name}")
