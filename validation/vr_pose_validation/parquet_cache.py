"""Parquet cache helpers for flat CMU-derived validation frames."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import polars as pl

from validation.vr_pose_validation.cmu_fk import CANONICAL_CMU_JOINTS
from validation.vr_pose_validation.cmu_metadata import ActivityClass, EnvironmentSplit
from validation.vr_pose_validation.geometry import Transform

CACHE_JOINTS = tuple(CANONICAL_CMU_JOINTS)
TRANSFORM_COMPONENTS = ("px", "py", "pz", "qx", "qy", "qz", "qw")
METADATA_COLUMNS = (
    "environment_split",
    "environment_tags",
    "activity_class",
    "activity_tags",
    "motion_description",
    "subject_description",
)
type CacheColumns = dict[str, list[Any]]
type ParquetCompression = Literal["lz4", "uncompressed", "snappy", "gzip", "brotli", "zstd"]


@dataclass(frozen=True, slots=True)
class CachedPoseFrame:
    dataset: str
    dataset_split: str
    subject: str
    clip: str
    frame_index: int
    timestamp_s: float
    joints: dict[str, Transform]
    environment_split: EnvironmentSplit = "unknown"
    environment_tags: tuple[str, ...] = ()
    activity_class: ActivityClass = "unknown"
    activity_tags: tuple[str, ...] = ()
    motion_description: str = ""
    subject_description: str = ""


def clip_cache_path(cache_dir: Path, subject: str, clip: str) -> Path:
    return cache_dir / f"subject={subject}" / f"{clip}.parquet"


def list_clip_cache_files(cache_dir: Path, subjects: set[str] | None = None) -> list[Path]:
    paths = sorted(cache_dir.glob("subject=*/*.parquet"))
    if not subjects:
        return paths
    return [path for path in paths if subject_from_cache_path(path) in subjects]


def filter_clip_cache_files_by_environment(
    paths: Iterable[Path],
    *,
    environment_splits: set[EnvironmentSplit] | None = None,
    activity_classes: set[ActivityClass] | None = None,
) -> list[Path]:
    if not environment_splits and not activity_classes:
        return list(paths)
    filtered_paths = []
    for path in paths:
        metadata = _clip_environment_from_cache(path)
        if environment_splits and metadata.environment_split not in environment_splits:
            continue
        if activity_classes and metadata.activity_class not in activity_classes:
            continue
        filtered_paths.append(path)
    return filtered_paths


def subject_from_cache_path(path: Path) -> str:
    parent = path.parent.name
    if parent.startswith("subject="):
        return parent.split("=", maxsplit=1)[1]
    return parent


def clip_from_cache_path(path: Path) -> str:
    return path.stem


def new_cache_columns() -> CacheColumns:
    columns: CacheColumns = {
        "dataset": [],
        "dataset_split": [],
        "subject": [],
        "clip": [],
        "frame_index": [],
        "timestamp_s": [],
        "environment_split": [],
        "environment_tags": [],
        "activity_class": [],
        "activity_tags": [],
        "motion_description": [],
        "subject_description": [],
    }
    for joint in CACHE_JOINTS:
        for component in TRANSFORM_COMPONENTS:
            columns[_transform_column(joint, component)] = []
    return columns


def append_cached_pose_frame(
    columns: CacheColumns,
    *,
    dataset: str = "cmu",
    dataset_split: str = "",
    subject: str,
    clip: str,
    frame_index: int,
    timestamp_s: float,
    joints: Mapping[str, Transform],
    environment_split: EnvironmentSplit = "unknown",
    environment_tags: tuple[str, ...] = (),
    activity_class: ActivityClass = "unknown",
    activity_tags: tuple[str, ...] = (),
    motion_description: str = "",
    subject_description: str = "",
) -> None:
    _append(columns, "dataset", dataset)
    _append(columns, "dataset_split", dataset_split)
    _append(columns, "subject", subject)
    _append(columns, "clip", clip)
    _append(columns, "frame_index", frame_index)
    _append(columns, "timestamp_s", timestamp_s)
    _append(columns, "environment_split", environment_split)
    _append(columns, "environment_tags", ";".join(environment_tags))
    _append(columns, "activity_class", activity_class)
    _append(columns, "activity_tags", ";".join(activity_tags))
    _append(columns, "motion_description", motion_description)
    _append(columns, "subject_description", subject_description)
    for joint in CACHE_JOINTS:
        transform = joints[joint]
        position = transform.position
        rotation = transform.rotation
        values = (
            position[0],
            position[1],
            position[2],
            rotation[0],
            rotation[1],
            rotation[2],
            rotation[3],
        )
        for component, value in zip(TRANSFORM_COMPONENTS, values, strict=True):
            _append(columns, _transform_column(joint, component), value)


def write_clip_cache(
    path: Path,
    columns: CacheColumns,
    *,
    compression: ParquetCompression = "zstd",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe = pl.DataFrame(columns)
    dataframe.write_parquet(path, compression=compression)


def iter_cached_pose_frames(path: Path) -> Iterator[CachedPoseFrame]:
    dataframe = pl.read_parquet(path)
    yield from iter_cached_pose_dataframe(dataframe)


def iter_cached_pose_dataframe(dataframe: pl.DataFrame) -> Iterator[CachedPoseFrame]:
    columns = dataframe.to_dict(as_series=False)
    row_count = dataframe.height
    available_joints = [
        joint
        for joint in CACHE_JOINTS
        if all(_transform_column(joint, component) in columns for component in TRANSFORM_COMPONENTS)
    ]
    for row_index in range(row_count):
        joints: dict[str, Transform] = {}
        for joint in available_joints:
            joints[joint] = Transform(
                position=(
                    float(columns[_transform_column(joint, "px")][row_index]),
                    float(columns[_transform_column(joint, "py")][row_index]),
                    float(columns[_transform_column(joint, "pz")][row_index]),
                ),
                rotation=(
                    float(columns[_transform_column(joint, "qx")][row_index]),
                    float(columns[_transform_column(joint, "qy")][row_index]),
                    float(columns[_transform_column(joint, "qz")][row_index]),
                    float(columns[_transform_column(joint, "qw")][row_index]),
                ),
            )
        yield CachedPoseFrame(
            dataset=_string_from_columns(columns, "dataset", row_index, fallback="cmu"),
            dataset_split=_string_from_columns(columns, "dataset_split", row_index),
            subject=str(columns["subject"][row_index]),
            clip=str(columns["clip"][row_index]),
            frame_index=int(columns["frame_index"][row_index]),
            timestamp_s=float(columns["timestamp_s"][row_index]),
            joints=joints,
            environment_split=_environment_split_from_columns(columns, row_index),
            environment_tags=_environment_tags_from_columns(columns, row_index),
            activity_class=_activity_class_from_columns(columns, row_index),
            activity_tags=_metadata_tags_from_columns(columns, "activity_tags", row_index),
            motion_description=_string_from_columns(columns, "motion_description", row_index),
            subject_description=_string_from_columns(columns, "subject_description", row_index),
        )


def cache_summary(paths: Iterable[Path]) -> pl.DataFrame:
    rows = [
        {
            "subject": subject_from_cache_path(path),
            "clip": clip_from_cache_path(path),
            "path": path.as_posix(),
        }
        for path in paths
    ]
    if not rows:
        return pl.DataFrame(
            schema={
                "subject": pl.String,
                "clip": pl.String,
                "path": pl.String,
            }
        )
    return pl.DataFrame(rows)


def _transform_column(joint: str, component: str) -> str:
    return f"{joint}_{component}"


def _append(
    columns: CacheColumns,
    column: str,
    value: float | int | str,
) -> None:
    columns[column].append(value)


def _clip_environment_from_cache(path: Path) -> CachedPoseFrame:
    frame = pl.read_parquet(path, n_rows=1)
    return next(iter_cached_pose_dataframe(frame))


def _environment_split_from_columns(columns: CacheColumns, row_index: int) -> EnvironmentSplit:
    value = _string_from_columns(columns, "environment_split", row_index)
    if value == "flat":
        return "flat"
    if value == "nonflat":
        return "nonflat"
    return "unknown"


def _environment_tags_from_columns(columns: CacheColumns, row_index: int) -> tuple[str, ...]:
    return _metadata_tags_from_columns(columns, "environment_tags", row_index)


def _metadata_tags_from_columns(
    columns: CacheColumns,
    column: str,
    row_index: int,
) -> tuple[str, ...]:
    value = _string_from_columns(columns, column, row_index)
    if not value:
        return ()
    return tuple(tag for tag in value.split(";") if tag)


def _activity_class_from_columns(columns: CacheColumns, row_index: int) -> ActivityClass:
    value = _string_from_columns(columns, "activity_class", row_index)
    if value == "idle_standing":
        return "idle_standing"
    if value == "sitting":
        return "sitting"
    if value == "stationary_upper_body":
        return "stationary_upper_body"
    if value == "locomotion":
        return "locomotion"
    if value == "dance_exercise":
        return "dance_exercise"
    if value == "swimming":
        return "swimming"
    if value == "floor_pose":
        return "floor_pose"
    if value == "sports":
        return "sports"
    return "unknown"


def _string_from_columns(
    columns: CacheColumns,
    column: str,
    row_index: int,
    *,
    fallback: str = "",
) -> str:
    if column not in columns:
        return fallback
    return str(columns[column][row_index])
