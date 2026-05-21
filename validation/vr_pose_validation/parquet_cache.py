"""Parquet cache helpers for flat CMU-derived validation frames."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import polars as pl

from validation.vr_pose_validation.cmu_fk import CANONICAL_CMU_JOINTS
from validation.vr_pose_validation.geometry import Transform

CACHE_JOINTS = tuple(CANONICAL_CMU_JOINTS)
TRANSFORM_COMPONENTS = ("px", "py", "pz", "qx", "qy", "qz", "qw")
type CacheColumns = dict[str, list[Any]]
type ParquetCompression = Literal["lz4", "uncompressed", "snappy", "gzip", "brotli", "zstd"]


@dataclass(frozen=True, slots=True)
class CachedPoseFrame:
    subject: str
    clip: str
    frame_index: int
    timestamp_s: float
    joints: dict[str, Transform]


def clip_cache_path(cache_dir: Path, subject: str, clip: str) -> Path:
    return cache_dir / f"subject={subject}" / f"{clip}.parquet"


def list_clip_cache_files(cache_dir: Path, subjects: set[str] | None = None) -> list[Path]:
    paths = sorted(cache_dir.glob("subject=*/*.parquet"))
    if not subjects:
        return paths
    return [path for path in paths if subject_from_cache_path(path) in subjects]


def subject_from_cache_path(path: Path) -> str:
    parent = path.parent.name
    if parent.startswith("subject="):
        return parent.split("=", maxsplit=1)[1]
    return parent


def clip_from_cache_path(path: Path) -> str:
    return path.stem


def new_cache_columns() -> CacheColumns:
    columns: CacheColumns = {
        "subject": [],
        "clip": [],
        "frame_index": [],
        "timestamp_s": [],
    }
    for joint in CACHE_JOINTS:
        for component in TRANSFORM_COMPONENTS:
            columns[_transform_column(joint, component)] = []
    return columns


def append_cached_pose_frame(
    columns: CacheColumns,
    *,
    subject: str,
    clip: str,
    frame_index: int,
    timestamp_s: float,
    joints: Mapping[str, Transform],
) -> None:
    _append(columns, "subject", subject)
    _append(columns, "clip", clip)
    _append(columns, "frame_index", frame_index)
    _append(columns, "timestamp_s", timestamp_s)
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
    for row_index in range(row_count):
        joints: dict[str, Transform] = {}
        for joint in CACHE_JOINTS:
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
            subject=str(columns["subject"][row_index]),
            clip=str(columns["clip"][row_index]),
            frame_index=int(columns["frame_index"][row_index]),
            timestamp_s=float(columns["timestamp_s"][row_index]),
            joints=joints,
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
