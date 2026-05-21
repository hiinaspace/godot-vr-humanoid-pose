from __future__ import annotations

from typing import cast

from validation.scripts.run_cached_validation import run_cached_validation
from validation.vr_pose_validation.geometry import Transform
from validation.vr_pose_validation.parquet_cache import (
    CACHE_JOINTS,
    append_cached_pose_frame,
    filter_clip_cache_files_by_environment,
    iter_cached_pose_frames,
    new_cache_columns,
    write_clip_cache,
)


def test_parquet_cache_round_trips_canonical_wide_columns(tmp_path) -> None:
    path = tmp_path / "subject=01" / "01_01.parquet"
    columns = new_cache_columns()
    joints = {
        joint: Transform(
            position=(1.0, 2.0, 3.0),
            rotation=(0.0, 0.0, 0.0, 1.0),
        )
        for joint in CACHE_JOINTS
    }
    append_cached_pose_frame(
        columns,
        subject="01",
        clip="01_01",
        frame_index=1,
        timestamp_s=0.0,
        joints=joints,
        environment_split="nonflat",
        environment_tags=("stairs",),
        motion_description="Walking up and down stairs",
    )

    write_clip_cache(path, columns)
    frames = list(iter_cached_pose_frames(path))

    assert len(frames) == 1
    assert frames[0].subject == "01"
    assert frames[0].clip == "01_01"
    assert frames[0].frame_index == 1
    assert frames[0].joints["head"].position == (1.0, 2.0, 3.0)
    assert frames[0].environment_split == "nonflat"
    assert frames[0].environment_tags == ("stairs",)
    assert frames[0].motion_description == "Walking up and down stairs"


def test_cached_validation_runs_one_clip_with_stateful_worker_boundary(tmp_path) -> None:
    cache_dir = tmp_path / "cache"
    path = cache_dir / "subject=01" / "01_01.parquet"
    columns = new_cache_columns()
    for frame_index in (1, 2):
        joints = {
            joint: Transform(
                position=(float(frame_index), 2.0, 3.0),
                rotation=(0.0, 0.0, 0.0, 1.0),
            )
            for joint in CACHE_JOINTS
        }
        append_cached_pose_frame(
            columns,
            subject="01",
            clip="01_01",
            frame_index=frame_index,
            timestamp_s=(frame_index - 1) / 120.0,
            joints=joints,
        )
    write_clip_cache(path, columns)

    result = run_cached_validation(
        cache_dir=cache_dir,
        subjects=set(),
        max_clips=None,
        workers=1,
    )
    summary = cast(dict[str, object], result["summary"])

    assert summary["clips"] == 1
    assert summary["evaluated_frames"] == 2
    assert summary["workers"] == 1


def test_cache_file_environment_filter_reads_first_row_metadata(tmp_path) -> None:
    cache_dir = tmp_path / "cache"
    flat_path = cache_dir / "subject=01" / "01_01.parquet"
    nonflat_path = cache_dir / "subject=01" / "01_02.parquet"
    joints = {
        joint: Transform(
            position=(1.0, 2.0, 3.0),
            rotation=(0.0, 0.0, 0.0, 1.0),
        )
        for joint in CACHE_JOINTS
    }
    for path, split in ((flat_path, "flat"), (nonflat_path, "nonflat")):
        columns = new_cache_columns()
        append_cached_pose_frame(
            columns,
            subject="01",
            clip=path.stem,
            frame_index=1,
            timestamp_s=0.0,
            joints=joints,
            environment_split=split,
        )
        write_clip_cache(path, columns)

    paths = filter_clip_cache_files_by_environment(
        [flat_path, nonflat_path],
        environment_splits={"nonflat"},
    )

    assert paths == [nonflat_path]
