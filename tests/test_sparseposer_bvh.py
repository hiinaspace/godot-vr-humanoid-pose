from __future__ import annotations

from typing import cast
from zipfile import ZipFile

import polars as pl
import pytest

from validation.scripts.build_sparseposer_parquet_cache import build_cache
from validation.vr_pose_validation.parquet_cache import iter_cached_pose_frames
from validation.vr_pose_validation.sparseposer_bvh import (
    SPARSEPOSER_DATASET,
    canonical_pose_from_sparseposer,
    iter_sparseposer_pose_frames,
    parse_sparseposer_bvh_text,
    sparseposer_clip_identity,
)


def test_sparseposer_clip_identity_preserves_split_and_subject() -> None:
    identity = sparseposer_clip_identity("data/xsens/eval/S01_A09_3.bvh")

    assert identity.dataset_split == "eval"
    assert identity.subject == "S01"
    assert identity.source_clip == "S01_A09_3"
    assert identity.clip == "eval_S01_A09_3"


def test_sparseposer_bvh_parser_solves_zero_rotation_chain() -> None:
    motion = parse_sparseposer_bvh_text(_minimal_sparseposer_bvh())

    frames = list(iter_sparseposer_pose_frames(motion))
    canonical = canonical_pose_from_sparseposer(frames[0][2])

    assert motion.source_frame_count == 1
    assert motion.channel_count == 69
    assert frames[0][0] == 0
    assert frames[0][1] == 0.0
    assert canonical["hips"].position == (0.0, 1.0, 0.0)
    assert canonical["head"].position == (0.0, 2.0, 0.0)
    assert canonical["left_hand"].position == (0.35, 1.8, 0.0)
    assert canonical["right_foot"].position == pytest.approx((-0.1, 0.2, 0.0))


def test_sparseposer_parquet_cache_builds_canonical_clip(tmp_path) -> None:
    archive_path = tmp_path / "sparseposer.zip"
    cache_dir = tmp_path / "cache"
    with ZipFile(archive_path, "w") as zf:
        zf.writestr("data/xsens/eval/S01_A09_3.bvh", _minimal_sparseposer_bvh())

    result = build_cache(
        archive_path=archive_path,
        input_dir=None,
        cache_dir=cache_dir,
        splits={"eval"},
        subjects=set(),
        dataset_split=None,
        filename_prefix_to_strip="",
        max_clips=None,
        frame_stride=1,
        max_frames_per_clip=None,
        force=True,
        compression="zstd",
    )
    summary = cast(dict[str, object], result["summary"])
    path = cache_dir / "subject=S01" / "eval_S01_A09_3.parquet"

    assert summary["dataset"] == SPARSEPOSER_DATASET
    assert summary["clips"] == 1
    assert summary["cached_frames"] == 1
    assert path.exists()

    dataframe = pl.read_parquet(path)
    assert dataframe["dataset"].to_list() == [SPARSEPOSER_DATASET]
    assert dataframe["dataset_split"].to_list() == ["eval"]

    frames = list(iter_cached_pose_frames(path))
    assert frames[0].dataset == SPARSEPOSER_DATASET
    assert frames[0].dataset_split == "eval"
    assert frames[0].subject == "S01"
    assert frames[0].clip == "eval_S01_A09_3"
    assert frames[0].joints["head"].position == (0.0, 2.0, 0.0)


def _minimal_sparseposer_bvh() -> str:
    frame_values = " ".join(str(value) for value in ([0, 1, 0, 0, 0, 0] + [0] * 63))
    return f"""HIERARCHY
ROOT Hips
{{
    OFFSET 0 0 0
    CHANNELS 6 Xposition Yposition Zposition Yrotation Xrotation Zrotation
    JOINT LeftHip
    {{
        OFFSET 0.1 0 0
        CHANNELS 3 Yrotation Xrotation Zrotation
        JOINT LeftKnee
        {{
            OFFSET 0 -0.4 0
            CHANNELS 3 Yrotation Xrotation Zrotation
            JOINT LeftAnkle
            {{
                OFFSET 0 -0.4 0
                CHANNELS 3 Yrotation Xrotation Zrotation
                JOINT LeftToe
                {{
                    OFFSET 0 0 0.1
                    CHANNELS 3 Yrotation Xrotation Zrotation
                    End Site
                    {{
                        OFFSET 0 0 0
                    }}
                }}
            }}
        }}
    }}
    JOINT RightHip
    {{
        OFFSET -0.1 0 0
        CHANNELS 3 Yrotation Xrotation Zrotation
        JOINT RightKnee
        {{
            OFFSET 0 -0.4 0
            CHANNELS 3 Yrotation Xrotation Zrotation
            JOINT RightAnkle
            {{
                OFFSET 0 -0.4 0
                CHANNELS 3 Yrotation Xrotation Zrotation
                JOINT RightToe
                {{
                    OFFSET 0 0 0.1
                    CHANNELS 3 Yrotation Xrotation Zrotation
                    End Site
                    {{
                        OFFSET 0 0 0
                    }}
                }}
            }}
        }}
    }}
    JOINT Chest
    {{
        OFFSET 0 0.25 0
        CHANNELS 3 Yrotation Xrotation Zrotation
        JOINT Chest3
        {{
            OFFSET 0 0.25 0
            CHANNELS 3 Yrotation Xrotation Zrotation
            JOINT Chest4
            {{
                OFFSET 0 0.25 0
                CHANNELS 3 Yrotation Xrotation Zrotation
                JOINT Neck
                {{
                    OFFSET 0 0.15 0
                    CHANNELS 3 Yrotation Xrotation Zrotation
                    JOINT Head
                    {{
                        OFFSET 0 0.1 0
                        CHANNELS 3 Yrotation Xrotation Zrotation
                        End Site
                        {{
                            OFFSET 0 0 0
                        }}
                    }}
                }}
                JOINT LeftCollar
                {{
                    OFFSET 0.05 0.05 0
                    CHANNELS 3 Yrotation Xrotation Zrotation
                    JOINT LeftShoulder
                    {{
                        OFFSET 0.1 0 0
                        CHANNELS 3 Yrotation Xrotation Zrotation
                        JOINT LeftElbow
                        {{
                            OFFSET 0.1 0 0
                            CHANNELS 3 Yrotation Xrotation Zrotation
                            JOINT LeftWrist
                            {{
                                OFFSET 0.1 0 0
                                CHANNELS 3 Yrotation Xrotation Zrotation
                                End Site
                                {{
                                    OFFSET 0 0 0
                                }}
                            }}
                        }}
                    }}
                }}
                JOINT RightCollar
                {{
                    OFFSET -0.05 0.05 0
                    CHANNELS 3 Yrotation Xrotation Zrotation
                    JOINT RightShoulder
                    {{
                        OFFSET -0.1 0 0
                        CHANNELS 3 Yrotation Xrotation Zrotation
                        JOINT RightElbow
                        {{
                            OFFSET -0.1 0 0
                            CHANNELS 3 Yrotation Xrotation Zrotation
                            JOINT RightWrist
                            {{
                                OFFSET -0.1 0 0
                                CHANNELS 3 Yrotation Xrotation Zrotation
                                End Site
                                {{
                                    OFFSET 0 0 0
                                }}
                            }}
                        }}
                    }}
                }}
            }}
        }}
    }}
}}
MOTION
Frames: 1
Frame Time: 0.016667
{frame_values}
"""
