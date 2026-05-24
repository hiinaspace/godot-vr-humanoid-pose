from __future__ import annotations

from pathlib import Path
from typing import cast

from pytest import approx

from validation.scripts.run_cached_validation import run_cached_validation
from validation.vr_pose_validation.cmu_metadata import ActivityClass
from validation.vr_pose_validation.geometry import Transform
from validation.vr_pose_validation.parquet_cache import (
    CACHE_JOINTS,
    append_cached_pose_frame,
    new_cache_columns,
    write_clip_cache,
)
from validation.vr_pose_validation.posture import (
    classify_reference_posture,
    summarize_reference_posture,
)


def test_reference_posture_classifies_upright_sitting_and_reclined_frames() -> None:
    upright = classify_reference_posture(
        _reference_frame(head=(0.0, 1.6, 0.0), hips=(0.0, 1.0, 0.0)),
        floor_height_m=0.0,
    )
    sitting = classify_reference_posture(
        _reference_frame(head=(0.0, 1.25, 0.0), hips=(0.0, 0.55, 0.0)),
        floor_height_m=0.0,
    )
    reclined = classify_reference_posture(
        _reference_frame(head=(1.0, 0.9, 0.0), hips=(0.0, 0.55, 0.0)),
        floor_height_m=0.0,
    )

    assert upright.posture_class == "upright"
    assert sitting.posture_class == "sitting"
    assert reclined.posture_class == "floor_reclined"


def test_posture_summary_separates_vr_stationary_locomotion_and_ood() -> None:
    stationary = summarize_reference_posture(
        [
            _reference_frame(head=(0.0, 1.6, 0.0), hips=(0.0, 1.0, 0.0)),
            _reference_frame(head=(0.0, 1.6, 0.0), hips=(0.0, 1.0, 0.0)),
        ],
        environment_split="flat",
        activity_class="stationary_upper_body",
    )
    locomotion = summarize_reference_posture(
        [
            _reference_frame(head=(0.0, 1.6, 0.0), hips=(0.0, 1.0, 0.0)),
            _reference_frame(head=(1.0, 1.6, 0.0), hips=(1.0, 1.0, 0.0)),
        ],
        environment_split="flat",
        activity_class="unknown",
        timestamps_seconds=(0.0, 2.0),
    )
    nonflat = summarize_reference_posture(
        [_reference_frame(head=(0.0, 1.6, 0.0), hips=(0.0, 1.0, 0.0))],
        environment_split="nonflat",
        activity_class="locomotion",
    )

    assert stationary.posture_subset == "vr_stationary"
    assert stationary.vr_candidate is True
    assert locomotion.posture_subset == "vr_locomotion"
    assert locomotion.hip_horizontal_displacement_m == approx(1.0)
    assert locomotion.mean_hip_horizontal_speed_m_per_s == approx(0.5)
    assert nonflat.posture_subset == "nonflat_ood"
    assert nonflat.vr_candidate is False


def test_posture_summary_rejects_inverted_head_hip_order() -> None:
    inverted = summarize_reference_posture(
        [
            _reference_frame(head=(0.0, 0.75, 0.0), hips=(0.0, 1.05, 0.0)),
            _reference_frame(head=(0.0, 0.70, 0.0), hips=(0.0, 1.00, 0.0)),
        ],
        environment_split="flat",
        activity_class="stationary_upper_body",
    )

    assert inverted.posture_class == "inverted"
    assert inverted.posture_subset == "inverted_ood"
    assert inverted.vr_candidate is False
    assert inverted.inverted_pct == approx(100.0)


def test_cached_validation_report_groups_posture_slices(tmp_path) -> None:
    cache_dir = tmp_path / "cache"
    _write_cached_clip(
        cache_dir,
        "01_01",
        "stationary_upper_body",
        [
            _reference_frame(head=(0.0, 1.6, 0.0), hips=(0.0, 1.0, 0.0)),
            _reference_frame(head=(0.0, 1.6, 0.0), hips=(0.0, 1.0, 0.0)),
        ],
    )
    _write_cached_clip(
        cache_dir,
        "01_02",
        "floor_pose",
        [
            _reference_frame(head=(1.0, 0.9, 0.0), hips=(0.0, 0.55, 0.0)),
            _reference_frame(head=(1.0, 0.9, 0.0), hips=(0.0, 0.55, 0.0)),
        ],
    )

    result = run_cached_validation(
        cache_dir=cache_dir,
        subjects=set(),
        max_clips=None,
        workers=1,
        estimator_name="basis",
    )
    summary = cast(dict[str, object], result["summary"])
    grouped = cast(dict[str, list[dict[str, object]]], summary["grouped_metrics"])
    posture_frame_counts = cast(dict[str, int], summary["posture_frame_counts"])
    clips = cast(list[dict[str, object]], result["clips"])

    assert summary["posture_subset_counts"] == {
        "floor_reclined_ood": 1,
        "vr_stationary": 1,
    }
    assert summary["posture_vr_candidate_clips"] == 1
    assert posture_frame_counts["inverted"] == 0
    assert grouped["posture_subset"][0]["posture_subset"] == "floor_reclined_ood"
    assert {clip["clip"]: clip["posture_subset"] for clip in clips} == {
        "01_01": "vr_stationary",
        "01_02": "floor_reclined_ood",
    }


def _write_cached_clip(
    cache_dir: Path,
    clip: str,
    activity_class: ActivityClass,
    frames: list[dict[str, Transform]],
) -> None:
    columns = new_cache_columns()
    for frame_index, joints in enumerate(frames):
        append_cached_pose_frame(
            columns,
            subject="01",
            clip=clip,
            frame_index=frame_index + 1,
            timestamp_s=frame_index / 120.0,
            joints=joints,
            environment_split="flat",
            activity_class=activity_class,
        )
    write_clip_cache(cache_dir / "subject=01" / f"{clip}.parquet", columns)


def _reference_frame(
    *,
    head: tuple[float, float, float],
    hips: tuple[float, float, float],
) -> dict[str, Transform]:
    joints = {
        joint: Transform(position=hips, rotation=(0.0, 0.0, 0.0, 1.0)) for joint in CACHE_JOINTS
    }
    joints.update(
        {
            "head": Transform(position=head),
            "hips": Transform(position=hips),
            "chest": Transform(position=(hips[0], (hips[1] + head[1]) * 0.5, hips[2])),
            "spine": Transform(position=(hips[0], hips[1] + 0.25, hips[2])),
            "left_shoulder": Transform(position=(hips[0] - 0.22, head[1] - 0.2, hips[2])),
            "right_shoulder": Transform(position=(hips[0] + 0.22, head[1] - 0.2, hips[2])),
            "left_hand": Transform(position=(hips[0] - 0.35, head[1] - 0.55, hips[2] - 0.2)),
            "right_hand": Transform(position=(hips[0] + 0.35, head[1] - 0.55, hips[2] - 0.2)),
            "left_foot": Transform(position=(hips[0] - 0.12, 0.0, hips[2])),
            "right_foot": Transform(position=(hips[0] + 0.12, 0.0, hips[2])),
            "left_knee": Transform(position=(hips[0] - 0.12, hips[1] * 0.5, hips[2])),
            "right_knee": Transform(position=(hips[0] + 0.12, hips[1] * 0.5, hips[2])),
        }
    )
    return joints
