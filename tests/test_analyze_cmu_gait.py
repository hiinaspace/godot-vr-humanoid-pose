from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import polars as pl
from pytest import approx

from validation.scripts.analyze_cmu_gait import (
    analyze_cmu_gait,
    extract_clip_gait_steps,
    render_gait_plot,
)
from validation.vr_pose_validation.geometry import Transform
from validation.vr_pose_validation.parquet_cache import (
    append_cached_pose_frame,
    new_cache_columns,
    write_clip_cache,
)


def test_extract_clip_gait_steps_detects_alternating_footfalls() -> None:
    dataframe = _synthetic_gait_dataframe()

    steps = extract_clip_gait_steps(
        dataframe,
        contact_height_threshold_m=0.05,
        contact_speed_threshold_m_s=0.25,
        min_contact_duration_s=0.0,
    )

    assert [step["foot"] for step in steps][:2] == ["right_foot", "left_foot"]
    assert steps[0]["step_abs_forward_m"] == approx(0.4, abs=0.05)
    assert steps[0]["step_width_m"] == approx(0.2, abs=0.05)
    assert steps[0]["step_interval_s"] is None
    assert steps[1]["step_interval_s"] == approx(0.1)
    assert cast(float, steps[1]["mean_head_speed_m_s"]) > 0.0


def test_analyze_cmu_gait_writes_summary_from_cache(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    columns = _synthetic_gait_columns()
    write_clip_cache(cache_dir / "subject=01" / "01_01.parquet", columns)

    report, steps = analyze_cmu_gait(
        cache_dir=cache_dir,
        environment_splits={"flat"},
        activity_classes={"locomotion"},
        contact_height_threshold_m=0.05,
        contact_speed_threshold_m_s=0.25,
        min_contact_duration_s=0.0,
    )

    summary = cast(dict[str, Any], report["summary"])
    assumptions = cast(dict[str, Any], summary["basis_footstep_assumptions"])
    distribution = cast(dict[str, Any], summary["distribution"])
    step_distribution = cast(dict[str, float], distribution["step_abs_forward_m"])

    assert summary["clips"] == 1
    assert summary["steps"] == len(steps)
    assert assumptions["footstep_forward_lead_m"] > 0.0
    assert step_distribution["p50"] > 0.0


def test_render_gait_plot_creates_png(tmp_path: Path) -> None:
    steps = extract_clip_gait_steps(
        _synthetic_gait_dataframe(),
        contact_height_threshold_m=0.05,
        contact_speed_threshold_m_s=0.25,
        min_contact_duration_s=0.0,
    )
    report, _ = analyze_cmu_gait(
        cache_dir=_write_synthetic_cache(tmp_path),
        environment_splits={"flat"},
        activity_classes={"locomotion"},
        contact_height_threshold_m=0.05,
        contact_speed_threshold_m_s=0.25,
        min_contact_duration_s=0.0,
    )
    output = tmp_path / "gait.png"

    render_gait_plot(report, steps, output)

    assert output.exists()
    assert output.stat().st_size > 0


def _write_synthetic_cache(tmp_path: Path) -> Path:
    cache_dir = tmp_path / "cache"
    write_clip_cache(cache_dir / "subject=01" / "01_01.parquet", _synthetic_gait_columns())
    return cache_dir


def _synthetic_gait_dataframe() -> pl.DataFrame:
    return pl.DataFrame(_synthetic_gait_columns())


def _synthetic_gait_columns() -> dict[str, list[object]]:
    columns = new_cache_columns()
    times = [index * 0.1 for index in range(8)]
    head_positions = [(0.0, 1.6, index * 0.2) for index in range(8)]
    hips_positions = [(0.0, 1.0, index * 0.2) for index in range(8)]
    left_hand_positions = [(-0.25, 1.2, index * 0.2 + 0.05) for index in range(8)]
    right_hand_positions = [(0.25, 1.2, index * 0.2 + 0.05) for index in range(8)]
    left_foot_positions = [
        (-0.1, 0.0, 0.0),
        (-0.1, 0.0, 0.0),
        (-0.1, 0.2, 0.4),
        (-0.1, 0.2, 0.6),
        (-0.1, 0.0, 0.8),
        (-0.1, 0.0, 0.8),
        (-0.1, 0.2, 1.0),
        (-0.1, 0.2, 1.2),
    ]
    right_foot_positions = [
        (0.1, 0.0, 0.0),
        (0.1, 0.0, 0.0),
        (0.1, 0.2, 0.2),
        (0.1, 0.0, 0.4),
        (0.1, 0.0, 0.4),
        (0.1, 0.2, 0.8),
        (0.1, 0.0, 1.2),
        (0.1, 0.0, 1.2),
    ]
    for index, timestamp in enumerate(times):
        append_cached_pose_frame(
            columns,
            subject="01",
            clip="01_01",
            frame_index=index,
            timestamp_s=timestamp,
            joints={
                "head": Transform(position=head_positions[index]),
                "chest": Transform(position=(0.0, 1.35, index * 0.2)),
                "spine": Transform(position=(0.0, 1.15, index * 0.2)),
                "left_shoulder": Transform(position=(-0.2, 1.3, index * 0.2)),
                "right_shoulder": Transform(position=(0.2, 1.3, index * 0.2)),
                "left_elbow": Transform(position=(-0.3, 1.15, index * 0.2)),
                "right_elbow": Transform(position=(0.3, 1.15, index * 0.2)),
                "left_hand": Transform(position=left_hand_positions[index]),
                "right_hand": Transform(position=right_hand_positions[index]),
                "hips": Transform(position=hips_positions[index]),
                "left_knee": Transform(position=(-0.1, 0.5, index * 0.2)),
                "right_knee": Transform(position=(0.1, 0.5, index * 0.2)),
                "left_foot": Transform(position=left_foot_positions[index]),
                "right_foot": Transform(position=right_foot_positions[index]),
            },
            environment_split="flat",
            activity_class="locomotion",
            motion_description="synthetic walk",
        )
    return columns
