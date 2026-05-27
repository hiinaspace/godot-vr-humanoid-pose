from __future__ import annotations

from math import pi, sin

from pytest import approx

from validation.vr_pose_validation.frequency import summarize_frequency_reference_sequence
from validation.vr_pose_validation.geometry import Transform
from validation.vr_pose_validation.types import HumanoidPoseFrame, JointPose


def test_frequency_analysis_places_sinusoidal_error_in_matching_band() -> None:
    frames = 120
    sample_rate = 60.0
    poses = []
    references = []
    for index in range(frames):
        timestamp = index / sample_rate
        error = 0.1 * sin(2.0 * pi * 8.0 * timestamp)
        poses.append(_pose(timestamp, head_x=error))
        references.append({"head": Transform(position=(0.0, 1.6, 0.0))})

    analysis = summarize_frequency_reference_sequence(
        poses,
        references,
        group_joints={"head": ("head",)},
    )

    metrics = analysis.metrics
    assert metrics["head_error_band_6_12hz_m2"] > metrics["head_error_band_0_2hz_m2"] * 100.0
    assert metrics["head_error_high_frequency_ratio"] == approx(1.0, abs=1e-6)


def test_frequency_analysis_reports_low_band_bias_when_error_drifts_slowly() -> None:
    frames = 120
    sample_rate = 60.0
    poses = []
    references = []
    for index in range(frames):
        timestamp = index / sample_rate
        error = 0.1 * sin(2.0 * pi * 1.0 * timestamp)
        poses.append(_pose(timestamp, head_x=error))
        references.append({"head": Transform(position=(0.0, 1.6, 0.0))})

    analysis = summarize_frequency_reference_sequence(
        poses,
        references,
        group_joints={"head": ("head",)},
    )

    metrics = analysis.metrics
    assert metrics["head_error_band_0_2hz_m2"] > metrics["head_error_band_6_12hz_m2"] * 100.0
    assert metrics["head_error_high_frequency_ratio"] == approx(0.0, abs=1e-6)


def test_frequency_analysis_returns_zero_metrics_for_short_sequences() -> None:
    analysis = summarize_frequency_reference_sequence(
        [_pose(0.0, head_x=0.0)],
        [{"head": Transform(position=(0.0, 1.6, 0.0))}],
        group_joints={"head": ("head",)},
    )

    assert analysis.metrics["head_error_total_power_m2"] == 0.0
    assert analysis.spectra == {}


def _pose(timestamp: float, *, head_x: float) -> HumanoidPoseFrame:
    return HumanoidPoseFrame(
        timestamp_seconds=timestamp,
        root=Transform(),
        joints={
            "head": JointPose(
                Transform(position=(head_x, 1.6, 0.0)),
            )
        },
    )
