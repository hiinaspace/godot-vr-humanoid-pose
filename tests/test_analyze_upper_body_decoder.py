from __future__ import annotations

import pytest

from validation.scripts.analyze_upper_body_decoder import (
    MetricAccumulator,
    _oracle_prediction,
    _reach_bucket,
)
from validation.vr_pose_validation.upper_body_ml import (
    TARGET_GROUPS,
    TARGET_NAMES,
    UpperBodyLatents,
)


def test_oracle_prediction_replaces_only_requested_group() -> None:
    heuristic = UpperBodyLatents(
        values={name: 1.0 for name in TARGET_NAMES},
        masks={name: 0.25 for name in TARGET_NAMES},
    )
    target = UpperBodyLatents(
        values={name: 2.0 for name in TARGET_NAMES},
        masks={name: 1.0 for name in TARGET_NAMES},
    )

    prediction = _oracle_prediction(
        heuristic_latents=heuristic,
        target_latents=target,
        groups={"elbow"},
    )

    for target_name in TARGET_GROUPS["elbow"]:
        assert prediction.values[target_name] == 2.0
        assert prediction.masks[target_name] == 1.0
    for group_name in ("torso", "shoulder"):
        for target_name in TARGET_GROUPS[group_name]:
            assert prediction.values[target_name] == 1.0
            assert prediction.masks[target_name] == 0.25


def test_reach_bucket_prioritizes_near_straight_singularity() -> None:
    assert _reach_bucket(reach_ratio=0.99, radius_ratio=0.10) == "near_straight"
    assert _reach_bucket(reach_ratio=0.99, radius_ratio=0.30) == "high_reach"
    assert _reach_bucket(reach_ratio=0.40, radius_ratio=0.30) == "low_reach"
    assert _reach_bucket(reach_ratio=0.75, radius_ratio=0.30) == "mid_reach"


def test_metric_accumulator_summary() -> None:
    accumulator = MetricAccumulator()
    for value in (1.0, 3.0, 5.0):
        accumulator.add(value)

    summary = accumulator.summary()

    assert summary["count"] == 3
    assert summary["mean"] == pytest.approx(3.0)
    assert summary["p50"] == pytest.approx(3.0)
    assert summary["p95"] == pytest.approx(5.0)
