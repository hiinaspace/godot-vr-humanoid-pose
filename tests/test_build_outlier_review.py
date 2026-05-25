from __future__ import annotations

import csv

from validation.scripts.build_outlier_review import (
    build_outlier_review,
    write_outlier_review_csv,
)


def test_build_outlier_review_selects_metric_union_and_buckets() -> None:
    rows = build_outlier_review(
        [
            {
                "clip": "07_12",
                "subject": "07",
                "motion_description": "brisk walk",
                "environment_split": "flat",
                "activity_class": "locomotion",
                "posture_subset": "vr_locomotion",
                "posture_vr_candidate": True,
                "reference_error_p95_m": 1.2,
                "mean_contact_foot_error_m": 0.8,
                "mean_contact_slip_m_per_s": 0.7,
            },
            {
                "clip": "105_50",
                "subject": "105",
                "motion_description": "jump turn",
                "environment_split": "flat",
                "activity_class": "stationary_upper_body",
                "posture_subset": "vr_stationary_upper_body",
                "pct_elbow_flexion_over_limit": 75.0,
                "reference_error_p95_m": 0.3,
                "mean_contact_slip_m_per_s": 0.1,
            },
            {
                "clip": "133_25",
                "subject": "133",
                "motion_description": "motorcycle",
                "environment_split": "unknown",
                "environment_tags": ("data_issue", "sideways_t_pose"),
                "activity_class": "unknown",
                "posture_subset": "unknown_ood",
                "reference_error_p95_m": 0.9,
                "mean_contact_slip_m_per_s": 0.2,
            },
            {
                "clip": "143_27",
                "subject": "143",
                "motion_description": "Washing Window",
                "environment_split": "flat",
                "activity_class": "stationary_upper_body",
                "posture_subset": "vr_stationary",
                "posture_inverted_pct": 8.0,
                "reference_error_p95_m": 0.8,
                "mean_contact_slip_m_per_s": 0.0,
            },
        ],
        metrics=(
            "reference_error_p95_m",
            "mean_contact_slip_m_per_s",
            "pct_elbow_flexion_over_limit",
        ),
        top_per_metric=3,
    )

    by_clip = {str(row["clip"]): row for row in rows}

    assert set(by_clip) == {"07_12", "105_50", "133_25", "143_27"}
    assert by_clip["07_12"]["review_hint"] == "candidate_footstep_failure"
    assert by_clip["105_50"]["review_hint"] == "candidate_arm_solver_failure"
    assert by_clip["133_25"]["review_hint"] == "possible_bad_data"
    assert by_clip["143_27"]["review_hint"] == "likely_ood_posture"
    assert "reference_error_p95_m#1" in str(by_clip["07_12"]["selection_metrics"])
    assert "pct_elbow_flexion_over_limit#1" in str(by_clip["105_50"]["selection_metrics"])


def test_write_outlier_review_csv_serializes_tags(tmp_path) -> None:
    output = tmp_path / "review.csv"
    write_outlier_review_csv(
        output,
        [
            {
                "clip": "133_25",
                "subject": "133",
                "review_hint": "possible_bad_data",
                "environment_tags": ("data_issue", "sideways_t_pose"),
                "activity_tags": ("parse_issue",),
            }
        ],
    )

    with output.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert rows[0]["environment_tags"] == "data_issue;sideways_t_pose"
    assert rows[0]["activity_tags"] == "parse_issue"


def test_build_outlier_review_can_bucket_and_exclude_bad_reference_clips() -> None:
    clips = [
        {
            "clip": "82_01",
            "subject": "82",
            "motion_description": "static pose",
            "environment_split": "flat",
            "activity_class": "idle_standing",
            "posture_subset": "vr_stationary",
            "reference_plausibility_bucket": "implausible_reference",
            "reference_plausibility_tags": ("head_below_floor",),
            "mean_elbow_error_m": 1.0,
        },
        {
            "clip": "80_16",
            "subject": "80",
            "motion_description": "mixing batter",
            "environment_split": "flat",
            "activity_class": "stationary_upper_body",
            "posture_subset": "vr_stationary",
            "reference_plausibility_bucket": "plausible",
            "mean_elbow_error_m": 0.8,
        },
    ]

    unfiltered = build_outlier_review(
        clips,
        metrics=("mean_elbow_error_m",),
        top_per_metric=2,
    )
    filtered = build_outlier_review(
        clips,
        metrics=("mean_elbow_error_m",),
        top_per_metric=2,
        excluded_reference_buckets={"implausible_reference"},
    )

    assert {row["clip"]: row["review_hint"] for row in unfiltered}["82_01"] == (
        "possible_bad_reference"
    )
    assert [row["clip"] for row in filtered] == ["80_16"]


def test_build_outlier_review_buckets_head_torso_orientation_failures() -> None:
    rows = build_outlier_review(
        [
            {
                "clip": "22_17",
                "subject": "22",
                "motion_description": "arm wrestle",
                "environment_split": "flat",
                "activity_class": "stationary_upper_body",
                "posture_subset": "vr_stationary",
                "head_forward_error_p95_deg": 145.0,
                "pct_head_forward_error_over_90deg": 12.0,
                "mean_neck_axis_error_deg": 50.0,
            }
        ],
        metrics=("head_forward_error_p95_deg",),
        top_per_metric=1,
    )

    assert rows[0]["review_hint"] == "candidate_head_torso_failure"
    assert rows[0]["primary_metric"] == "head_forward_error_p95_deg"


def test_build_outlier_review_identifies_natural_reference_turns() -> None:
    rows = build_outlier_review(
        [
            {
                "clip": "75_08",
                "subject": "75",
                "motion_description": "jump turn",
                "environment_split": "flat",
                "activity_class": "locomotion",
                "posture_subset": "vr_locomotion",
                "max_head_forward_angular_speed_deg_s": 1400.0,
                "max_reference_head_forward_angular_speed_deg_s": 1300.0,
                "pct_reference_fast_head_turns_with_torso_follow": 100.0,
                "pct_estimated_fast_head_turns_without_reference_motion": 0.0,
            }
        ],
        metrics=("max_head_forward_angular_speed_deg_s",),
        top_per_metric=1,
    )

    assert rows[0]["review_hint"] == "likely_natural_reference_turn"


def test_build_outlier_review_identifies_reference_torso_follow_metric_as_natural_turn() -> None:
    rows = build_outlier_review(
        [
            {
                "clip": "07_12",
                "subject": "07",
                "motion_description": "brisk walk",
                "environment_split": "flat",
                "activity_class": "locomotion",
                "posture_subset": "vr_locomotion",
                "max_head_forward_angular_speed_deg_s": 1400.0,
                "max_reference_head_forward_angular_speed_deg_s": 1300.0,
                "pct_reference_fast_head_turns_with_torso_follow": 100.0,
                "pct_estimated_fast_head_turns_without_reference_motion": 0.0,
            }
        ],
        metrics=("pct_reference_fast_head_turns_with_torso_follow",),
        top_per_metric=1,
    )

    assert rows[0]["review_hint"] == "likely_natural_reference_turn"
