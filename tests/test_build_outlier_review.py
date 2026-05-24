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
        ],
        metrics=(
            "reference_error_p95_m",
            "mean_contact_slip_m_per_s",
            "pct_elbow_flexion_over_limit",
        ),
        top_per_metric=2,
    )

    by_clip = {str(row["clip"]): row for row in rows}

    assert set(by_clip) == {"07_12", "105_50", "133_25"}
    assert by_clip["07_12"]["review_hint"] == "candidate_footstep_failure"
    assert by_clip["105_50"]["review_hint"] == "candidate_arm_solver_failure"
    assert by_clip["133_25"]["review_hint"] == "possible_bad_data"
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
