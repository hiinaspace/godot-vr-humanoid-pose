from __future__ import annotations

import json

from validation.scripts.plot_cached_validation_report import render_report


def test_plot_cached_validation_report_writes_png(tmp_path) -> None:
    report = tmp_path / "report.json"
    output = tmp_path / "report.png"
    report.write_text(
        json.dumps(
            {
                "summary": {
                    "estimator": "basis",
                    "evaluated_frames": 120,
                    "total_seconds": 1.0,
                    "mean_reference_error_m": 0.4,
                    "mean_pelvis_error_m": 0.2,
                    "mean_pelvis_horizontal_error_m": 0.15,
                    "mean_pelvis_vertical_error_m": 0.05,
                    "mean_chest_error_m": 0.25,
                    "mean_spine_error_m": 0.3,
                    "mean_torso_error_m": 0.25,
                    "mean_shoulder_error_m": 0.35,
                    "mean_elbow_error_m": 0.45,
                    "mean_knee_error_m": 0.55,
                    "reference_error_p95_m": 0.7,
                    "pct_reference_error_over_50cm": 25.0,
                    "mean_joint_velocity_error_m_per_s": 0.8,
                    "mean_joint_acceleration_error_m_per_s2": 1.2,
                    "mean_joint_estimated_jerk_m_per_s3": 2.5,
                    "mean_left_foot_error_m": 0.5,
                    "mean_right_foot_error_m": 0.6,
                    "mean_contact_foot_error_m": 0.3,
                    "mean_contact_slip_m_per_s": 0.1,
                    "mean_low_foot_slip_m_per_s": 0.12,
                    "mean_hips_support_distance_m": 0.2,
                    "grouped_metrics": {
                        "environment_split": [
                            {
                                "environment_split": "flat",
                                "mean_reference_error_m": 0.4,
                                "mean_torso_error_m": 0.25,
                                "mean_contact_foot_error_m": 0.3,
                            }
                        ],
                        "activity_class": [
                            {
                                "activity_class": "sitting",
                                "mean_reference_error_m": 0.5,
                                "mean_torso_error_m": 0.3,
                                "mean_contact_foot_error_m": 0.35,
                            }
                        ],
                    },
                    "worst_clips": {
                        "reference_error_p95_m": [
                            {
                                "clip": "01_01",
                                "subject": "01",
                                "environment_split": "flat",
                                "activity_class": "sitting",
                                "mean_reference_error_m": 0.8,
                                "reference_error_p95_m": 1.0,
                            }
                        ]
                    },
                },
                "clips": [
                    {
                        "environment_split": "flat",
                        "mean_reference_error_m": 0.4,
                        "mean_pelvis_error_m": 0.2,
                        "mean_torso_error_m": 0.25,
                        "mean_shoulder_error_m": 0.35,
                        "mean_elbow_error_m": 0.45,
                        "mean_knee_error_m": 0.55,
                        "reference_error_p95_m": 0.7,
                        "mean_joint_velocity_error_m_per_s": 0.8,
                        "mean_joint_estimated_jerk_m_per_s3": 2.5,
                        "mean_contact_foot_error_m": 0.3,
                        "mean_contact_slip_m_per_s": 0.1,
                        "mean_low_foot_slip_m_per_s": 0.12,
                        "mean_hips_support_distance_m": 0.2,
                    },
                    {
                        "environment_split": "nonflat",
                        "mean_reference_error_m": 0.8,
                        "mean_pelvis_error_m": 0.5,
                        "mean_torso_error_m": 0.45,
                        "mean_shoulder_error_m": 0.55,
                        "mean_elbow_error_m": 0.65,
                        "mean_knee_error_m": 0.75,
                        "reference_error_p95_m": 1.1,
                        "mean_joint_velocity_error_m_per_s": 1.4,
                        "mean_joint_estimated_jerk_m_per_s3": 3.5,
                        "mean_contact_foot_error_m": 0.7,
                        "mean_contact_slip_m_per_s": 0.2,
                        "mean_low_foot_slip_m_per_s": 0.22,
                        "mean_hips_support_distance_m": 0.45,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    render_report(report, output)

    assert output.exists()
    assert output.stat().st_size > 0
