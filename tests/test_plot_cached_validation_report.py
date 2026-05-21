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
                    "mean_left_foot_error_m": 0.5,
                    "mean_right_foot_error_m": 0.6,
                    "mean_contact_foot_error_m": 0.3,
                    "mean_contact_slip_m_per_s": 0.1,
                    "grouped_metrics": {
                        "environment_split": [
                            {
                                "environment_split": "flat",
                                "mean_reference_error_m": 0.4,
                                "mean_contact_foot_error_m": 0.3,
                            }
                        ],
                        "activity_class": [
                            {
                                "activity_class": "sitting",
                                "mean_reference_error_m": 0.5,
                                "mean_contact_foot_error_m": 0.35,
                            }
                        ],
                    },
                    "worst_clips": {
                        "mean_reference_error_m": [
                            {
                                "clip": "01_01",
                                "subject": "01",
                                "environment_split": "flat",
                                "activity_class": "sitting",
                                "mean_reference_error_m": 0.8,
                            }
                        ]
                    },
                },
                "clips": [],
            }
        ),
        encoding="utf-8",
    )

    render_report(report, output)

    assert output.exists()
    assert output.stat().st_size > 0
