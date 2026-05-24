"""Build a compact outlier review table from a cached validation report."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

DEFAULT_REPORT_PATH = Path("validation/reports/outliers_basis_flat_current.json")
DEFAULT_OUTPUT_PATH = Path("validation/reports/outlier_review_flat.csv")
DEFAULT_REVIEW_METRICS = (
    "reference_error_p95_m",
    "mean_reference_error_m",
    "mean_contact_foot_error_m",
    "mean_contact_slip_m_per_s",
    "mean_feet_velocity_error_m_per_s",
    "mean_knee_error_m",
    "pct_knee_flexion_over_limit",
    "mean_elbow_error_m",
    "pct_elbow_flexion_over_limit",
    "mean_shoulder_error_m",
    "head_forward_error_p95_deg",
    "pct_head_forward_error_over_90deg",
    "head_up_error_p95_deg",
    "pct_head_up_error_over_90deg",
    "mean_neck_axis_error_deg",
    "mean_spine_axis_error_deg",
    "max_head_forward_angular_speed_deg_s",
    "mean_joint_estimated_jerk_m_per_s3",
    "avatar_implausibility_score_0_100",
    "avatar_joint_limit_penalty",
    "avatar_endpoint_stretch_penalty",
    "avatar_temporal_jerk_penalty",
)
REVIEW_FIELDS = (
    "clip",
    "subject",
    "motion_description",
    "review_hint",
    "review_reasons",
    "selection_metrics",
    "primary_metric",
    "primary_value",
    "environment_split",
    "environment_tags",
    "activity_class",
    "activity_tags",
    "posture_subset",
    "posture_vr_candidate",
    "reference_plausibility_bucket",
    "reference_plausibility_tags",
    "avatar_plausibility_bucket",
    "avatar_plausibility_tags",
    "avatar_plausibility_score_0_100",
    "avatar_implausibility_score_0_100",
    "avatar_joint_limit_penalty",
    "avatar_sustained_awkwardness_penalty",
    "avatar_endpoint_stretch_penalty",
    "avatar_temporal_jerk_penalty",
    "avatar_pct_joint_limit_violation_frames",
    "avatar_pct_sustained_awkward_frames",
    "avatar_max_endpoint_stretch_ratio",
    "avatar_p95_endpoint_stretch_ratio",
    "avatar_p95_joint_jerk_m_per_s3",
    "avatar_max_joint_jerk_m_per_s3",
    "reference_max_joint_speed_m_per_s",
    "reference_pct_frames_over_20m_s",
    "reference_min_head_height_m",
    "reference_max_bone_length_cv",
    "reference_error_p95_m",
    "mean_reference_error_m",
    "mean_contact_foot_error_m",
    "mean_contact_slip_m_per_s",
    "mean_feet_velocity_error_m_per_s",
    "mean_knee_error_m",
    "mean_knee_flexion_error_deg",
    "pct_knee_flexion_over_limit",
    "mean_elbow_error_m",
    "mean_elbow_flexion_error_deg",
    "pct_elbow_flexion_over_limit",
    "mean_shoulder_error_m",
    "mean_head_forward_error_deg",
    "head_forward_error_p95_deg",
    "pct_head_forward_error_over_90deg",
    "mean_head_up_error_deg",
    "head_up_error_p95_deg",
    "pct_head_up_error_over_90deg",
    "mean_neck_axis_error_deg",
    "neck_axis_error_p95_deg",
    "pct_neck_axis_error_over_45deg",
    "mean_spine_axis_error_deg",
    "spine_axis_error_p95_deg",
    "pct_spine_axis_error_over_45deg",
    "p95_head_forward_angular_speed_deg_s",
    "max_head_forward_angular_speed_deg_s",
    "pct_head_forward_angular_speed_over_720deg_s",
    "mean_joint_estimated_jerk_m_per_s3",
    "hip_horizontal_path_m",
    "mean_hip_horizontal_speed_m_per_s",
    "posture_reclined_pct",
    "posture_floor_reclined_pct",
    "posture_inverted_pct",
    "fixture_command",
)
BAD_DATA_TAGS = frozenset({"data_issue", "parse_issue", "sideways_t_pose"})
BAD_DATA_DESCRIPTION_TERMS = ("motorcycle", "t pose", "t-pose")
OOD_ACTIVITY_CLASSES = frozenset({"dance_exercise", "floor_pose", "sports", "swimming"})
OOD_POSTURE_SUBSETS = frozenset(
    {"dynamic_ood", "floor_reclined_ood", "inverted_ood", "nonflat_ood"}
)
VR_CANDIDATE_ACTIVITY_CLASSES = frozenset(
    {"idle_standing", "sitting", "stationary_upper_body", "locomotion"}
)


@dataclass(frozen=True, slots=True)
class MetricSelection:
    metric: str
    rank: int
    value: float


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", nargs="?", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--plausibility-report", type=Path)
    parser.add_argument(
        "--exclude-reference-bucket",
        action="append",
        dest="excluded_reference_buckets",
        help="Reference plausibility bucket to exclude after optional plausibility merge.",
    )
    parser.add_argument("--top-per-metric", type=int, default=12)
    parser.add_argument(
        "--metric",
        action="append",
        dest="metrics",
        help="Metric to rank by. Can be repeated; defaults cover tail, limb, temporal, and feet.",
    )
    args = parser.parse_args(argv)

    clips = load_report_clips(args.report)
    if args.plausibility_report is not None:
        merge_plausibility_report(clips, args.plausibility_report)
    rows = build_outlier_review(
        clips,
        metrics=tuple(args.metrics or DEFAULT_REVIEW_METRICS),
        top_per_metric=max(0, args.top_per_metric),
        excluded_reference_buckets=set(args.excluded_reference_buckets or ()),
    )
    write_outlier_review_csv(args.output, rows)
    if args.json_output is not None:
        write_outlier_review_json(args.json_output, rows)

    hint_counts = Counter(str(row["review_hint"]) for row in rows)
    print(
        json.dumps(
            {
                "outlier_rows": len(rows),
                "output": args.output.as_posix(),
                "review_hints": dict(sorted(hint_counts.items())),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def load_report_clips(report_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    clips = payload.get("clips")
    if not isinstance(clips, list):
        raise ValueError(f"{report_path} does not contain a clips list")
    return [cast(dict[str, Any], clip) for clip in clips if isinstance(clip, dict)]


def merge_plausibility_report(clips: Sequence[dict[str, Any]], report_path: Path) -> None:
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    plausibility_clips = payload.get("clips")
    if not isinstance(plausibility_clips, list):
        raise ValueError(f"{report_path} does not contain a clips list")
    plausibility_by_key = {
        _clip_key(cast(dict[str, Any], clip)): cast(dict[str, Any], clip)
        for clip in plausibility_clips
        if isinstance(clip, dict)
    }
    for clip in clips:
        plausibility = plausibility_by_key.get(_clip_key(clip))
        if plausibility is None:
            continue
        for key, value in plausibility.items():
            if key.startswith("reference_"):
                clip[key] = value


def build_outlier_review(
    clips: Sequence[dict[str, Any]],
    *,
    metrics: Sequence[str] = DEFAULT_REVIEW_METRICS,
    top_per_metric: int = 12,
    excluded_reference_buckets: set[str] | None = None,
) -> list[dict[str, object]]:
    if top_per_metric <= 0:
        return []

    excluded_buckets = excluded_reference_buckets or set()
    candidate_clips = [
        clip
        for clip in clips
        if str(clip.get("reference_plausibility_bucket", "")) not in excluded_buckets
    ]
    clips_by_key = {_clip_key(clip): clip for clip in clips}
    selections_by_key: dict[str, list[MetricSelection]] = {}
    for metric in metrics:
        ranked_clips = sorted(
            (clip for clip in candidate_clips if _number(clip.get(metric)) is not None),
            key=lambda clip: cast(float, _number(clip.get(metric))),
            reverse=True,
        )
        for rank, clip in enumerate(ranked_clips[:top_per_metric], start=1):
            value = _number(clip.get(metric))
            if value is None:
                continue
            selections_by_key.setdefault(_clip_key(clip), []).append(
                MetricSelection(metric=metric, rank=rank, value=value)
            )

    rows = [
        _review_row(clips_by_key[key], selections)
        for key, selections in selections_by_key.items()
        if selections
    ]
    return sorted(
        rows,
        key=lambda row: (
            min(
                _selection_rank(selection)
                for selection in str(row["selection_metrics"]).split("; ")
            ),
            str(row["review_hint"]),
            str(row["clip"]),
        ),
    )


def write_outlier_review_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(REVIEW_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field, "")) for field in REVIEW_FIELDS})


def write_outlier_review_json(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(rows), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _review_row(clip: dict[str, Any], selections: Sequence[MetricSelection]) -> dict[str, object]:
    sorted_selections = sorted(selections, key=lambda selection: (selection.rank, selection.metric))
    primary = sorted_selections[0]
    hint, reasons = _classify_outlier(clip, primary_metric=primary.metric)
    row: dict[str, object] = {
        "clip": str(clip.get("clip", "")),
        "subject": str(clip.get("subject", "")),
        "motion_description": str(clip.get("motion_description", "")),
        "review_hint": hint,
        "review_reasons": "; ".join(reasons),
        "selection_metrics": "; ".join(
            f"{selection.metric}#{selection.rank}={selection.value:.4g}"
            for selection in sorted_selections
        ),
        "primary_metric": primary.metric,
        "primary_value": primary.value,
        "environment_split": str(clip.get("environment_split", "unknown")),
        "environment_tags": tuple(_strings(clip.get("environment_tags"))),
        "activity_class": str(clip.get("activity_class", "unknown")),
        "activity_tags": tuple(_strings(clip.get("activity_tags"))),
        "posture_subset": str(clip.get("posture_subset", "unknown_ood")),
        "posture_vr_candidate": bool(clip.get("posture_vr_candidate", False)),
        "fixture_command": _fixture_command(clip),
    }
    for field in REVIEW_FIELDS:
        if field in row:
            continue
        value = clip.get(field)
        number = _number(value)
        row[field] = number if number is not None else value if value is not None else ""
    return row


def _classify_outlier(clip: dict[str, Any], *, primary_metric: str = "") -> tuple[str, list[str]]:
    environment_split = str(clip.get("environment_split", "unknown"))
    activity_class = str(clip.get("activity_class", "unknown"))
    posture_subset = str(clip.get("posture_subset", "unknown_ood"))
    tags = {
        *_strings(clip.get("environment_tags")),
        *_strings(clip.get("activity_tags")),
    }
    description = str(clip.get("motion_description", "")).lower()

    if tags & BAD_DATA_TAGS or any(term in description for term in BAD_DATA_DESCRIPTION_TERMS):
        return "possible_bad_data", sorted(tags & BAD_DATA_TAGS) or ["description_keyword"]
    reference_bucket = str(clip.get("reference_plausibility_bucket", ""))
    if reference_bucket == "implausible_reference":
        reference_tags = _strings(clip.get("reference_plausibility_tags"))
        return "possible_bad_reference", reference_tags or ["reference_plausibility"]
    if environment_split != "flat":
        return "likely_ood_environment", [f"environment_split={environment_split}"]
    if activity_class in OOD_ACTIVITY_CLASSES:
        return "likely_ood_activity", [f"activity_class={activity_class}"]
    if _metric_at_least(clip, "posture_inverted_pct", 5.0):
        return "likely_ood_posture", ["inverted head/hip frames"]
    if posture_subset in OOD_POSTURE_SUBSETS:
        return "likely_ood_posture", [f"posture_subset={posture_subset}"]

    candidate_reasons: list[tuple[str, str]] = []
    if activity_class == "locomotion" and (
        _metric_at_least(clip, "mean_contact_foot_error_m", 0.35)
        or _metric_at_least(clip, "mean_contact_slip_m_per_s", 0.4)
        or _metric_at_least(clip, "mean_feet_velocity_error_m_per_s", 0.8)
    ):
        candidate_reasons.append(("candidate_footstep_failure", "locomotion foot/contact error"))
    if _metric_at_least(clip, "pct_elbow_flexion_over_limit", 20.0) or _metric_at_least(
        clip, "mean_elbow_error_m", 0.2
    ):
        candidate_reasons.append(("candidate_arm_solver_failure", "arm/elbow error"))
    if _metric_at_least(clip, "pct_knee_flexion_over_limit", 20.0) or _metric_at_least(
        clip, "mean_knee_error_m", 0.35
    ):
        candidate_reasons.append(("candidate_knee_solver_failure", "knee error"))
    if (
        _metric_at_least(clip, "pct_head_forward_error_over_90deg", 5.0)
        or _metric_at_least(clip, "pct_head_up_error_over_90deg", 5.0)
        or _metric_at_least(clip, "pct_neck_axis_error_over_45deg", 10.0)
        or _metric_at_least(clip, "pct_spine_axis_error_over_45deg", 10.0)
    ):
        candidate_reasons.append(("candidate_head_torso_failure", "head/torso orientation"))
    if _metric_at_least(clip, "mean_joint_estimated_jerk_m_per_s3", 3000.0):
        candidate_reasons.append(("candidate_temporal_failure", "temporal jerk"))
    if str(clip.get("avatar_plausibility_bucket", "")) == "implausible_output" or _metric_at_least(
        clip, "avatar_implausibility_score_0_100", 25.0
    ):
        candidate_reasons.append(("candidate_avatar_plausibility_failure", "avatar plausibility"))
    if candidate_reasons:
        reasons = [reason for _hint, reason in candidate_reasons]
        preferred_hint = _hint_from_primary_metric(primary_metric)
        if preferred_hint is not None and any(
            hint == preferred_hint for hint, _reason in candidate_reasons
        ):
            return preferred_hint, reasons
        if preferred_hint is not None:
            return preferred_hint, [f"selected by {primary_metric}", *reasons]
        return candidate_reasons[0][0], reasons

    if (
        bool(clip.get("posture_vr_candidate", False))
        or activity_class in VR_CANDIDATE_ACTIVITY_CLASSES
    ):
        return "candidate_solver_failure", ["flat VR-candidate clip"]
    return "needs_visual_review", ["no automatic bucket"]


def _hint_from_primary_metric(metric: str) -> str | None:
    if metric.startswith("avatar_"):
        return "candidate_avatar_plausibility_failure"
    if "head" in metric or "neck" in metric or "spine_axis" in metric:
        return "candidate_head_torso_failure"
    if "elbow" in metric or "shoulder" in metric:
        return "candidate_arm_solver_failure"
    if "knee" in metric:
        return "candidate_knee_solver_failure"
    if "foot" in metric or "feet" in metric or "contact" in metric:
        return "candidate_footstep_failure"
    if "velocity" in metric or "acceleration" in metric or "jerk" in metric:
        return "candidate_temporal_failure"
    return None


def _metric_at_least(clip: dict[str, Any], metric: str, threshold: float) -> bool:
    value = _number(clip.get(metric))
    return value is not None and value >= threshold


def _number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _strings(value: object) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, (list, tuple)):
        return [item for item in (str(item) for item in value) if item]
    return []


def _clip_key(clip: dict[str, Any]) -> str:
    return f"{clip.get('subject', '')}:{clip.get('clip', '')}"


def _selection_rank(selection_text: str) -> int:
    marker = selection_text.split("=", maxsplit=1)[0]
    if "#" not in marker:
        return 9999
    try:
        return int(marker.rsplit("#", maxsplit=1)[1])
    except ValueError:
        return 9999


def _csv_value(value: object) -> object:
    if isinstance(value, (list, tuple)):
        return ";".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return value


def _fixture_command(clip: dict[str, Any]) -> str:
    subject = str(clip.get("subject", ""))
    clip_name = str(clip.get("clip", ""))
    if not subject or not clip_name:
        return ""
    return (
        "uv run python -m validation.scripts.export_godot_replay_fixture "
        "--cache-dir .cache/cmu/parquet/canonical_full "
        f"--subject {subject} --clip {clip_name} "
        "--max-frames 180 --frame-stride 2 --estimator basis"
    )


if __name__ == "__main__":
    raise SystemExit(main())
