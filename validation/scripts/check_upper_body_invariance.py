"""Check upper-body ML feature and latent invariance conventions."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

from validation.vr_pose_validation.geometry import Transform, look_rotation
from validation.vr_pose_validation.synthetic_trackers import generate_three_point_samples
from validation.vr_pose_validation.upper_body_ml import (
    TARGET_NAMES,
    extract_upper_body_latents,
    sparse_window_features,
    yaw_rotated_pose_frame,
    yaw_rotated_sample,
)

DEFAULT_REPORT_PATH = Path("validation/reports/upper_body_invariance.json")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--yaw-degrees", type=float, default=137.0)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    args = parser.parse_args(argv)

    report = check_upper_body_representation_invariance(
        window_size=max(2, args.window_size),
        yaw_degrees=args.yaw_degrees,
        tolerance=max(0.0, args.tolerance),
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    summary = cast(Mapping[str, object], report["summary"])
    return 0 if bool(summary["passed"]) else 1


def check_upper_body_representation_invariance(
    *,
    window_size: int = 5,
    yaw_degrees: float = 137.0,
    tolerance: float = 1e-5,
) -> dict[str, object]:
    frames = [_pose_frame(index * 0.02) for index in range(window_size)]
    samples = generate_three_point_samples(frames)
    yaw_samples = [yaw_rotated_sample(sample, yaw_degrees) for sample in samples]
    center_index = window_size - 1

    feature_delta = _max_abs_delta(
        sparse_window_features(samples, center_index, window_size),
        sparse_window_features(yaw_samples, center_index, window_size),
    )
    target_delta = _max_abs_target_delta(
        extract_upper_body_latents(frames[center_index]).values,
        extract_upper_body_latents(
            yaw_rotated_pose_frame(frames[center_index], yaw_degrees)
        ).values,
    )
    mirror_delta = _mirror_selected_target_delta(frames[center_index])
    passed = feature_delta <= tolerance and target_delta <= tolerance and mirror_delta <= tolerance
    return {
        "summary": {
            "passed": passed,
            "window_size": window_size,
            "yaw_degrees": yaw_degrees,
            "tolerance": tolerance,
            "max_feature_yaw_abs_delta": feature_delta,
            "max_target_yaw_abs_delta": target_delta,
            "max_mirror_selected_abs_delta": mirror_delta,
        }
    }


def _pose_frame(offset_z: float) -> dict[str, Transform]:
    rotation = look_rotation((0.0, 0.0, 1.0))
    return {
        "hips": Transform(position=(0.03, 0.92, offset_z), rotation=rotation),
        "spine": Transform(position=(0.05, 1.12, offset_z + 0.01), rotation=rotation),
        "chest": Transform(position=(0.07, 1.34, offset_z + 0.02), rotation=rotation),
        "head": Transform(position=(0.11, 1.62, offset_z + 0.04), rotation=rotation),
        "left_shoulder": Transform(position=(0.31, 1.41, offset_z - 0.03), rotation=rotation),
        "right_shoulder": Transform(position=(-0.14, 1.36, offset_z + 0.05), rotation=rotation),
        "left_elbow": Transform(position=(0.43, 1.17, offset_z - 0.09), rotation=rotation),
        "right_elbow": Transform(position=(-0.26, 1.09, offset_z + 0.01), rotation=rotation),
        "left_hand": Transform(position=(0.51, 1.03, offset_z + 0.10), rotation=rotation),
        "right_hand": Transform(position=(-0.34, 1.00, offset_z + 0.18), rotation=rotation),
    }


def _mirror_pose_x_swap_sides(joints: Mapping[str, Transform]) -> dict[str, Transform]:
    mirrored: dict[str, Transform] = {}
    for name, transform in joints.items():
        target_name = _swap_side_name(name)
        mirrored[target_name] = Transform(
            position=(-transform.position[0], transform.position[1], transform.position[2]),
            rotation=transform.rotation,
        )
    return mirrored


def _swap_side_name(name: str) -> str:
    if name.startswith("left_"):
        return "right_" + name.removeprefix("left_")
    if name.startswith("right_"):
        return "left_" + name.removeprefix("right_")
    return name


def _mirror_selected_target_delta(joints: Mapping[str, Transform]) -> float:
    original = extract_upper_body_latents(joints).values
    mirrored = extract_upper_body_latents(_mirror_pose_x_swap_sides(joints)).values
    expected = {
        "torso_heading_sin": -original["torso_heading_sin"],
        "torso_heading_cos": original["torso_heading_cos"],
        "torso_lean_x": -original["torso_lean_x"],
        "torso_lean_z": original["torso_lean_z"],
        "left_shoulder_offset_x": -original["right_shoulder_offset_x"],
        "left_shoulder_offset_y": original["right_shoulder_offset_y"],
        "left_shoulder_offset_z": original["right_shoulder_offset_z"],
        "right_shoulder_offset_x": -original["left_shoulder_offset_x"],
        "right_shoulder_offset_y": original["left_shoulder_offset_y"],
        "right_shoulder_offset_z": original["left_shoulder_offset_z"],
    }
    return max(abs(mirrored[name] - value) for name, value in expected.items())


def _max_abs_target_delta(
    first: Mapping[str, float],
    second: Mapping[str, float],
) -> float:
    return max(abs(first[name] - second[name]) for name in TARGET_NAMES)


def _max_abs_delta(first: Sequence[float], second: Sequence[float]) -> float:
    return max(abs(a - b) for a, b in zip(first, second, strict=True))


if __name__ == "__main__":
    raise SystemExit(main())
