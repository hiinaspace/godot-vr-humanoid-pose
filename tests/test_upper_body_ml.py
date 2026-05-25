from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from pytest import approx

from validation.scripts.check_upper_body_invariance import (
    check_upper_body_representation_invariance,
)
from validation.vr_pose_validation.baseline import BasisPortEstimator
from validation.vr_pose_validation.cmu_metadata import ActivityClass, EnvironmentSplit
from validation.vr_pose_validation.geometry import Transform, distance, look_rotation
from validation.vr_pose_validation.parquet_cache import (
    CACHE_JOINTS,
    append_cached_pose_frame,
    new_cache_columns,
    write_clip_cache,
)
from validation.vr_pose_validation.synthetic_trackers import generate_three_point_samples
from validation.vr_pose_validation.types import AvatarCalibration, GroundingContext
from validation.vr_pose_validation.upper_body_ml import (
    SparseWindowConfig,
    StaticUpperBodyLatentProvider,
    apply_upper_body_latents,
    build_sparse_window_dataset,
    elbow_swivel_sin_cos,
    extract_upper_body_latents,
    feature_schema,
    rotation6d,
    split_for_subject,
    yaw_basis_from_rotation,
)


def test_sparse_window_export_writes_causal_rows_and_heuristic_latents(tmp_path) -> None:
    cache_dir = tmp_path / "cache"
    _write_clip(cache_dir / "subject=01" / "01_01.parquet", subject="01", clip="01_01")
    _write_clip(
        cache_dir / "subject=02" / "02_01.parquet",
        subject="02",
        clip="02_01",
        environment_split="nonflat",
    )
    output_dir = tmp_path / "ml"

    summary = build_sparse_window_dataset(
        cache_dir=cache_dir,
        output_dir=output_dir,
        config=SparseWindowConfig(window_size=3, stride=2, holdout_folds=3),
    )

    import polars as pl

    dataframe = pl.read_parquet(output_dir / "windows.parquet")
    rows = dataframe.to_dicts()

    assert summary.clips_seen == 1
    assert summary.clips_exported == 1
    assert summary.windows == 2
    assert [row["frame_index"] for row in rows] == [3, 5]
    assert all(len(row["features"]) == len(feature_schema(3)) for row in rows)
    assert rows[0]["split"] == split_for_subject("01", 3)
    assert "heuristic_left_elbow_swivel_sin" in rows[0]
    assert "left_shoulder_elevation" in rows[0]
    assert rows[0]["reference_left_elbow_valid"] == 1.0
    assert rows[0]["reference_neck_valid"] == 1.0
    assert rows[0]["heuristic_left_elbow_valid"] == 1.0
    assert rows[0]["reference_left_shoulder_position_x"] == approx(0.22)


def test_sparse_window_export_writes_loadable_shards(tmp_path) -> None:
    from validation.scripts.train_upper_body_latents import load_dataset

    cache_dir = tmp_path / "cache"
    _write_clip(cache_dir / "subject=01" / "01_01.parquet", subject="01", clip="01_01")
    _write_clip(cache_dir / "subject=03" / "03_01.parquet", subject="03", clip="03_01")
    output_dir = tmp_path / "ml"

    summary = build_sparse_window_dataset(
        cache_dir=cache_dir,
        output_dir=output_dir,
        config=SparseWindowConfig(
            window_size=3,
            stride=2,
            holdout_folds=3,
            shard_rows=1,
        ),
    )
    dataset = load_dataset(output_dir)

    assert summary.windows == 4
    assert summary.shards == 2
    assert not (output_dir / "windows.parquet").exists()
    assert (output_dir / "windows_shards").exists()
    assert dataset.features.shape[0] == 4
    assert dataset.decoded_reference_points is not None
    assert dataset.decoded_reference_points.shape == (4, 6, 3)


def test_rotation6d_is_independent_of_quaternion_sign() -> None:
    rotation = look_rotation((0.0, 0.0, 1.0))
    negated = cast(tuple[float, float, float, float], tuple(-value for value in rotation))
    basis = yaw_basis_from_rotation(rotation)

    assert rotation6d(rotation, basis=basis) == approx(rotation6d(negated, basis=basis))


def test_upper_body_sparse_representation_invariance_smoke() -> None:
    report = check_upper_body_representation_invariance(window_size=5, yaw_degrees=123.0)
    summary = cast(Mapping[str, object], report["summary"])

    assert summary["passed"]
    assert summary["max_feature_yaw_abs_delta"] == approx(0.0, abs=1e-6)
    assert summary["max_target_yaw_abs_delta"] == approx(0.0, abs=1e-6)
    assert summary["max_mirror_selected_abs_delta"] == approx(0.0, abs=1e-6)


def test_elbow_swivel_labels_lower_elbow_as_downward_orbit() -> None:
    swivel_sin, swivel_cos, valid = elbow_swivel_sin_cos(
        shoulder=(0.25, 1.4, 0.0),
        elbow=(0.35, 1.1, -0.05),
        hand=(0.45, 1.0, 0.2),
    )

    assert valid
    assert swivel_cos > 0.6
    assert abs(swivel_sin) < 0.8


def test_learned_latent_application_keeps_controller_hand_target_fixed() -> None:
    frame = _pose_frame(0.0)
    sample = generate_three_point_samples([frame])[0]
    calibration = AvatarCalibration(human_height_m=1.8)
    base_pose = BasisPortEstimator().estimate(sample, calibration, GroundingContext())
    latents = extract_upper_body_latents(
        {name: joint.transform for name, joint in base_pose.joints.items()},
        calibration,
    )
    provider = StaticUpperBodyLatentProvider(latents.values, latents.masks)

    adjusted = apply_upper_body_latents(
        base_pose,
        provider.predict(sample, calibration, GroundingContext()),
        calibration=calibration,
    )

    assert adjusted.joints["left_hand"].transform.position == sample.left_hand.transform.position
    assert adjusted.joints["right_hand"].transform.position == sample.right_hand.transform.position
    assert adjusted.joints["left_elbow"].transform.position == approx(
        base_pose.joints["left_elbow"].transform.position,
        abs=1e-6,
    )


def test_torso_lean_latents_move_runtime_torso_chain() -> None:
    frame = _pose_frame(0.0)
    sample = generate_three_point_samples([frame])[0]
    calibration = AvatarCalibration(human_height_m=1.8)
    base_pose = BasisPortEstimator().estimate(sample, calibration, GroundingContext())
    latents = extract_upper_body_latents(
        {name: joint.transform for name, joint in base_pose.joints.items()},
        calibration,
    )
    values = dict(latents.values)
    masks = dict(latents.masks)
    values["torso_lean_z"] = 0.35
    masks["torso_lean_z"] = 1.0
    provider = StaticUpperBodyLatentProvider(values, masks)

    adjusted = apply_upper_body_latents(
        base_pose,
        provider.predict(sample, calibration, GroundingContext()),
        calibration=calibration,
    )

    base_head = base_pose.joints["head"].transform.position
    base_hips = base_pose.joints["hips"].transform.position
    adjusted_head = adjusted.joints["head"].transform.position
    adjusted_hips = adjusted.joints["hips"].transform.position

    assert adjusted_head == base_head
    assert distance(adjusted_head, adjusted_hips) == approx(distance(base_head, base_hips))
    assert adjusted_hips[2] < base_hips[2]
    assert (
        adjusted.joints["chest"].transform.position[2]
        < base_pose.joints["chest"].transform.position[2]
    )


def _write_clip(
    path,
    *,
    subject: str,
    clip: str,
    environment_split: EnvironmentSplit = "flat",
    activity_class: ActivityClass = "stationary_upper_body",
) -> None:
    columns = new_cache_columns()
    for frame_index in range(1, 7):
        append_cached_pose_frame(
            columns,
            subject=subject,
            clip=clip,
            frame_index=frame_index,
            timestamp_s=(frame_index - 1) / 120.0,
            joints={joint: _pose_frame((frame_index - 1) * 0.01)[joint] for joint in CACHE_JOINTS},
            environment_split=environment_split,
            activity_class=activity_class,
            motion_description="upper body test",
        )
    write_clip_cache(path, columns)


def _pose_frame(offset_z: float) -> dict[str, Transform]:
    rotation = look_rotation((0.0, 0.0, 1.0))
    joints = {
        "hips": Transform(position=(0.0, 0.92, offset_z), rotation=rotation),
        "spine": Transform(position=(0.0, 1.12, offset_z), rotation=rotation),
        "chest": Transform(position=(0.0, 1.34, offset_z), rotation=rotation),
        "head": Transform(position=(0.0, 1.62, offset_z), rotation=rotation),
        "left_shoulder": Transform(position=(0.22, 1.38, offset_z), rotation=rotation),
        "right_shoulder": Transform(position=(-0.22, 1.38, offset_z), rotation=rotation),
        "left_elbow": Transform(position=(0.34, 1.14, offset_z - 0.08), rotation=rotation),
        "right_elbow": Transform(position=(-0.34, 1.14, offset_z - 0.08), rotation=rotation),
        "left_hand": Transform(position=(0.42, 1.02, offset_z + 0.12), rotation=rotation),
        "right_hand": Transform(position=(-0.42, 1.02, offset_z + 0.12), rotation=rotation),
        "left_knee": Transform(position=(0.11, 0.48, offset_z), rotation=rotation),
        "right_knee": Transform(position=(-0.11, 0.48, offset_z), rotation=rotation),
        "left_foot": Transform(position=(0.11, 0.05, offset_z + 0.05), rotation=rotation),
        "right_foot": Transform(position=(-0.11, 0.05, offset_z + 0.05), rotation=rotation),
    }
    return {joint: joints[joint] for joint in CACHE_JOINTS}
