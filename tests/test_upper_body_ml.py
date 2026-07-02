from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from pytest import approx

from validation.scripts.check_upper_body_invariance import (
    check_upper_body_representation_invariance,
)
from validation.vr_pose_validation.baseline import BasisPortEstimator
from validation.vr_pose_validation.cmu_metadata import ActivityClass, EnvironmentSplit
from validation.vr_pose_validation.geometry import Transform, look_rotation
from validation.vr_pose_validation.parquet_cache import (
    CACHE_JOINTS,
    append_cached_pose_frame,
    new_cache_columns,
    write_clip_cache,
)
from validation.vr_pose_validation.synthetic_trackers import generate_three_point_samples
from validation.vr_pose_validation.types import (
    AvatarCalibration,
    GroundingContext,
    TrackedPoseSample,
    TrackerState,
)
from validation.vr_pose_validation.upper_body_ml import (
    FK_TARGET_NAMES,
    FK_TARGET_SCHEMA,
    RUNTIME_DECODER_CONTEXT_JOINTS,
    TARGET_SCHEMA,
    EmaUpperBodyLatentProvider,
    SparseWindowConfig,
    StaticUpperBodyLatentProvider,
    UpperBodyLatentPrediction,
    apply_upper_body_fk_latents,
    apply_upper_body_latents,
    build_sparse_window_dataset,
    elbow_pole_vector,
    extract_upper_body_fk_targets,
    extract_upper_body_latents,
    feature_schema,
    precompute_sparse_tracker_features,
    rotation6d,
    sparse_window_features,
    sparse_window_features_from_cache,
    split_for_subject,
    tracking_frame_for_joints,
    tracking_frame_for_sample,
    window_size_from_feature_names,
    yaw_basis_from_rotation,
    yaw_rotated_pose_frame,
    yaw_rotated_sample,
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
    assert "heuristic_left_elbow_pole_x" in rows[0]
    assert "left_shoulder_rel_y" in rows[0]
    schema = __import__("json").loads((output_dir / "schema.json").read_text())
    assert schema["target_schema"] == TARGET_SCHEMA
    assert schema["fk_target_schema"] == FK_TARGET_SCHEMA
    assert schema["fk_target_names"] == list(FK_TARGET_NAMES)
    assert rows[0]["reference_left_elbow_valid"] == 1.0
    assert rows[0]["reference_neck_valid"] == 1.0
    assert rows[0]["heuristic_left_elbow_valid"] == 1.0
    assert rows[0]["heuristic_context_hips_valid"] == 1.0
    assert rows[0]["heuristic_context_left_hand_valid"] == 1.0
    assert rows[0]["reference_left_shoulder_position_x"] == approx(0.22)
    assert rows[0]["fk_hips_rotation6d_0_mask"] == 1.0
    assert rows[0]["fk_left_elbow_local_rotation6d_0_mask"] == 1.0


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
    assert dataset.runtime_context_points is not None
    assert dataset.runtime_context_points.shape == (4, len(RUNTIME_DECODER_CONTEXT_JOINTS), 3)


def test_sparse_window_cli_caps_dense_default_workers(monkeypatch) -> None:
    from validation.scripts import build_sparse_window_dataset as script

    monkeypatch.setattr(script.os, "cpu_count", lambda: 64)

    assert script._default_workers(stride=1) == 2
    assert script._default_workers(stride=4) == 16


def test_window_size_can_be_inferred_from_artifact_feature_names() -> None:
    assert window_size_from_feature_names(feature_schema(7)) == 7


def test_rotation6d_is_independent_of_quaternion_sign() -> None:
    rotation = look_rotation((0.0, 0.0, 1.0))
    negated = cast(tuple[float, float, float, float], tuple(-value for value in rotation))
    basis = yaw_basis_from_rotation(rotation)

    assert rotation6d(rotation, basis=basis) == approx(rotation6d(negated, basis=basis))


def test_upper_body_fk_targets_are_yaw_invariant() -> None:
    frame = _pose_frame(0.0)
    rotated = yaw_rotated_pose_frame(frame, 137.0)

    original_targets = extract_upper_body_fk_targets(
        frame,
        AvatarCalibration(human_height_m=1.8),
    )
    rotated_targets = extract_upper_body_fk_targets(
        rotated,
        AvatarCalibration(human_height_m=1.8),
    )

    assert original_targets.masks == rotated_targets.masks
    for name in FK_TARGET_NAMES:
        assert rotated_targets.values[name] == approx(original_targets.values[name], abs=1e-6)


def test_upper_body_fk_targets_encode_local_identity_rotations() -> None:
    targets = extract_upper_body_fk_targets(
        _pose_frame(0.0),
        AvatarCalibration(human_height_m=1.8),
    )

    assert (
        targets.values["fk_left_elbow_local_rotation6d_0"],
        targets.values["fk_left_elbow_local_rotation6d_1"],
        targets.values["fk_left_elbow_local_rotation6d_2"],
        targets.values["fk_left_elbow_local_rotation6d_3"],
        targets.values["fk_left_elbow_local_rotation6d_4"],
        targets.values["fk_left_elbow_local_rotation6d_5"],
    ) == approx((0.0, 0.0, 1.0, 0.0, 1.0, 0.0), abs=1e-6)


def test_upper_body_sparse_representation_invariance_smoke() -> None:
    report = check_upper_body_representation_invariance(window_size=5, yaw_degrees=123.0)
    summary = cast(Mapping[str, object], report["summary"])

    assert summary["passed"]
    assert summary["max_feature_yaw_abs_delta"] == approx(0.0, abs=1e-6)
    assert summary["max_target_yaw_abs_delta"] == approx(0.0, abs=1e-6)
    assert summary["max_mirror_selected_abs_delta"] == approx(0.0, abs=1e-6)


def test_calibration_yaw_features_use_stable_axes_for_head_turns() -> None:
    samples = [
        _sample_with_head_yaw(0.0, look_rotation((0.0, 0.0, 1.0))),
        _sample_with_head_yaw(1.0 / 120.0, look_rotation((1.0, 0.0, 0.0))),
    ]
    calibration_frame = tracking_frame_for_sample(
        samples[0],
        tracking_space="calibration_yaw",
    )
    stable_frame = tracking_frame_for_sample(
        samples[1],
        tracking_space="calibration_yaw",
        calibration_frame=calibration_frame,
    )
    moving_frame = tracking_frame_for_sample(samples[1], tracking_space="head_yaw")
    names = feature_schema(2)

    stable_features = sparse_window_features(
        samples,
        1,
        2,
        tracking_space="calibration_yaw",
        tracking_frame=stable_frame,
    )
    cached_features = sparse_window_features_from_cache(
        precompute_sparse_tracker_features(samples, calibration_frame.basis),
        1,
        2,
    )
    moving_features = sparse_window_features(
        samples,
        1,
        2,
        tracking_space="head_yaw",
        tracking_frame=moving_frame,
    )

    left_x = names.index("t+00_left_hand_position_x")
    left_z = names.index("t+00_left_hand_position_z")
    assert cached_features == approx(stable_features, abs=1e-6)
    assert stable_features[left_x] == approx(1.0, abs=1e-6)
    assert stable_features[left_z] == approx(0.0, abs=1e-6)
    assert moving_features[left_x] == approx(0.0, abs=1e-6)
    assert moving_features[left_z] == approx(1.0, abs=1e-6)


def test_calibration_yaw_targets_remain_global_yaw_invariant() -> None:
    frame = _pose_frame(0.0)
    samples = generate_three_point_samples([frame])
    calibration_frame = tracking_frame_for_sample(
        samples[0],
        tracking_space="calibration_yaw",
    )
    rotated_frame = yaw_rotated_pose_frame(frame, 91.0)
    rotated_sample = yaw_rotated_sample(samples[0], 91.0)
    rotated_calibration_frame = tracking_frame_for_sample(
        rotated_sample,
        tracking_space="calibration_yaw",
    )

    original = extract_upper_body_latents(
        frame,
        AvatarCalibration(human_height_m=1.8),
        tracking_frame=tracking_frame_for_joints(
            frame,
            tracking_space="calibration_yaw",
            calibration_frame=calibration_frame,
        ),
    )
    rotated = extract_upper_body_latents(
        rotated_frame,
        AvatarCalibration(human_height_m=1.8),
        tracking_frame=tracking_frame_for_joints(
            rotated_frame,
            tracking_space="calibration_yaw",
            calibration_frame=rotated_calibration_frame,
        ),
    )

    assert original.masks == rotated.masks
    for name, value in original.values.items():
        assert rotated.values[name] == approx(value, abs=1e-6)


def test_elbow_pole_labels_lower_elbow_as_downward_bend() -> None:
    pole, valid = elbow_pole_vector(
        shoulder=(0.25, 1.4, 0.0),
        elbow=(0.35, 1.1, -0.05),
        hand=(0.45, 1.0, 0.2),
        basis=yaw_basis_from_rotation(look_rotation((0.0, 0.0, 1.0))),
    )

    assert valid
    assert pole[1] < -0.4


def test_elbow_pole_mask_turns_off_for_near_straight_arm() -> None:
    frame = _pose_frame(0.0)
    frame["left_elbow"] = Transform(
        position=(0.44, 1.03, 0.11),
        rotation=frame["left_elbow"].rotation,
    )

    latents = extract_upper_body_latents(frame, AvatarCalibration(human_height_m=1.8))

    assert latents.masks["left_elbow_pole_x"] == approx(0.0)
    assert latents.masks["left_elbow_pole_y"] == approx(0.0)
    assert latents.masks["left_elbow_pole_z"] == approx(0.0)


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


def test_chain_latents_move_runtime_torso_anchors() -> None:
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
    values["chest_rel_z"] += 0.2
    values["neck_rel_z"] += 0.2
    masks["chest_rel_z"] = 1.0
    masks["neck_rel_z"] = 1.0
    provider = StaticUpperBodyLatentProvider(values, masks)

    adjusted = apply_upper_body_latents(
        base_pose,
        provider.predict(sample, calibration, GroundingContext()),
        calibration=calibration,
    )

    base_head = base_pose.joints["head"].transform.position
    adjusted_head = adjusted.joints["head"].transform.position

    assert adjusted_head == base_head
    assert (
        adjusted.joints["chest"].transform.position[2]
        > base_pose.joints["chest"].transform.position[2]
    )


def test_fk_latents_move_runtime_upper_body_but_keep_tracked_endpoints() -> None:
    frame = _pose_frame(0.0)
    sample = generate_three_point_samples([frame])[0]
    calibration = AvatarCalibration(human_height_m=1.8)
    base_pose = BasisPortEstimator().estimate(sample, calibration, GroundingContext())
    fk_targets = extract_upper_body_fk_targets(
        {name: joint.transform for name, joint in base_pose.joints.items()},
        calibration,
    )
    values = dict(fk_targets.values)
    masks = dict(fk_targets.masks)
    values["fk_hips_rel_z"] += 0.2

    adjusted = apply_upper_body_fk_latents(
        base_pose,
        UpperBodyLatentPrediction(values=values, masks=masks),
        calibration=calibration,
    )

    assert adjusted.joints["head"].transform == base_pose.joints["head"].transform
    assert adjusted.joints["left_hand"].transform == base_pose.joints["left_hand"].transform
    assert adjusted.joints["right_hand"].transform == base_pose.joints["right_hand"].transform
    assert (
        adjusted.joints["hips"].transform.position[2]
        > base_pose.joints["hips"].transform.position[2]
    )
    assert (
        adjusted.joints["left_shoulder"].transform.position[2]
        > base_pose.joints["left_shoulder"].transform.position[2]
    )


def test_fk_latents_with_zero_masks_leave_runtime_pose_unchanged() -> None:
    frame = _pose_frame(0.0)
    sample = generate_three_point_samples([frame])[0]
    calibration = AvatarCalibration(human_height_m=1.8)
    base_pose = BasisPortEstimator().estimate(sample, calibration, GroundingContext())
    fk_targets = extract_upper_body_fk_targets(
        {name: joint.transform for name, joint in base_pose.joints.items()},
        calibration,
    )
    adjusted = apply_upper_body_fk_latents(
        base_pose,
        UpperBodyLatentPrediction(
            values=fk_targets.values,
            masks={name: 0.0 for name in fk_targets.masks},
        ),
        calibration=calibration,
    )

    assert adjusted.joints == base_pose.joints


def test_ema_latent_provider_smooths_valid_values_and_resets() -> None:
    provider = _SequenceLatentProvider(
        [
            UpperBodyLatentPrediction(values={"x": 0.0}, masks={"x": 1.0}),
            UpperBodyLatentPrediction(values={"x": 10.0}, masks={"x": 1.0}),
            UpperBodyLatentPrediction(values={"x": 10.0}, masks={"x": 0.0}),
        ]
    )
    ema = EmaUpperBodyLatentProvider(provider=provider, alpha=0.25)
    sample = generate_three_point_samples([_pose_frame(0.0)])[0]

    first = ema.predict(sample, AvatarCalibration(), GroundingContext())
    second = ema.predict(sample, AvatarCalibration(), GroundingContext())
    masked = ema.predict(sample, AvatarCalibration(), GroundingContext())

    assert first.values["x"] == 0.0
    assert second.values["x"] == approx(2.5)
    assert masked.values["x"] == 10.0

    ema.reset()
    assert provider.index == 0


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


def _sample_with_head_yaw(
    timestamp: float,
    head_rotation: tuple[float, float, float, float],
) -> TrackedPoseSample:
    return TrackedPoseSample(
        timestamp_seconds=timestamp,
        hmd=TrackerState(
            role="head",
            transform=Transform(position=(0.0, 1.62, 0.0), rotation=head_rotation),
        ),
        left_hand=TrackerState(
            role="left_hand",
            transform=Transform(position=(1.0, 1.1, 0.0), rotation=head_rotation),
        ),
        right_hand=TrackerState(
            role="right_hand",
            transform=Transform(position=(-1.0, 1.1, 0.0), rotation=head_rotation),
        ),
    )


class _SequenceLatentProvider:
    def __init__(self, predictions: list[UpperBodyLatentPrediction]) -> None:
        self.predictions = predictions
        self.index = 0

    def predict(
        self,
        sample,
        calibration,
        grounding,
    ) -> UpperBodyLatentPrediction:
        del sample, calibration, grounding
        prediction = self.predictions[self.index]
        self.index += 1
        return prediction

    def reset(self) -> None:
        self.index = 0


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
