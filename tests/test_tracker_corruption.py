from __future__ import annotations

from math import cos, sin

import polars as pl
from pytest import approx

from validation.scripts.fit_tracker_noise import (
    _apply_run_filters,
    fit_tracker_noise,
)
from validation.vr_pose_validation.geometry import Transform
from validation.vr_pose_validation.noise_profile_io import (
    FITTED_PROFILE_SCHEMA,
    config_from_fitted_profiles,
    fitted_profile_document,
    fitted_profiles_from_document,
    load_fitted_profiles,
    write_fitted_profile,
)
from validation.vr_pose_validation.synthetic_trackers import (
    CONTROLLER_KIND,
    STRAP_TRACKER_KIND,
    NoiseProfile,
    SyntheticTrackerConfig,
    generate_multi_point_samples,
    generate_three_point_samples,
    role_default_profile,
)
from validation.vr_pose_validation.types import TrackerAttachment, TrackerRole


def _walking_frames(count: int) -> list[dict[str, Transform]]:
    frames = []
    for i in range(count):
        t = i * 0.05
        frames.append(
            {
                "head": Transform(position=(0.2 * sin(t), 1.6 + 0.02 * cos(t), 0.1 * sin(t))),
                "left_hand": Transform(position=(-0.3 + 0.1 * sin(t), 1.2, -0.2)),
                "right_hand": Transform(position=(0.3 + 0.1 * cos(t), 1.2, -0.2)),
            }
        )
    return frames


def _config(profile: NoiseProfile, seed: int = 7) -> SyntheticTrackerConfig:
    attachments: dict[TrackerRole, TrackerAttachment] = {
        "head": TrackerAttachment(role="head", source_joint="head", noise_profile_id="p"),
        "left_hand": TrackerAttachment(
            role="left_hand", source_joint="left_hand", noise_profile_id="p"
        ),
        "right_hand": TrackerAttachment(
            role="right_hand", source_joint="right_hand", noise_profile_id="p"
        ),
    }
    return SyntheticTrackerConfig(
        attachments=attachments,
        noise_profiles={"p": profile},
        seed=seed,
        frame_rate_hz=90.0,
    )


def test_corruption_is_deterministic_under_seed() -> None:
    profile = role_default_profile("left_hand", device_kind=CONTROLLER_KIND)
    frames = _walking_frames(200)
    first = generate_three_point_samples(frames, _config(profile))
    second = generate_three_point_samples(frames, _config(profile))

    assert [s.left_hand.transform.position for s in first] == [
        s.left_hand.transform.position for s in second
    ]
    assert [s.left_hand.valid for s in first] == [s.left_hand.valid for s in second]


def test_dropout_produces_multi_frame_runs() -> None:
    profile = NoiseProfile(
        dropout_probability=0.05,
        dropout_mean_run_frames=8.0,
        invalid_confidence=0.0,
    )
    frames = _walking_frames(1500)
    samples = generate_three_point_samples(frames, _config(profile))
    valids = [s.left_hand.valid for s in samples]

    runs = []
    run = 0
    for valid in valids:
        if not valid:
            run += 1
        elif run:
            runs.append(run)
            run = 0
    if run:
        runs.append(run)

    assert runs, "expected at least one dropout run"
    assert max(runs) > 1, "dropout should span multiple consecutive frames"
    assert sum(runs) / len(runs) > 2.0


def test_hold_last_variant_freezes_position_while_invalid() -> None:
    profile = NoiseProfile(
        dropout_probability=0.03,
        dropout_mean_run_frames=6.0,
        dropout_hold_last_probability=1.0,
        invalid_confidence=0.0,
    )
    frames = _walking_frames(600)
    samples = generate_three_point_samples(frames, _config(profile))

    for index in range(1, len(samples)):
        current = samples[index].left_hand
        previous = samples[index - 1].left_hand
        if not current.valid and not previous.valid:
            assert current.transform.position == previous.transform.position


def test_invalid_confidence_variant_reports_configured_confidence() -> None:
    profile = NoiseProfile(
        dropout_probability=1.0,
        dropout_mean_run_frames=1.0,
        dropout_hold_last_probability=0.0,
        invalid_confidence=0.2,
    )
    samples = generate_three_point_samples(_walking_frames(3), _config(profile))
    assert all(not s.left_hand.valid for s in samples)
    assert all(s.left_hand.confidence == approx(0.2) for s in samples)


def test_native_velocity_matches_pose_finite_difference_when_uncorrupted() -> None:
    profile = NoiseProfile(position_std_m=0.0, velocity_std_mps=0.0)
    config = _config(profile)
    frames = _walking_frames(50)
    samples = generate_three_point_samples(frames, config)
    dt = 1.0 / config.frame_rate_hz

    for index in range(2, len(samples)):
        current = samples[index].left_hand
        previous = samples[index - 1].left_hand
        expected = tuple(
            (current.transform.position[axis] - previous.transform.position[axis]) / dt
            for axis in range(3)
        )
        assert current.linear_velocity == approx(expected, abs=1e-9)


def test_native_velocity_carries_its_own_error_term() -> None:
    frames = _walking_frames(200)
    clean = generate_three_point_samples(frames, _config(NoiseProfile(velocity_std_mps=0.0)))
    noisy = generate_three_point_samples(frames, _config(NoiseProfile(velocity_std_mps=0.2)))

    # Native velocity error is independent of pose: poses match, velocities do not.
    assert [s.left_hand.transform.position for s in clean] == [
        s.left_hand.transform.position for s in noisy
    ]
    differences = [
        clean[i].left_hand.linear_velocity != noisy[i].left_hand.linear_velocity
        for i in range(2, len(frames))
    ]
    assert any(differences)


def test_finite_difference_of_corrupted_pose_is_self_consistent() -> None:
    # With a calibration bias the reported poses shift, and a downstream
    # finite-difference over consecutive reported samples still cancels the
    # constant bias out (the corruption is consistent frame-to-frame).
    profile = NoiseProfile(calibration_bias_std_m=0.03, velocity_std_mps=0.0)
    config = _config(profile)
    clean = generate_three_point_samples(_walking_frames(50), _config(NoiseProfile()))
    biased = generate_three_point_samples(_walking_frames(50), config)
    dt = 1.0 / config.frame_rate_hz

    for index in range(2, len(clean)):
        clean_fd = tuple(
            (
                clean[index].left_hand.transform.position[axis]
                - clean[index - 1].left_hand.transform.position[axis]
            )
            / dt
            for axis in range(3)
        )
        biased_fd = tuple(
            (
                biased[index].left_hand.transform.position[axis]
                - biased[index - 1].left_hand.transform.position[axis]
            )
            / dt
            for axis in range(3)
        )
        assert biased_fd == approx(clean_fd, abs=1e-9)


def test_calibration_bias_is_constant_within_session_and_varies_across_seeds() -> None:
    profile = NoiseProfile(calibration_bias_std_m=0.02)
    frames = _walking_frames(10)

    seed_a = generate_three_point_samples(frames, _config(profile, seed=1))
    seed_b = generate_three_point_samples(frames, _config(profile, seed=2))

    bias_a = tuple(
        seed_a[0].left_hand.transform.position[axis] - frames[0]["left_hand"].position[axis]
        for axis in range(3)
    )
    for sample, frame in zip(seed_a, frames, strict=True):
        offset = tuple(
            sample.left_hand.transform.position[axis] - frame["left_hand"].position[axis]
            for axis in range(3)
        )
        assert offset == approx(bias_a, abs=1e-9)

    bias_b = tuple(
        seed_b[0].left_hand.transform.position[axis] - frames[0]["left_hand"].position[axis]
        for axis in range(3)
    )
    assert bias_a != approx(bias_b)


def test_latency_delays_reported_position() -> None:
    profile = NoiseProfile(latency_frames=2)
    frames = _walking_frames(20)
    samples = generate_three_point_samples(frames, _config(profile))

    # With a 2-frame latency the reported position at frame i tracks the true
    # position at frame i-2 (offset-free profile).
    for index in range(2, len(frames)):
        expected = frames[index - 2]["left_hand"].position
        assert samples[index].left_hand.transform.position == approx(expected, abs=1e-9)


def test_multi_point_samples_include_optional_strap_trackers() -> None:
    attachments: dict[TrackerRole, TrackerAttachment] = {
        "head": TrackerAttachment(role="head", source_joint="head", noise_profile_id="hmd"),
        "left_hand": TrackerAttachment(
            role="left_hand", source_joint="left_hand", noise_profile_id="ctrl"
        ),
        "right_hand": TrackerAttachment(
            role="right_hand", source_joint="right_hand", noise_profile_id="ctrl"
        ),
        "pelvis": TrackerAttachment(role="pelvis", source_joint="hips", noise_profile_id="strap"),
    }
    config = SyntheticTrackerConfig(
        attachments=attachments,
        noise_profiles={
            "hmd": role_default_profile("head"),
            "ctrl": role_default_profile("left_hand"),
            "strap": role_default_profile("pelvis", device_kind=STRAP_TRACKER_KIND),
        },
        seed=3,
        frame_rate_hz=90.0,
    )
    frames = [
        {
            "head": Transform(position=(0.0, 1.6, 0.0)),
            "left_hand": Transform(position=(-0.3, 1.2, 0.0)),
            "right_hand": Transform(position=(0.3, 1.2, 0.0)),
            "hips": Transform(position=(0.0, 1.0, 0.0)),
        }
        for _ in range(5)
    ]
    samples = generate_multi_point_samples(frames, config)
    assert "pelvis" in samples[0].optional_trackers
    assert samples[0].optional_trackers["pelvis"].role == "pelvis"


def test_role_profiles_differ_by_device_kind() -> None:
    hmd = role_default_profile("head")
    controller = role_default_profile("left_hand")
    strap = role_default_profile("pelvis")
    assert hmd.device_kind == "hmd"
    assert controller.device_kind == "controller"
    assert strap.device_kind == "strap_tracker"
    # Strap trackers drift and drop out more than the HMD.
    assert strap.drift_amplitude_m > hmd.drift_amplitude_m
    assert strap.dropout_probability > hmd.dropout_probability


def test_fitted_profile_document_round_trips() -> None:
    profiles = {
        "head": role_default_profile("head"),
        "left_hand": role_default_profile("left_hand"),
    }
    document = fitted_profile_document(profiles, metadata={"note": "test"})
    assert document["schema"] == FITTED_PROFILE_SCHEMA
    restored = fitted_profiles_from_document(document)
    assert restored["head"] == profiles["head"]
    assert restored["left_hand"] == profiles["left_hand"]


def test_write_and_load_fitted_profile(tmp_path) -> None:
    profiles = {"head": role_default_profile("head")}
    path = tmp_path / "profile.json"
    write_fitted_profile(path, profiles, metadata={"sessions": 1})
    loaded = load_fitted_profiles(path)
    assert loaded["head"] == profiles["head"]

    config = config_from_fitted_profiles(loaded, frame_rate_hz=90.0)
    assert config.attachments["head"].noise_profile_id == "fitted_head"
    assert config.noise_profiles["fitted_head"] == profiles["head"]


def _devices_dataframe_from_samples(
    samples,
    *,
    run_name: str | None = None,
    run_kind: str | None = None,
    clip_index: int = 0,
) -> pl.DataFrame:
    role_to_state = {
        "head": lambda s: s.hmd,
        "left_hand": lambda s: s.left_hand,
        "right_hand": lambda s: s.right_hand,
    }
    class_by_kind = {
        "hmd": "HMD",
        "controller": "Controller",
        "strap_tracker": "CalibratedTracker",
    }
    rows = []
    for frame_index, sample in enumerate(samples):
        for role, accessor in role_to_state.items():
            state = accessor(sample)
            kind = state.attachment.noise_profile_id if state.attachment else "controller"
            row = {
                "session_id": "synthetic",
                "clip_index": clip_index,
                "frame_index": frame_index,
                "timestamp_seconds": sample.timestamp_seconds,
                "role": role,
                "device_class": class_by_kind.get(kind, "Controller"),
                "valid": state.valid,
                "confidence": state.confidence,
                "device_px": state.transform.position[0],
                "device_py": state.transform.position[1],
                "device_pz": state.transform.position[2],
                "linear_velocity_x": state.linear_velocity[0],
                "linear_velocity_y": state.linear_velocity[1],
                "linear_velocity_z": state.linear_velocity[2],
                "angular_velocity_x": state.angular_velocity[0],
                "angular_velocity_y": state.angular_velocity[1],
                "angular_velocity_z": state.angular_velocity[2],
            }
            if run_name is not None:
                row["run_name"] = run_name
            if run_kind is not None:
                row["run_kind"] = run_kind
            rows.append(row)
    return pl.DataFrame(rows)


def test_fitter_recovers_known_dropout_parameters() -> None:
    profile = NoiseProfile(
        position_std_m=0.0,
        dropout_probability=0.02,
        dropout_mean_run_frames=8.0,
        dropout_hold_last_probability=0.5,
        invalid_confidence=0.0,
        velocity_std_mps=0.0,
    )
    attachments: dict[TrackerRole, TrackerAttachment] = {
        "head": TrackerAttachment(role="head", source_joint="head", noise_profile_id="controller"),
        "left_hand": TrackerAttachment(
            role="left_hand", source_joint="left_hand", noise_profile_id="controller"
        ),
        "right_hand": TrackerAttachment(
            role="right_hand", source_joint="right_hand", noise_profile_id="controller"
        ),
    }
    config = SyntheticTrackerConfig(
        attachments=attachments,
        noise_profiles={"controller": profile},
        seed=11,
        frame_rate_hz=90.0,
    )
    frames = _walking_frames(6000)
    samples = generate_three_point_samples(frames, config)
    devices = _devices_dataframe_from_samples(samples)

    result = fit_tracker_noise(devices, frame_rate_hz=90.0)
    fitted = result["roles"]["left_hand"]["profile"]

    assert fitted.dropout_probability == approx(0.02, rel=0.4)
    assert fitted.dropout_mean_run_frames == approx(8.0, rel=0.4)
    assert fitted.dropout_hold_last_probability == approx(0.5, abs=0.25)
    assert fitted.device_kind == CONTROLLER_KIND
    assert fitted.velocity_std_mps == approx(0.0, abs=0.02)


def test_fitter_recovers_jitter_and_velocity_error() -> None:
    profile = NoiseProfile(
        position_std_m=0.002,
        velocity_std_mps=0.05,
        dropout_probability=0.0,
    )
    attachments: dict[TrackerRole, TrackerAttachment] = {
        "head": TrackerAttachment(role="head", source_joint="head", noise_profile_id="strap"),
        "left_hand": TrackerAttachment(
            role="left_hand", source_joint="left_hand", noise_profile_id="strap"
        ),
        "right_hand": TrackerAttachment(
            role="right_hand", source_joint="right_hand", noise_profile_id="strap"
        ),
    }
    config = SyntheticTrackerConfig(
        attachments=attachments,
        noise_profiles={"strap": profile},
        seed=5,
        frame_rate_hz=90.0,
    )
    # Idle frames so jitter dominates the position residual.
    frames = [
        {
            "head": Transform(position=(0.0, 1.6, 0.0)),
            "left_hand": Transform(position=(-0.3, 1.2, -0.2)),
            "right_hand": Transform(position=(0.3, 1.2, -0.2)),
        }
        for _ in range(3000)
    ]
    samples = generate_three_point_samples(frames, config)
    devices = _devices_dataframe_from_samples(samples)

    fitted = fit_tracker_noise(devices, frame_rate_hz=90.0)["roles"]["left_hand"]["profile"]
    assert fitted.position_std_m == approx(0.002, rel=0.5)
    # Velocity discrepancy combines native velocity noise and jitter finite diff.
    assert fitted.velocity_std_mps > 0.05


def test_fitter_recovers_native_velocity_sigma() -> None:
    # No jitter/drift/latency: the only velocity discrepancy is the per-axis
    # native-velocity Gaussian, so the fitter recovers its sigma directly (the
    # 3D-magnitude -> per-axis-sigma conversion is what makes this exact).
    profile = NoiseProfile(
        position_std_m=0.0,
        velocity_std_mps=0.05,
        dropout_probability=0.0,
    )
    attachments: dict[TrackerRole, TrackerAttachment] = {
        "head": TrackerAttachment(role="head", source_joint="head", noise_profile_id="ctrl"),
        "left_hand": TrackerAttachment(
            role="left_hand", source_joint="left_hand", noise_profile_id="ctrl"
        ),
        "right_hand": TrackerAttachment(
            role="right_hand", source_joint="right_hand", noise_profile_id="ctrl"
        ),
    }
    config = SyntheticTrackerConfig(
        attachments=attachments,
        noise_profiles={"ctrl": profile},
        seed=5,
        frame_rate_hz=90.0,
    )
    frames = _walking_frames(4000)
    samples = generate_three_point_samples(frames, config)
    devices = _devices_dataframe_from_samples(samples)

    fitted = fit_tracker_noise(devices, frame_rate_hz=90.0)["roles"]["left_hand"]["profile"]
    # Per-axis sigma recovered, not the ~1.60x-inflated mean magnitude: rel=0.1.
    assert fitted.velocity_std_mps == approx(0.05, rel=0.1)


def test_fitter_recovers_excursion_magnitude_sigma() -> None:
    # Frequent short dropouts so reacquire excursions are plentiful; excursion
    # offset is a per-axis Gaussian, and the fitter must divide the mean
    # reacquire displacement by 2*sqrt(2/pi) to recover the generator's sigma.
    profile = NoiseProfile(
        position_std_m=0.0,
        velocity_std_mps=0.0,
        dropout_probability=0.05,
        dropout_mean_run_frames=3.0,
        dropout_hold_last_probability=0.0,
        invalid_confidence=0.0,
        excursion_magnitude_m=0.1,
        excursion_frames=2,
    )
    attachments: dict[TrackerRole, TrackerAttachment] = {
        "head": TrackerAttachment(role="head", source_joint="head", noise_profile_id="ctrl"),
        "left_hand": TrackerAttachment(
            role="left_hand", source_joint="left_hand", noise_profile_id="ctrl"
        ),
        "right_hand": TrackerAttachment(
            role="right_hand", source_joint="right_hand", noise_profile_id="ctrl"
        ),
    }
    # Stationary frames so reacquire displacement is dominated by the excursion,
    # not by true motion between the pre-dropout and reacquire frame.
    frames = [
        {
            "head": Transform(position=(0.0, 1.6, 0.0)),
            "left_hand": Transform(position=(-0.3, 1.2, -0.2)),
            "right_hand": Transform(position=(0.3, 1.2, -0.2)),
        }
        for _ in range(8000)
    ]
    config = SyntheticTrackerConfig(
        attachments=attachments,
        noise_profiles={"ctrl": profile},
        seed=13,
        frame_rate_hz=90.0,
    )
    samples = generate_three_point_samples(frames, config)
    devices = _devices_dataframe_from_samples(samples)

    fitted = fit_tracker_noise(devices, frame_rate_hz=90.0)["roles"]["left_hand"]["profile"]
    assert fitted.excursion_magnitude_m == approx(0.1, rel=0.15)


def test_hold_last_runs_do_not_contaminate_invalid_confidence() -> None:
    # Half the dropouts hold-last at full confidence (1.0); the fitted
    # invalid_confidence must reflect only the genuine invalid-confidence runs
    # (0.0 here), not the ~0.5 average of mixing both classes.
    profile = NoiseProfile(
        position_std_m=0.0,
        velocity_std_mps=0.0,
        dropout_probability=0.05,
        dropout_mean_run_frames=5.0,
        dropout_hold_last_probability=0.5,
        invalid_confidence=0.0,
    )
    attachments: dict[TrackerRole, TrackerAttachment] = {
        "head": TrackerAttachment(role="head", source_joint="head", noise_profile_id="ctrl"),
        "left_hand": TrackerAttachment(
            role="left_hand", source_joint="left_hand", noise_profile_id="ctrl"
        ),
        "right_hand": TrackerAttachment(
            role="right_hand", source_joint="right_hand", noise_profile_id="ctrl"
        ),
    }
    config = SyntheticTrackerConfig(
        attachments=attachments,
        noise_profiles={"ctrl": profile},
        seed=17,
        frame_rate_hz=90.0,
    )
    frames = _walking_frames(8000)
    samples = generate_three_point_samples(frames, config)
    devices = _devices_dataframe_from_samples(samples)

    fitted = fit_tracker_noise(devices, frame_rate_hz=90.0)["roles"]["left_hand"]["profile"]
    assert fitted.invalid_confidence == approx(0.0, abs=0.02)
    assert fitted.dropout_hold_last_probability == approx(0.5, abs=0.2)


def test_invalid_confidence_recovered_above_half() -> None:
    # A device whose genuine invalid confidence exceeds 0.5 must still be
    # classified as invalid-confidence (not hold-last): the relative threshold
    # anchors on the valid-frame level (1.0), not a hardcoded 0.5 cut.
    profile = NoiseProfile(
        position_std_m=0.0,
        velocity_std_mps=0.0,
        dropout_probability=0.05,
        dropout_mean_run_frames=4.0,
        dropout_hold_last_probability=0.0,
        invalid_confidence=0.7,
    )
    attachments: dict[TrackerRole, TrackerAttachment] = {
        "head": TrackerAttachment(role="head", source_joint="head", noise_profile_id="ctrl"),
        "left_hand": TrackerAttachment(
            role="left_hand", source_joint="left_hand", noise_profile_id="ctrl"
        ),
        "right_hand": TrackerAttachment(
            role="right_hand", source_joint="right_hand", noise_profile_id="ctrl"
        ),
    }
    config = SyntheticTrackerConfig(
        attachments=attachments,
        noise_profiles={"ctrl": profile},
        seed=19,
        frame_rate_hz=90.0,
    )
    # Mix of valid and invalid frames so both a valid-frame level and invalid
    # runs exist for the relative classifier to anchor against.
    frames = _walking_frames(4000)
    samples = generate_three_point_samples(frames, config)
    devices = _devices_dataframe_from_samples(samples)

    fitted = fit_tracker_noise(devices, frame_rate_hz=90.0)["roles"]["left_hand"]["profile"]
    assert fitted.invalid_confidence == approx(0.7, abs=0.02)
    assert fitted.dropout_hold_last_probability == approx(0.0, abs=0.05)


def _dropout_config(profile: NoiseProfile, seed: int) -> SyntheticTrackerConfig:
    attachments: dict[TrackerRole, TrackerAttachment] = {
        "head": TrackerAttachment(role="head", source_joint="head", noise_profile_id="ctrl"),
        "left_hand": TrackerAttachment(
            role="left_hand", source_joint="left_hand", noise_profile_id="ctrl"
        ),
        "right_hand": TrackerAttachment(
            role="right_hand", source_joint="right_hand", noise_profile_id="ctrl"
        ),
    }
    return SyntheticTrackerConfig(
        attachments=attachments,
        noise_profiles={"ctrl": profile},
        seed=seed,
        frame_rate_hz=90.0,
    )


def test_run_filters_restrict_rows() -> None:
    devices = pl.DataFrame(
        {
            "role": ["left_hand", "left_hand", "left_hand"],
            "run_name": ["idle", "range_of_motion", "idle"],
            "run_kind": ["idle", "motion", "idle"],
            "device_px": [0.0, 1.0, 2.0],
        }
    )

    by_name = _apply_run_filters(devices, ["idle"], [])
    assert by_name["device_px"].to_list() == [0.0, 2.0]

    by_kind = _apply_run_filters(devices, [], ["motion"])
    assert by_kind["device_px"].to_list() == [1.0]

    both = _apply_run_filters(devices, ["idle"], ["motion"])
    assert both.is_empty()


def test_run_filter_missing_column_yields_empty_selection(capsys) -> None:
    devices = pl.DataFrame({"role": ["left_hand"], "device_px": [0.0]})
    filtered = _apply_run_filters(devices, ["idle"], [])
    assert filtered.is_empty()
    assert "run_name" in capsys.readouterr().out


def test_dropout_fit_only_from_occlusion_rows() -> None:
    # A run with heavy dropout but kind="motion" must NOT feed dropout params;
    # only the occlusion run does, so the fitted dropout reflects the occlusion
    # rows despite the motion rows having their own (different) dropout.
    occlusion_profile = NoiseProfile(
        position_std_m=0.0,
        velocity_std_mps=0.0,
        dropout_probability=0.05,
        dropout_mean_run_frames=8.0,
        dropout_hold_last_probability=0.5,
        invalid_confidence=0.0,
    )
    motion_profile = NoiseProfile(
        position_std_m=0.0,
        velocity_std_mps=0.0,
        dropout_probability=0.0,
    )
    frames = _walking_frames(6000)
    occlusion_samples = generate_three_point_samples(
        frames, _dropout_config(occlusion_profile, seed=11)
    )
    motion_samples = generate_three_point_samples(frames, _dropout_config(motion_profile, seed=23))
    occlusion_devices = _devices_dataframe_from_samples(
        occlusion_samples, run_name="occlusion", run_kind="occlusion", clip_index=0
    )
    motion_devices = _devices_dataframe_from_samples(
        motion_samples, run_name="range_of_motion", run_kind="motion", clip_index=1
    )
    devices = pl.concat([occlusion_devices, motion_devices])

    entry = fit_tracker_noise(devices, frame_rate_hz=90.0)["roles"]["left_hand"]
    fitted = entry["profile"]

    assert not entry["dropout_from_priors"]
    assert fitted.dropout_probability == approx(0.05, rel=0.4)
    assert fitted.dropout_mean_run_frames == approx(8.0, rel=0.4)
    # Motion rows contributed zero dropout; if they leaked in, run count would be
    # unchanged but p_onset would drop toward half. Occlusion-only keeps it near.
    assert fitted.dropout_probability > 0.02


def test_dropout_falls_back_to_priors_without_occlusion_rows() -> None:
    profile = NoiseProfile(
        position_std_m=0.0,
        velocity_std_mps=0.0,
        dropout_probability=0.05,
        dropout_mean_run_frames=8.0,
        invalid_confidence=0.0,
    )
    frames = _walking_frames(4000)
    samples = generate_three_point_samples(frames, _dropout_config(profile, seed=11))
    devices = _devices_dataframe_from_samples(
        samples, run_name="range_of_motion", run_kind="motion"
    )

    entry = fit_tracker_noise(devices, frame_rate_hz=90.0)["roles"]["left_hand"]
    fitted = entry["profile"]
    prior = role_default_profile("left_hand", device_kind=CONTROLLER_KIND)

    assert entry["dropout_from_priors"]
    assert fitted.dropout_probability == prior.dropout_probability
    assert fitted.dropout_mean_run_frames == prior.dropout_mean_run_frames
    assert fitted.excursion_magnitude_m == prior.excursion_magnitude_m
    assert "dropout_probability" in entry["unfitted_fields"]
    assert "excursion_magnitude_m" in entry["unfitted_fields"]


def test_legacy_no_run_kind_column_fits_dropout_from_all_rows() -> None:
    profile = NoiseProfile(
        position_std_m=0.0,
        velocity_std_mps=0.0,
        dropout_probability=0.05,
        dropout_mean_run_frames=8.0,
        dropout_hold_last_probability=0.5,
        invalid_confidence=0.0,
    )
    frames = _walking_frames(6000)
    samples = generate_three_point_samples(frames, _dropout_config(profile, seed=11))
    devices = _devices_dataframe_from_samples(samples)
    assert "run_kind" not in devices.columns

    entry = fit_tracker_noise(devices, frame_rate_hz=90.0)["roles"]["left_hand"]
    fitted = entry["profile"]

    assert not entry["dropout_from_priors"]
    assert fitted.dropout_probability == approx(0.05, rel=0.4)
    assert fitted.dropout_mean_run_frames == approx(8.0, rel=0.4)
    assert "dropout_probability" not in entry["unfitted_fields"]


def test_three_point_and_multi_point_agree_on_shared_roles() -> None:
    # Per-role derived RNGs make each role's corruption stream independent of the
    # role set, so the shared head/hand roles corrupt identically whether or not
    # extra strap trackers are present.
    profile = role_default_profile("left_hand", device_kind=CONTROLLER_KIND)
    frames = _walking_frames(300)

    three_point_attachments: dict[TrackerRole, TrackerAttachment] = {
        "head": TrackerAttachment(role="head", source_joint="head", noise_profile_id="p"),
        "left_hand": TrackerAttachment(
            role="left_hand", source_joint="left_hand", noise_profile_id="p"
        ),
        "right_hand": TrackerAttachment(
            role="right_hand", source_joint="right_hand", noise_profile_id="p"
        ),
    }
    multi_attachments = dict(three_point_attachments)
    multi_attachments["pelvis"] = TrackerAttachment(
        role="pelvis", source_joint="left_hand", noise_profile_id="p"
    )

    three_config = SyntheticTrackerConfig(
        attachments=three_point_attachments,
        noise_profiles={"p": profile},
        seed=7,
        frame_rate_hz=90.0,
    )
    multi_config = SyntheticTrackerConfig(
        attachments=multi_attachments,
        noise_profiles={"p": profile},
        seed=7,
        frame_rate_hz=90.0,
    )

    three = generate_three_point_samples(frames, three_config)
    multi = generate_multi_point_samples(frames, multi_config)

    assert [s.left_hand.transform.position for s in three] == [
        s.left_hand.transform.position for s in multi
    ]
    assert [s.left_hand.valid for s in three] == [s.left_hand.valid for s in multi]
    assert [s.hmd.transform.position for s in three] == [s.hmd.transform.position for s in multi]
