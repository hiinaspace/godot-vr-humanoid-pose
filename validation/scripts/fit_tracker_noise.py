"""Fit a tracker-noise profile from one or more live-capture session directories.

Reads the Parquet output of the live-capture recorder (devices.parquet), groups
device rows by tracker role, and estimates the corruption statistics used by
``synthetic_trackers.NoiseProfile``: run-length dropout, excursion magnitude,
position jitter, native-vs-finite-difference velocity error, and low-frequency
drift. Roles absent from the sessions are skipped; per-role sample counts are
reported. Parameters that cannot be fit without ground truth (calibration bias,
latency, prediction overshoot) fall back to the invented role priors and are
flagged in the summary.

Run-kind provenance policy: each parameter is fit only from the run whose motion
exercises it representatively.
- Dropout/excursion fields (dropout_probability, dropout_mean_run_frames,
  dropout_hold_last_probability, invalid_confidence, excursion_magnitude_m) fit
  only from rows whose ``run_kind == "occlusion"``.
- Stationary-noise fields (position_std_m, drift_amplitude_m, velocity_std_mps,
  angular_velocity_std_radps) fit only from rows whose ``run_kind == "idle"``.
  Pooling motion runs folds real body motion into these terms and inflates the
  fitted jitter/velocity by an order of magnitude.
When a role has no rows of the required kind, the invented role prior is used for
those specific fields and they are flagged in the summary. Legacy captures with
no ``run_kind`` column preserve the original behavior (every field fit from all
rows). The ``--run``/``--run-kind`` filters restrict which rows feed the fit
before this policy is applied.

Base-station rows (role ``tracking_reference``) are excluded: they define the
tracking origin rather than reporting a tracked pose, so they are never a
corruption-model attachment and their in-stream jitter/drift is zero by
construction.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from math import pi
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

import polars as pl

from validation.vr_pose_validation.noise_profile_io import write_fitted_profile
from validation.vr_pose_validation.synthetic_trackers import (
    DEFAULT_ROLE_DEVICE_KINDS,
    NoiseProfile,
    role_default_profile,
)

DEVICE_CLASS_TO_KIND = {
    "HMD": "hmd",
    "Controller": "controller",
    "CalibratedTracker": "strap_tracker",
    "GenericTracker": "strap_tracker",
}
IDLE_SPEED_THRESHOLD_MPS = 0.05
OCCLUSION_RUN_KIND = "occlusion"
# Dropout/excursion statistics only have representative provenance when captured
# in a dedicated occlusion run; for any other run_kind we fall back to the role
# prior for these fields (see the provenance policy in the module docstring).
OCCLUSION_ONLY_FIELDS = (
    "dropout_probability",
    "dropout_mean_run_frames",
    "dropout_hold_last_probability",
    "invalid_confidence",
    "excursion_magnitude_m",
)
STATIONARY_RUN_KIND = "idle"
# Jitter, drift, velocity, and angular noise are only measurable while the body
# is still; a moving run folds real motion into them. Fit from the idle run and
# fall back to the role prior when no idle run is present.
STATIONARY_ONLY_FIELDS = (
    "position_std_m",
    "drift_amplitude_m",
    "velocity_std_mps",
    "angular_velocity_std_radps",
)
# Base stations define the tracking origin; they are not a corruption-model
# attachment role and their in-stream pose is static by construction.
EXCLUDED_ROLES = frozenset({"tracking_reference"})
UNFITTABLE_FIELDS = (
    "calibration_bias_std_m",
    "latency_frames",
    "overshoot_gain_seconds",
    "excursion_frames",
    "drift_period_seconds",
)

# The generator draws excursion offsets and velocity error as per-axis Gaussian
# vectors of standard deviation sigma (see synthetic_trackers._gaussian_vector).
# The fitter observes only the Euclidean magnitude of such a 3D vector, whose
# mean is sigma * 2 * sqrt(2 / pi). Dividing the mean magnitude by this factor
# recovers the per-axis sigma the generator consumes, so fitted parameters mean
# the same thing the generator was fed.
_GAUSSIAN_VECTOR_MAGNITUDE_MEAN_OVER_SIGMA = 2.0 * (2.0 / pi) ** 0.5


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "sessions",
        nargs="+",
        type=Path,
        help="Live-capture session directories (or parents containing session=* folders).",
    )
    parser.add_argument("--output", type=Path, help="Fitted-profile JSON output path.")
    parser.add_argument(
        "--frame-rate-hz",
        type=float,
        default=90.0,
        help="Assumed capture frame rate when device timestamps are unavailable.",
    )
    parser.add_argument(
        "--run",
        dest="runs",
        action="append",
        default=[],
        metavar="NAME",
        help="Only fit device rows whose run_name matches NAME (repeatable).",
    )
    parser.add_argument(
        "--run-kind",
        dest="run_kinds",
        action="append",
        default=[],
        metavar="KIND",
        help="Only fit device rows whose run_kind matches KIND (repeatable).",
    )
    args = parser.parse_args(argv)

    session_dirs = _resolve_sessions(args.sessions)
    devices = _load_devices(session_dirs)
    devices = _apply_run_filters(devices, args.runs, args.run_kinds)
    summary = fit_tracker_noise(devices, frame_rate_hz=args.frame_rate_hz)

    if args.output is not None:
        write_fitted_profile(
            args.output,
            {role: entry["profile"] for role, entry in summary["roles"].items()},
            metadata={
                "sessions": [str(path) for path in session_dirs],
                "frame_rate_hz": args.frame_rate_hz,
                "device_rows": devices.height,
            },
        )
    _print_summary(summary, session_dirs, args.output)
    return 0


def _resolve_sessions(paths: Sequence[Path]) -> list[Path]:
    resolved: list[Path] = []
    for path in paths:
        if (path / "devices.parquet").exists():
            resolved.append(path)
            continue
        resolved.extend(
            sorted(
                candidate
                for candidate in path.glob("session=*")
                if (candidate / "devices.parquet").exists()
            )
        )
    if not resolved:
        raise FileNotFoundError("No live-capture sessions with devices.parquet found")
    return resolved


def _apply_run_filters(
    devices: pl.DataFrame,
    runs: Sequence[str],
    run_kinds: Sequence[str],
) -> pl.DataFrame:
    for values, column in ((runs, "run_name"), (run_kinds, "run_kind")):
        if not values:
            continue
        if column not in devices.columns:
            print(
                f"Filter --{column.replace('_', '-')} given but column "
                f"'{column}' is absent; no rows match."
            )
            return devices.clear()
        devices = devices.filter(pl.col(column).cast(pl.Utf8).is_in(list(values)))
    return devices


def _load_devices(session_dirs: Sequence[Path]) -> pl.DataFrame:
    frames = []
    for session_dir in session_dirs:
        table = pl.read_parquet(session_dir / "devices.parquet")
        if table.is_empty():
            continue
        table = table.with_columns(pl.lit(str(session_dir)).alias("_session_dir"))
        frames.append(table)
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal_relaxed")


def fit_tracker_noise(
    devices: pl.DataFrame,
    *,
    frame_rate_hz: float = 90.0,
) -> dict[str, Any]:
    roles: dict[str, dict[str, Any]] = {}
    if devices.is_empty() or "role" not in devices.columns:
        return {"roles": roles, "frame_rate_hz": frame_rate_hz}

    for role in sorted(str(role) for role in devices["role"].unique() if role):
        if role in EXCLUDED_ROLES:
            continue
        role_rows = devices.filter(pl.col("role") == role)
        entry = _fit_role(role, role_rows, frame_rate_hz)
        if entry is not None:
            roles[role] = entry
    return {"roles": roles, "frame_rate_hz": frame_rate_hz}


def _fit_role(role: str, rows: pl.DataFrame, frame_rate_hz: float) -> dict[str, Any] | None:
    if rows.is_empty():
        return None
    kind = _device_kind(role, rows)
    prior = role_default_profile(role, device_kind=kind)
    dt = 1.0 / frame_rate_hz if frame_rate_hz > 0.0 else 0.0

    # Legacy captures without a run_kind column fit every field from all rows.
    # When the column is present, dropout/excursion fit only from occlusion rows
    # and stationary noise only from idle rows; the rest fall back to the prior.
    has_run_kind = "run_kind" in rows.columns
    if has_run_kind:
        run_kind = pl.col("run_kind").cast(pl.Utf8)
        dropout_rows = rows.filter(run_kind == OCCLUSION_RUN_KIND)
        stationary_rows = rows.filter(run_kind == STATIONARY_RUN_KIND)
    else:
        dropout_rows = rows
        stationary_rows = rows

    samples = rows.height
    jitter_values: list[float] = []
    drift_values: list[float] = []
    velocity_errors: list[float] = []
    angular_speeds: list[float] = []

    for _, group in _iter_tracks(stationary_rows):
        positions, valids, _confidences, native_velocities, angular = _track_arrays(group)
        _accumulate_jitter_and_drift(positions, valids, dt, jitter_values, drift_values)
        _accumulate_velocity_error(positions, native_velocities, valids, dt, velocity_errors)
        _accumulate_angular(angular, valids, angular_speeds)

    dropout_samples = 0
    dropout_frames = 0
    run_lengths: list[int] = []
    hold_runs = 0
    invalid_confidences: list[float] = []
    excursion_distances: list[float] = []

    for _, group in _iter_tracks(dropout_rows):
        positions, valids, confidences, _native_velocities, _angular = _track_arrays(group)
        dropout_samples += len(positions)
        _accumulate_dropout(valids, run_lengths)
        dropout_frames += sum(1 for valid in valids if not valid)
        holds, invalid_conf = _classify_invalid_runs(valids, confidences)
        hold_runs += holds
        invalid_confidences.extend(invalid_conf)
        _accumulate_excursion(positions, valids, excursion_distances)

    total_runs = len(run_lengths)
    eligible = max(dropout_samples - dropout_frames, 1)

    # When a run_kind column exists but yields no occlusion rows, dropout and
    # excursion parameters fall back to the role prior rather than to the empty
    # non-occlusion accumulation.
    dropout_from_priors = has_run_kind and dropout_rows.is_empty()
    if dropout_from_priors:
        dropout_probability = prior.dropout_probability
        dropout_mean_run_frames = prior.dropout_mean_run_frames
        dropout_hold_last_probability = prior.dropout_hold_last_probability
        invalid_confidence = prior.invalid_confidence
        excursion_magnitude_m = prior.excursion_magnitude_m
    else:
        dropout_probability = (total_runs / eligible) if total_runs else prior.dropout_probability
        dropout_mean_run_frames = (
            fmean(run_lengths) if run_lengths else prior.dropout_mean_run_frames
        )
        dropout_hold_last_probability = (
            hold_runs / total_runs if total_runs else prior.dropout_hold_last_probability
        )
        invalid_confidence = _stat(invalid_confidences, prior.invalid_confidence, use_mean=True)
        excursion_magnitude_m = _axis_sigma_from_magnitudes(
            excursion_distances, prior.excursion_magnitude_m
        )

    profile = NoiseProfile(
        position_std_m=_stat(jitter_values, prior.position_std_m, use_mean=True),
        dropout_probability=dropout_probability,
        invalid_confidence=invalid_confidence,
        device_kind=kind,
        dropout_mean_run_frames=dropout_mean_run_frames,
        dropout_hold_last_probability=dropout_hold_last_probability,
        excursion_magnitude_m=excursion_magnitude_m,
        excursion_frames=prior.excursion_frames,
        calibration_bias_std_m=prior.calibration_bias_std_m,
        drift_amplitude_m=_stat(drift_values, prior.drift_amplitude_m, use_mean=True),
        drift_period_seconds=prior.drift_period_seconds,
        latency_frames=prior.latency_frames,
        overshoot_gain_seconds=prior.overshoot_gain_seconds,
        velocity_std_mps=_axis_sigma_from_magnitudes(velocity_errors, prior.velocity_std_mps),
        angular_velocity_std_radps=_axis_sigma_from_magnitudes(
            angular_speeds, prior.angular_velocity_std_radps
        ),
    )
    # The stationary accumulators are empty when no idle rows exist, so the
    # profile already falls back to the prior via _stat; flag it for the summary.
    stationary_from_priors = has_run_kind and stationary_rows.is_empty()

    unfitted_fields = list(UNFITTABLE_FIELDS)
    if dropout_from_priors:
        unfitted_fields.extend(OCCLUSION_ONLY_FIELDS)
    if stationary_from_priors:
        unfitted_fields.extend(STATIONARY_ONLY_FIELDS)
    return {
        "profile": profile,
        "samples": samples,
        "dropout_frames": dropout_frames,
        "dropout_runs": total_runs,
        "dropout_from_priors": dropout_from_priors,
        "stationary_from_priors": stationary_from_priors,
        "sessions": rows["_session_dir"].n_unique() if "_session_dir" in rows.columns else 1,
        "unfitted_fields": unfitted_fields,
    }


def _device_kind(role: str, rows: pl.DataFrame) -> str:
    if "device_class" in rows.columns:
        classes = [str(value) for value in rows["device_class"].drop_nulls().unique() if value]
        for device_class in classes:
            if device_class in DEVICE_CLASS_TO_KIND:
                return DEVICE_CLASS_TO_KIND[device_class]
    for known_role, kind in DEFAULT_ROLE_DEVICE_KINDS.items():
        if known_role == role:
            return kind
    return "controller"


def _iter_tracks(rows: pl.DataFrame):
    # A track is one physical device's continuous stream. Several devices can
    # share a role (e.g. two base stations under tracking_reference, or several
    # uncalibrated straps under generic_tracker); without a per-device key they
    # interleave frame-by-frame and the finite differences jump between devices,
    # inflating jitter and velocity into physically impossible values.
    device_key = next((key for key in ("device_id", "serial_hash") if key in rows.columns), None)
    keys = [key for key in ("_session_dir", "session_id", "clip_index") if key in rows.columns]
    if device_key:
        keys.append(device_key)
    sort_keys = keys + (["frame_index"] if "frame_index" in rows.columns else [])
    ordered = rows.sort(sort_keys) if sort_keys else rows
    if not keys:
        yield (None, ordered)
        return
    yield from ordered.group_by(keys, maintain_order=True)


def _track_arrays(group: pl.DataFrame):
    positions = list(
        zip(
            group["device_px"].to_list(),
            group["device_py"].to_list(),
            group["device_pz"].to_list(),
            strict=False,
        )
    )
    valids = (
        [bool(value) for value in group["valid"].to_list()]
        if "valid" in group.columns
        else [True] * len(positions)
    )
    confidences = (
        [float(value) for value in group["confidence"].to_list()]
        if "confidence" in group.columns
        else [1.0] * len(positions)
    )
    native_velocities = _optional_vectors(group, "linear_velocity")
    angular = _optional_vectors(group, "angular_velocity")
    return positions, valids, confidences, native_velocities, angular


def _optional_vectors(group: pl.DataFrame, prefix: str) -> list[tuple[float, float, float]] | None:
    columns = [f"{prefix}_{axis}" for axis in ("x", "y", "z")]
    if not all(column in group.columns for column in columns):
        return None
    values = list(zip(*(group[column].to_list() for column in columns), strict=False))
    if any(any(component is None for component in row) for row in values):
        return None
    return [(float(x), float(y), float(z)) for x, y, z in values]


def _accumulate_dropout(
    valids: Sequence[bool],
    run_lengths: list[int],
) -> None:
    run = 0
    for valid in valids:
        if not valid:
            run += 1
        elif run > 0:
            run_lengths.append(run)
            run = 0
    if run > 0:
        run_lengths.append(run)


def _invalid_runs(
    valids: Sequence[bool],
    confidences: Sequence[float],
) -> list[tuple[int, int, float]]:
    """Return ``(start, stop, mean_confidence)`` for each invalid run."""

    runs: list[tuple[int, int, float]] = []
    index = 0
    n = len(valids)
    while index < n:
        if valids[index]:
            index += 1
            continue
        start = index
        while index < n and not valids[index]:
            index += 1
        runs.append((start, index, fmean(confidences[start:index])))
    return runs


def _classify_invalid_runs(
    valids: Sequence[bool],
    confidences: Sequence[float],
) -> tuple[int, list[float]]:
    """Split invalid runs into hold-last vs invalid-confidence variants.

    A hold-last dropout keeps streaming the last pose at the valid-frame
    confidence level, while an invalid-confidence dropout reports a distinctly
    lower confidence. Rather than a hardcoded 0.5 cut (which misclassifies
    devices whose genuine invalid confidence exceeds 0.5), classify each run by
    whether its mean confidence sits closer to the valid-frame confidence level
    or to the lowest observed invalid-run confidence.

    Returns the hold-last run count and the per-frame confidences from the
    invalid-confidence runs only, so the fitted ``invalid_confidence`` is not
    contaminated by full-confidence hold-last frames.
    """

    runs = _invalid_runs(valids, confidences)
    if not runs:
        return 0, []

    valid_confidences = [confidences[i] for i, valid in enumerate(valids) if valid]
    valid_level = fmean(valid_confidences) if valid_confidences else max(c for _, _, c in runs)
    lowest_invalid = min(c for _, _, c in runs)

    # When every invalid run streams at the valid-frame level there is no
    # invalid-confidence class to anchor against: treat them all as hold-last.
    span = valid_level - lowest_invalid
    if span <= 1e-9:
        return len(runs), []

    holds = 0
    invalid_confidences: list[float] = []
    for start, stop, mean_conf in runs:
        closer_to_valid = (valid_level - mean_conf) <= (mean_conf - lowest_invalid)
        if closer_to_valid:
            holds += 1
        else:
            invalid_confidences.extend(confidences[start:stop])
    return holds, invalid_confidences


def _accumulate_excursion(
    positions: Sequence[tuple[float, float, float]],
    valids: Sequence[bool],
    out: list[float],
) -> None:
    # Only reacquire (invalid -> valid) adds a measurable excursion: the generator
    # applies the offset on the first valid frame after a dropout. Measure it
    # against the last *valid* pose before the dropout, not the preceding frame:
    # real captures stream a null position while invalid (so positions[index - 1]
    # is None), and even held-last dropouts equal that last valid pose. Measuring
    # at onset too would fold ~0-displacement samples into the mean and halve the
    # fitted magnitude.
    last_valid: tuple[float, float, float] | None = None
    prev_valid = False
    for index in range(len(positions)):
        position = positions[index]
        usable = valids[index] and position is not None and None not in position
        if usable:
            if not prev_valid and last_valid is not None:
                out.append(_distance(position, last_valid))
            last_valid = position
        prev_valid = usable


def _accumulate_jitter_and_drift(
    positions: Sequence[tuple[float, float, float]],
    valids: Sequence[bool],
    dt: float,
    jitter_out: list[float],
    drift_out: list[float],
) -> None:
    if dt <= 0.0 or len(positions) < 5:
        return
    window = min(9, len(positions) // 2 * 2 + 1)
    for axis in range(3):
        series = [positions[i][axis] for i in range(len(positions))]
        # A frame is usable only when marked valid and carrying a real position.
        # Dropout frames stream a null (or held-last) position; letting them into
        # the windowed mean corrupts both the jitter residual and the drift
        # estimate — and a null trips fmean outright.
        usable = [valids[i] and series[i] is not None for i in range(len(series))]
        speeds: list[float] = [
            abs(series[i] - series[i - 1]) / dt if usable[i] and usable[i - 1] else float("inf")
            for i in range(1, len(series))
        ]
        residuals = []
        low_freq = []
        for i in range(len(series)):
            if not usable[i]:
                continue
            lo = max(0, i - window // 2)
            hi = min(len(series), i + window // 2 + 1)
            window_values = [series[j] for j in range(lo, hi) if usable[j]]
            if not window_values:
                continue
            local_mean = fmean(window_values)
            if i >= 1 and speeds[i - 1] < IDLE_SPEED_THRESHOLD_MPS:
                residuals.append(series[i] - local_mean)
                low_freq.append(local_mean)
        if len(residuals) >= 4:
            jitter_out.append(pstdev(residuals))
        if len(low_freq) >= 4:
            drift_out.append(pstdev(low_freq))


def _accumulate_velocity_error(
    positions: Sequence[tuple[float, float, float]],
    native_velocities: Sequence[tuple[float, float, float]] | None,
    valids: Sequence[bool],
    dt: float,
    out: list[float],
) -> None:
    if native_velocities is None or dt <= 0.0:
        return
    for index in range(1, len(positions)):
        if not (valids[index] and valids[index - 1]):
            continue
        finite = (
            (positions[index][0] - positions[index - 1][0]) / dt,
            (positions[index][1] - positions[index - 1][1]) / dt,
            (positions[index][2] - positions[index - 1][2]) / dt,
        )
        native = native_velocities[index]
        error = _distance(finite, native)
        out.append(error)


def _accumulate_angular(
    angular: Sequence[tuple[float, float, float]] | None,
    valids: Sequence[bool],
    out: list[float],
) -> None:
    if angular is None:
        return
    for index, valid in enumerate(valids):
        if valid:
            out.append(_magnitude(angular[index]))


def _axis_sigma_from_magnitudes(magnitudes: Sequence[float], fallback: float) -> float:
    """Recover per-axis Gaussian sigma from observed 3D-vector magnitudes.

    The generator draws excursion, velocity-error, and angular-velocity noise as
    per-axis Gaussian vectors of standard deviation sigma; only their Euclidean
    magnitude is observable. The mean magnitude of such a vector is
    ``sigma * 2 * sqrt(2 / pi)``, so dividing the sample mean by that factor
    recovers the sigma the generator consumes.
    """

    if not magnitudes:
        return fallback
    return float(fmean(magnitudes)) / _GAUSSIAN_VECTOR_MAGNITUDE_MEAN_OVER_SIGMA


def _stat(values: Sequence[float], fallback: float, *, use_mean: bool = False) -> float:
    if not values:
        return fallback
    if use_mean:
        return float(fmean(values))
    if len(values) < 2:
        return float(values[0])
    return float(pstdev(values))


def _distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5


def _magnitude(a: tuple[float, float, float]) -> float:
    return (a[0] ** 2 + a[1] ** 2 + a[2] ** 2) ** 0.5


def _print_summary(
    summary: dict[str, Any],
    session_dirs: Sequence[Path],
    output: Path | None,
) -> None:
    roles = summary["roles"]
    print(f"Fitted tracker noise from {len(session_dirs)} session(s)")
    print(f"Assumed frame rate: {summary['frame_rate_hz']} Hz")
    if not roles:
        print("No device roles found.")
        return
    for role in sorted(roles):
        entry = roles[role]
        profile = entry["profile"]
        print(f"\n[{role}] kind={profile.device_kind} samples={entry['samples']}")
        dropout_note = (
            " [from priors: no occlusion rows]" if entry.get("dropout_from_priors") else ""
        )
        print(
            f"  dropout: p_onset={profile.dropout_probability:.4f} "
            f"mean_run={profile.dropout_mean_run_frames:.2f} frames "
            f"hold_last={profile.dropout_hold_last_probability:.2f} "
            f"(runs={entry['dropout_runs']}){dropout_note}"
        )
        stationary_note = (
            "  [from priors: no idle rows]" if entry.get("stationary_from_priors") else ""
        )
        print(
            f"  jitter_std={profile.position_std_m * 1000:.2f} mm "
            f"drift_amp={profile.drift_amplitude_m * 1000:.2f} mm "
            f"excursion={profile.excursion_magnitude_m * 1000:.1f} mm{stationary_note}"
        )
        print(
            f"  velocity_std={profile.velocity_std_mps:.3f} m/s "
            f"angular_std={profile.angular_velocity_std_radps:.3f} rad/s"
        )
        print(f"  not fit from data (using priors): {', '.join(entry['unfitted_fields'])}")
    if output is not None:
        print(f"\nWrote fitted profile to {output}")


if __name__ == "__main__":
    raise SystemExit(main())
