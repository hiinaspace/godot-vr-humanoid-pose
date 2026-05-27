"""Frequency-domain diagnostics for estimator output."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Any, cast

import numpy as np

from validation.vr_pose_validation.geometry import Transform
from validation.vr_pose_validation.types import HumanoidPoseFrame

type FrequencyBand = tuple[str, float, float | None]

DEFAULT_FREQUENCY_BANDS: tuple[FrequencyBand, ...] = (
    ("0_2hz", 0.0, 2.0),
    ("2_6hz", 2.0, 6.0),
    ("6_12hz", 6.0, 12.0),
    ("12hz_plus", 12.0, None),
)
FREQUENCY_GROUP_JOINTS = {
    "upper_body": (
        "head",
        "neck",
        "chest",
        "spine",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_hand",
        "right_hand",
    ),
    "torso": ("head", "neck", "chest", "spine"),
    "shoulder": ("left_shoulder", "right_shoulder"),
    "elbow": ("left_elbow", "right_elbow"),
    "hand": ("left_hand", "right_hand"),
}
SIGNAL_NAMES = ("error", "estimated", "reference")


@dataclass(frozen=True, slots=True)
class FrequencyAnalysis:
    """Aggregated position spectra for one estimated/reference pose sequence."""

    metrics: dict[str, float]
    spectra: dict[str, list[dict[str, float]]]


def summarize_frequency_reference_sequence(
    poses: Iterable[HumanoidPoseFrame],
    reference_frames: Iterable[Mapping[str, Transform]],
    *,
    bands: Sequence[FrequencyBand] = DEFAULT_FREQUENCY_BANDS,
    group_joints: Mapping[str, Sequence[str]] = FREQUENCY_GROUP_JOINTS,
    include_spectra: bool = False,
) -> FrequencyAnalysis:
    """Summarize estimator/reference position spectra with coarse bandpowers.

    Bandpowers are mean squared meters per joint component. The high-frequency
    ratios are dimensionless fractions of total position variance, which makes
    them easier to compare across clips with different motion amplitude.
    """

    pose_list = list(poses)
    reference_list = list(reference_frames)
    if len(pose_list) < 4 or len(reference_list) < 4:
        return FrequencyAnalysis(metrics=_zero_metrics(bands, group_joints), spectra={})

    timestamps = [pose.timestamp_seconds for pose in pose_list]
    sample_rate_hz = _sample_rate_hz(timestamps)
    if sample_rate_hz <= 0.0:
        return FrequencyAnalysis(metrics=_zero_metrics(bands, group_joints), spectra={})

    metrics: dict[str, float] = {}
    spectra: dict[str, list[dict[str, float]]] = {}
    for group_name, joints in group_joints.items():
        series = _group_series(pose_list, reference_list, joints)
        group_result = _summarize_group(series, sample_rate_hz, bands)
        for signal_name in SIGNAL_NAMES:
            total_power = group_result["total_power"][signal_name]
            metrics[f"{group_name}_{signal_name}_total_power_m2"] = total_power
            for band_name, _low_hz, _high_hz in bands:
                metrics[f"{group_name}_{signal_name}_band_{band_name}_m2"] = group_result[
                    "band_power"
                ][signal_name][band_name]
            high_power = sum(
                group_result["band_power"][signal_name][band_name]
                for band_name, low_hz, _high_hz in bands
                if low_hz >= 6.0
            )
            metrics[f"{group_name}_{signal_name}_high_frequency_ratio"] = (
                high_power / total_power if total_power > 1e-12 else 0.0
            )
        if include_spectra and group_name == "upper_body":
            spectra[group_name] = _spectrum_rows(group_result["spectrum"])

    return FrequencyAnalysis(metrics=metrics, spectra=spectra)


def _group_series(
    poses: Sequence[HumanoidPoseFrame],
    reference_frames: Sequence[Mapping[str, Transform]],
    joints: Sequence[str],
) -> dict[str, list[list[float]]]:
    series = {signal_name: [] for signal_name in SIGNAL_NAMES}
    for joint_name in joints:
        estimated_positions: list[tuple[float, float, float]] = []
        reference_positions: list[tuple[float, float, float]] = []
        for pose, reference in zip(poses, reference_frames, strict=False):
            estimated_joint = pose.joints.get(joint_name)
            reference_joint = reference.get(joint_name)
            if estimated_joint is None or reference_joint is None:
                continue
            estimated_positions.append(estimated_joint.transform.position)
            reference_positions.append(reference_joint.position)
        if len(estimated_positions) < 4 or len(reference_positions) < 4:
            continue
        for axis in range(3):
            estimated_axis = [position[axis] for position in estimated_positions]
            reference_axis = [position[axis] for position in reference_positions]
            series["estimated"].append(estimated_axis)
            series["reference"].append(reference_axis)
            series["error"].append(
                [
                    estimated_value - reference_value
                    for estimated_value, reference_value in zip(
                        estimated_axis,
                        reference_axis,
                        strict=True,
                    )
                ]
            )
    return series


def _summarize_group(
    series: Mapping[str, Sequence[Sequence[float]]],
    sample_rate_hz: float,
    bands: Sequence[FrequencyBand],
) -> dict[str, Any]:
    band_power = {
        signal_name: {band_name: 0.0 for band_name, _low_hz, _high_hz in bands}
        for signal_name in SIGNAL_NAMES
    }
    total_power = {signal_name: 0.0 for signal_name in SIGNAL_NAMES}
    spectra: dict[str, tuple[np.ndarray, np.ndarray] | None] = {
        signal_name: None for signal_name in SIGNAL_NAMES
    }

    for signal_name in SIGNAL_NAMES:
        signal_series = series.get(signal_name, ())
        if not signal_series:
            continue
        powers_by_series: list[np.ndarray] = []
        freqs: np.ndarray | None = None
        for values in signal_series:
            spectrum = _power_spectrum(values, sample_rate_hz)
            if spectrum is None:
                continue
            freqs, powers = spectrum
            powers_by_series.append(powers)
            total_power[signal_name] += float(np.sum(powers))
            for band_name, low_hz, high_hz in bands:
                band_mask = _band_mask(freqs, low_hz, high_hz)
                band_power[signal_name][band_name] += float(np.sum(powers[band_mask]))
        count = len(powers_by_series)
        if count == 0:
            continue
        total_power[signal_name] /= count
        for band_name, _low_hz, _high_hz in bands:
            band_power[signal_name][band_name] /= count
        spectra[signal_name] = (
            cast(np.ndarray, freqs),
            cast(np.ndarray, np.mean(np.vstack(powers_by_series), axis=0)),
        )

    return {
        "band_power": band_power,
        "total_power": total_power,
        "spectrum": spectra,
    }


def _power_spectrum(
    values: Sequence[float],
    sample_rate_hz: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    if len(values) < 4 or sample_rate_hz <= 0.0:
        return None
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        return None
    array = array - float(np.mean(array))
    spectrum = np.fft.rfft(array)
    power = (np.abs(spectrum) ** 2) / (len(array) * len(array))
    if len(power) > 2:
        power[1:-1] *= 2.0
    freqs = np.fft.rfftfreq(len(array), d=1.0 / sample_rate_hz)
    return freqs, power


def _sample_rate_hz(timestamps_seconds: Sequence[float]) -> float:
    dts = [
        current - previous
        for previous, current in zip(timestamps_seconds, timestamps_seconds[1:], strict=False)
        if current > previous and isfinite(current) and isfinite(previous)
    ]
    if not dts:
        return 0.0
    return 1.0 / float(np.median(np.asarray(dts, dtype=np.float64)))


def _band_mask(freqs: np.ndarray, low_hz: float, high_hz: float | None) -> np.ndarray:
    mask = freqs >= low_hz
    if high_hz is not None:
        mask = mask & (freqs < high_hz)
    return mask


def _spectrum_rows(
    spectra: Mapping[str, tuple[np.ndarray, np.ndarray] | None],
) -> list[dict[str, float]]:
    error_spectrum = spectra.get("error")
    if error_spectrum is None:
        return []
    freqs, _error_power = error_spectrum
    rows = []
    for index, frequency_hz in enumerate(freqs):
        row = {"frequency_hz": float(frequency_hz)}
        for signal_name in SIGNAL_NAMES:
            spectrum = spectra.get(signal_name)
            row[f"{signal_name}_power_m2"] = (
                float(spectrum[1][index]) if spectrum is not None else 0.0
            )
        rows.append(row)
    return rows


def _zero_metrics(
    bands: Sequence[FrequencyBand],
    group_joints: Mapping[str, Sequence[str]],
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for group_name in group_joints:
        for signal_name in SIGNAL_NAMES:
            metrics[f"{group_name}_{signal_name}_total_power_m2"] = 0.0
            for band_name, _low_hz, _high_hz in bands:
                metrics[f"{group_name}_{signal_name}_band_{band_name}_m2"] = 0.0
            metrics[f"{group_name}_{signal_name}_high_frequency_ratio"] = 0.0
    return metrics
