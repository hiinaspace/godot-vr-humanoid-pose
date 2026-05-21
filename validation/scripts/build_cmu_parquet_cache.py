"""Build a flat Parquet cache from the official CMU ASF/AMC archive."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from zipfile import ZipFile

from validation.vr_pose_validation.cmu import (
    CmuSkeleton,
    parse_amc_text_sampled,
    parse_asf_text,
)
from validation.vr_pose_validation.cmu_fk import (
    canonical_pose_from_cmu,
    compile_cmu_fk_plan,
    solve_cmu_pose_frame,
)
from validation.vr_pose_validation.cmu_metadata import (
    DEFAULT_CMU_METADATA_PATH,
    default_clip_metadata,
    read_cmu_metadata,
)
from validation.vr_pose_validation.parquet_cache import (
    CACHE_JOINTS,
    ParquetCompression,
    append_cached_pose_frame,
    clip_cache_path,
    new_cache_columns,
    write_clip_cache,
)

DEFAULT_ARCHIVE_PATH = Path(".cache/cmu/allasfamc.zip")
DEFAULT_CACHE_DIR = Path(".cache/cmu/parquet/canonical")
CMU_FRAME_RATE_HZ = 120.0


@dataclass(frozen=True, slots=True)
class CacheClipResult:
    clip: str
    subject: str
    source_frames: int
    cached_frames: int
    seconds: float
    path: str
    environment_split: str
    environment_tags: tuple[str, ...]
    motion_description: str
    skipped: bool = False


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE_PATH)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--subjects", nargs="*", help="Subject ids such as 07 08 09.")
    parser.add_argument("--max-clips", type=int)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames-per-clip", type=int)
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_CMU_METADATA_PATH)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--compression", default="zstd")
    args = parser.parse_args(argv)

    result = build_cache(
        archive_path=args.archive,
        cache_dir=args.cache_dir,
        subjects=set(args.subjects or []),
        max_clips=args.max_clips,
        frame_stride=args.frame_stride,
        max_frames_per_clip=args.max_frames_per_clip,
        metadata_path=args.metadata_path,
        force=args.force,
        compression=args.compression,
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0


def build_cache(
    *,
    archive_path: Path,
    cache_dir: Path,
    subjects: set[str],
    max_clips: int | None,
    frame_stride: int,
    max_frames_per_clip: int | None,
    metadata_path: Path,
    force: bool,
    compression: str,
) -> dict[str, object]:
    start = time.perf_counter()
    cache_dir.mkdir(parents=True, exist_ok=True)
    results: list[CacheClipResult] = []
    metadata_by_clip = read_cmu_metadata(metadata_path) if metadata_path.exists() else {}

    with ZipFile(archive_path) as zf:
        names = zf.namelist()
        asf_by_subject = {
            _subject_from_asf(name): name for name in names if name.lower().endswith(".asf")
        }
        amc_names = [
            name
            for name in sorted(names)
            if name.lower().endswith(".amc")
            and (not subjects or _subject_from_amc(name) in subjects)
        ]
        if max_clips is not None:
            amc_names = amc_names[:max_clips]

        skeleton_cache: dict[str, CmuSkeleton] = {}
        for amc_name in amc_names:
            subject = _subject_from_amc(amc_name)
            clip = Path(amc_name).stem
            metadata = metadata_by_clip.get(clip, default_clip_metadata(subject, clip))
            output_path = clip_cache_path(cache_dir, subject, clip)
            if output_path.exists() and not force:
                results.append(
                    CacheClipResult(
                        clip=clip,
                        subject=subject,
                        source_frames=0,
                        cached_frames=0,
                        seconds=0.0,
                        path=output_path.as_posix(),
                        environment_split=metadata.environment_split,
                        environment_tags=metadata.environment_tags,
                        motion_description=metadata.motion_description,
                        skipped=True,
                    )
                )
                continue

            clip_start = time.perf_counter()
            if subject not in skeleton_cache:
                skeleton_cache[subject] = parse_asf_text(
                    zf.read(asf_by_subject[subject]).decode("utf-8", errors="replace")
                )
            skeleton = skeleton_cache[subject]
            fk_plan = compile_cmu_fk_plan(skeleton)
            frame_selection = parse_amc_text_sampled(
                zf.read(amc_name).decode("utf-8", errors="replace"),
                frame_stride=frame_stride,
                max_frames=max_frames_per_clip,
            )
            columns = new_cache_columns()
            for frame in frame_selection.frames:
                pose = canonical_pose_from_cmu(solve_cmu_pose_frame(skeleton, frame, fk_plan))
                append_cached_pose_frame(
                    columns,
                    subject=subject,
                    clip=clip,
                    frame_index=frame.index,
                    timestamp_s=(frame.index - 1) / CMU_FRAME_RATE_HZ,
                    joints=pose,
                    environment_split=metadata.environment_split,
                    environment_tags=metadata.environment_tags,
                    motion_description=metadata.motion_description,
                    subject_description=metadata.subject_description,
                )
            write_clip_cache(
                output_path,
                columns,
                compression=cast(ParquetCompression, compression),
            )
            results.append(
                CacheClipResult(
                    clip=clip,
                    subject=subject,
                    source_frames=frame_selection.source_frame_count,
                    cached_frames=len(frame_selection.frames),
                    seconds=time.perf_counter() - clip_start,
                    path=output_path.as_posix(),
                    environment_split=metadata.environment_split,
                    environment_tags=metadata.environment_tags,
                    motion_description=metadata.motion_description,
                )
            )

    payload = {
        "summary": {
            "archive": archive_path.as_posix(),
            "cache_dir": cache_dir.as_posix(),
            "clips": len(results),
            "skipped_clips": sum(1 for result in results if result.skipped),
            "source_frames": sum(result.source_frames for result in results),
            "cached_frames": sum(result.cached_frames for result in results),
            "frame_stride": frame_stride,
            "max_frames_per_clip": max_frames_per_clip,
            "compression": compression,
            "joints": list(CACHE_JOINTS),
            "metadata_path": metadata_path.as_posix() if metadata_path.exists() else None,
            "environment_splits": _split_counts(results),
            "total_seconds": time.perf_counter() - start,
            "generated_at_utc": datetime.now(UTC).isoformat(),
        },
        "clips": [asdict(result) for result in results],
    }
    (cache_dir / "_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return cast(dict[str, object], payload)


def _subject_from_asf(name: str) -> str:
    return Path(name).stem


def _subject_from_amc(name: str) -> str:
    return Path(name).stem.split("_", maxsplit=1)[0]


def _split_counts(results: list[CacheClipResult]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        counts[result.environment_split] = counts.get(result.environment_split, 0) + 1
    return counts


if __name__ == "__main__":
    raise SystemExit(main())
