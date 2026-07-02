"""Build a flat Parquet cache from the SparsePoser Xsens BVH archive."""

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

from validation.vr_pose_validation.parquet_cache import (
    CACHE_JOINTS,
    ParquetCompression,
    append_cached_pose_frame,
    clip_cache_path,
    new_cache_columns,
    write_clip_cache,
)
from validation.vr_pose_validation.sparseposer_bvh import (
    SPARSEPOSER_DATASET,
    canonical_pose_from_sparseposer,
    iter_sparseposer_pose_frames,
    parse_sparseposer_bvh_text,
    sparseposer_clip_identity,
)

DEFAULT_ARCHIVE_PATH = Path(".cache/sparseposer/data.zip")
DEFAULT_CACHE_DIR = Path(".cache/sparseposer/parquet/canonical")


@dataclass(frozen=True, slots=True)
class SparseposerCacheClipResult:
    clip: str
    source_clip: str
    subject: str
    dataset_split: str
    source_path: str
    source_frames: int
    cached_frames: int
    frame_time_s: float
    parse_seconds: float
    fk_seconds: float
    write_seconds: float
    seconds: float
    path: str
    skipped: bool = False


@dataclass(frozen=True, slots=True)
class SparseposerBvhSource:
    path: str
    identity_path: str
    text: str | None = None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE_PATH)
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Read loose BVH files from a directory instead of the SparsePoser zip archive.",
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--splits", nargs="*", choices=("train", "eval"), default=("train", "eval"))
    parser.add_argument("--subjects", nargs="*", help="Subject ids such as S01 S02.")
    parser.add_argument(
        "--dataset-split",
        choices=("train", "eval"),
        help="Assign loose input BVHs to this SparsePoser split.",
    )
    parser.add_argument(
        "--filename-prefix-to-strip",
        default="",
        help="Strip this prefix from loose input BVH stems before deriving source clip ids.",
    )
    parser.add_argument("--max-clips", type=int)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames-per-clip", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--compression", default="zstd")
    args = parser.parse_args(argv)

    result = build_cache(
        archive_path=args.archive,
        input_dir=args.input_dir,
        cache_dir=args.cache_dir,
        splits=set(args.splits or []),
        subjects=set(args.subjects or []),
        dataset_split=args.dataset_split,
        filename_prefix_to_strip=args.filename_prefix_to_strip,
        max_clips=args.max_clips,
        frame_stride=args.frame_stride,
        max_frames_per_clip=args.max_frames_per_clip,
        force=args.force,
        compression=args.compression,
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0


def build_cache(
    *,
    archive_path: Path,
    input_dir: Path | None = None,
    cache_dir: Path,
    splits: set[str],
    subjects: set[str],
    dataset_split: str | None = None,
    filename_prefix_to_strip: str = "",
    max_clips: int | None,
    frame_stride: int,
    max_frames_per_clip: int | None,
    force: bool,
    compression: str,
) -> dict[str, object]:
    start = time.perf_counter()
    cache_dir.mkdir(parents=True, exist_ok=True)
    results: list[SparseposerCacheClipResult] = []

    with _open_sparseposer_sources(
        archive_path=archive_path,
        input_dir=input_dir,
        dataset_split=dataset_split,
        filename_prefix_to_strip=filename_prefix_to_strip,
    ) as source_reader:
        sources = [
            source
            for source in source_reader.sources
            if _include_source(source, splits=splits, subjects=subjects)
        ]
        if max_clips is not None:
            sources = sources[:max_clips]

        for source in sources:
            identity = sparseposer_clip_identity(source.identity_path)
            output_path = clip_cache_path(cache_dir, identity.subject, identity.clip)
            if output_path.exists() and not force:
                results.append(
                    SparseposerCacheClipResult(
                        clip=identity.clip,
                        source_clip=identity.source_clip,
                        subject=identity.subject,
                        dataset_split=identity.dataset_split,
                        source_path=source.path,
                        source_frames=0,
                        cached_frames=0,
                        frame_time_s=0.0,
                        parse_seconds=0.0,
                        fk_seconds=0.0,
                        write_seconds=0.0,
                        seconds=0.0,
                        path=output_path.as_posix(),
                        skipped=True,
                    )
                )
                continue

            clip_start = time.perf_counter()
            parse_start = time.perf_counter()
            text = source_reader.read_text(source)
            motion = parse_sparseposer_bvh_text(text)
            parse_seconds = time.perf_counter() - parse_start

            fk_start = time.perf_counter()
            columns = new_cache_columns()
            cached_frames = 0
            for frame_index, timestamp_s, pose in iter_sparseposer_pose_frames(
                motion,
                frame_stride=frame_stride,
                max_frames=max_frames_per_clip,
            ):
                append_cached_pose_frame(
                    columns,
                    dataset=SPARSEPOSER_DATASET,
                    dataset_split=identity.dataset_split,
                    subject=identity.subject,
                    clip=identity.clip,
                    frame_index=frame_index,
                    timestamp_s=timestamp_s,
                    joints=canonical_pose_from_sparseposer(pose),
                    environment_split="flat",
                    environment_tags=("vr_capture",),
                    activity_class="unknown",
                    activity_tags=(),
                    motion_description=(
                        f"SparsePoser Xsens {identity.dataset_split}/{identity.source_clip}"
                    ),
                )
                cached_frames += 1
            fk_seconds = time.perf_counter() - fk_start

            write_start = time.perf_counter()
            write_clip_cache(
                output_path,
                columns,
                compression=cast(ParquetCompression, compression),
            )
            write_seconds = time.perf_counter() - write_start

            results.append(
                SparseposerCacheClipResult(
                    clip=identity.clip,
                    source_clip=identity.source_clip,
                    subject=identity.subject,
                    dataset_split=identity.dataset_split,
                    source_path=source.path,
                    source_frames=motion.source_frame_count,
                    cached_frames=cached_frames,
                    frame_time_s=motion.frame_time_s,
                    parse_seconds=parse_seconds,
                    fk_seconds=fk_seconds,
                    write_seconds=write_seconds,
                    seconds=time.perf_counter() - clip_start,
                    path=output_path.as_posix(),
                )
            )

    payload = {
        "summary": {
            "archive": archive_path.as_posix(),
            "cache_dir": cache_dir.as_posix(),
            "dataset": SPARSEPOSER_DATASET,
            "clips": len(results),
            "skipped_clips": sum(1 for result in results if result.skipped),
            "source_frames": sum(result.source_frames for result in results),
            "cached_frames": sum(result.cached_frames for result in results),
            "frame_stride": frame_stride,
            "max_frames_per_clip": max_frames_per_clip,
            "compression": compression,
            "joints": list(CACHE_JOINTS),
            "splits": _split_counts(results),
            "subjects": _subject_counts(results),
            "parse_seconds": sum(result.parse_seconds for result in results),
            "fk_seconds": sum(result.fk_seconds for result in results),
            "write_seconds": sum(result.write_seconds for result in results),
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


@dataclass(slots=True)
class _SparseposerSourceReader:
    sources: list[SparseposerBvhSource]
    zip_file: ZipFile | None = None

    def __enter__(self) -> _SparseposerSourceReader:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        if self.zip_file is not None:
            self.zip_file.close()

    def read_text(self, source: SparseposerBvhSource) -> str:
        if source.text is not None:
            return source.text
        if self.zip_file is None:
            raise ValueError(f"Loose BVH source has no loaded text: {source.path}")
        return self.zip_file.read(source.path).decode("utf-8", errors="replace")


def _open_sparseposer_sources(
    *,
    archive_path: Path,
    input_dir: Path | None,
    dataset_split: str | None,
    filename_prefix_to_strip: str,
) -> _SparseposerSourceReader:
    if input_dir is not None:
        sources = [
            SparseposerBvhSource(
                path=path.as_posix(),
                identity_path=_loose_bvh_identity_path(
                    path,
                    dataset_split=dataset_split,
                    filename_prefix_to_strip=filename_prefix_to_strip,
                ),
                text=path.read_text(encoding="utf-8", errors="replace"),
            )
            for path in sorted(input_dir.glob("*.bvh"))
        ]
        return _SparseposerSourceReader(sources=sources)

    zip_file = ZipFile(archive_path)
    return _SparseposerSourceReader(
        zip_file=zip_file,
        sources=[
            SparseposerBvhSource(path=name, identity_path=name)
            for name in sorted(zip_file.namelist())
            if name.lower().endswith(".bvh")
        ],
    )


def _loose_bvh_identity_path(
    path: Path,
    *,
    dataset_split: str | None,
    filename_prefix_to_strip: str,
) -> str:
    source_stem = path.stem
    if filename_prefix_to_strip and source_stem.startswith(filename_prefix_to_strip):
        source_stem = source_stem[len(filename_prefix_to_strip) :]
    if dataset_split:
        return f"data/xsens/{dataset_split}/{source_stem}.bvh"
    return path.as_posix()


def _include_source(
    source: SparseposerBvhSource,
    *,
    splits: set[str],
    subjects: set[str],
) -> bool:
    identity = sparseposer_clip_identity(source.identity_path)
    return (not splits or identity.dataset_split in splits) and (
        not subjects or identity.subject in subjects
    )


def _split_counts(results: list[SparseposerCacheClipResult]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        counts[result.dataset_split] = counts.get(result.dataset_split, 0) + 1
    return counts


def _subject_counts(results: list[SparseposerCacheClipResult]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        counts[result.subject] = counts.get(result.subject, 0) + 1
    return counts


if __name__ == "__main__":
    raise SystemExit(main())
