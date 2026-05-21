"""Fetch CMU mocap trial descriptions and derive validation split labels."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from zipfile import ZipFile

from validation.scripts.fetch_cmu import ARCHIVE_NAME, DEFAULT_CACHE_DIR
from validation.vr_pose_validation.cmu_metadata import (
    DEFAULT_CMU_METADATA_PATH,
    CmuClipMetadata,
    default_clip_metadata,
    fetch_cmu_metadata,
    write_cmu_metadata,
)

DEFAULT_ARCHIVE_PATH = DEFAULT_CACHE_DIR / ARCHIVE_NAME


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE_PATH)
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_CMU_METADATA_PATH)
    parser.add_argument("--subjects", nargs="*", help="Subject ids such as 07 08 09.")
    args = parser.parse_args(argv)

    archive_clips = _archive_clips(args.archive)
    subjects = set(args.subjects or {subject for subject, _clip in archive_clips})
    metadata = _complete_archive_metadata(
        archive_clips,
        fetch_cmu_metadata(subjects),
    )
    write_cmu_metadata(args.metadata_path, metadata)

    split_counts: dict[str, int] = {}
    for row in metadata.values():
        split_counts[row.environment_split] = split_counts.get(row.environment_split, 0) + 1
    print(
        json.dumps(
            {
                "metadata_path": args.metadata_path.as_posix(),
                "subjects": len(subjects),
                "clips": len(metadata),
                "environment_splits": split_counts,
                "missing_live_metadata": sum(
                    1 for row in metadata.values() if row.label_source == "missing_metadata"
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _archive_clips(archive_path: Path) -> list[tuple[str, str]]:
    with ZipFile(archive_path) as zf:
        return sorted(
            {
                (_subject_from_amc(name), Path(name).stem)
                for name in zf.namelist()
                if name.lower().endswith(".amc")
            },
            key=lambda value: (int(value[0]), int(value[1].rsplit("_", maxsplit=1)[1])),
        )


def _complete_archive_metadata(
    archive_clips: list[tuple[str, str]],
    fetched_metadata: dict[str, CmuClipMetadata],
) -> dict[str, CmuClipMetadata]:
    return {
        clip: fetched_metadata.get(clip, default_clip_metadata(subject, clip))
        for subject, clip in archive_clips
    }


def _subject_from_amc(name: str) -> str:
    return Path(name).stem.split("_", maxsplit=1)[0]


if __name__ == "__main__":
    raise SystemExit(main())
