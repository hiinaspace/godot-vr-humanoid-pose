"""CMU clip metadata and environment-split labeling."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Literal
from urllib.request import urlopen

CMU_SEARCH_URL = "https://mocap.cs.cmu.edu/search.php?subjectnumber={subject}"
DEFAULT_CMU_METADATA_PATH = Path("validation/cmu_clip_metadata.json")
type EnvironmentSplit = Literal["flat", "nonflat", "unknown"]

NONFLAT_KEYWORDS: Mapping[str, tuple[str, ...]] = {
    "stairs": ("stair",),
    "uneven_terrain": (
        "uneven terrain",
        "uneven ground",
        "rough terrain",
        "slope",
        "ramp",
        "incline",
        "decline",
    ),
    "step_obstacle": (
        "stepstool",
        "step stool",
        "step over",
        "varying height",
        "varying length",
        "obstacle",
    ),
    "ledge": ("ledge",),
    "ladder": ("ladder",),
    "playground": ("playground", "climb", "climbing", "hanging", "monkey bars", "slide"),
}

GENERIC_MOTION_DESCRIPTIONS = {
    "",
    "clean",
    "tvd",
}


@dataclass(frozen=True, slots=True)
class CmuClipMetadata:
    subject: str
    clip: str
    trial: int
    motion_description: str
    subject_description: str
    environment_split: EnvironmentSplit
    environment_tags: tuple[str, ...]
    label_source: str


def label_environment(
    motion_description: str,
    subject_description: str = "",
) -> tuple[EnvironmentSplit, tuple[str, ...], str]:
    motion_tags = _nonflat_tags(motion_description)
    if motion_tags:
        return "nonflat", motion_tags, "motion_description"

    normalized_motion = _normalize(motion_description)
    if normalized_motion in GENERIC_MOTION_DESCRIPTIONS:
        subject_tags = _nonflat_tags(subject_description)
        if subject_tags:
            return "nonflat", subject_tags, "subject_description"

    if normalized_motion:
        return "flat", (), "motion_description"
    return "unknown", (), "missing_description"


def clip_metadata_from_subject_page(subject: str, html: str) -> list[CmuClipMetadata]:
    parser = _SubjectPageParser()
    parser.feed(html)
    subject_description = parser.subject_description
    return [
        CmuClipMetadata(
            subject=subject,
            clip=f"{subject}_{trial:02d}",
            trial=trial,
            motion_description=description,
            subject_description=subject_description,
            environment_split=split,
            environment_tags=tags,
            label_source=label_source,
        )
        for trial, description in parser.trials
        for split, tags, label_source in [label_environment(description, subject_description)]
    ]


def fetch_subject_metadata(subject: str, *, timeout_seconds: float = 20.0) -> list[CmuClipMetadata]:
    url = CMU_SEARCH_URL.format(subject=int(subject))
    with urlopen(url, timeout=timeout_seconds) as response:
        html = response.read().decode("utf-8", errors="replace")
    return clip_metadata_from_subject_page(subject, html)


def fetch_cmu_metadata(
    subjects: Iterable[str],
    *,
    timeout_seconds: float = 20.0,
) -> dict[str, CmuClipMetadata]:
    metadata: dict[str, CmuClipMetadata] = {}
    for subject in sorted(subjects, key=lambda value: int(value)):
        for clip_metadata in fetch_subject_metadata(subject, timeout_seconds=timeout_seconds):
            metadata[clip_metadata.clip] = clip_metadata
    return metadata


def read_cmu_metadata(path: Path) -> dict[str, CmuClipMetadata]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("clips", [])
    return {
        str(row["clip"]): CmuClipMetadata(
            subject=str(row["subject"]),
            clip=str(row["clip"]),
            trial=int(row["trial"]),
            motion_description=str(row.get("motion_description", "")),
            subject_description=str(row.get("subject_description", "")),
            environment_split=_environment_split(row.get("environment_split")),
            environment_tags=tuple(str(tag) for tag in row.get("environment_tags", [])),
            label_source=str(row.get("label_source", "unknown")),
        )
        for row in rows
    }


def write_cmu_metadata(path: Path, metadata: Mapping[str, CmuClipMetadata]) -> None:
    payload = {
        "source": CMU_SEARCH_URL,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "label_rules": {
            "nonflat_keywords": {tag: list(keywords) for tag, keywords in NONFLAT_KEYWORDS.items()},
            "generic_motion_descriptions": sorted(GENERIC_MOTION_DESCRIPTIONS),
        },
        "clips": [asdict(metadata[clip]) for clip in sorted(metadata, key=_clip_sort_key)],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def default_clip_metadata(subject: str, clip: str) -> CmuClipMetadata:
    return CmuClipMetadata(
        subject=subject,
        clip=clip,
        trial=_trial_from_clip(clip),
        motion_description="",
        subject_description="",
        environment_split="unknown",
        environment_tags=(),
        label_source="missing_metadata",
    )


def _nonflat_tags(text: str) -> tuple[str, ...]:
    normalized = _normalize(text)
    return tuple(
        tag
        for tag, keywords in NONFLAT_KEYWORDS.items()
        if any(keyword in normalized for keyword in keywords)
    )


def _normalize(text: str) -> str:
    lowered = text.lower().replace("\xa0", " ")
    return re.sub(r"\s+", " ", lowered).strip()


def _trial_from_clip(clip: str) -> int:
    try:
        return int(clip.rsplit("_", maxsplit=1)[1])
    except (IndexError, ValueError):
        return 0


def _clip_sort_key(clip: str) -> tuple[int, int, str]:
    subject, _, trial = clip.partition("_")
    try:
        subject_number = int(subject)
    except ValueError:
        subject_number = 0
    try:
        trial_number = int(trial)
    except ValueError:
        trial_number = 0
    return subject_number, trial_number, clip


def _environment_split(value: object) -> EnvironmentSplit:
    if value == "flat":
        return "flat"
    if value == "nonflat":
        return "nonflat"
    return "unknown"


class _SubjectPageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._row: list[str] = []
        self._cell_parts: list[str] = []
        self._in_td = False
        self.subject_description = ""

    @property
    def trials(self) -> list[tuple[int, str]]:
        trials: list[tuple[int, str]] = []
        for row in self.rows:
            if len(row) >= 3 and row[1].isdigit():
                trials.append((int(row[1]), row[2]))
        return trials

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "tr":
            self._row = []
        if tag.lower() == "td":
            self._cell_parts = []
            self._in_td = True

    def handle_data(self, data: str) -> None:
        if self._in_td:
            self._cell_parts.append(data.strip())

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "td":
            self._row.append(" ".join(part for part in self._cell_parts if part))
            self._in_td = False
        if tag.lower() == "tr" and self._row:
            self.rows.append(self._row)
            self._maybe_update_subject_description(self._row)

    def _maybe_update_subject_description(self, row: list[str]) -> None:
        if self.subject_description or not row:
            return
        match = re.search(r"Subject\s+#\d+\s+\(([^)]*)\)", row[0])
        if match is not None:
            self.subject_description = match.group(1).strip()
