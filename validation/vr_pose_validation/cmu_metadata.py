"""CMU clip metadata and environment-split labeling."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Literal
from urllib.request import urlopen

CMU_SEARCH_URL = "https://mocap.cs.cmu.edu/search.php?subjectnumber={subject}"
DEFAULT_CMU_METADATA_PATH = Path("validation/cmu_clip_metadata.json")
type EnvironmentSplit = Literal["flat", "nonflat", "unknown"]
type ActivityClass = Literal[
    "idle_standing",
    "sitting",
    "stationary_upper_body",
    "locomotion",
    "dance_exercise",
    "swimming",
    "floor_pose",
    "sports",
    "unknown",
]

NONFLAT_KEYWORDS: Mapping[str, tuple[str, ...]] = {
    "stairs": ("stair", "stairs"),
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
    "bad amc",
    "clean",
    "cleaned",
    "lost marker",
    "movie and trial dont match",
    "no movie",
    "tvd",
}

SWIMMING_MOTION_KEYWORDS = (
    "swimming",
    "back stroke",
    "breast stroke",
    "break stroke",
    "fly stroke",
)
SWIMMING_CONTEXTUAL_MOTION_KEYWORDS = ("butterfly", "free style", "freestyle")
SWIMMING_SUBJECT_KEYWORDS = ("swimming",)

ACTIVITY_KEYWORDS: Mapping[ActivityClass, Mapping[str, tuple[str, ...]]] = {
    "floor_pose": {
        "lying": ("laying", "lying", "lie down", "lay down", "lay down", "prone", "supine"),
        "floor_transition": (
            "getting up from floor",
            "get up from floor",
            "getting up from ground",
            "get up from ground",
            "get up face down",
            "from laying",
            "face down",
        ),
        "crawl_roll": ("crawl", "crawling", "rolling", "roll ", "somersault"),
        "kneel_crouch": ("kneel", "crouch"),
    },
    "sitting": {
        "chair": ("chair", "seat", "seated"),
        "vehicle": ("motorcycle", "motor cycle", "driving"),
        "sit": ("sit", "sitting", "sits"),
        "stand_from_sit": ("stand up from chair", "getting up from chair", "getting out of chair"),
    },
    "dance_exercise": {
        "dance": (
            "dance",
            "salsa",
            "charleston",
            "charelston",
            "lindy",
            "pirouette",
            "arabesque",
            "jete",
            "ballet",
            "curtsey",
            "ring around the rosie",
        ),
        "exercise": (
            "exercise",
            "stretch",
            "yoga",
            "tai chi",
            "jumping jack",
            "range of motion",
            "stretching",
            "squats",
            "squat",
            "flexing",
            "swimming",
            "rowing",
            "breakdance",
            "break dance",
            "flip",
            "flips",
            "handstand",
            "hand stand",
        ),
        "martial_arts": (
            "martial",
            "kick",
            "kicks",
            "kicking",
            "punch",
            "punching",
            "punch/strike",
            "fight",
            "boxing",
            "karate",
        ),
        "pantomime": (
            "nursery rhyme",
            "teapot",
            "cock robin",
            "alaska vacation",
            "alaskan vacation",
            "pantomime",
            "animal behavior",
            "prairie dog",
            "whale",
            "bear",
            "dog",
            "snake",
            "chicken",
            "monkey",
            "mouse",
            "cat",
            "elephant",
            "dinosaur",
            "tyrannosaurus",
            "t-rex",
            "fish",
            "chipmunk",
            "chip monk",
            "bird",
            "cow",
            "tiger",
            "horse",
            "gorilla",
            "panther",
            "penguin",
            "airplane",
        ),
    },
    "sports": {
        "ball_sport": (
            "basketball",
            "football",
            "soccer",
            "baseball",
            "tennis",
            "volleyball",
            "skateboard",
        ),
        "golf": ("golf",),
        "throwing": ("throw", "throwing", "catch", "frisbee", "toss", "pitch"),
        "gymnastics": ("cartwheel", "cartwheels", "acrobat", "acrobatics", "gymnast"),
        "tag_game": ("tag", "blind man's bluff"),
    },
    "stationary_upper_body": {
        "weapon": ("sword", "swordplay", "bow", "arrow", "rifle", "gun", "shoot"),
        "object_interaction": (
            "pick",
            "pickup",
            "pick up",
            "picking",
            "object",
            "box",
            "carry",
            "lift",
            "lifting",
            "place",
            "placing",
            "put",
            "puts",
            "putting",
            "scoop",
            "lean",
            "leans",
            "reach",
            "reaching",
            "pull",
            "pulls",
            "resist",
            "resists",
            "comfort",
            "comforts",
            "shelter",
            "shelters",
            "shoulder",
            "shoulders",
            "arm wrestle",
            "stare",
            "stares",
            "laugh",
            "threaten",
            "threatens",
            "moving",
            "grip",
            "pushing",
            "paying",
            "buying",
            "holding",
            "taking",
            "using",
            "setting",
            "coffee",
            "mug",
        ),
        "gesture": (
            "gesture",
            "gestures",
            "wave",
            "signal",
            "signals",
            "point",
            "conversation",
            "shake hands",
            "shaking hands",
            "hand shake",
            "high five",
            "look",
            "looking",
            "searching",
            "poking",
            "investigating",
            "flashlight",
            "teaching",
            "defensive",
            "guard",
            "avoid attacker",
            "peekaboo",
            "peek a boo",
            "shrug",
            "waving",
        ),
        "daily_action": (
            "wash",
            "washing",
            "drink",
            "drinking",
            "eat",
            "sipping",
            "phone",
            "sweep",
            "mop",
            "clean",
            "cleaning",
            "hammer",
            "hammering",
            "saw",
            "wrench",
            "bolt",
            "screw",
            "screwing",
            "unscrewing",
            "nail",
            "umbrella",
            "rope",
            "chopping",
            "digging",
            "sewing",
            "slicing",
            "eating",
            "mixing",
            "making",
            "playing",
            "hanging",
            "planting",
            "writing",
            "chug",
            "fishing",
            "painting",
            "sweeping",
            "vacuuming",
            "brushing",
            "shaving",
            "combing",
            "typing",
            "raking",
            "rake",
            "deoderant",
            "drumming",
            "violin",
            "piano",
            "tv",
            "smoking",
            "chugging",
            "swatting",
            "arguing",
            "channel surfing",
            "deodorant",
        ),
    },
    "locomotion": {
        "walk": ("walk", "walks", "walking", "stride", "march"),
        "run": ("run", "running", "jog", "jogging"),
        "jump": ("jump", "jumping", "hop", "hopping"),
        "climb": ("climb", "climbing"),
        "step": (
            "step",
            "steps",
            "stepping",
            "creep",
            "creeping",
            "limp",
            "limping",
            "stumble",
            "stumbles",
            "pacing",
            "sneak",
            "hopscotch",
            "duck under",
            "duck underneath",
        ),
        "turn": (
            "90",
            "180",
            "270",
            "360",
            "stop",
            "turn",
            "turns",
            "turning",
            "sidestep",
            "side step",
            "backwards",
            "backward",
            "navigate",
            "forward",
        ),
    },
    "idle_standing": {
        "stand": (
            "stand",
            "standing",
            "neutral",
            "wait",
            "waiting",
            "stance",
            "calibration",
            "static pose",
            "idle",
        ),
        "emotion": (
            "happy",
            "sad",
            "crying",
            "scared",
            "upset",
            "cold",
            "laughing",
            "freezing",
            "mope",
        ),
        "balance": ("balance",),
    },
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
    activity_class: ActivityClass = "unknown"
    activity_tags: tuple[str, ...] = ()
    activity_label_source: str = "missing_description"


@dataclass(frozen=True, slots=True)
class ClipLabelOverride:
    environment_split: EnvironmentSplit | None = None
    environment_tags: tuple[str, ...] | None = None
    label_source: str | None = None
    activity_class: ActivityClass | None = None
    activity_tags: tuple[str, ...] | None = None
    activity_label_source: str | None = None


KNOWN_CLIP_LABEL_OVERRIDES: Mapping[str, ClipLabelOverride] = {
    # Visual review: ground truth appears suspended/floating rather than flat-floor locomotion.
    "104_17": ClipLabelOverride(
        environment_split="nonflat",
        environment_tags=("suspended",),
        label_source="manual_clip_override",
    ),
    "104_18": ClipLabelOverride(
        environment_split="nonflat",
        environment_tags=("suspended",),
        label_source="manual_clip_override",
    ),
    # Visual review: ground truth appears to be a sideways T-pose translating through space.
    "133_25": ClipLabelOverride(
        environment_split="unknown",
        environment_tags=("data_issue", "sideways_t_pose"),
        label_source="manual_clip_override",
        activity_class="unknown",
        activity_tags=("parse_issue", "sideways_t_pose"),
        activity_label_source="manual_clip_override",
    ),
}


def label_environment(
    motion_description: str,
    subject_description: str = "",
) -> tuple[EnvironmentSplit, tuple[str, ...], str]:
    swimming_tags = _swimming_tags(motion_description, subject_description)
    if swimming_tags:
        return "nonflat", swimming_tags, "motion_description"

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
        apply_known_clip_label_overrides(
            CmuClipMetadata(
                subject=subject,
                clip=f"{subject}_{trial:02d}",
                trial=trial,
                motion_description=description,
                subject_description=subject_description,
                environment_split=split,
                environment_tags=tags,
                label_source=label_source,
                activity_class=activity_class,
                activity_tags=activity_tags,
                activity_label_source=activity_label_source,
            )
        )
        for trial, description in parser.trials
        for split, tags, label_source in [label_environment(description, subject_description)]
        for activity_class, activity_tags, activity_label_source in [
            label_activity(description, subject_description)
        ]
    ]


def label_activity(
    motion_description: str,
    subject_description: str = "",
) -> tuple[ActivityClass, tuple[str, ...], str]:
    normalized_motion = _normalize(motion_description)
    swimming_tags = _swimming_tags(motion_description, subject_description)
    if swimming_tags:
        return "swimming", swimming_tags, "motion_description"

    if normalized_motion in GENERIC_MOTION_DESCRIPTIONS:
        subject_class, subject_tags = _activity_match(subject_description)
        if subject_class != "unknown":
            return subject_class, subject_tags, "subject_description"
        return "unknown", (), "missing_description"

    motion_class, motion_tags = _activity_match(motion_description)
    if motion_class != "unknown":
        return motion_class, motion_tags, "motion_description"

    subject_class, subject_tags = _activity_match(subject_description)
    if subject_class != "unknown":
        return subject_class, subject_tags, "subject_description"

    return "unknown", (), "missing_description" if not normalized_motion else "keyword_miss"


def apply_known_clip_label_overrides(row: CmuClipMetadata) -> CmuClipMetadata:
    override = KNOWN_CLIP_LABEL_OVERRIDES.get(row.clip)
    if override is None:
        return row
    return replace(
        row,
        environment_split=override.environment_split or row.environment_split,
        environment_tags=override.environment_tags
        if override.environment_tags is not None
        else row.environment_tags,
        label_source=override.label_source or row.label_source,
        activity_class=override.activity_class or row.activity_class,
        activity_tags=override.activity_tags
        if override.activity_tags is not None
        else row.activity_tags,
        activity_label_source=override.activity_label_source or row.activity_label_source,
    )


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
            activity_class=_activity_class(row.get("activity_class")),
            activity_tags=tuple(str(tag) for tag in row.get("activity_tags", [])),
            activity_label_source=str(row.get("activity_label_source", "unknown")),
        )
        for row in rows
    }


def write_cmu_metadata(path: Path, metadata: Mapping[str, CmuClipMetadata]) -> None:
    payload = {
        "source": CMU_SEARCH_URL,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "label_rules": {
            "nonflat_keywords": {tag: list(keywords) for tag, keywords in NONFLAT_KEYWORDS.items()},
            "activity_keywords": {
                activity_class: {tag: list(keywords) for tag, keywords in tag_keywords.items()}
                for activity_class, tag_keywords in ACTIVITY_KEYWORDS.items()
            },
            "swimming_keywords": {
                "contextual_motion": list(SWIMMING_CONTEXTUAL_MOTION_KEYWORDS),
                "motion": list(SWIMMING_MOTION_KEYWORDS),
                "subject": list(SWIMMING_SUBJECT_KEYWORDS),
            },
            "clip_overrides": {
                clip: asdict(override) for clip, override in KNOWN_CLIP_LABEL_OVERRIDES.items()
            },
            "generic_motion_descriptions": sorted(GENERIC_MOTION_DESCRIPTIONS),
        },
        "clips": [asdict(metadata[clip]) for clip in sorted(metadata, key=_clip_sort_key)],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


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
        activity_class="unknown",
        activity_tags=(),
        activity_label_source="missing_metadata",
    )


def _nonflat_tags(text: str) -> tuple[str, ...]:
    normalized = _normalize(text)
    return tuple(
        tag
        for tag, keywords in NONFLAT_KEYWORDS.items()
        if any(_keyword_matches(normalized, keyword) for keyword in keywords)
    )


def _activity_match(text: str) -> tuple[ActivityClass, tuple[str, ...]]:
    normalized = _normalize(text)
    for activity_class, tags in ACTIVITY_KEYWORDS.items():
        matched_tags = tuple(
            tag
            for tag, keywords in tags.items()
            if any(_keyword_matches(normalized, keyword) for keyword in keywords)
        )
        if matched_tags:
            return activity_class, matched_tags
    return "unknown", ()


def _swimming_tags(motion_description: str, subject_description: str) -> tuple[str, ...]:
    normalized_motion = _normalize(motion_description)
    normalized_subject = _normalize(subject_description)
    if any(_keyword_matches(normalized_motion, keyword) for keyword in SWIMMING_MOTION_KEYWORDS):
        return ("swimming",)
    subject_is_swimming = any(
        _keyword_matches(normalized_subject, keyword) for keyword in SWIMMING_SUBJECT_KEYWORDS
    )
    if subject_is_swimming and any(
        _keyword_matches(normalized_motion, keyword)
        for keyword in SWIMMING_CONTEXTUAL_MOTION_KEYWORDS
    ):
        return ("swimming",)
    return ()


def _normalize(text: str) -> str:
    camel_split = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
    alpha_digit_split = re.sub(
        r"(?<=[A-Za-z])(?=[0-9])|(?<=[0-9])(?=[A-Za-z])",
        " ",
        camel_split,
    )
    lowered = alpha_digit_split.lower().replace("\xa0", " ")
    return re.sub(r"\s+", " ", lowered).strip()


def _keyword_matches(normalized_text: str, keyword: str) -> bool:
    normalized_keyword = _normalize(keyword)
    if not normalized_keyword:
        return False
    return (
        re.search(
            rf"(?<![a-z0-9]){re.escape(normalized_keyword)}(?![a-z0-9])",
            normalized_text,
        )
        is not None
    )


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


def _activity_class(value: object) -> ActivityClass:
    if value == "idle_standing":
        return "idle_standing"
    if value == "sitting":
        return "sitting"
    if value == "stationary_upper_body":
        return "stationary_upper_body"
    if value == "locomotion":
        return "locomotion"
    if value == "dance_exercise":
        return "dance_exercise"
    if value == "swimming":
        return "swimming"
    if value == "floor_pose":
        return "floor_pose"
    if value == "sports":
        return "sports"
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
