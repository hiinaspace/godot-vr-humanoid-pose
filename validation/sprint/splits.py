# ruff: noqa: E501
"""Frozen subject-disjoint splits: splits_v1.json (id-level) and splits_v2.json (= v1 minus content duplicates).

Rebuild v1 with ``python -m validation.sprint.splits``; v2 with ``python -m validation.sprint.splits
--v2 PAIRS.json DROP.json`` (outputs of ``dedupe.py``). Round 3 onward uses v2: the audit found CMU test
subject 84 == train subject 74, val subject 141 == train subject 106 and Xsens eval_S02_A04 == train copy,
frame for frame, which an id-level split cannot see.

CMU:   test = sha1(subject) % 10 == 0, val = % 10 == 1, train = the rest (subject-level).
Xsens: the official SparsePoser train/eval lists, plus one train subject (orig + mirror coupled)
       held out of train as ``xsens_val_subject``. The official eval shares subjects with train;
       that overlap is recorded, not hidden.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

SPLITS_VERSION = "splits-v2"
REPO = Path(__file__).resolve().parents[2]
SPLITS_V1_PATH = Path(__file__).with_name("splits_v1.json")
SPLITS_PATH = Path(__file__).with_name("splits_v2.json")
CMU_DIR = REPO / ".cache/cmu/parquet/canonical_full"
XSENS_DIR = REPO / ".cache/sparseposer/parquet/canonical"
ZENODO_RECORD = "8427980"
ZENODO_MD5_DATA_ZIP = "c9f08cf985e2b3bac324c0a4e17d4fbc"  # from the Zenodo API, record revision 1


def cmu_bucket(subject_id: str) -> int:
    return int(hashlib.sha1(subject_id.encode("utf-8")).hexdigest(), 16) % 10


def _clips(root: Path, prefix: str | None = None) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for d in sorted(root.glob("subject=*")):
        subject = d.name.split("=", 1)[1]
        names = sorted(
            p.stem for p in d.glob("*.parquet") if prefix is None or p.stem.startswith(prefix)
        )
        if names:
            out[subject] = names
    return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _md5(path: Path) -> str:
    h = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def build() -> dict:
    cmu = _clips(CMU_DIR)
    buckets = {s: cmu_bucket(s) for s in cmu}
    cmu_split = {"test": [], "val": [], "train": []}
    for s, b in buckets.items():
        cmu_split["test" if b == 0 else "val" if b == 1 else "train"].append(s)

    xs_train = _clips(XSENS_DIR, "train_")
    xs_eval = _clips(XSENS_DIR, "eval_")
    eval_overlap = sorted(set(xs_eval) & set(xs_train))
    # Held-out subject: a train subject that is NOT in the official eval; lowest sha1 wins.
    eligible = sorted(
        set(xs_train) - set(xs_eval), key=lambda s: hashlib.sha1(s.encode()).hexdigest()
    )
    val_subject = eligible[0]

    def ids(dataset: str, by_subject: dict[str, list[str]], subjects=None) -> list[str]:
        return [
            f"{dataset}/{s}/{c}"
            for s, cs in sorted(by_subject.items())
            if subjects is None or s in subjects
            for c in cs
        ]

    doc = {
        "version": "splits-v1",
        "rule": {
            "cmu": "bucket = int(sha1(subject_id).hexdigest(), 16) % 10; test=0, val=1, train=rest",
            "xsens": "official train/eval lists; xsens_val_subject = train subject absent from "
            "official eval with the lowest sha1(subject), removed from xsens_train (mirror clips "
            "coupled)",
        },
        "cmu": {
            "subjects": {k: sorted(v) for k, v in cmu_split.items()},
            "clips": {k: ids("cmu", cmu, set(v)) for k, v in cmu_split.items()},
        },
        "xsens": {
            "train_subjects": sorted(set(xs_train) - {val_subject}),
            "val_subject": val_subject,
            "eval_subjects": sorted(xs_eval),
            "eval_subject_overlap_with_train": eval_overlap,
            "eval_overlap_note": "official eval shares these subjects with official train; "
            "'named' clips have unknown subject identity (cannot be checked)",
            "clips": {
                "train": ids("xsens", xs_train, set(xs_train) - {val_subject}),
                "val_subject": ids("xsens", xs_train, {val_subject}),
                "eval": ids("xsens", xs_eval),
            },
        },
        "source_pins": {
            "zenodo_record": ZENODO_RECORD,
            "zenodo_revision": 1,
            "zenodo_modified": "2023-10-12T09:30:59Z",
            "zenodo_license": "cc-by-4.0",
            "xsens_data_zip": {
                "path": ".cache/sparseposer/data.zip",
                "bytes": (REPO / ".cache/sparseposer/data.zip").stat().st_size,
                "md5": _md5(REPO / ".cache/sparseposer/data.zip"),
                "md5_expected_zenodo": ZENODO_MD5_DATA_ZIP,
                "sha256": _sha256(REPO / ".cache/sparseposer/data.zip"),
            },
            "xsens_parquet_manifest_sha256": _sha256(XSENS_DIR / "_manifest.json"),
            "cmu_allasfamc_zip": {
                "path": ".cache/cmu/allasfamc.zip",
                "bytes": (REPO / ".cache/cmu/allasfamc.zip").stat().st_size,
                "sha256": _sha256(REPO / ".cache/cmu/allasfamc.zip"),
            },
            "cmu_parquet_manifest_sha256": _sha256(CMU_DIR / "_manifest.json"),
            "git_head": subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True
            ).stdout.strip(),
        },
    }
    assert_disjoint(doc)
    return doc


def assert_disjoint(doc: dict) -> None:
    c = doc["cmu"]["subjects"]
    sets = {k: set(v) for k, v in c.items()}
    assert not (sets["test"] & sets["val"]) and not (sets["test"] & sets["train"])
    assert not (sets["val"] & sets["train"])
    cl = {k: set(v) for k, v in doc["cmu"]["clips"].items()}
    assert (
        not (cl["test"] & cl["val"])
        and not (cl["test"] & cl["train"])
        and not (cl["val"] & cl["train"])
    )
    for name, subs in sets.items():
        for s in subs:
            assert (cmu_bucket(s) == 0) == (name == "test") and (cmu_bucket(s) == 1) == (
                name == "val"
            )
    x = doc["xsens"]
    xt, xv, xe = (set(x["clips"][k]) for k in ("train", "val_subject", "eval"))
    assert not (xt & xv) and not (xv & xe), "xsens val subject must be disjoint from train and eval"
    sub = lambda cs: {c.split("/")[1] for c in cs}  # noqa: E731
    assert not (sub(xt) & sub(xv)) and not (sub(xv) & sub(xe))
    assert x["val_subject"] not in x["eval_subjects"]
    # mirror coupling: a clip and its mirror always land in the same set
    for group in (xt, xv, xe):
        for cid in group:
            twin = cid[: -len("_mirror")] if cid.endswith("_mirror") else cid + "_mirror"
            if twin in xt | xv | xe:
                assert twin in group, f"mirror pair split across sets: {cid}"


def load_splits() -> dict:
    return json.loads(SPLITS_PATH.read_text())


def clip_path(clip_id: str) -> Path:
    dataset, subject, clip = clip_id.split("/")
    root = CMU_DIR if dataset == "cmu" else XSENS_DIR
    return root / f"subject={subject}" / f"{clip}.parquet"


def clip_ids(split: str) -> list[str]:
    """Split names: cmu_test, cmu_val, cmu_train, xsens_eval, xsens_val, xsens_train, cmu_all."""
    doc = load_splits()
    if split == "cmu_all":
        return [c for k in ("test", "val", "train") for c in doc["cmu"]["clips"][k]]
    if split.startswith("cmu_"):
        return list(doc["cmu"]["clips"][split[4:]])
    key = {"xsens_eval": "eval", "xsens_val": "val_subject", "xsens_train": "train"}[split]
    return list(doc["xsens"]["clips"][key])


def build_v2(pairs_path: Path, drop_path: Path) -> dict:
    """splits_v1 with every val / test / eval clip that duplicates a clip of another split removed."""
    doc = json.loads(SPLITS_V1_PATH.read_text())
    drop = set(json.loads(drop_path.read_text()))
    pairs = json.loads(pairs_path.read_text())
    removed: dict[str, list[str]] = {}
    for group in (doc["cmu"]["clips"], doc["xsens"]["clips"]):
        for k, v in group.items():
            kept = [c for c in v if c not in drop]
            if len(kept) != len(v):
                removed[k] = sorted(set(v) - set(kept))
            group[k] = kept
    for k, subs in doc["cmu"]["subjects"].items():  # subjects left without clips
        have = {c.split("/")[1] for c in doc["cmu"]["clips"][k]}
        doc["cmu"]["subjects"][k] = [s for s in subs if s in have]
    doc["version"] = "splits-v2"
    doc["dedupe"] = {
        "method": "validation/sprint/dedupe.py: root-relative 13-joint pose, stride-10 KD-tree radius "
        "1.8 cm, per-pair dominant offset, verified over the overlap (>= 60 frames, >= 90 % of frames "
        "with mean joint distance < 5 mm, pose std > 2 mm); mirrored queries for held-out clips. "
        "Rule: drop the held-out copy (val < test/eval when both held out); train copies stay.",
        "removed": removed,
        "pairs": pairs,
    }
    assert_disjoint(doc)
    return doc


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--v2":
        out = build_v2(Path(sys.argv[2]), Path(sys.argv[3]))
        SPLITS_PATH.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
        print("wrote", SPLITS_PATH, {k: len(v) for k, v in out["dedupe"]["removed"].items()})
        sys.exit(0)
    out = build()
    SPLITS_V1_PATH.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    c, x = out["cmu"], out["xsens"]
    print("wrote", SPLITS_V1_PATH)
    print({k: (len(c["subjects"][k]), len(c["clips"][k])) for k in c["subjects"]})
    print(
        {k: len(v) for k, v in x["clips"].items()},
        "val_subject",
        x["val_subject"],
        "eval overlap",
        x["eval_subject_overlap_with_train"],
    )
    sys.exit(0)
