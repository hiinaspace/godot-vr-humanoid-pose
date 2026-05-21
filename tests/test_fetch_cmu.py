from __future__ import annotations

import hashlib
import json

from validation.scripts.fetch_cmu import (
    LICENSE_NOTE,
    build_manifest,
    manifest_matches_archive,
    sha256_file,
)
from validation.vr_pose_validation.cmu import CMU_SCALE_TO_METERS


def test_manifest_matches_archive_when_size_and_hash_match(tmp_path) -> None:
    archive = tmp_path / "allasfamc.zip"
    archive.write_bytes(b"small fixture")
    digest = hashlib.sha256(b"small fixture").hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"bytes": archive.stat().st_size, "sha256": digest}))

    assert sha256_file(archive) == digest
    assert manifest_matches_archive(manifest, archive)


def test_build_manifest_records_source_scale_and_license(tmp_path) -> None:
    archive = tmp_path / "allasfamc.zip"
    manifest = build_manifest(
        archive_path=archive,
        url="http://example.invalid/allasfamc.zip",
        headers={"Content-Length": "12"},
        sha256="abc123",
        byte_count=12,
    )

    assert manifest["archive_path"] == archive.as_posix()
    assert manifest["cmu_scale_to_meters"] == CMU_SCALE_TO_METERS
    assert manifest["license_note"] == LICENSE_NOTE
    assert manifest["headers"] == {"Content-Length": "12"}
