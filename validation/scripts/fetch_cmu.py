"""Fetch the official CMU ASF/AMC archive into a local cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from validation.vr_pose_validation.cmu import CMU_SCALE_TO_METERS

CMU_ASF_AMC_URL = "http://mocap.cs.cmu.edu/allasfamc.zip"
DEFAULT_CACHE_DIR = Path(".cache/cmu")
DEFAULT_MANIFEST_PATH = Path("validation/cmu_manifest.json")
ARCHIVE_NAME = "allasfamc.zip"

LICENSE_NOTE = (
    "CMU states that the motion capture data may be copied, modified, or "
    "redistributed without permission."
)


@dataclass(frozen=True, slots=True)
class FetchResult:
    archive_path: Path
    manifest_path: Path
    sha256: str
    bytes: int
    downloaded: bool


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=CMU_ASF_AMC_URL)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args(argv)

    try:
        result = fetch_cmu_archive(
            url=args.url,
            cache_dir=args.cache_dir,
            manifest_path=args.manifest_path,
            force=args.force,
            timeout=args.timeout,
        )
    except (OSError, URLError) as exc:
        print(f"fetch failed: {exc}", file=sys.stderr)
        return 1

    action = "downloaded" if result.downloaded else "verified"
    print(f"{action} {result.archive_path} ({result.bytes} bytes, sha256={result.sha256})")
    return 0


def fetch_cmu_archive(
    *,
    url: str = CMU_ASF_AMC_URL,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    force: bool = False,
    timeout: float = 60.0,
) -> FetchResult:
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    archive_path = cache_dir / ARCHIVE_NAME

    if not force and manifest_matches_archive(manifest_path, archive_path):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return FetchResult(
            archive_path=archive_path,
            manifest_path=manifest_path,
            sha256=str(manifest["sha256"]),
            bytes=int(manifest["bytes"]),
            downloaded=False,
        )

    headers = read_remote_headers(url, timeout=timeout)
    downloaded = False
    if force or not archive_path.exists():
        download_file(url, archive_path, timeout=timeout)
        downloaded = True

    sha256 = sha256_file(archive_path)
    byte_count = archive_path.stat().st_size
    manifest = build_manifest(
        archive_path=archive_path,
        url=url,
        headers=headers,
        sha256=sha256,
        byte_count=byte_count,
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return FetchResult(
        archive_path=archive_path,
        manifest_path=manifest_path,
        sha256=sha256,
        bytes=byte_count,
        downloaded=downloaded,
    )


def read_remote_headers(url: str, timeout: float = 60.0) -> dict[str, str]:
    request = Request(url, method="HEAD")
    with urlopen(request, timeout=timeout) as response:
        return {key: value for key, value in response.headers.items()}


def download_file(url: str, destination: Path, timeout: float = 60.0) -> None:
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = Request(url, method="GET")
    with urlopen(request, timeout=timeout) as response, temporary.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
    os.replace(temporary, destination)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_matches_archive(manifest_path: Path, archive_path: Path) -> bool:
    if not manifest_path.exists() or not archive_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return int(manifest.get("bytes", -1)) == archive_path.stat().st_size and str(
        manifest.get("sha256", "")
    ) == sha256_file(archive_path)


def build_manifest(
    *,
    archive_path: Path,
    url: str,
    headers: Mapping[str, str],
    sha256: str,
    byte_count: int,
) -> dict[str, object]:
    return {
        "dataset": "CMU Graphics Lab Motion Capture Database ASF/AMC archive",
        "url": url,
        "archive_path": archive_path.as_posix(),
        "bytes": byte_count,
        "sha256": sha256,
        "fetched_at_utc": datetime.now(UTC).isoformat(),
        "headers": dict(headers),
        "cmu_scale_to_meters": CMU_SCALE_TO_METERS,
        "license_note": LICENSE_NOTE,
    }


if __name__ == "__main__":
    raise SystemExit(main())
