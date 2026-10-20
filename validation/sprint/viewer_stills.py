# ruff: noqa: E501
"""Headless-browser stills of the review viewer (proof the export is not garbled).

    uv run python -m validation.sprint.viewer_stills [--clips 3] [--cams 3q,side,top,fp]

Needs a headless chromium: ``nix shell nixpkgs#chromium`` is used when ``chromium`` is not on PATH.
Renders each owner camera at two timepoints (lowest and highest learned error frame) per clip, with
GT / Basis / learned overlaid, by driving the viewer through its URL hash (``#clip=..&t=..&cam=..&still=1``).
The viewer needs the pinned three.js from cdnjs (network) unless ``--three`` points at a local copy.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from validation.sprint import splits as S

HERE = Path(__file__).resolve().parent
OUT = S.REPO / "validation/reports/sprint-2026-10/viewer-stills"


def _nix_chromium() -> str:
    out = subprocess.run(
        ["nix", "shell", "nixpkgs#chromium", "-c", "sh", "-c", "command -v chromium"],
        check=True, capture_output=True, text=True,
    )  # fmt: skip
    return out.stdout.strip()


THREE_URL = "https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"


def prepare_local_html(args) -> Path:
    """Copy of the viewer pointing at a local three.js (headless chromium stalls on the CDN fetch)."""
    three = args.three
    if not three.exists():
        import urllib.request

        three.parent.mkdir(parents=True, exist_ok=True)
        three.write_bytes(urllib.request.urlopen(THREE_URL, timeout=60).read())
    out = three.parent / "viewer_local.html"
    out.write_text(args.html.read_text().replace(THREE_URL, three.name))
    return out.resolve()


def pick_clips(blob: dict, n: int) -> list[int]:
    """One CMU worst, one Xsens worst, one learned-beats-basis (then others), no live."""
    want = ["worst-cmu-test", "worst-xsens-s09", "learned-beats-basis", "sit-crouch"]
    out: list[int] = []
    for g in want:
        for i, c in enumerate(blob["clips"]):
            if c["group"] == g and c["has_gt"] and i not in out:
                out.append(i)
                break
    return out[:n]


def err_curve(c: dict, m: str) -> np.ndarray:
    u = np.frombuffer(base64.b64decode(c["err"][m]), dtype="<u2")
    return u / 1000.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--html", type=Path, default=HERE / "viewer/index.html")
    ap.add_argument("--data", type=Path, default=S.REPO / ".cache/r4b/review_data.json")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--clips", type=int, default=3)
    ap.add_argument("--cams", default="3q,side,top,fp")
    ap.add_argument("--size", default="900,700")
    ap.add_argument("--theme", default="light")
    ap.add_argument("--three", type=Path, default=S.REPO / ".cache/r4b/three.min.js")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--profile", type=Path, default=S.REPO / ".cache/r4b/chprofile")
    args = ap.parse_args(argv)
    blob = json.loads(args.data.read_text())
    html = prepare_local_html(args)
    args.out.mkdir(parents=True, exist_ok=True)
    cmd0 = [shutil.which("chromium") or _nix_chromium()]
    jobs = []
    for ci in pick_clips(blob, args.clips):
        c = blob["clips"][ci]
        e = err_curve(c, "learned")
        if "invalid" in c:  # never pick a GT glitch frame
            bad = np.frombuffer(base64.b64decode(c["invalid"]), dtype=np.uint8) > 0
            e = np.where(bad, np.nan, e)
        shots = {
            "good": int(np.nanargmin(e[30:-30]) + 30),
            "worst": int(np.nanargmax(e)),
        }
        slug = re.sub(r"[^A-Za-z0-9]+", "_", c["id"]).strip("_")
        for tag, t in shots.items():
            for cam in args.cams.split(","):
                jobs.append((ci, t, cam, args.out / f"{slug}_{cam}_{tag}_f{t}.png"))

    def shoot(job):
        ci, t, cam, png = job
        prof = args.profile.with_name(f"{args.profile.name}_{png.stem}")
        url = f"file://{html}#clip={ci}&t={t}&cam={cam}&still=1&theme={args.theme}"
        cmd = cmd0 + [
            "--headless=new", "--no-sandbox", "--disable-gpu", "--use-gl=angle",
            "--use-angle=swiftshader", "--enable-unsafe-swiftshader",
            f"--user-data-dir={prof}", f"--screenshot={png}",
            f"--window-size={args.size}", "--virtual-time-budget=2500", url,
        ]  # fmt: skip
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=400)
        finally:
            shutil.rmtree(prof, ignore_errors=True)
        print(png, flush=True)
        return png

    with ThreadPoolExecutor(args.jobs) as ex:
        made = list(ex.map(shoot, jobs))
    print(f"{len(made)} stills in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
