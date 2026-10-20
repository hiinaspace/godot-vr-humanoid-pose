"""One command: ``uv run python -m validation.sprint.eval --method basis --split test``."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from validation.sprint import baselines, oracle
from validation.sprint import splits as S

SPLIT_GROUPS = {
    "test": ["cmu_test", "xsens_eval", "xsens_val"],
    "val": ["cmu_val", "xsens_val"],
}
OUT_DIR = S.REPO / "validation/reports/sprint-2026-10"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--method", default="basis", choices=sorted(baselines._REGISTRY))
    ap.add_argument(
        "--split",
        default="test",
        help="test | val | cmu_test | cmu_val | cmu_all | xsens_eval | xsens_val",
    )
    ap.add_argument("--variant", choices=(*oracle.VARIANTS, "both"), default="both")
    ap.add_argument("--quick", action="store_true", help="6 clips x <=30 s per set (smoke test)")
    ap.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args(argv)

    sets = SPLIT_GROUPS.get(args.split, [args.split])
    variants = oracle.VARIANTS if args.variant == "both" else (args.variant,)
    results = []
    for split in sets:
        for variant in variants:
            r = oracle.evaluate_set(
                args.method, split, variant, workers=args.workers, quick=args.quick
            )
            print(
                f"{split:11s} {variant:9s} {r['clips']:4d} clips {r['frames']:8d} fr  "
                f"full {r['full_m'] * 100:6.2f} upper {r['upper_m'] * 100:6.2f} "
                f"lower {r['lower_m'] * 100:6.2f} cm  "
                f"jitter {r['jitter']:.2f} (GT {r['jitter_gt']:.2f})  {r['seconds']:.0f}s",
                flush=True,
            )
            results.append(r)
    tag = f"{args.method}_{args.split}" + ("_quick" if args.quick else "")
    path = oracle.write_report(args.method, results, args.out, tag)
    print(f"wrote {path}")
    print(path.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
