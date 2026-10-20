"""One command: ``uv run python -m validation.sprint.eval --method basis --split test``.

``--oracle v2`` (default) adds the perceptual columns and GT-bone-length FK for rotation models;
``--oracle v1`` is the frozen Round 1 table. ``--ckpt PATH`` evaluates a trained model checkpoint
(``train.py``) instead of a baseline.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from validation.sprint import baselines, oracle, oracle_v2
from validation.sprint import splits as S

SPLIT_GROUPS = {
    "test": ["cmu_test", "xsens_eval", "xsens_val"],
    "val": ["cmu_val", "xsens_val"],
}
OUT_DIR = S.REPO / "validation/reports/sprint-2026-10"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--method", default="basis", choices=sorted(baselines._REGISTRY))
    ap.add_argument("--ckpt", type=Path, help="model checkpoint from train.py (overrides --method)")
    ap.add_argument("--oracle", choices=("v1", "v2"), default="v2")
    ap.add_argument("--postfix", action="store_true", help="two-bone hand post-fix (model ckpt)")
    ap.add_argument(
        "--honest-fk", action="store_true", help="FK with lengths from noisy calibration"
    )
    ap.add_argument("--smooth", default=None, help="causal One-Euro output filter: MINCUT_HZ,BETA")
    ap.add_argument("--footlock", action="store_true", help="causal foot lock (needs contact head)")
    ap.add_argument(
        "--split",
        default="test",
        help="test | val | cmu_test | cmu_val | cmu_all | xsens_eval | xsens_val",
    )
    ap.add_argument("--variant", choices=(*oracle.VARIANTS, "both"), default="both")
    ap.add_argument("--quick", action="store_true", help="6 clips x <=30 s per set (smoke test)")
    ap.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--tag", default=None, help="report file stem (default method_split)")
    args = ap.parse_args(argv)

    mod = oracle_v2 if args.oracle == "v2" else oracle
    predictor, name = args.method, args.method
    if args.ckpt:
        if args.oracle != "v2":
            ap.error("--ckpt needs --oracle v2")
        from validation.sprint import train

        predictor = train.ModelPredictor.load(
            args.ckpt,
            postfix=args.postfix,
            footlock=args.footlock,
            honest_fk=args.honest_fk,
            smooth=tuple(map(float, args.smooth.split(","))) if args.smooth else None,
        )
        name = predictor.name
    sets = SPLIT_GROUPS.get(args.split, [args.split])
    variants = oracle.VARIANTS if args.variant == "both" else (args.variant,)
    results = []
    for split in sets:
        for variant in variants:
            r = mod.evaluate_set(predictor, split, variant, workers=args.workers, quick=args.quick)
            print(
                f"{split:11s} {variant:9s} {r['clips']:4d} clips {r['frames']:8d} fr  "
                f"full {r['full_m'] * 100:6.2f} upper {r['upper_m'] * 100:6.2f} "
                f"lower {r['lower_m'] * 100:6.2f} cm  "
                f"jitter {r['jitter']:.2f} (GT {r['jitter_gt']:.2f})  {r['seconds']:.0f}s",
                flush=True,
            )
            results.append(r)
    tag = args.tag or f"{name}_{args.split}" + ("_quick" if args.quick else "")
    if args.oracle == "v2":
        tag += "_v2" if not args.tag else ""
    path = mod.write_report(name, results, args.out, tag)
    print(f"wrote {path}")
    print(path.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
