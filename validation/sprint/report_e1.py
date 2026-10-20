"""Assemble validation/reports/sprint-2026-10/e1.md from oracle-v2 JSON reports.

``python -m validation.sprint.report_e1 --e1 e1_test_v2.json --e1-val e1_val_v2.json
--basis basis_test_v2.json --basis-val basis_val_v2.json``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from validation.sprint import oracle_v2 as O2
from validation.sprint import splits as S

OUT = S.REPO / "validation/reports/sprint-2026-10"


def rows(path: Path, label: str, splits: list[str]) -> list[str]:
    res = json.loads(path.read_text())["results"]
    out = []
    for r in res:
        if r["split"] not in splits:
            continue
        cells = [O2.O._cell(r, k, f) for k, _, f in O2.COLS]
        out.append(f"| {r['split']} | {r['variant']} | {label} | " + " | ".join(cells) + " |")
    return sorted(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--e1", required=True)
    ap.add_argument("--e1-val", required=True)
    ap.add_argument("--basis", required=True)
    ap.add_argument("--basis-val", required=True)
    ap.add_argument("--notes", default="")
    a = ap.parse_args()
    head = "| set | variant | method | " + " | ".join(c[1] for c in O2.COLS) + " |"
    sep = "|---|---|---|" + "---|" * len(O2.COLS)
    lines = ["# E1: hmdposer (CMU + Xsens train, commercial-clean) vs Basis", ""]
    lines += [a.notes, "", "## Validation sets (used for selection)", "", head, sep]
    for sp in ("cmu_val", "xsens_val"):
        lines += rows(OUT / a.basis_val, "basis", [sp]) + rows(OUT / a.e1_val, "E1", [sp])
    lines += ["", "## Test sets (run once for the selected checkpoint)", "", head, sep]
    for sp in ("cmu_test", "xsens_eval"):
        lines += rows(OUT / a.basis, "basis", [sp]) + rows(OUT / a.e1, "E1", [sp])
    (OUT / "e1.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
