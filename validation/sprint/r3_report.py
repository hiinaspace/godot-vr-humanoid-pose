# ruff: noqa: E501, B007, F841, E731, E741, F541, E402
# ruff: noqa: E501
"""Round 3 report tables from ``.cache/sprint/r3/eval/*.json`` (oracle-v2 evaluations).

``python -m validation.sprint.r3_report [--test]`` prints markdown tables: the seed-noise table, the
val ablation table with deltas against the 3-seed base (flagged when larger than 2x the seed sd),
the post-process rows, E2's 2x2, and (with ``--test``) the final test table.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
import os

EVAL = Path(os.environ.get("R3_EVAL", REPO / ".cache/sprint/r3/eval"))
REPORTS = REPO / "validation/reports/sprint-2026-10"

COLS = [  # key, header, scale, fmt
    ("full_m", "full", 100, "{:.2f}"),
    ("upper_m", "upper", 100, "{:.2f}"),
    ("lower_m", "lower", 100, "{:.2f}"),
    ("rootrel_m", "root-rel", 100, "{:.2f}"),
    ("mpjre_m", "MPJRE deg", 1, "{:.1f}"),
    ("hand_m", "hand", 100, "{:.2f}"),
    ("jit_ratio", "jit/GT", 1, "{:.2f}"),
    ("p95_full_m", "p95 full", 100, "{:.1f}"),
    ("skate_cm_s", "skate cm/s", 1, "{:.1f}"),
    ("pen_p95_cm", "pen p95", 1, "{:.1f}"),
    ("float_p95_cm", "float p95", 1, "{:.1f}"),
    ("pop_per_min", "pops/min", 1, "{:.1f}"),
    ("swivel_err_deg", "swivel deg", 1, "{:.1f}"),
]
DELTA = [("upper_m", "upper", 100), ("lower_m", "lower", 100), ("skate_cm_s", "skate", 1)]


def load(stem: str) -> dict | None:
    p = EVAL / f"{stem}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    out = {}
    for r in d["results"]:
        r = dict(r)
        r["jit_ratio"] = r["jitter"] / r["jitter_gt"]
        out[(r["split"], r["variant"])] = r
    return out


def val(r: dict | None, key: str, scale: float):
    if r is None or key not in r or r[key] is None:
        return float("nan")
    return r[key] * scale


def cell(r, key, scale, fmt):
    v = val(r, key, scale)
    return "n/a" if v != v else fmt.format(v)


def rows(label: str, stem: str, variant="clean", sets=("cmu_val", "xsens_val")) -> list[str]:
    d = load(stem)
    if d is None:
        return []
    return [
        f"| {label} | {s.replace('_val', '')} | {variant} | "
        + " | ".join(cell(d.get((s, variant)), k, sc, f) for k, _, sc, f in COLS)
        + " |"
        for s in sets
    ]


def header(first="arm") -> list[str]:
    return [
        f"| {first} | set | variant | " + " | ".join(h for _, h, _, _ in COLS) + " |",
        "|---|---|---|" + "---|" * len(COLS),
    ]


def seed_stats(stems: list[str], variant="clean") -> dict:
    """{(set, key): (mean, sd, n)} over runs that exist."""
    out = {}
    ds = [d for d in (load(s) for s in stems) if d]
    for s in ("cmu_val", "xsens_val"):
        for k, _, sc, _ in COLS:
            xs = [val(d.get((s, variant)), k, sc) for d in ds]
            xs = [x for x in xs if x == x]
            if xs:
                out[(s, k)] = (st.mean(xs), st.stdev(xs) if len(xs) > 1 else float("nan"), len(xs))
    return out


def seed_table(stems: list[str]) -> list[str]:
    lines = header("run")
    for s in stems:
        lines += rows(s.replace("r3_", ""), f"{s}__val")
    ss = seed_stats([f"{s}__val" for s in stems])
    for s in ("cmu_val", "xsens_val"):
        cells = []
        for k, _, sc, f in COLS:
            if (s, k) in ss:
                m, sd, n = ss[(s, k)]
                cells.append((f + " ± " + f).format(m, sd) if sd == sd else f.format(m))
            else:
                cells.append("n/a")
        lines.append(
            f"| **mean ± sd (n={ss.get((s, 'full_m'), (0, 0, 0))[2]})** | {s.replace('_val', '')} | clean | "
            + " | ".join(cells)
            + " |"
        )
    return lines


def delta_table(arms: list[tuple[str, str]], base: list[str], variant="clean") -> list[str]:
    """arms: (label, stem). Delta vs the base mean; ``*`` when |delta| > 2 x seed sd."""
    ss = seed_stats([f"{b}__val" for b in base], variant)
    hdr = [
        "| arm | "
        + " | ".join(f"{s[:3]} d{h}" for s in ("cmu", "xsens") for _, h, _ in DELTA)
        + " |",
        "|---|" + "---|" * (2 * len(DELTA)),
    ]
    lines = list(hdr)
    for label, stem in arms:
        d = load(stem)
        if d is None:
            continue
        cells = []
        for s in ("cmu_val", "xsens_val"):
            for k, _, sc in DELTA:
                x = val(d.get((s, variant)), k, sc)
                m, sd, _ = ss.get((s, k), (float("nan"),) * 3)
                dv = x - m
                flag = "*" if (dv == dv and sd == sd and abs(dv) > 2 * sd) else ""
                cells.append(f"{dv:+.2f}{flag}" if dv == dv else "n/a")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append(
        f"\n`*` = |delta| > 2x the seed sd of the 3 base runs (sd: "
        + ", ".join(
            f"{s[:3]} {h} {ss[(s, k)][1]:.2f}"
            for s in ("cmu_val", "xsens_val")
            for k, h, _ in DELTA
            if (s, k) in ss
        )
        + ")."
    )
    return lines


def per_activity(
    stems: list[tuple[str, str]], split="cmu_val", variant="clean", key="lower_m"
) -> list[str]:
    """Per-activity table of one metric (cm) for several (label, stem); marks where the first is worse."""
    ds = [(lab, load_raw(st)) for lab, st in stems]
    ds = [(lab, d) for lab, d in ds if d]
    if not ds:
        return []
    acts = (
        sorted(next(iter(ds[0][1].values()))["per_activity"])
        if False
        else sorted(ds[0][1][(split, variant)]["per_activity"])
    )
    lines = [
        "| activity | frames | " + " | ".join(lab for lab, _ in ds) + " |",
        "|---|---|" + "---|" * len(ds),
    ]
    for act in acts:
        fr = ds[0][1][(split, variant)]["per_activity"][act]["frames"]
        vals = [
            d[(split, variant)]["per_activity"].get(act, {}).get(key, float("nan")) * 100
            for _, d in ds
        ]
        lines.append(f"| {act} | {fr} | " + " | ".join(f"{v:.1f}" for v in vals) + " |")
    return lines


def load_raw(stem: str):
    p = EVAL / f"{stem}.json"
    if not p.exists():
        return None
    return {(r["split"], r["variant"]): r for r in json.loads(p.read_text())["results"]}


def twobytwo(clean_stem: str, aug_stem: str) -> list[str]:
    lines = [
        "| train \\ eval | set | clean upper / lower / hand | corrupted upper / lower / hand | jit/GT clean | jit/GT corr. |",
        "|---|---|---|---|---|---|",
    ]
    for lab, stem in (("clean", clean_stem), ("corruption-aug", aug_stem)):
        d = load(stem)
        if d is None:
            continue
        for s in ("cmu_val", "xsens_val"):
            c, k = d.get((s, "clean")), d.get((s, "corrupted"))
            f = lambda r: (
                f"{r['upper_m'] * 100:.2f} / {r['lower_m'] * 100:.2f} / {r['hand_m'] * 100:.2f}"
            )  # noqa: E731
            lines.append(
                f"| {lab} | {s[:-4]} | {f(c)} | {f(k)} | {c['jit_ratio']:.2f} | {k['jit_ratio']:.2f} |"
            )
    return lines


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "section", choices=("seeds", "arms", "all", "post", "e2", "test"), default="all", nargs="?"
    )
    args = ap.parse_args()
    base = ["r3_base_s0", "r3_base_s1", "r3_base_s2"]
    print("## Seed noise (base = E1 config, 15k steps, 3 seeds, clean)\n")
    print("\n".join(seed_table(base)))
    arms = [
        ("(f) contact head + skate loss", "r3_contact__val"),
        ("(f') contact weight 5", "r3_contact_w5__val"),
        ("jitter losses (acc + consistency)", "r3_jit__val"),
        ("(e) avatarposer, equal wall-clock", "r3_avatar__val"),
        ("(g) no skeleton-id", "r3_noskel__val"),
        ("(h) no wrist aug", "r3_nowrist__val"),
        ("(b) inputs: pos only", "r3_in_pos__val"),
        ("(b) inputs: pos+rot", "r3_in_posrot__val"),
        ("all-in windowed (mask+jit+contact)", "r3_win_full__val"),
        ("all-in streaming", "r3_stream_full__val"),
        ("all-in streaming + corruption aug (E2)", "r3_stream_aug__val"),
    ]
    print("\n## Ablation arms, absolute (clean)\n")
    print("\n".join(header("arm") + sum((rows(l, st) for l, st in arms), [])))
    print("\n## Ablation arms, delta vs base mean\n")
    print("\n".join(delta_table(arms, base)))
    print("\n## Post-processes (inference only)\n")
    post_rows = [
        ("base s0", "r3_base_s0__val"),
        ("base s0 + postfix (d)", "r3_base_s0__val_postfix"),
        ("contact", "r3_contact__val"),
        ("contact + footlock 0.5", "r3_contact__val_footlock"),
        ("contact + footlock 0.35", "r3_contact__val_footlock035"),
        ("contact + footlock 0.25", "r3_contact__val_footlock025"),
        ("contact + postfix", "r3_contact__val_postfix"),
        ("contact + footlock + postfix", "r3_contact__val_footlock_postfix"),
        ("E1", "r3_e1__val"),
        ("E1 + postfix", "r3_e1__val_postfix"),
        ("E1 honest FK", "r3_e1__val_honestfk"),
        ("Basis", "r3_basis__val"),
    ]
    print("\n".join(header("row") + sum((rows(l, st) for l, st in post_rows), [])))
    print("\n## Per-activity MPJPE-lower, CMU val (cm)\n")
    print(
        "\n".join(
            per_activity(
                [("Basis", "r3_basis__val"), ("E1", "r3_e1__val"), ("base s0", "r3_base_s0__val")]
            )
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
