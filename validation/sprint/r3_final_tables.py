# ruff: noqa: E501, B007, F841, E731, E741, F541
"""Round 3 final tables (stream vs windowed, E2 2x2, post-processes on the final model, test table, per-activity,
per-clip) from ``.cache/sprint/r3/eval``. ``python -m validation.sprint.r3_final_tables > tables.md``."""

from __future__ import annotations

from validation.sprint import r3_report as RP

load, rows, header, load_raw = RP.load, RP.rows, RP.header, RP.load_raw
VAL = ("cmu_val", "xsens_val")
TEST = ("cmu_test", "xsens_eval", "xsens_val")


def block(title: str, items: list[tuple[str, str]], variant="clean", sets=VAL) -> list[str]:
    out = [f"### {title}", ""] + header("row")
    for lab, stem in items:
        out += rows(lab, stem, variant, sets)
    return out + [""]


def masked(stem: str, sets=VAL) -> str:
    d = load_raw(stem)
    if d is None:
        return ""
    return ", ".join(
        f"{s} {100 * d[(s, 'clean')]['masked_frac']:.2f} %" for s in sets if (s, "clean") in d
    )


def per_activity_rows(stems: list[tuple[str, str]], key: str, split="cmu_val") -> list[str]:
    ds = [(l, load_raw(s)) for l, s in stems]
    ds = [(l, d) for l, d in ds if d]
    acts = sorted(ds[0][1][(split, "clean")]["per_activity"])
    out = [
        "| activity | frames | " + " | ".join(l for l, _ in ds) + " |",
        "|---|---|" + "---|" * len(ds),
    ]
    for a in acts:
        fr = ds[0][1][(split, "clean")]["per_activity"][a]["frames"]
        out.append(
            f"| {a} | {fr} | "
            + " | ".join(f"{d[(split, 'clean')]['per_activity'][a][key] * 100:.1f}" for _, d in ds)
            + " |"
        )
    return out


def clip_rows(clips: list[str], stems: list[tuple[str, str]], split="cmu_test") -> list[str]:
    ds = [(l, load_raw(s)) for l, s in stems]
    out = [
        "| clip | " + " | ".join(f"{l} upper / lower" for l, _ in ds) + " |",
        "|---|" + "---|" * len(ds),
    ]
    for c in clips:
        cells = []
        for _, d in ds:
            pc = {x["clip"]: x for x in d[(split, "clean")]["per_clip"]} if d else {}
            x = pc.get(c)
            cells.append(f"{x['upper_cm']:.1f} / {x['lower_cm']:.1f}" if x else "n/a")
        out.append(f"| {c} | " + " | ".join(cells) + " |")
    return out


def twobytwo2() -> list[str]:
    out = [
        "| train data | set | eval clean: upper / lower / hand / jit | eval corrupted: upper / lower / hand / jit / pops | "
    ] + ["|---|---|---|---|"]
    for lab, cs, ks in (
        ("clean", "r3_stream_full__val", "r3_stream_full__val"),
        ("corruption-aug (E2)", "r3_stream_aug__val", "r3_stream_aug__val_corr"),
    ):
        dc, dk = load(cs), load(ks)
        for s in VAL:
            c, k = dc[(s, "clean")], dk[(s, "corrupted")]
            out.append(
                f"| {lab} | {s[:-4]} | {c['upper_m'] * 100:.2f} / {c['lower_m'] * 100:.2f} / {c['hand_m'] * 100:.2f} / {c['jit_ratio']:.2f} | {k['upper_m'] * 100:.2f} / {k['lower_m'] * 100:.2f} / {k['hand_m'] * 100:.2f} / {k['jit_ratio']:.2f} / {k['pop_per_min']:.1f} |"
            )
    return out


def main() -> int:
    P = print
    P(
        "## Streaming vs windowed (same all-in recipe: GT-mask + jitter + contact losses; val, clean)\n"
    )
    P(
        "\n".join(
            block(
                "",
                [
                    ("windowed all-in (14k steps, time-capped)", "r3_win_full__val"),
                    ("streaming all-in (final)", "r3_stream_full__val"),
                ],
            )
        )
    )
    P(
        "## E2 2x2: training data clean vs corruption-augmented, evaluated clean vs corrupted (streaming all-in)\n"
    )
    P("\n".join(twobytwo2()))
    P("\n(all-in windowed, clean-trained: corrupted-eval rows below)\n")
    P(
        "\n".join(
            header("row")
            + rows("windowed all-in", "r3_win_full__val_corr", "corrupted")
            + rows("streaming all-in", "r3_stream_full__val", "corrupted")
            + rows("streaming + aug (E2)", "r3_stream_aug__val_corr", "corrupted")
        )
    )
    P("\n## Post-processes on the final (streaming) model, val clean\n")
    items = [
        ("final raw", "r3_stream_full__val"),
        ("+ footlock p>0.5", "r3_stream_full__val_footlock"),
        ("+ footlock p>0.35", "r3_stream_full__val_footlock035"),
        ("+ postfix", "r3_stream_full__val_postfix"),
        ("+ One-Euro 12 Hz / 0.3", "r3_stream_full__val_smooth"),
        ("+ footlock .35 + postfix + One-Euro", "r3_stream_full__val_post"),
        ("footlock .35 + postfix (pre-registered 'final')", "r3_stream_full__val_final"),
        ("+ lock released beyond 5 cm drift", "r3_stream_full__val_final_drift0.05"),
        ("+ lock released beyond 10 cm drift", "r3_stream_full__val_final_drift0.10"),
        (
            "**final = footlock .35 + postfix + 20 cm drift cap**",
            "r3_stream_full__val_final_drift0.20",
        ),
        ("Basis", "r3_basis__val"),
        ("Basis debiased", "r3_basis_deb__val"),
    ]
    P("\n".join(header("row") + sum((rows(l, s) for l, s in items), [])))
    P("\nfinal, corrupted val:\n")
    P(
        "\n".join(
            header("row")
            + rows("final", "r3_stream_full__val_final", "corrupted")
            + rows("final raw", "r3_stream_full__val", "corrupted")
            + rows("Basis", "r3_basis__val", "corrupted")
            + rows("Basis debiased", "r3_basis_deb__val", "corrupted")
        )
    )
    for variant in ("clean", "corrupted"):
        P(f"\n## TEST (oracle-v3, splits-v2, run once; {variant})\n")
        titems = [
            ("Basis", "r3_test_basis__basis"),
            ("Basis debiased", "r3_test_basis__basis_deb"),
            ("E1 (Round 2 recipe, windowed)", "r3_test_e1__e1"),
            ("E1, honest FK", "r3_test_e1__e1_honestfk"),
            (
                "pre-registered final (streaming + footlock .35 + postfix, no drift cap)",
                "r3_test_final__final",
            ),
            (
                "**final (+ 20 cm lock-drift cap; second test look, see text)**",
                "r3_test_final__final_cap",
            ),
            ("final raw (no post-process)", "r3_test_final__final_raw"),
            ("final raw, honest FK", "r3_test_final__final_honestfk"),
            ("windowed all-in raw (reported, not selected)", "r3_test_extras__win_full"),
            ("windowed all-in + post", "r3_test_extras__win_full_post"),
            ("streaming + corruption-aug raw (E2)", "r3_test_extras__stream_aug"),
            ("streaming + corruption-aug + post", "r3_test_extras__stream_aug_post"),
        ]
        P(
            "\n".join(
                header("method")
                + sum((rows(l, s, variant, ("cmu_test", "xsens_eval")) for l, s in titems), [])
            )
        )
    P(
        "\nMasked fraction (GT-validity): test "
        + masked("r3_test_basis__basis", ("cmu_test", "xsens_eval"))
        + "; val "
        + masked("r3_basis__val")
        + "\n"
    )
    P("## oracle-v2 (no GT-validity mask) rows for Basis and the final model, test\n")
    P(
        "\n".join(
            header("method")
            + sum(
                (
                    rows(l, s, v, ("cmu_test", "xsens_eval"))
                    for v in ("clean",)
                    for l, s in [
                        ("Basis (v2)", "r3_test_basis__basis_v2"),
                        ("pre-registered final (v2)", "r3_test_final__final_v2"),
                        ("final with drift cap (v2)", "r3_test_final__final_cap_v2"),
                    ]
                ),
                [],
            )
        )
    )
    P("\n## Per-activity MPJPE-lower / upper, CMU val clean (cm)\n")
    st = [
        ("Basis", "r3_basis__val"),
        ("Basis debiased", "r3_basis_deb__val"),
        ("E1", "r3_e1__val"),
        ("final raw", "r3_stream_full__val"),
        ("final (drift cap)", "r3_stream_full__val_final_drift0.20"),
    ]
    P("lower:\n")
    P("\n".join(per_activity_rows(st, "lower_m")))
    P("\nupper:\n")
    P("\n".join(per_activity_rows(st, "upper_m")))
    P("\n## Sitting clip 114_05 and friends (CMU test, clean, upper / lower cm)\n")
    ts = [
        ("Basis", "r3_test_basis__basis"),
        ("Basis deb.", "r3_test_basis__basis_deb"),
        ("E1", "r3_test_e1__e1"),
        ("final (cap)", "r3_test_final__final_cap"),
    ]
    d = load_raw("r3_test_basis__basis")
    sitting = (
        sorted(
            [x["clip"] for x in d[("cmu_test", "clean")]["per_clip"] if x["activity"] == "sitting"],
            key=str,
        )[:8]
        if d
        else []
    )
    P("\n".join(clip_rows(["cmu/114/114_05"] + sitting, ts)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
