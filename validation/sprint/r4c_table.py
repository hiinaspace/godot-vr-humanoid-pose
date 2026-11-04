"""Round 4c ablation table from .cache/sprint/r3/eval/r4c_*__{val,test}.json (raw model output, clean inputs)."""
import json, sys
from pathlib import Path
EV = Path(sys.argv[1] if len(sys.argv) > 1 else ".cache/sprint/r3/eval")
ARMS = [
 ("r4c_ref_stream", "REF r3_stream_full (streaming, mask+jit+contact), re-eval"),
 ("r4c_ref_win", "REF r3_win_full (windowed all-in), re-eval"),
 ("r4c_stream_s1", "streaming all-in, seed 1 (seed noise vs REF)"),
 ("r4c_mask_only", "windowed, GT-mask only"),
 ("r4c_stream_only", "streaming only (no mask/jit/contact)"),
 ("r4c_cmuonly", "data: CMU only (xs-frac 0)"),
 ("r4c_xs025", "data: xs-frac 0.25"),
 ("r4c_xsonly", "data: Xsens only (xs-frac 1)"),
 ("r4c_hmd_w20", "window 20 (hmdposer)"),
 ("r4c_hmd_w80", "window 80 (hmdposer)"),
 ("r4c_avatar_w10", "avatarposer window 10"),
]
def load(a, s):
    f = EV / f"{a}__{s}.json"
    if not f.exists(): return None
    return {r["split"]: r for r in json.loads(f.read_text())["results"]}
def c(r, k, f=100): return "—" if r is None else f"{r[k]*f:.2f}"
rows = []
for a, label in ARMS:
    v, t = load(a, "val"), load(a, "test")
    g = lambda d, sp: None if d is None else d.get(sp)
    cv, xv, ct, xt = g(v, "cmu_val"), g(v, "xsens_val"), g(t, "cmu_test"), g(t, "xsens_eval")
    rows.append(f"| {label} | {c(cv,'upper_m')} / {c(cv,'lower_m')} | {c(xv,'upper_m')} / {c(xv,'lower_m')} | "
                f"{c(ct,'upper_m')} / {c(ct,'lower_m')} | {c(xt,'upper_m')} / {c(xt,'lower_m')} | "
                f"{'—' if ct is None else '%.2f' % (ct['jitter']/ct['jitter_gt'])} | {c(ct,'skate_cm_s',1)} | {c(ct,'pen_p95_cm',1)} |")
print("| arm | CMU val up/low | Xsens S09 val up/low | CMU test up/low | Xsens eval up/low | CMU test jit/GT | skate cm/s | pen p95 cm |")
print("|---|---|---|---|---|---|---|---|")
print("\n".join(rows))
