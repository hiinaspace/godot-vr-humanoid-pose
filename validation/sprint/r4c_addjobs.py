"""Append the Round-4c ablation jobs to the r3 queue (val v3 = splits-v3, then test, clean, raw model output)."""
import json, sys
p = ".cache/sprint/r3/jobs.json"
j = json.load(open(p))
have = {x["name"] for x in j}
BASE = ["--steps", "15000", "--val-every", "1500"]
FULL = ["--gt-mask", "--jit-loss", "--contact-loss"]
def ev(ck=None):
    out = []
    for tag, split in (("val", "val"), ("test", "test")):
        e = {"tag": tag, "split": split, "variant": "clean"}
        if ck: e["ckpt"] = ck
        out.append(e)
    return out
R = ".cache/sprint/runs/"
jobs = [
 {"name": "r4c_ref_stream", "evals": ev(R + "r3_stream_full/best.pt")},
 {"name": "r4c_ref_win", "evals": ev(R + "r3_win_full/best.pt")},
 {"name": "r4c_stream_s1", "train": BASE + ["--streaming", "--seed", "1"] + FULL, "evals": ev()},
 {"name": "r4c_mask_only", "train": BASE + ["--gt-mask"], "evals": ev()},
 {"name": "r4c_stream_only", "train": BASE + ["--streaming"], "evals": ev()},
 {"name": "r4c_cmuonly", "train": BASE + ["--xs-frac", "0"], "evals": ev()},
 {"name": "r4c_xs025", "train": BASE + ["--xs-frac", "0.25"], "evals": ev()},
 {"name": "r4c_xsonly", "train": BASE + ["--xs-frac", "1"], "evals": ev()},
 {"name": "r4c_hmd_w20", "train": BASE + ["--window", "20"], "evals": ev()},
 {"name": "r4c_avatar_w10", "train": ["--model", "avatarposer", "--steps", "140000", "--val-every", "7000", "--window", "10"], "evals": ev()},
 {"name": "r4c_hmd_w80", "train": BASE + ["--window", "80"], "evals": ev()},
]
for x in jobs:
    if x["name"] not in have: j.append(x)
json.dump(j, open(p, "w"), indent=1)
