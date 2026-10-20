# oracle-v1 lock

These files define the frozen evaluation. Later rounds must not change them; any change means a
new version (oracle-v2) with its own tag and a re-run of every reported number.

| file | sha256 |
|---|---|
| validation/sprint/oracle.py | 6bf031954f35b34155a7bf448892d906a09fa58d80507c5859a120d8cc891e70 |
| validation/sprint/conventions.py | bd9569415e609297319efb2f0d7b4ec0df18882affbaeaae00937492c99e6ed8 |
| validation/sprint/splits_v1.json | f7106767622a838569421ed1f0617eebb3040b9ccea98481c3accdcbf576809c |

Check: `sha256sum validation/sprint/oracle.py validation/sprint/conventions.py validation/sprint/splits_v1.json`.
Tag: `oracle-v1`. Not locked (may change without a version bump as long as they do not change
the numbers): eval.py, baselines.py (new methods may be added; `gt`, `gt_rot`, `basis` must not
change), splits.py, probe_conventions.py, repro_legacy.py.

# oracle-v2 lock (2026-10-20)

oracle-v2 imports oracle-v1 unchanged (hashes above still hold) and only adds columns:
GT-bone-length FK for canonical-rest rotation outputs and the skate / ground / pop / swivel
columns. `eval.py` now defaults to `--oracle v2`; `--oracle v1` reproduces the Round 1 tables.
Gate: v1 numeric columns of gt, gt_rot and basis are identical under v2 (checked key by key).
Tag: `oracle-v2`.

| file | sha256 |
|---|---|
| validation/sprint/oracle_v2.py | fcde4cbc2f86c2a7d2dea4f4f4a41acb3c91953b97f41d943e13e73fce04bda9 |
| validation/sprint/rest.py | f18004216711d083bcd0c72b640ecc83c5f519daecfc22ddf4a2b862119d907c |

# splits-v2 (2026-10-20, Round 3, after the Round 2 audit)

The Round 2 audit found content-level duplicates across splits that the id-level split cannot see:
CMU test subject 84 == train subject 74 and CMU val subject 141 == train subject 106 (frame for
frame), plus Xsens `eval_S02_A04` == train `S02_A04` (and the mirror pair). `splits_v2.json` is
`splits_v1.json` minus every val / test / eval clip that duplicates a clip in another split (train
copies stay). v1 is untouched and stays locked. `splits.py` now loads v2 (`SPLITS_VERSION = "splits-v2"`);
every Round 3 number uses v2. Round 2 numbers (E1, Basis on val / test) were on v1 and are recomputed.

Method (`dedupe.py`, `dedupe_pairs_v1.json` = all 56 pairs): root-relative 13-joint poses of all 2631
clips (CMU + Xsens, incl. mirror clips); stride-10 frames in one KD-tree (radius 1.8 cm); per candidate
pair (>= 5 hits) the dominant frame offset is verified over the full overlap: >= 60 frames, >= 90 % of
frames with mean joint distance < 5 mm, pose std > 2 mm; held-out clips are also queried mirrored.
Result: 56 cross-split pairs, 54 clips removed (CMU test 20 = all of subject 84, CMU val 32 of
subject 141, Xsens eval 2). Remaining: CMU test 99 clips / 7 subjects, CMU val 290 clips, Xsens eval 9
clips, S09 12. Not detectable by this method: re-solved or re-captured takes of the same motion.
`probe_conventions.py` thresholds / quick-sample step were adjusted for the deduped clip lists
(walking-forward fraction 0.848 vs 0.85 after removing the duplicated subject; the quick probe sample
step 40 -> 25); the full probe passes.

| file | sha256 |
|---|---|
| validation/sprint/splits_v2.json | ea8c92e17c09c0b487cd26723d15494185002ef5fb423bc24471c8d8ba60a0c7 |
| validation/sprint/dedupe_pairs_v1.json | eceef758aa60f44f190822768ec7778534a083bbe630f014a31b45e130fae8fd |

# oracle-v3 (2026-10-20, Round 3)

Add-only over oracle-v2 (v1 and v2 files and hashes above unchanged). Adds a per-frame GT-validity mask
(`oracle_v3.valid_mask`): a frame is invalid if any joint jumps > 0.5 m to the previous or next frame, or
any bone length is > 35 % off the clip median (a tighter 15 % bound flagged 20-36 % of frames in legitimate
CMU clips, whose virtual joints are not rigid); the mask is dilated by +-5 frames. Masked frames are
excluded from every metric (jitter: fully valid 4-frame windows; skate / pen / float: valid t-1, t, t+1 and
the floor percentile over valid frames; pops: both frames valid; swivel: valid frames). `eval.py`
defaults to `--oracle v3`; the masked fraction per set is printed and stored (`masked_frac`, and
`masked_clips` when a set exceeds 2 %). Splits are splits-v2. Masked fraction: CMU val 0.07 %, S09 0.58 %.
Round 2 / early Round 3 numbers (v2) are kept under `eval_v2` for Basis and the final model only.

| file | sha256 |
|---|---|
| validation/sprint/oracle_v3.py | $H |
