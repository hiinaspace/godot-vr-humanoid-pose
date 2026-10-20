# Sprint 2026-10 round log

Branch `sprint/2026-10-ml-direction`. Direction memo and digest live in the org vault
(`projects/vr-humanoid-pose/sources/2026-10-19-*`).

## Round 1: frozen evaluation oracle (tag `oracle-v1`)

Builder: Claude Sonnet 5.5 (`claude-sonnet-5-5`). Owner-judged property: one command prints a frozen
evaluation table (Basis vs GT) on subject-disjoint held-out CMU and Xsens data, and the table is
trustworthy (GT-as-prediction scores 0, Basis reproduces documented values, left/right/axis
conventions verified on both datasets). Strike counter: none.

What was built (all under `validation/sprint/`, locked files listed in `LOCK.md`):

| file | role |
|---|---|
| `conventions.py` | canonical 14-joint skeleton, axes, units, fps, rotation convention, FK templates, clip loader (locked) |
| `splits.py`, `splits_v1.json` | subject-disjoint splits + Zenodo/CMU source pins (json locked) |
| `oracle.py` | calibration, tracker-input generator (`inputs-v1`), metrics, aggregation, report writer (locked) |
| `baselines.py` | `gt`, `gt_rot` (GT rotations through the oracle's template FK), `basis` (`basis_research`, untuned) |
| `eval.py` | `uv run python -m validation.sprint.eval --method {gt,gt_rot,basis} --split test [--quick]` |
| `probe_conventions.py` | asserts conventions on real CMU/Xsens clips |
| `repro_legacy.py` | reproduces the pre-sprint documented Basis numbers (unlocked helper) |
| `tests/test_sprint_oracle.py` | 11 tests: split disjointness, GT -> 0, known-offset MPJPE, jitter on a sine, conventions, determinism |

Command: `UV_PROJECT_ENVIRONMENT=$HOME/.cache/venvs/godot-vr-humanoid-pose uv run python -m validation.sprint.eval --method basis --split test`
writes `validation/reports/sprint-2026-10/basis_test.{md,json}`. Wall clock on sayu (16 workers,
CPU only): `gt` 3 s, `gt_rot` 4 s, `basis` 23 s for the whole `test` group (3 sets x 2 variants);
per set 1-7 s. `--split cmu_all` (2.08 M frames) takes 26 s.

### Gates

| gate | result |
|---|---|
| GT scores 0 on every metric except GT jitter | PASS: MPJPE full/upper/lower/root-rel/hand/p95 are exactly 0.00, MPJRE and bone-dir < 1e-9 deg (float round-off), jitter equals GT jitter, on all 6 set x variant rows |
| Basis reproduces documented numbers within 10 % | CMU PASS, Xsens FAIL AS DOCUMENTED but explained (below) |
| Convention probe passes on both datasets | PASS (CMU 210 clips, Xsens 56 + 3 mirror clips, 10 mirror pairs) |
| tests | 11/11 sprint tests pass; full suite 240 pass, 1 pre-existing unrelated failure (`test_live_capture::test_tcp_recorder_flushes_after_abrupt_client_disconnect`: `struct.pack("hh")` for SO_LINGER is invalid on this kernel) |
| ruff format/check, ty on `validation/sprint` | clean |

### Basis reproduction

Documented (macro mean over clips of the old `upper_body` set = head, chest, spine, L/R shoulder,
L/R elbow, L/R hand; the head term carries the 8.25 cm tracker mounting offset): CMU all clips
7.67 cm; Xsens official eval 9.03 cm (lower 27.28).

| pipeline | CMU all (2514 clips) | Xsens eval (11 clips) |
|---|---|---|
| documented (old runner, 120 Hz CMU) | 7.67 | 9.03 |
| old runner re-run today on the eval clips | n/a | 9.03 (bit-identical: code and cache unchanged) |
| oracle loaders + old whole-clip calibration + old floor, T-pose frame **kept** | n/a | **9.03** (lower 27.28) |
| oracle loaders + old calibration, T-pose frame dropped | 8.01 (60 Hz) | 6.68 (lower 23.44) |
| oracle (`basis`, own calibration, clean inputs), same legacy joint set, macro | **7.71** | 7.10 (frame-pooled 7.83) |

Diagnosis of the Xsens gap (verified to the digit above): every Xsens BVH starts with a T-pose
reference frame at the origin, followed by a 3-7 m jump into the take. The old pipeline fed that
frame to the stateful Basis estimator. In short clips (`eval_S01_A09_3`: 11.70 vs 7.16 cm;
`eval_S02_A09_3`: 13.49 vs 5.46 cm) Basis never recovers from the teleport, which also doubles
its lower-body error (27.4 vs 14.2 cm). So 9.03 cm was partly a harness artifact. The oracle uses
that frame as the T-pose calibration pose and drops it from the evaluated clip. Nothing in Basis
was tuned. Consequence for the memo's predictions: the Basis Xsens-eval baseline is 7.8 cm
legacy-set / 10.4 cm in the new MPJPE-upper set, not 9.0; the earlier GRU-vs-Basis Xsens
comparison (0.0903 vs 0.0907) used the same poisoned frame 0 and should not be cited.

### Basis table (`--split test`, `basis_research`, untuned; cm unless noted)

| set | variant | clips | frames | MPJPE-full | upper | lower | root-rel | jitter (GT) | p95 full | hand | legacy-upper |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cmu_test | clean | 118 | 96 204 | 17.65 | 10.86 | 25.81 | 18.11 | 2.72 (2.26) | 31.62 | 0.00 | 8.15 |
| cmu_test | corrupted | 118 | 96 204 | 18.69 | 12.52 | 26.10 | 18.95 | 5.12 (2.26) | 31.69 | 14.36 | 12.55 |
| xsens_eval | clean | 11 | 126 192 | 18.26 | 10.37 | 27.72 | 19.25 | 3.34 (2.04) | 27.78 | 0.00 | 7.83 |
| xsens_eval | corrupted | 11 | 126 192 | 19.39 | 12.16 | 28.07 | 19.64 | 5.76 (2.04) | 29.34 | 14.40 | 12.31 |
| xsens_val (S09) | clean | 12 | 156 879 | 15.59 | 7.73 | 25.03 | 17.77 | 2.14 (2.22) | 30.98 | 0.00 | 6.07 |
| xsens_val (S09) | corrupted | 12 | 156 879 | 16.74 | 9.72 | 25.16 | 18.25 | 4.66 (2.22) | 32.11 | 14.36 | 10.68 |

Headline numbers are frame-pooled; clip-macro means are in the JSON. Jitter unit: 1e2 m/s^3 (AGRoL), all 14 joints, GT in brackets. MPJRE is n/a for Basis (it emits
positions plus a few rotations in its own frame convention); the bone-direction error column
(20.5-24.0 deg) is the rotation-free proxy. Per-activity CMU tables are in the report.
`gt_rot` floor (GT rotations through the oracle template FK): CMU test 5.75 cm full / 5.20 upper,
Xsens eval 5.64 / 2.82, Xsens val 6.85 / 3.50.

Splits: CMU test 8 subjects / 119 clips (118 evaluated, one under 16 frames), val 15 subjects /
322 clips, train 89 subjects / 2073 clips (the sha1 rule yields 8, not ~12, test subjects).
Xsens: official train 106 clips and eval 11; val subject S09 (12 clips with mirrors) removed from
train (94 clips left). Official eval subjects S01, S02, S05 are also in train (recorded in
`splits_v1.json`); the 3 `named` eval clips have unknown subject identity and make up 54 % of eval
frames. Xsens source pinned: Zenodo record 8427980 revision 1 (modified 2023-10-12, CC BY 4.0),
`data.zip` 573 126 833 bytes, md5 `c9f08cf9...` matches Zenodo, sha256 in `splits_v1.json`.

### Oracle design decisions worth knowing (all frozen in oracle-v1)

* Joints: the cached canonical skeleton has 14 joints (no neck, pelvis or upper-leg joints), so
  full = 11 non-input joints, upper = 6 (chest, spine, L/R shoulder, L/R elbow), lower = 5 (hips,
  L/R knee, L/R foot). `foot` is the CMU foot-bone end (mid-foot) but the Xsens ankle: a few-cm
  convention difference between the datasets.
* Calibration is pose-invariant rigid lengths (bone-chain height, arm length, shoulder width) on
  the calibration pose (Xsens T-pose frame; CMU first 10 frames), each x(1 +- 5 %), seeded per clip.
  Height above the floor is never read from the clip: CMU clips can start lying on the floor.
  Playspace floor is Y = 0.
* Inputs `inputs-v1`: 24 floats per frame (3 trackers x pos+quat, 3 validity flags), seed 20261019
  mixed with clip id and variant. `corrupted` = fitted model with Gaussian jitter and the
  `overshoot` term disabled (overshoot = 6 ms x finite-difference acceleration; on mocap it made
  3 m hand steps from 0.2 m clean steps). Dropout, bias, drift, 1-frame latency and re-acquire
  excursions are kept. Tracker rotations are never corrupted.
* Rotations: "incoming-bone" convention (see `conventions.py`), template FK per dataset.

### Doubts and design questions

1. The fitted controller drift amplitude (hands 9-14 cm, documented as an over-estimate) makes the
   `corrupted` hand error 14.4 cm, larger than any upper-body error. The corrupted column measures
   drift tolerance more than dropout tolerance. Fixing it means oracle-v2.
2. Template FK from three noisy scalars leaves a 5-7 cm floor for rotation-output models, 3-3.5 cm
   on upper joints. Models should return their own positions (FK with their own bone lengths) or
   the calibration should carry more lengths; both are oracle-v2 questions.
3. Hand tracker rotations are native joint frames (CMU forearm bone, Xsens hand), not comparable
   across datasets; only head rotation was verified (+Z forward, median cos 0.97-0.98).
4. Xsens A09 train clips have the feet up to 0.8 m below Y = 0 for thousands of frames, and the
   val subject S09 has 118 GT discontinuities (> 0.5 m in one frame). Kept as is; they raise
   Basis lower-body and p95 on `xsens_val`.
5. Leg rest directions differ by 15-20 degrees between CMU and Xsens; upper-body ones agree within
   ~8 degrees. Joint-rotation targets for lower body are therefore not cross-dataset canonical.
