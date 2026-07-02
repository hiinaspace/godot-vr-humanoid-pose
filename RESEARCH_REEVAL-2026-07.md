# Research re-evaluation — 2026-07-01

A fresh literature pass over PLAN.md / validation/RESEARCH_NOTES.md, run as
four parallel reviews (SOTA lineage, missed method branches, shipping-practice
+ perception, data/licensing) grounded in the repo's stated position. This is
a **delta document**: it records what the earlier research passes missed or
under-weighted, what they got right, and the recommended plan changes.

> **Status 2026-07-02: applied.** The recommendations below have been folded
> into `PLAN.md`, which is now the authoritative current plan. This file is
> kept as the dated rationale and source-citation record; do not treat its
> "recommended changes" framing as pending work.

## TL;DR — recommended plan changes, ranked

1. **Adopt an explicit end-to-end latency budget: ≤75–80 ms movement→avatar
   pose, and count input windowing against it.** Detectability starts at
   ~75 ms (Waltemate VRST'16) / ~80 ms fast-motion (Hoyet AIVR'19); Meta
   judged 160 ms unshippable for self-view legs. At 90 Hz every frame of
   future-context lookahead costs ~11 ms. This constrains model choice more
   than inference cost does. Corollary: replace fixed EMA with a
   speed-adaptive 1€-style filter (latency provably trades worse than
   sub-mm jitter for the self-avatar), and consider **asymmetric self/remote
   treatment** — Meta ships generated legs to observers and hides them from
   self-view; remote avatars sit behind 50+ ms of network latency where
   aggressive smoothing is free.
2. **Recalibrate the numeric target to real-signal SOTA: ~7.4 cm full-body
   MPJPE** (Meta RPM on GORP, real Quest 3 controller signals), not the
   ~3.3 cm synthetic-AMASS protocol numbers. The sim-to-real gap is ~2x.
   Also promote **tracking-loss/transition handling** (holstered controllers,
   hands out of FOV) to a first-class design requirement — RPM's rolling
   prediction with consistency anchoring is the reference design (code/data
   CC BY-NC: knowledge only, no reuse).
3. **Replace the single head-yaw torso heuristic with a DivaTrack-style
   dual-reference-frame blend** (head-projection frame + head-trajectory
   frame + learned/heuristic blend). This directly addresses the documented
   failure of head-derived forward frames (look-back during backward
   walking), which is also the weak point of the current
   residual-on-heuristic decoder design. DivaTrack code is **MIT** and its
   training data is proprietary-not-AMASS, making it the cleanest
   architecture reference legally. Treat lying-down (head-forward ≈ gravity)
   as an explicitly detected mode with its own fallback — even DivaTrack's
   dual frames fail there.
4. **Promote seated/floor poses from deferred to an early milestone.**
   Sitting is the #1 quality cliff in every shipping system (Standable's
   known breakage; VRChat's upright-scalar snap at ~0.65 with foot-IK
   cutoff; ENGAGE's manual mode switch). Nobody has shipped good automatic
   posture-mode inference — this is the cheapest differentiation available,
   and seated desk data is uniquely cheap to self-capture. Sketch: small
   posture classifier (head height vs calibrated standing height, velocity
   stats, dwell) → dedicated seated decoder → optional chair-plane estimate.
5. **Rebuild the data strategy around newly-verified commercially-safe
   sources** (see Data section): Kimodo-SOMA as a synthetic-data engine,
   SparsePoser's Zenodo corpus under CC BY, 100STYLE for locomotion breadth,
   and own-capture as the VR-telemetry source of record. The CMU-only
   distribution-shift problem the notes already admit is now addressable
   without license risk.
6. **Rebalance the synthetic corruption model toward dropout/occlusion, away
   from Gaussian jitter** — lighthouse jitter is sub-mm (0.18–0.3 mm RMS)
   and nearly irrelevant, while 5% dropout produces >19 cm errors and
   velocity metrics fail first (arXiv:2404.18628 injects exactly our
   artifact taxonomy into sparse solvers). And **fit the noise model to our
   own device logs** — no published fitted model exists for strap-mounted
   trackers/held controllers (Brekel OpenVR Recorder is the standard logging
   tool; note SteamVR poses embed photon-time prediction overshoot, itself
   worth modeling).
7. **A/B motion matching against the procedural gait planner** before
   investing further in the planner. 100STYLE (CC BY 4.0, 4M frames styled
   locomotion) + Holden's BSD-2 Motion-Matching/LMM reference implementation
   means MM is not license-blocked, and MMVR (UPC-ViRVIG, method reference
   only — code NC) already demonstrated MM legs from HMD velocity in
   consumer VR. Keep the procedural planner as fallback/blend target.

## What held up (no change needed)

- **Grouped body-partition architecture** (the LGSIK-Poser bet): 2025-26 work
  independently converges on it — SAGE (upper-then-lower stratified
  diffusion), DivaTrack (lower conditioned on upper + foot contact),
  AtomicMotion (five functional clusters + kinematic-tree attention).
- **Hard analytic hand/controller constraint**: the perception literature
  rewards exactly this (fingertip JND ~5 cm; end-effector error is what
  users notice); the MANIKIN-lite differentiable decoder stays.
- **Avoiding SMPL as a runtime/training dependency** is not just convenience
  — it is the single thing keeping the whole pipeline commercially clean.
  Every strong public 3-point model is AMASS-trained and therefore excluded;
  the raw-BVH/Godot-humanoid formulation sidesteps the trap entirely.
- **Staged sparse→denoised-virtual-trackers→latents→analytic-IK pipeline**
  remains well-precedented; no reviewed work contradicts it.
- **Training-recipe upgrades available within the current design**: KineST's
  geometric angular-velocity loss (a more principled multi-scale velocity
  loss) and AtomicMotion's masked full-body pre-conditioning are drop-in
  improvements to the LGSIK-style recipe. (Neither has released code yet;
  papers only.)

## Data & licensing (verified 2026-07)

**Adopt:**
- **NVIDIA Kimodo-SOMA-RP-v1.1** — synthetic-data engine. Apache-2.0 code;
  NVIDIA Open Model License: commercially usable, "NVIDIA claims no
  ownership rights in outputs." Trained on NVIDIA's proprietary 700 h Bones
  Rigplay set — no AMASS/SMPL lineage. SMPL-free 30-joint SOMA skeleton.
  Generate text-conditioned VR-style motion ("seated, gesturing while
  talking") to attack the CMU distribution-shift problem directly.
  **Trap: only the SOMA variants — Kimodo-SMPLX-RP is research-only.**
- **SparsePoser Xsens corpus under CC BY 4.0** (Zenodo record 8427980
  metadata; the repo's CC BY-SA LICENSE_data conflicts, but the Zenodo
  record is the distribution channel and the same rights-holder published
  it there — dual-license analysis favors taking it under CC BY). It is the
  only commercially-usable VR-native GT corpus found (~3 h, Xsens while
  using VR apps). **Action: one confirmation email to the UPC-ViRVIG
  authors (J. L. Ponton) converts "defensible" to "clean."** Even worst-case
  BY-SA, current authority (CC's own 2025 guidance; Getty v. Stability UK)
  says weights are likely not adaptations — the SA exposure is contractual
  and survivable, unlike NC.
- **100STYLE** — CC BY 4.0, commercial explicitly invited. 4M+ frames /
  100 locomotion styles, single Xsens skeleton, Holden maintains a
  common-skeleton retarget. Feeds both a learned model and motion matching.
- **Own capture scales up as the VR-telemetry source of record** — no
  commercially-licensed VR-native corpus at scale exists (BOXRR-23 and
  Who-Is-Alyx are both CC BY-NC-SA: legal as *held-out eval* distribution
  references only). GORP proves real-signal GT capture matters; the cheap
  local version is extra Vive trackers as pseudo-GT during normal VR use.
- Worth an email: **BONES-SEED** (Bones Studio + NVIDIA, 142k Vicon-quality
  annotated sequences, SOMA skeleton — same retarget path as Kimodo) has a
  "qualifying startups" tier; the LLC may qualify cheaply
  (licensing@bones.studio).
- Purchasable gap-filler: MoCap Online now sells an explicit **AI Usage
  Permit** on top of its packs — first mainstream vendor selling ML rights.

**Drop / never use:**
- LAFAN1 (CC BY-NC-**ND** — commonly mistaken for permissive), Bandai Namco
  (NC), ZeroEGGS (NC-ND), Motorica (NC by default), Mixamo (Adobe ToS has an
  explicit ML-training ban), Motion-X/EgoBody/Nymeria (NC + SMPL taint).
- **Distillation does not launder AMASS.** Training from a teacher obtained
  under research terms binds contractually regardless of copyright; NVIDIA's
  MaskedMimic checkpoint is explicitly NC; third-party "MIT" or "CC BY"
  relicensings of AMASS-trained weights (LGSIK-Poser's repo weights,
  DragPoser's weights) are risk transfer, not clean title. Use the code and
  architectures freely; never the published checkpoints. Real unlocks:
  Meshcapade/MPI commercial licensing, or the permissive source corpora
  directly (which is the current strategy — validated).

## New method branches (assessed)

- **Physics trackers (QuestSim/QuestEnvSim lineage, MaskedMimic/
  ProtoMotions)**: don't replace the v1 procedural lower body — sim-in-loop
  runtime, engine-specific policies, and Meta itself never shipped QuestSim.
  But **ProtoMotions is Apache-2.0 with a MuJoCo backend and ONNX export**,
  and MaskedMimic treats sparse VR input as masked inpainting (sitting
  included). Retrained on CMU + own capture (full-AMASS training was 12 h on
  4×A100, so CMU-scale is plausibly consumer-GPU), it is (a) the credible v2
  plausibility layer, and (b) immediately useful **offline** as a
  physics-based data cleaner/augmenter — grounding and contact-correcting
  CMU/Kimodo clips into VR-conditioned training pairs.
- **DragPoser-style test-time latent optimization** (MIT code, retrain on
  CMU): pose constraints as runtime losses in a learned latent manifold.
  Architecturally attractive as the *seam* between the learned upper body
  and procedural foot grounding — the planner's foot targets become runtime
  constraints instead of a hand-authored blend.
- **Environment-aware motion matching** (Ponton SIGGRAPH Asia 2025) points
  where scene-aware grounding is heading; watch it.
- Skipped after review: video/world-model priors (WHAM/TRAM — SMPL-native +
  research licenses), XR-MBT-style depth-sensor legs (no SteamVR sensor
  path), pressure/chair sensing (~5 s/frame), further IMU-phone lineage
  (ideas already absorbed via HMD-Poser).

## Evaluation additions

- Add the **shipping-system failure-mode list** as named qualitative
  regression cases: crouch/prone state snap + ground clip, seated
  leg-flip/stretch/cross, foot skate under stick locomotion, spine twist /
  chest cave on look-up, elbow-through-torso, proportion mismatch ("arm
  pull"/"stilts"), self-legs proprioception mismatch.
- Track **velocity-domain metrics under dropout** explicitly (they fail
  first — arXiv:2404.18628).
- Latency ledger per estimator config: window lookahead + filter lag +
  solve time vs the 75–80 ms budget, self and remote tunings separately.
- Real-signal eval split: BOXRR-23 / Who-Is-Alyx as NC held-out
  distribution references (eval-only; note in license log).

## Concrete next experiments (ordered)

1. Log real device sessions (Brekel OpenVR Recorder or the existing overlay
   capture): idle, dance, sit/stand transitions, controllers occluded behind
   back → fit dropout/excursion/velocity-error statistics → replace the
   invented corruption priors.
2. Implement the dual-reference-frame torso base (DivaTrack MIT reference)
   under the existing residual decoder; re-run the upper-body eval suite;
   add look-back and seated clips to the hard-clip set.
3. Stand up the Kimodo-SOMA → SOMA→Godot-humanoid retarget → synthetic
   sparse-input pipeline; generate a seated/gesture/locomotion-mix corpus;
   measure whether it closes the CMU→Xsens distribution gap the notes
   already document.
4. Posture classifier + seated decoder MVP from self-captured desk sessions
   + CMU seated clips.
5. Motion-matching A/B: 100STYLE + BSD-2 LMM controller vs the procedural
   gait planner on the locomotion eval set (joystick + roomscale).
6. Send the two emails: UPC-ViRVIG (SparsePoser CC BY confirmation),
   Bones Studio (BONES-SEED startup tier).

## Source notes

Key primary sources: RPM/GORP arXiv:2504.05265 (github.com/facebookresearch/
motion_rolling_prediction, CC BY-NC); DivaTrack arXiv:2402.09211
(github.com/robindsyang/DivaTrack, MIT); SAGE arXiv:2405.20786 (CC BY-NC);
AtomicMotion arXiv:2605.22631; KineST arXiv:2512.16791; SSD-Poser
arXiv:2504.18332; MotionMAR arXiv:2606.23000; EnvPoser arXiv:2412.10235;
MaskedMimic arXiv:2409.14393 + github.com/NVlabs/ProtoMotions (Apache-2.0;
HF checkpoint NC — verify exact NVIDIA NC-source-license text before citing);
QuestSim arXiv:2209.09391; QuestEnvSim arXiv:2306.05666; DragPoser
arXiv:2406.14567 (MIT code); MMVR upc-virvig.github.io/MMVR (code NC);
Motion-Matching github.com/orangeduck/Motion-Matching (BSD-2); 100STYLE
ianxmason.com/100style (CC BY 4.0); Kimodo github.com/nv-tlabs/kimodo +
huggingface.co/nvidia/Kimodo-SOMA-RP-v1.1 (NVIDIA Open Model License);
SparsePoser data zenodo.org/records/8427980 (CC BY 4.0 metadata) vs repo
LICENSE_data (CC BY-SA); AMASS license amass.is.tue.mpg.de/license.html
(prohibits training for commercial use); latency: Waltemate VRST'16
(10.1145/2993369.2993381), Hoyet AIVR'19, Teather VR'09, Casiez CHI'12 (1€
filter); tracker noise: Bauer Sensors 2023 (PMC9862184), arXiv:2404.18628,
arXiv:2104.11523; practice: VRChat IK 2.0 blog/docs + feedback board,
Standable Steam reviews, Meta Movement SDK "generative legs" blog +
AGRoL arXiv:2304.08577, Resonite issue #130, SlimeVR docs. Unverified:
QuestSim end-to-end latency figure; MaskedMimic HF license exact text.
