# Godot VR Humanoid Pose Estimation Plan

Authoritative plan as of 2026-07-02. The 2026-07 literature re-evaluation
(`RESEARCH_REEVAL-2026-07.md`) has been folded into this file; read that file
for the rationale and primary sources behind the newer decisions, and this
file for the current direction.

## Document Map

- `PLAN.md` (this file) — current goals, architecture direction, data policy,
  and ordered milestones. If another document disagrees with this one, this
  one wins.
- `RESEARCH_REEVAL-2026-07.md` — dated 2026-07 literature pass: what earlier
  research missed, what held up, and source citations. Applied here; kept as
  the rationale record.
- `validation/RESEARCH_NOTES.md` — implementation-level reference notes:
  input representations, architecture templates, the SparsePoser/Xsens spike,
  per-paper licensing notes.
- `validation/DATA_STRATEGY.md` — dataset, capture, and augmentation details.
- `validation/README.md` and `godot/README.md` — operational how-to for the
  validation harness and the Godot scenes/sidecar.

## Goal

Build a better Godot humanoid VR pose estimation system for PC VR avatars. In
this context, "IK" means driving a full humanoid skeleton from sparse
mainstream VR tracking data: usually HMD plus two controllers, with optional
hip, chest, elbow, knee, and foot trackers. The minimum useful target is a
convincing 3-point mode with procedural lower body and foot grounding, not a
general-purpose inverse kinematics library.

The practical benchmark is convincing social-VR style avatar pose quality from
sparse VR tracking, while keeping the implementation open source and suitable
for commercial games.

Quality and latency targets (2026-07):

- **End-to-end latency budget: ≤75–80 ms movement→avatar pose**, counting
  model window lookahead (each frame of future context at 90 Hz costs ~11 ms),
  filter lag, and solve time. Detectability starts at ~75 ms (Waltemate
  VRST'16) / ~80 ms for fast motion (Hoyet AIVR'19). Maintain a latency ledger
  per estimator configuration, with self-view and remote-view tunings tracked
  separately.
- **Accuracy reference is real-signal SOTA: ~7.4 cm full-body MPJPE** (Meta
  RPM on GORP, real Quest 3 signals) — not the ~3.3 cm synthetic-AMASS
  protocol numbers. The sim-to-real gap is roughly 2x.
- **Tracking-loss and transition handling** (holstered controllers, hands out
  of FOV, stale trackers) is a first-class design requirement, not an edge
  case. RPM's rolling prediction with consistency anchoring is the reference
  design (knowledge only — its code/data are CC BY-NC).
- **Seated and floor poses are an early milestone**, not deferred polish.
  Sitting is the #1 quality cliff in every shipping system, and nobody ships
  good automatic posture-mode inference; it is the cheapest differentiation
  available.

## Current Status (2026-07)

Built and active:

- CMU ASF/AMC intake → canonical humanoid wide-Parquet cache, with clip
  environment/activity metadata and manifest checksums.
- SparsePoser Xsens BVH importer into the same canonical cache
  (license-gated; see Data Policy).
- Synthetic 3-point/multi-point tracker generation with per-role attachment
  and corruption profiles (`validation/vr_pose_validation/synthetic_trackers.py`).
- Basis-style procedural solver: calibration, virtual spine/hips, two-bone
  arms with swivel strategies, sticky-feet gait planner
  (`validation/vr_pose_validation/basis_estimator.py`).
- Upper-body latent ML: GRU/TCN/MLP/linear over causal sparse windows, a
  differentiable MANIKIN-lite decoder and FK rotation targets in the training
  loop, decoded temporal losses matched to runtime smoothing, MLflow
  tracking, subject-fold cross-validation
  (`validation/scripts/train_upper_body_latents.py`).
- Metrics and reporting: regional MPJPE/MPJVE, contact/slip/grounding,
  flexion/hyperextension, gait plausibility, frequency-band diagnostics,
  worst-clip tables, plots, outlier review.
- Live VR capture: Godot overlay scene + Python sidecar over a JSON socket
  protocol, T-pose calibration with tracker role labeling, per-session
  Parquet output; optional six-tracker SparsePoser live bridge.

Benchmark results worth remembering:

- SparsePoser released weights: ~15 mm MPJPE on their own 11-clip Xsens eval
  (six trackers).
- Our Basis-style 3-point estimator on the same canonicalized eval: ~288 mm
  mean reference error (expected — three trackers vs six, GT-matching is not
  its objective).
- The CMU-trained GRU upper-body model transfers poorly to Xsens eval:
  distribution-shift evidence, not a model-quality conclusion. This motivates
  the data strategy below.

## Starting Assumptions

- Godot target is `Skeleton3D` plus the newer humanoid skeleton/modifier
  pipeline where practical.
- Desktop VR is the primary runtime target; Quest/mobile compatibility is
  valuable but secondary.
- 3-point HMD plus hands must work well enough without external trackers.
- Optional trackers should improve results without becoming required.
- Scene access matters. Foot grounding and lower-body plausibility need
  collision/raycast access, character-controller state, and avatar scale, so
  the main runtime API should not be VMC-only. VMC remains valuable as an
  interoperability, recording, and comparison format.
- Latency is a harder constraint than inference cost. Model choice is
  constrained by the 75–80 ms budget more than by FLOPs; future-context
  windows and heavy smoothing spend the same budget as slow solves.
- Self-view and remote-view avatars can be treated asymmetrically: remote
  avatars sit behind 50+ ms of network latency where aggressive smoothing is
  free; self-view must stay under the detectability threshold (Meta ships
  generated legs to observers and hides them from self-view).
- Commercially clean data is a hard constraint, and avoiding SMPL as a
  runtime/training dependency is what keeps the pipeline clean: every strong
  public 3-point model is AMASS-trained and therefore excluded. Distillation
  does not launder AMASS (contractual, not just copyright). See Data Policy.
- VRChat black-box reverse engineering is not worth the ToS/account risk.

## Key References

Local reference checkouts and their roles:

- `S:\code\Basis` — closest open-source implementation to the desired
  procedural behavior; MIT. Virtual spine, foot placement, full-body IK.
- `.cache/references/LGSIK-Poser` — MIT code; the grouped-partition
  architecture template (grouped LSTMs, masked skeleton graph convolution,
  partition IK refinement heads with sparse-input skips, anthropometric
  priors, multi-scale velocity loss). AMASS-trained weights excluded.
- `S:\code\ik-pose-research\SparsePoser` — MIT code; BVH evaluation,
  six-tracker live calibration, skeleton-proportion conditioning, learned
  generator + learned end-effector correction.
- `S:\code\ik-pose-research\GodotXRVmcTracker`, `S:\code\godot-renik`,
  `S:\code\ik-pose-research\LabRCSF`, `kalidokit`, VRArmIK — Godot
  integration, pose-write patterns, joint mapping, deterministic arm
  heuristics.
- HMD-Poser / AvatarPoser / EgoPoser / AvatarJLM / SSR-HMR papers and code —
  architecture and metrics references; dataset/license paths not reusable.

Added by the 2026-07 re-evaluation (see `RESEARCH_REEVAL-2026-07.md` for
citations):

- **DivaTrack** (MIT code, non-AMASS proprietary training data) — the
  dual-reference-frame torso design this plan adopts; the cleanest
  architecture reference legally.
- **RPM/GORP** (CC BY-NC — knowledge only) — rolling prediction with
  consistency anchoring for tracking loss; GORP is the real-signal benchmark
  that sets the 7.4 cm reference.
- **ProtoMotions/MaskedMimic** (code Apache-2.0; published checkpoints NC) —
  v2 physics plausibility layer candidate (MuJoCo backend, ONNX export) if
  retrained on clean data; immediately useful offline as a physics-based data
  cleaner/augmenter for CMU/Kimodo clips.
- **DragPoser** (MIT code; weights excluded) — test-time latent optimization
  with dynamic hard constraints; candidate seam between learned upper body
  and procedural foot grounding.
- **Motion-Matching / LMM (Holden, BSD-2) + 100STYLE (CC BY 4.0)** — the
  license-clean motion-matching path for the lower-body A/B.
- **Kimodo-SOMA-RP-v1.1** (Apache-2.0 code; NVIDIA Open Model License, no
  AMASS/SMPL lineage) — text-conditioned synthetic-data engine. Only the
  SOMA variants; Kimodo-SMPLX-RP is research-only.
- **KineST** (geometric angular-velocity loss) and **AtomicMotion** (masked
  full-body pre-conditioning) — drop-in training-recipe upgrades; papers
  only, no code releases yet.

## Architecture Direction

Use an internal pose-estimation API, with VMC as one possible input/output
adapter. Development is Python-first learned estimation with analytic/runtime
correction; the procedural solver remains the baseline and fallback.

Core data types (implemented in `validation/vr_pose_validation/types.py`):
`TrackedPoseSample`, `AvatarCalibration`, `GroundingContext`,
`HumanoidPoseFrame`, and the `PoseEstimator` protocol.

The staged pipeline held up under re-review and stays: **sparse trackers →
denoised/body-attached virtual trackers → grouped latents → analytic IK**,
with grouped body partitions (independently re-validated by SAGE, DivaTrack,
and AtomicMotion) and a hard analytic hand/controller constraint (end-effector
error is what users notice; fingertip JND ~5 cm).

Changes adopted from the 2026-07 re-evaluation:

- **Torso base: dual-reference-frame blend, not head yaw.** Replace the
  single head-yaw forward-frame heuristic with a DivaTrack-style blend of a
  head-projection frame and a head-trajectory frame with a learned or
  heuristic blending weight. Look-back during backward walking is the
  canonical failure of the current design and becomes a named regression
  case. Lying down (head-forward ≈ gravity) breaks even dual frames — detect
  it explicitly and use a dedicated fallback.
- **Posture modes are first-class.** A small posture classifier (head height
  vs calibrated standing height, velocity statistics, dwell time) routes to
  standing/seated/floor handling; a dedicated seated decoder is an early
  milestone, with an optional chair-plane estimate.
  `validation/vr_pose_validation/posture.py` exists and needs promotion into
  the main estimator/validation path.
- **Filtering: speed-adaptive 1€-style filter replaces fixed EMA** on latents
  and outputs. Latency provably trades worse than sub-mm jitter for the
  self-avatar. Filter lag is counted in the latency ledger; self and remote
  get separate tunings.
- **Training recipe upgrades within the current design**: KineST's geometric
  angular-velocity loss as the principled multi-scale velocity loss;
  AtomicMotion's masked full-body pre-conditioning.
- **DragPoser-style test-time latent optimization** is the candidate
  mechanism for joining the learned upper body with procedural foot
  grounding: the gait planner's foot targets become runtime constraints in
  the latent solve instead of a hand-authored blend.
- **Lower body: A/B motion matching against the procedural gait planner**
  before investing further in the planner (100STYLE + BSD-2 LMM controller;
  MMVR demonstrated MM legs from HMD velocity in consumer VR). The planner
  stays as fallback/blend target either way.
- **v2 plausibility layer candidate**: ProtoMotions/MaskedMimic-style physics
  policy retrained on CMU + own capture (full-AMASS training was 12 h on
  4×A100, so CMU-scale is plausibly consumer-GPU). Not a v1 dependency;
  immediately useful offline for grounding/contact-correcting training clips.

Godot integration options (unchanged):

- Near-term live-test path: the Python sidecar owns the learned estimator and
  streams estimated pose/debug data to the Godot overlay.
- Preferred production runtime path: a `SkeletonModifier3D`-based node,
  probably backed later by a Rust or C++ GDExtension once the model/decoder
  shape stabilizes.
- Optional interop: publish `HumanoidPoseFrame` to an `XRBodyTracker`;
  receive/transmit VMC for tools and comparison, not as the only runtime API.

## Data Policy (2026-07)

Adopt (verified licenses; details and caveats in
`RESEARCH_REEVAL-2026-07.md` and `validation/DATA_STRATEGY.md`):

- **CMU raw ASF/AMC** — broad motion priors; already the validation base.
- **Own VR capture** — the VR-telemetry source of record; no
  commercially-licensed VR-native corpus at scale exists. Cheap pseudo-GT:
  extra Vive trackers worn during normal VR use. Seated desk sessions are
  uniquely cheap to self-capture and feed the seated milestone.
- **SparsePoser Xsens corpus** — the only commercially-usable VR-native GT
  corpus found (~9.8 h). Zenodo metadata says CC BY 4.0; the repo's
  LICENSE_data says CC BY-SA. Dual-license analysis favors CC BY via the
  Zenodo channel; **action: one confirmation email to UPC-ViRVIG (J. L.
  Ponton) converts "defensible" to "clean."** Until then keep the training
  split license-gated as implemented.
- **Kimodo-SOMA-RP-v1.1** — synthetic-data engine for VR-style motion
  ("seated, gesturing while talking") to attack the CMU distribution-shift
  problem. SOMA variants only.
- **100STYLE** — CC BY 4.0, 4M+ frames styled locomotion; feeds both learned
  models and motion matching.
- Eval-only distribution references: BOXRR-23, Who-Is-Alyx (both CC BY-NC-SA
  — held-out eval only, never training; note in the license log).
- Worth an email: BONES-SEED "qualifying startups" tier
  (licensing@bones.studio).

Never use: LAFAN1 (CC BY-NC-ND), Bandai Namco (NC), ZeroEGGS (NC-ND),
Motorica (NC), Mixamo (explicit ML-training ban), Motion-X/EgoBody/Nymeria
(NC + SMPL taint), AMASS-trained published checkpoints regardless of the
repo's claimed license (LGSIK-Poser weights, DragPoser weights — risk
transfer, not clean title). Distillation from research-terms teachers binds
contractually; it does not launder anything. Use architectures and code
freely; never the checkpoints.

## Tracker Corruption And Noise Model

Model trackers as noisy devices attached to body-local frames (implemented in
`synthetic_trackers.py`: per-role attachments, noise profiles, dropout,
confidence). Re-evaluation rebalance:

- **Dropout/occlusion dominates; Gaussian jitter nearly doesn't matter.**
  Lighthouse jitter is sub-mm (0.18–0.3 mm RMS) while 5% dropout produces
  >19 cm errors, and velocity metrics fail first (arXiv:2404.18628 injects
  exactly this artifact taxonomy into sparse solvers). Weight the corruption
  budget accordingly.
- **Fit the noise model to our own device logs** — no published fitted model
  exists for strap-mounted trackers or held controllers. Log real sessions
  (the existing overlay capture path, or Brekel OpenVR Recorder) covering
  idle, dance, sit/stand transitions, and controllers occluded behind the
  back; fit dropout/excursion/velocity-error statistics; replace the invented
  priors. Note: SteamVR poses embed photon-time prediction overshoot, itself
  worth modeling.
- Keep the existing taxonomy: per-session calibration bias, per-frame
  measurement noise, low-frequency strap drift, per-tracker latency, dropout
  windows (hold-last-value and invalid-confidence variants). Corrupt velocity
  channels too, not just poses.
- Preserve native tracker linear/angular velocity in live capture; synthetic
  CMU trackers keep finite-difference velocities, and training distinguishes
  the two.

## Evaluation

The metrics suite (regional MPJPE/MPJVE, hand target error, foot
slip/penetration/hover, flexion/side-flip counts, jerk, gait plausibility,
frequency bands, worst-clip tables, distribution plots) is implemented; see
`validation/README.md`. Additions from the re-evaluation:

- **Named qualitative regression cases from shipping-system failure modes**:
  crouch/prone state snap + ground clip, seated leg-flip/stretch/cross, foot
  skate under stick locomotion, spine twist / chest cave on look-up,
  elbow-through-torso, proportion mismatch ("arm pull"/"stilts"), self-legs
  proprioception mismatch, look-back during backward walking.
- **Velocity-domain metrics under dropout** tracked explicitly (they fail
  first).
- **Latency ledger** per estimator config: window lookahead + filter lag +
  solve time vs the 75–80 ms budget, self and remote separately.
- **Real-signal eval split**: BOXRR-23 / Who-Is-Alyx as NC held-out
  distribution references (eval only).
- Hard-clip set gains look-back and seated clips.

## Milestones — Ordered Next Experiments

1. **Fit the corruption model to real logs.** Record real device sessions via
   the existing overlay capture (or Brekel OpenVR Recorder): idle, dance,
   sit/stand transitions, controllers occluded behind back. Fit
   dropout/excursion/velocity-error statistics; replace the invented
   corruption priors in `synthetic_trackers.py`.
2. **Dual-reference-frame torso base** (DivaTrack MIT reference) under the
   existing residual decoder; re-run the upper-body eval suite; add look-back
   and seated clips to the hard-clip set.
3. **Kimodo-SOMA synthetic corpus pipeline**: SOMA→Godot-humanoid retarget
   (same canonical-cache pattern as the SparsePoser importer) → synthetic
   sparse-input generation → seated/gesture/locomotion-mix corpus; measure
   whether it closes the documented CMU→Xsens distribution gap.
4. **Posture classifier + seated decoder MVP** from self-captured desk
   sessions + CMU seated clips; promote `posture.py` into the estimator and
   validation path.
5. **Motion-matching A/B**: 100STYLE + BSD-2 LMM controller vs the procedural
   gait planner on the locomotion eval set (joystick + roomscale).
6. **Send the two license emails**: UPC-ViRVIG (SparsePoser CC BY
   confirmation), Bones Studio (BONES-SEED startup tier).

Cross-cutting, start alongside milestone 1: the 1€ filter swap and the
latency ledger (both are measurement infrastructure the other milestones
report against).

## Open Questions

- Whether the first Godot integration should be direct `SkeletonModifier3D`
  only, or also publish `XRBodyTracker` from the start.
- Which Godot version is the minimum target once the newer humanoid modifier
  APIs are used (currently developing against 4.6, verifying on 4.7 dev).
- Which avatar should be the first visual regression target.
- Whether motion matching (if it wins the A/B) runs in the Python sidecar or
  needs an early native port for latency reasons.
- How the posture classifier's mode switches interact with the latency
  budget — dwell-time hysteresis vs snap transitions.
