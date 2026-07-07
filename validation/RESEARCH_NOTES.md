# Research Notes

These notes track sparse-VR pose-estimation references that are directly useful
for this repo. The goal is not a complete literature survey; it is to preserve
implementation ideas, evaluation ideas, and licensing constraints for the
CMU/Godot humanoid path.

Plan-level decisions live in `PLAN.md` (authoritative) and the 2026-07
literature re-evaluation lives in `RESEARCH_REEVAL-2026-07.md` (rationale and
citations for the newer decisions: latency budget, dual-frame torso,
seated-first, data policy, dropout-first corruption). These notes stay at the
implementation-reference level.

## Project Constraints

- Runtime target is a Godot humanoid skeleton, not SMPL. SMPL papers and body
  formats are useful for representation and interchange ideas, but a direct
  SMPL output head would need retargeting before it helps the plugin.
- Commercial availability matters. CMU raw mocap and our own VR capture data
  are preferred training/evaluation sources. AMASS, DMPL, Human Body Prior, and
  pretrained weights built from them should be treated as restricted research
  references unless their licenses explicitly allow the intended use.
- Do not collapse all SMPL-family assets into one license bucket. Official
  `SMPL-Body` and `SMPL-X Body` exports are CC BY 4.0, but the full
  SMPL/SMPL-X model/software downloads remain under non-commercial research
  licenses. A transient SMPL-body interchange layer may be acceptable; training
  commercial-bound models with the full model/software still needs review.
- SparsePoser's public Xsens dataset is much closer to VR usage than CMU.
  **License resolved 2026-07: J. L. Ponton (UPC-ViRVIG) confirmed CC BY (no
  ShareAlike) over email, and that open-source weights/library are an approved
  use.** The earlier Zenodo-vs-`LICENSE_data` ambiguity is settled in favor of
  CC BY; the training split is unblocked (attribution required, no SA on
  derived weights).
- The immediate product case is 3-point VR tracking: HMD plus left/right hands.
  Extra trackers in live capture are useful as withheld ground truth, not as
  required estimator inputs.

## References

| Reference | Useful idea | Local cache | Public URL |
| --- | --- | --- | --- |
| AvatarPoser, ECCV 2022 | Transformer over head/hand pose windows; separate global orientation stabilization; post-network arm IK to recover hand alignment. | [paper text](../.cache/references/papers/avatarposer_2207.13784.txt), [repo README](../.cache/references/AvatarPoser/README.md), [license](../.cache/references/AvatarPoser/LICENSE) | https://github.com/eth-siplab/AvatarPoser |
| AvatarJLM, ICCV 2023 | Two-stage sparse VR model: initial pose plus joint-level spatiotemporal tokens for alternating spatial/temporal Transformer blocks; useful reference for moving beyond low-dimensional latents. | n/a | https://arxiv.org/abs/2308.08855, https://github.com/zxz267/AvatarJLM |
| EgoPoser, ECCV 2024 | Sparse/intermittent hand tracking, global motion decomposition, 80-frame windows, SlowFast temporal fusion, FoV masking. | [paper text](../.cache/references/papers/egoposer_2308.06493.txt), [repo README](../.cache/references/EgoPoser/README.md), [network](../.cache/references/EgoPoser/networks/egoposer.py), [training config](../.cache/references/EgoPoser/options/train_egoposer.yaml) | https://github.com/eth-siplab/EgoPoser, https://arxiv.org/abs/2308.06493 |
| HMD-Poser, CVPR 2024 | Scalable sparse-observation model for HMD plus optional IMUs; shape estimation and real-captured free-dancing dataset are useful for the synthetic-vs-real gap. | n/a | https://arxiv.org/abs/2403.03561, https://pico-ai-team.github.io/hmd-poser |
| LGSIK-Poser, AI 2025 | HMD/controller plus optional IMU inputs; grouped LSTMs, skeleton-aware graph reasoning, partition IK refinement, anthropometric priors, and multi-scale velocity loss. Strong current architecture reference. | [paper text](../.cache/references/papers/lgsik-poser_ai2025_6_180.txt), [repo](../.cache/references/LGSIK-Poser) | https://www.mdpi.com/2673-2688/6/8/180, https://github.com/Lucifer-G0/LGSIK-Poser |
| SSR-HMR, Electronics 2025 | Sparse-node model using root-centered dual quaternions, spatiotemporal graph convolution, torso adjustment, pose optimization, and multi-scale velocity loss. Useful graph/representation reference, but six-DoF tracker focused. | [repo](../.cache/references/SSR-HMR) | https://www.mdpi.com/2079-9292/14/18/3664, https://github.com/Lucifer-G0/SSR-HMR |
| SparsePoser, TOG 2023 | Six-6DoF-tracker VR reconstruction; convolutional skeleton autoencoder plus learned limb IK; includes public Xsens BVH train/eval data and Unity live-calibration flow. | [repo](../.cache/references/SparsePoser), [data zip](../.cache/sparseposer/data.zip) | https://upc-virvig.github.io/SparsePoser/, https://github.com/UPC-ViRVIG/SparsePoser |
| MANIKIN, ECCV 2024 | Neural-analytic IK: predict base joints, foot poses, and limb swivel angles, then solve limbs analytically to match end effectors. | [paper text](../.cache/references/papers/manikin_eccv2024_00194.txt), [paper PDF](../.cache/references/papers/manikin_eccv2024_00194.pdf) | https://static.siplab.org/papers/eccv2024-manikin.pdf |
| DragPoser, CGF 2025 | Structured latent-space optimization with dynamic hard constraints and a temporal predictor; useful for live optional tracker constraints and IK correction philosophy. | n/a | https://arxiv.org/abs/2406.14567, https://upc-virvig.github.io/DragPoser |
| Garment Inertial Poser, IJCAI 2025 | Loose sensor modeling: simulate/synthesize garment-mounted IMU signals, condition pose prediction on loose IMU data and garment parameters, and evaluate simulated/synthetic/real data separately. | n/a | https://siplab.org/projects/GarmentInertialPoser, https://www.ijcai.org/proceedings/2025/135 |
| Garment Inertial Denoiser, arXiv 2026 | Treat loose sensor corruption as a front-end denoising problem before pose estimation: location-specific denoising, cross-sensor fusion, then general pose prediction. | n/a | https://arxiv.org/abs/2601.01360 |
| VRArmIK, 2018 | Deterministic shoulder/elbow heuristics for 3-point VR arms: shoulder pose, elbow angle, hand-rotation influence, stretch handling. | [repo README](../.cache/references/VRArmIK/README.md), [arm solver](../.cache/references/VRArmIK/Assets/Plugins/VRArmIK/Scripts/VRArmIK.cs), [license](../.cache/references/VRArmIK/LICENSE) | https://github.com/dabeschte/VRArmIK, https://dabeschte.github.io/paper/2018/12/01/vrarmik.html |
| Existing data strategy | Separates CMU gait priors, in-distribution VR capture, augmentation, and plausibility metrics. | [DATA_STRATEGY.md](DATA_STRATEGY.md) | n/a |

## SparsePoser/Xsens Dataset Spike

SparsePoser is the first non-AMASS public dataset found so far that is both
close to our target usage and immediately parseable. The downloaded archive is
`.cache/sparseposer/data.zip` from Zenodo record `8427980`.

Inventory:

- Zip size: `573,126,833` bytes.
- MD5: `c9f08cf985e2b3bac324c0a4e17d4fbc`.
- Uncompressed size: `1,438,096,348` bytes.
- Files: `117` BVH clips, `106` under `data/xsens/train` and `11` under
  `data/xsens/eval`.
- Duration: about `9.24` train hours and `0.58` eval hours at 60 Hz.
- Canonical Parquet cache: `2,122,704` frames in `117` clips, about
  `1.57 GB` on disk with Zstandard compression.
- Cache build profile on this machine: about `259 s` total, with `247 s` in FK,
  `5.4 s` parsing, and `6.7 s` Parquet writing.
- Subjects: `S01` through `S09`, plus named eval clips such as `game_1`,
  `squat_1`, and `body_directions_1`.
- Skeleton: 22 named joints with meter-scale offsets, root `Hips`, root
  channels `Xposition Yposition Zposition Yrotation Xrotation Zrotation`, and
  local joint channels `Yrotation Xrotation Zrotation`.

Usefulness:

- The data is Xsens Awinda motion captured while actors performed VR-relevant
  actions, including VR gaming, sitting, warmups, exercise, and dancing. This is
  more in-distribution than CMU for social-VR body inference.
- The reference six sparse joints are root/pelvis, left foot, right foot, head,
  left hand, and right hand. We can derive our 3-point setting by dropping root
  and feet, then use those withheld joints as optional-tracker labels and
  lower-body diagnostics.
- The mirrored clips are already included, so subject-held-out splits must treat
  original and `_mirror` clips as a coupled unit.
- The BVH format is easier to normalize into our Parquet cache than AMASS/SMPL,
  but the repo depends on `upc-pymotion`; importing only the format and math
  behavior is preferable to depending on its full training stack.

License/status caveats:

- SparsePoser code is MIT.
- The repo's `LICENSE_data` says model weights and data are CC BY-SA 4.0.
- Zenodo metadata for record `8427980` currently reports `cc-by-4.0`.
- 2026-07 review: the Zenodo record is the distribution channel and the same
  rights-holder published it there, so dual-license analysis favors taking
  the data under CC BY. Even worst-case BY-SA, current authority (CC's 2025
  guidance; Getty v. Stability UK) says trained weights are likely not
  adaptations — the SA exposure is contractual and survivable, unlike NC.
- **Resolved 2026-07 (PLAN.md milestone 7): J. L. Ponton (UPC-ViRVIG)
  confirmed CC BY, no ShareAlike, open-source weights/library approved.** The
  training split is unblocked; the license gate below is now lifted for
  committed artifacts (keep attribution in the license log).

Importer status:

1. The fixed-skeleton BVH reader is implemented in
   `validation.vr_pose_validation.sparseposer_bvh` without a runtime
   `upc-pymotion` dependency.
2. `validation.scripts.build_sparseposer_parquet_cache` exports the Xsens BVH
   files into the same canonical wide-Parquet pose-cache schema as CMU, with
   `dataset=sparseposer_xsens` and `dataset_split=train|eval`.
3. The full train/eval cache builds correctly, but FK remains Python-loop bound.
   Optimize this path before using SparsePoser/Xsens as a frequent cache
   rebuild target.
4. The official SparsePoser code can be run in an ignored reference checkout.
   Its generated BVH outputs can be converted back into our canonical cache with
   `--input-dir`, then compared against the reference cache with
   `validation.scripts.run_pose_cache_comparison`.
5. Training split unblocked (license settled 2026-07, see caveats above): the
   data may now feed committed model artifacts. Keep SparsePoser/Xsens
   attribution in the license log; no ShareAlike obligation on derived weights.

SparsePoser weights benchmark:

- On the official 11-clip Xsens eval split, the released SparsePoser weights
  report about `15 mm` MPJPE in our canonical cache comparison. Their own
  eval script reports the same order of magnitude: about `15.0 mm` MPJPE and
  `20.9 mm` end-effector error averaged over the 11 eval files.
- On the same canonicalized Xsens eval cache, our Basis-style 3-point estimator
  is much worse on GT matching: about `288 mm` mean reference error, `90 mm`
  upper-body error, `138 mm` shoulder error, and `137 mm` elbow error. This is
  not surprising: SparsePoser uses six tracked nodes, while the Basis-style
  estimator is intentionally constrained to HMD plus hands.
- The current CMU-trained GRU upper-body model only slightly improves shoulder
  and elbow error over the Basis baseline on Xsens eval, and worsens spine-axis
  and plausibility metrics. Treat that as distribution-shift evidence rather
  than a model-quality conclusion.
- Running SparsePoser weights directly on CMU remains a separate adapter task.
  Their runtime path expects 64-frame windows of root, feet, head, and hand
  6DoF tracker poses in the fixed Xsens skeleton space. Our current adapter maps
  SparsePoser BVH output into canonical Godot-style joints; CMU-to-SparsePoser
  input retargeting is the opposite direction and needs a careful coordinate,
  scale, and orientation check.

## Input Representation

AvatarPoser and EgoPoser both use sparse 6DoF head/hand inputs, not just
positions. Their AMASS preprocessing builds, per frame, global 6D rotations,
angular velocity, global positions, and linear velocity for head, left hand, and
right hand. EgoPoser's default config uses `input_dim: 60`, `num_input: 3`, and
`window_size: 80`; AvatarPoser uses 40-frame chunks.

LGSIK-Poser and HMD-Poser reinforce this input shape. LGSIK keeps 6D rotation,
angular velocity, position, and linear velocity for sparse VR devices, adds IMU
acceleration when IMUs exist, and includes head-relative cues for hand motion.
The released preprocessing code materializes a 135-wide feature vector made from
global 6D rotations and relative rotation deltas for six sparse joints, global
positions and finite differences for head and hands, head-relative hand rotation
and position plus their deltas, and three acceleration channels for pelvis/knee
IMU-style inputs. It then zeroes unavailable channel ranges to simulate HMD-only,
HMD+2IMU, and HMD+3IMU modes.

The current sparse-window exporter already emits HMD/left-hand/right-hand
position, 6D rotation, finite-difference linear velocity, finite-difference
6D-rotation velocity, validity masks, and calibration features. Two important
caveats remain:

- `rotation6d_velocity` is a model-friendly finite difference, not true angular
  velocity. Live VR trackers can provide physical angular velocity; the capture
  schema should preserve it separately when available.
- CMU-derived synthetic trackers get velocities from differentiated poses. Live
  tracker velocities will include sensor filtering, prediction, and strap
  motion, so synthetic corruption/noise should include velocity channels too.

For this repo, the durable exporter surface should produce Parquet or tensor
windows from the existing CMU cache with:

- HMD, left-hand, and right-hand position.
- HMD, left-hand, and right-hand orientation in a continuous representation,
  preferably 6D rotation or quaternion with sign continuity.
- Linear and angular velocities.
- Temporal deltas from the first frame in the window.
- Horizontal head-relative positions while preserving world/floor height.
- Optional visibility/dropout masks for live hand tracking experiments.

This representation is valuable even before training. It gives deterministic
estimators and learned predictors the same input surface, which makes report
comparisons cleaner.

## Tracker Attachment Drift And Sensor Denoising

Garment Inertial Poser and Garment Inertial Denoiser target loose IMUs rather
than optical VR trackers, but their core assumption transfers: a wearable sensor
is not the body joint. It is a noisy, drifting, body-local attachment with
location-specific corruption. VR straps are usually tighter than garments, but
they can still shift, occlude, jitter, and drift relative to the underlying bone.

Useful patterns:

- Model sensor corruption separately from pose inference. A small denoiser or
  calibration layer can map raw tracker poses to body-attached virtual tracker
  poses before the pose estimator consumes them.
- Make corruption location-specific. HMD, controller, elbow, pelvis, knee, and
  foot trackers have different mounting error, latency, and occlusion behavior.
- Train with synthetic corruption even when the target runtime is deterministic:
  mounting offsets, soft strap drift, per-tracker latency, dropout masks,
  low-frequency yaw/position bias, and short high-frequency jitter bursts.
- Keep clean sparse-tracker reconstruction and downstream pose quality as
  separate reports. A denoiser that reduces tracker error can still hurt the
  avatar if the pose layer over-trusts it.

Implementation implication:

- For learned experiments, prefer a staged model: raw sparse tracker window to
  denoised/body-attached virtual trackers, then residual upper-body latents,
  then analytic IK.
- For live capture, keep optional physical trackers as withheld labels for the
  denoiser when they are not used by the 3-point estimator.

## Global Orientation And Torso

AvatarPoser explicitly warns that deriving pelvis/global orientation directly
from head orientation is brittle: users can rotate or pitch their head without
rotating their torso. EgoPoser extends this with global motion decomposition:
positions are normalized over a temporal window, and horizontal translation is
handled separately from vertical/floor information.

Implementation implication:

- Keep root/pelvis/torso orientation as an estimator state, not a direct copy of
  HMD yaw/pitch.
- Use temporal evidence from head trajectory and both hands to infer body
  facing. For purely deterministic work, this can start as filtered heading from
  horizontal motion plus hand symmetry.
- Report torso/root diagnostics separately from whole-body MPJPE so head
  orientation mistakes do not hide as generic joint error.

## Upper Body IK

All four references point to the same core issue: sparse trackers specify the
hands, but elbow and shoulder placement are underconstrained.

Useful patterns:

- AvatarPoser predicts a body pose, then applies IK over shoulder/elbow joints
  to reduce hand target error. That suggests exact or near-exact controller
  alignment should be a postcondition for our Godot runtime.
- VRArmIK is a practical deterministic baseline. It computes a two-bone arm
  solve from shoulder to hand, then adjusts elbow swivel from hand position,
  hand orientation, side-specific signs, and clamps.
- MANIKIN gives a cleaner abstraction: predict the base joint and limb swivel
  angle, then use analytic geometry to solve the limb. It represents swivel as
  `[cos(phi), sin(phi)]` and regularizes both swivel and mid-joint position.

Implementation implication:

- A good next experiment is a "MANIKIN-lite" arm solver for our skeleton:
  choose or predict shoulder position plus elbow swivel, solve the two-bone arm
  analytically, and enforce the controller/wrist target.
- Condition learned upper-body residuals on avatar/user calibration. EgoPoser's
  body-shape result is a useful warning here: shoulder and elbow latents should
  know the target shoulder width, arm lengths, and height instead of baking
  those proportions into raw meter-space labels.
- Start with deterministic swivel priors from CMU and live capture, then compare
  cheap predictors: constant default, VRArmIK-like heuristic, linear model, and
  small MLP over sparse temporal windows.
- The current Torch training path now has a differentiable runtime decoder for
  this MANIKIN-lite shape: torso heading/lean, shoulder base offsets, and elbow
  swivel are decoded against the heuristic base pose before applying decoded
  point, direction, stretch, and temporal losses.
- Temporal decoded losses should match runtime state policy. Residual artifacts
  use causal EMA smoothing at inference, so training now evaluates decoded
  temporal losses after the same local EMA and can add an output-side excess jerk
  loss instead of only matching reference jerk direction and magnitude.
- Metrics should include hand target error, elbow position error, elbow swivel
  error, elbow side-flip rate, and upper-body MPJPE.

## Cross-Method Read: Augmentation, Spine, Heading (2026-07)

A focused read of the local reference code (SparsePoser, EgoPoser, AvatarPoser,
DivaTrack, LGSIK-Poser, SSR-HMR, VRArmIK, MANIKIN) plus a transient clone of
DragPoser, run to ground three open forks and to scope questions for the
UPC-ViRVIG domain-expert channel. Findings are code-grounded; the residual
judgment calls are collected as Open Questions in `PLAN.md`.

- **Sim-to-real tracker noise (augmentation fork).** None of the read methods
  inject fitted or stochastic positional/rotational tracker noise in training —
  all train on clean mocap. The only input corruption anyone trains with is
  structured *dropout*: EgoPoser's deterministic field-of-view hand masking (zero
  the hand channels when the wrist leaves a modeled camera frustum; optional, on
  only in its `handtracking` configs), with FLAG's stochastic Bernoulli hand
  masking as the cited precedent. EgoPoser's ablation reports realistic FoV
  masking beating uniform-random masking, and its authors explicitly argue
  against generic augmentation ("impossible to cover the entire infinite
  3-space"). SparsePoser handles real hardware entirely with rigid T-pose
  calibration at inference, no learned noise model. Implication: the field's
  sim-to-real answer is architectural + structured-dropout, not data-level
  jitter; our fitted-from-real-logs corruption model (esp. dropout run-length /
  hold-last) is under-explored, and if used as augmentation should favor the
  structured occlusion terms over generic jitter. See "Tracker Attachment Drift
  And Sensor Denoising".

- **Spine / shoulders: analytic vs learned (6-point upper-body fork).** VRArmIK
  is the only fully-analytic method and has no spine at all — the torso is welded
  below the head, shoulder yaw is the hands-averaged direction clamped to ±80° of
  head yaw. Every learned method regresses spine/shoulder *orientation*; where
  analytic components survive they only recover *position* via FK, never
  chest/shoulder orientation. SparsePoser generates the spine purely learned and
  refines shoulders/clavicles with a learned per-limb IK MLP (not analytic). The
  one direct ablation (LGSIK "noIK") shows the learned upper-body refinement
  helps but only modestly (~0.1 cm). No source runs the clean experiment —
  waist-tracked analytic spine vs learned spine, error decomposed by chest lean
  and head-vs-neck twist. A learned upper-body residual is thus consistent with
  the field, and the DoF argument (chest lean + neck twist stay underdetermined
  from endpoints even with head+hips) supports keeping it, but its magnitude of
  benefit given a waist tracker is unproven. See "Upper Body IK".

- **Torso heading / hip-yaw (dual-frame heuristic fork).** Three families:
  observe it (waist tracker — the most effective lever; it largely dissolves the
  ambiguity), learn it implicitly from temporal history (SparsePoser: 64-frame
  temporal conv, no RNN, plus a physical hips tracker as input; AvatarPoser
  Stabilizer; LGSIK LSTM), or an explicit heading heuristic. Only VRArmIK and
  DivaTrack use an explicit heuristic — and DivaTrack independently arrived at our
  dual-reference-frame design (head-forward frame vs velocity-direction frame)
  but *learns* the blend weight via a Transformer rather than hand-tuning it.
  DragPoser (Ponton's newer, genuinely no-pelvis system; SparsePoser always has a
  hips tracker) coded an explicit head-hips-forward heading loss plus feet-floor
  and colinearity losses, then shipped with all of them disabled in favor of the
  learned temporal prior; its per-tracker configs drop the temporal-prior weight
  (0.15→0.02) and lookahead (16→0) as trackers go 3→6. Static-pose heading drift
  is unmeasured across the entire literature — every temporal prior is trained on
  moving data. Implication: our heuristic is a known-good idea, not idiosyncratic;
  the 6-point waist tracker makes it much less load-bearing; the open fork is
  learn-the-blend vs hand-tune, and static behavior is the one thing no code can
  settle. See "Global Orientation And Torso".

**Domain-expert questions (UPC-ViRVIG channel).** Distilled to what is
design-judgment or unobservable-from-code, SparsePoser/DragPoser specific:

1. Static-pose heading — does hip/torso heading stay locked when the user stands
   still and only turns their head, or follow/wobble; was it ever measured?
2. DragPoser's disabled heading-loss family — did the learned temporal prior
   strictly dominate, or did the heuristics cause artifacts; was the 3-tracker
   `lambda_temporal`/lookahead gradient tuned mainly against pelvis-yaw ambiguity?
3. SparsePoser learned vs analytic spine — was a cheap analytic spine
   (interpolated twist between pelvis and head) ever tried as a baseline; how
   much of the learned spine's benefit is orientation accuracy vs end-effector
   position fitting?
4. Clean-training sim-to-real — was the T-pose calibration load-bearing on the
   Vive, or did the temporal window smooth over tracker noise; did controller
   dropout cause visible failures?

## Learned Temporal-Skeleton Architecture

The project has moved from "classical Basis port plus small residuals" toward a
full learned pose model with analytic correction at runtime. The strongest
nearby references now suggest the following shape:

- Use a causal sequence model over sparse tracker windows. GRU is a reasonable
  local baseline, but LGSIK-Poser uses grouped unidirectional LSTMs and
  HMD-Poser also uses recurrent temporal modeling before spatial reasoning.
- Keep direct sparse input skip connections into later pose or IK refinement
  layers. LGSIK-Poser feeds raw sparse end-effector cues into partition MLPs;
  this matches our need to preserve controller alignment instead of asking the
  network to remember it through a hidden state.
- Add skeleton-aware spatial structure. Options in increasing complexity:
  per-region heads, masked group graph over torso/head/arms/legs/shape, or
  joint-level tokens with alternating spatial/temporal attention as in
  AvatarJLM.
- LGSIK's released model is a practical template for a first grouped graph:
  eight dynamic groups plus one shape group are embedded independently, each
  group gets a unidirectional LSTM, a masked skeleton convolution fuses nine
  groups down to six anatomical partitions, then another set of partition LSTMs
  and MLP heads predicts normalized 6D joint rotations. The heads concatenate
  partition features with raw sparse input channels, giving the model direct
  end-effector access instead of forcing all controller alignment through the
  recurrent state.
- Keep anthropometric calibration as first-class model input. LGSIK-Poser uses
  height and limb lengths for shape; SSR-HMR feeds offsets/skeleton structure
  into pose refinement. For Godot, this should mean avatar/user height,
  shoulder width, arm lengths, leg lengths, and controller grip offsets.
- Train decoded geometry and temporal behavior, not only latent coordinates.
  LGSIK-Poser and SSR-HMR both use joint position, sparse/end-effector position,
  and multi-scale velocity losses; SSR-HMR reports that FK losses can be
  unstable in the generator stage but uses FK positions in pose optimization.
- LGSIK's loss stack uses root orientation, local rotation, global rotation,
  joint position, sparse root/hand position, shape consistency, and multi-scale
  velocity over intervals 1, 3, and 5. Its paper reports AdamW, 40-frame windows,
  stride 20 for training segmentation, batch size 64, and 150 epochs; the public
  config differs in some training-loop details, so treat those as defaults to
  inspect rather than canonical paper values.
- SparsePoser's released code uses 64-frame windows, 16-frame stride, skeleton
  convolution/unpool layers over root-space dual quaternions, a static encoder
  for offsets, and a separate learned IK module for limb end-effectors. SSR-HMR
  keeps that two-stage shape but adds torso-first refinement, hierarchical limb
  adjustment, root-centered dual-quaternion features, and multi-scale velocity
  loss. These are stronger evidence for a graph/sequence model plus IK
  refinement than for continuing to hand-design low-dimensional latents.
- Treat smoothing as a runtime prior with a measured latency budget. Our GRU
  results currently improve low-frequency elbow/torso position but add
  high-frequency error; EMA reduces that noise without losing most positional
  gain. A learned model should still train on temporal losses, but output
  filters remain a practical deployment control.

Near-term architecture implication:

1. Keep the current GRU/TCN/MLP ladder as the baseline bench.
2. Add an LSTM model for parity with LGSIK/HMD-Poser.
3. Preserve native live tracker linear/angular velocity when available, while
   keeping finite-difference velocity for CMU-derived synthetic trackers.
4. Add a skeleton-aware grouped model before trying a full Transformer:
   per-group temporal encoders for torso, head, left arm, right arm, lower body,
   and calibration/shape; masked message passing across those groups; then
   region-specific heads.
5. Keep analytic two-bone or IK correction for hands as a hard postcondition,
   but train the model with the correction in the loop.
6. Preserve the graph path as Godot-humanoid native. Avoid making SMPL a runtime
   dependency; use SMPL/AMASS papers for architecture and metrics only unless
   licensing is resolved.

## Lower Body, Feet, And Contact

MANIKIN predicts foot pose and leg swivel, then solves legs analytically from
hip to foot. EgoPoser evaluates ground penetration and foot floating in addition
to joint/velocity error. This matches our current observation that GT-matching
CMU locomotion is not always the same objective as plausible VR avatar feet.

Implementation implication:

- Keep the procedural gait planner as the real-time baseline, but evaluate it
  with plausibility metrics as well as GT errors.
- Treat foot targets and contact state as estimator state. A stateless lower
  body will keep producing foot drag, lazy trailing feet, or too-fast tiny steps.
- Use CMU flat locomotion to fit broad cadence, stance width, step length, and
  contact-duration priors, but validate against live VR capture because joystick
  locomotion and room-scale stepping are different distributions.

## Metrics And Reports

The papers repeatedly use MPJPE, MPJVE, rotation error, upper/lower body splits,
hand error, foot error, ground penetration, and jitter/jerk. The current reports
already cover several of these; useful additions or refinements are:

- MPJPE split by body region: torso, upper body, lower body, hands, feet.
- MPJVE and jerk/jitter split by the same regions.
- Hand target error for exact controller alignment.
- Foot ground penetration and foot floating above ground during intended
  contact.
- Foot skating distance while contact is active.
- Elbow/knee side-flip or hyperextension counts.
- Limb stretch ratio against calibrated bone lengths.
- Worst-clip tables by metric plus rendered outlier videos.

Distribution plots are more useful than summary bars for this project. Keep
violin/scatter/histogram views in the dashboard so rare catastrophic poses stay
visible.

## Licensing And Reuse Notes

- AvatarPoser has an MIT license file in the cloned repo.
- VRArmIK has an MIT license file in the cloned repo.
- EgoPoser's README says MIT and that it is based on AvatarPoser, but the cached
  clone does not currently include a separate license file. Verify before
  copying code.
- MANIKIN's paper advertises a GitHub URL, but no local clone is available in
  the current cache. Treat it as paper-only until a public repo is available.
- AMASS, SMPL+H, DMPL, and Human Body Prior dependencies should not become
  runtime dependencies or sources of commercial training artifacts without a
  separate license review.

## Near-Term Experiments

1. Add a CMU sparse-window exporter that emits the 3-point temporal features
   above from our existing flat-transform Parquet cache.
2. Implement and evaluate a MANIKIN-lite two-bone arm solver with explicit
   elbow swivel against the current Basis/VRArmIK-style arm estimates.
3. Run simple black-box or grid optimization over deterministic upper-body
   constants using CMU, then rerun on clean live capture clips when available.
4. Add report panels for hand target error, elbow side flips, region MPJVE,
   jerk, foot skating, and ground/floating errors.
5. Train residual, confidence-gated upper-body latent predictors against a
   differentiable Torch surrogate of the analytic IK layer, with temporal losses
   on decoded shoulder/elbow/chest trajectories.
6. Keep learned models isolated from the runtime path at first: train small
   predictors for low-dimensional parameters such as normalized shoulder
   elevation/protraction, torso facing, elbow swivel, cadence, stance width, and
   step length before trying a full pose network.
