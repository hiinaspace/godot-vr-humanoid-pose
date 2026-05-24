# Research Notes

These notes track sparse-VR pose-estimation references that are directly useful
for this repo. The goal is not a complete literature survey; it is to preserve
implementation ideas, evaluation ideas, and licensing constraints for the
CMU/Godot humanoid path.

## Project Constraints

- Runtime target is a Godot humanoid skeleton, not SMPL. SMPL papers are useful
  for representation and training ideas, but a direct SMPL output head would
  need retargeting before it helps the plugin.
- Commercial availability matters. CMU raw mocap and our own VR capture data
  are preferred training/evaluation sources. AMASS, SMPL, DMPL, Human Body
  Prior, and pretrained weights built from them should be treated as restricted
  research references unless their licenses explicitly allow the intended use.
- The immediate product case is 3-point VR tracking: HMD plus left/right hands.
  Extra trackers in live capture are useful as withheld ground truth, not as
  required estimator inputs.

## References

| Reference | Useful idea | Local cache | Public URL |
| --- | --- | --- | --- |
| AvatarPoser, ECCV 2022 | Transformer over head/hand pose windows; separate global orientation stabilization; post-network arm IK to recover hand alignment. | [paper text](../.cache/references/papers/avatarposer_2207.13784.txt), [repo README](../.cache/references/AvatarPoser/README.md), [license](../.cache/references/AvatarPoser/LICENSE) | https://github.com/eth-siplab/AvatarPoser |
| EgoPoser, ECCV 2024 | Sparse/intermittent hand tracking, global motion decomposition, 80-frame windows, SlowFast temporal fusion, FoV masking. | [paper text](../.cache/references/papers/egoposer_2308.06493.txt), [repo README](../.cache/references/EgoPoser/README.md), [network](../.cache/references/EgoPoser/networks/egoposer.py), [training config](../.cache/references/EgoPoser/options/train_egoposer.yaml) | https://github.com/eth-siplab/EgoPoser, https://arxiv.org/abs/2308.06493 |
| MANIKIN, ECCV 2024 | Neural-analytic IK: predict base joints, foot poses, and limb swivel angles, then solve limbs analytically to match end effectors. | [paper text](../.cache/references/papers/manikin_eccv2024_00194.txt), [paper PDF](../.cache/references/papers/manikin_eccv2024_00194.pdf) | https://static.siplab.org/papers/eccv2024-manikin.pdf |
| VRArmIK, 2018 | Deterministic shoulder/elbow heuristics for 3-point VR arms: shoulder pose, elbow angle, hand-rotation influence, stretch handling. | [repo README](../.cache/references/VRArmIK/README.md), [arm solver](../.cache/references/VRArmIK/Assets/Plugins/VRArmIK/Scripts/VRArmIK.cs), [license](../.cache/references/VRArmIK/LICENSE) | https://github.com/dabeschte/VRArmIK, https://dabeschte.github.io/paper/2018/12/01/vrarmik.html |
| Existing data strategy | Separates CMU gait priors, in-distribution VR capture, augmentation, and plausibility metrics. | [DATA_STRATEGY.md](DATA_STRATEGY.md) | n/a |

## Input Representation

AvatarPoser and EgoPoser both use sparse 6DoF head/hand inputs, not just
positions. Their AMASS preprocessing builds, per frame, global 6D rotations,
angular velocity, global positions, and linear velocity for head, left hand, and
right hand. EgoPoser's default config uses `input_dim: 60`, `num_input: 3`, and
`window_size: 80`; AvatarPoser uses 40-frame chunks.

For this repo, a useful exporter would produce Parquet or tensor windows from
the existing CMU cache with:

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
- Start with deterministic swivel priors from CMU and live capture, then compare
  cheap predictors: constant default, VRArmIK-like heuristic, linear model, and
  small MLP over sparse temporal windows.
- Metrics should include hand target error, elbow position error, elbow swivel
  error, elbow side-flip rate, and upper-body MPJPE.

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
5. Keep learned models isolated from the runtime path at first: train small
   predictors for low-dimensional parameters such as shoulder offset, torso
   facing, elbow swivel, cadence, stance width, and step length before trying a
   full pose network.
