# Data Strategy Notes

This validation harness should separate three data questions:

1. What can CMU tell us about human gait priors?
2. What in-distribution VR tracker data do we need for joystick and room-scale use?
3. What plausibility metrics can catch visually bad poses even when GT matching is not the right objective?

## CMU Gait Analysis

Use `validation.scripts.analyze_cmu_gait` to derive approximate footfall events
from the cached flat locomotion split. The output is intentionally low
dimensional: step length, step width, cadence, swing duration, contact duration,
and sparse upper-body features around each footfall.

The first use is to compare the GT distribution against procedural assumptions:

- `footstep_forward_lead_m`
- `footstep_max_forward_lead_m`
- `footstep_max_step_length_m`
- `stance_width_ratio`
- `footstep_min_interval_s`
- `step_duration_s`

The second use is to test cheap predictors. If head/controller-window features
predict step length, width, or cadence poorly on held-out subjects, then a
deterministic solver should avoid depending too heavily on those quantities.
If a target is predictable, it is a good candidate for a small parameter
predictor feeding the existing procedural planner.

## In-Distribution VR Capture

CMU is useful for broad human-motion priors, but it is not a direct match for
VRChat-style use. Physical locomotion ties head, hands, hips, and feet together;
joystick locomotion often leaves the physical feet stationary while the avatar
moves virtually. A real capture path should record both sparse inputs and
withheld trackers:

- HMD, left controller, right controller.
- Optional pelvis, chest, knees, and feet when available.
- Tracker validity/confidence/dropouts if the API exposes them.
- Floor/chaperone origin and calibration pose.
- User height, tracker layout, and a short consent/session metadata block.
- Activity labels such as idle, joystick locomotion, room-scale walking,
  stationary upper-body motion, dance/exercise, sitting, crouching, and prone.

A friend-friendly capture app should be boring: start/stop, short prompted
clips, local preview, local review/delete, and an export bundle. Parquet is a
good internal format for analysis; a JSON manifest should carry labels,
hardware layout, and redistribution consent.

## Synthetic Augmentation

Synthetic generation can be useful, but should be treated as augmentation, not
ground truth replacement. NVIDIA Kimodo is relevant because its public project
page describes text-conditioned motion generation with kinematic constraints
such as keyframe poses, end-effector targets, and 2D paths, and its current page
lists an NVIDIA Open Model License. Verify the exact license and redistribution
terms before using outputs in any committed dataset or commercial runtime.

Possible augmentation uses:

- Generate motion paths and prompts that resemble VR sessions.
- Distill only low-dimensional gait parameters, not full skeleton outputs.
- Blend generated lower-body motion with real sparse-controller traces.
- Stress-test procedural constraints with adversarial but plausible motions.

## Plausibility Evaluation

GT matching is necessary but not sufficient. For VR avatars, a result can be
acceptable when it differs from the mocap clip but still looks physically
plausible and stable. OpenCapBench is relevant because it evaluates pose
estimation for biomechanical usefulness with kinematic metrics rather than only
visual keypoint error.

Cheap plausibility checks to add before any heavier evaluator:

- Foot skating during estimated contact.
- Knee/elbow hyperextension and side flips.
- Center-of-mass support violations.
- Pelvis/head acceleration spikes.
- Foot plant jerk at step transitions.
- Limb length stretch and floor penetration.

A heavier benchmark model could later score short output windows for
biomechanical plausibility, but it should complement the current JSON metrics
and qualitative videos rather than replace them.

References:

- Kimodo project page: https://kimodo.net/
- Kimodo technical report: https://research.nvidia.com/labs/sil/projects/kimodo/assets/kimodo_tech_report.pdf
- OpenCapBench arXiv: https://arxiv.org/abs/2406.09788
- OpenCapBench WACV paper: https://openaccess.thecvf.com/content/WACV2025/papers/Gozlan_OpenCapBench_A_Benchmark_to_Bridge_Pose_Estimation_and_Biomechanics_WACV_2025_paper.pdf
