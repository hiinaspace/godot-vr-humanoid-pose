# Godot VR Humanoid Pose Estimation Plan

## Goal

Build a better Godot humanoid VR pose estimation system for PC VR avatars. In this context, "IK" means driving a full humanoid skeleton from sparse mainstream VR tracking data: usually HMD plus two controllers, with optional hip, chest, elbow, knee, and foot trackers. The minimum useful target is a convincing 3-point mode with procedural lower body and foot grounding, not a general-purpose inverse kinematics library.

The practical benchmark is convincing social-VR style avatar pose quality from sparse VR tracking, while keeping the implementation open source and suitable for commercial games.

## Starting Assumptions

- Godot target is `Skeleton3D` plus the newer humanoid skeleton/modifier pipeline where practical.
- Desktop VR is the primary runtime target; Quest/mobile compatibility is valuable but secondary.
- 3-point HMD plus hands must work well enough without external trackers.
- Optional trackers should improve results without becoming required.
- Scene access matters. Foot grounding and lower-body plausibility need collision/raycast access, character-controller state, and avatar scale, so the main runtime API should not be VMC-only.
- VMC is still valuable as an interoperability, recording, and comparison format.
- CMU mocap is the first validation dataset candidate. AMASS/SMPL-derived assets are useful research references but have licensing constraints for commercial-bound work.
- VRChat black-box reverse engineering is not worth the ToS/account risk for this project.

## Key References

- `S:\code\Basis`
  - Closest open-source implementation to the desired behavior.
  - MIT licensed.
  - Relevant files:
    - `Basis\Packages\com.basis.framework\IK\BasisFullBodyIK.cs`
    - `Basis\Packages\com.basis.framework\Drivers\Local\BasisLocalVirtualSpineDriver.cs`
    - `Basis\Packages\com.basis.framework\IK\FootPlacement\BasisLocalFootDriver.cs`
- `S:\code\ik-pose-research\GodotXRVmcTracker`
  - MIT licensed Godot VMC receiver.
  - Good reference for mapping external humanoid pose streams into `XRBodyTracker`.
  - Predates some newer Godot humanoid modifier work, so use it as architecture/reference rather than a final API shape.
- `S:\code\godot-renik`
  - Reference material only.
  - Useful for direct `Skeleton3D` pose-write patterns and known failure cases.
  - Do not spend early implementation time adapting RenIK to CMU data, `XRBodyTracker`, or the new validation harness.
- `S:\code\ik-pose-research\LabRCSF`
  - Useful survey and joint-name mapping reference across Godot, VRM, Unity humanoid, SMPL, BVH, and related formats.
- `S:\code\ik-pose-research\HMD-Poser`, `AvatarPoser`, `EgoPoser`
  - Useful model architectures and metrics references.
  - Do not assume their dataset/license path is suitable for commercial-bound training.
- `S:\code\ik-pose-research\SparsePoser`
  - MIT code with CC BY-SA model/data assets.
  - Useful reference for BVH evaluation, six-tracker live calibration, skeleton-proportion conditioning, and a learned generator plus learned end-effector correction architecture.
  - More focused on 6-point tracking than the required 3-point MVP, so use it as a validation/training design reference rather than the first runtime target.
- `S:\code\ik-pose-research\kalidokit`
  - MIT VTuber/VRM pose heuristics reference.
  - Useful for readable landmark-to-humanoid math, not enough as the main 3-point VR solver.
- `S:\code\ik-pose-research\OpenWBT`, `xr_teleoperate`, `VisionProTeleop`
  - Useful robotics teleoperation references for optimizer costs, smoothing, calibration, and streaming schemas.
  - Not direct lower-body avatar estimation ports.

## Architecture Direction

Use an internal pose-estimation API, with VMC as one possible input/output adapter.

Core data types:

- `TrackedPoseSample`
  - HMD transform.
  - Left/right controller or hand transforms.
  - Optional tracker transforms by semantic role: hip, chest, left/right foot, left/right knee, left/right elbow.
  - Tracking confidence, timestamps, and velocities where available.
- `AvatarCalibration`
  - Rest pose, bone lengths, humanoid bone map, avatar scale, eye/head offset, controller-to-wrist offset, foot dimensions, knee bend preferences.
- `GroundingContext`
  - Physics space access, floor normal, collision mask, character velocity, platform transform if any, and optional locomotion state.
- `HumanoidPoseFrame`
  - Root transform plus Godot humanoid joint transforms in a canonical space.
  - Per-joint confidence and flags for estimated vs. tracked data.
- `PoseEstimator`
  - Pure-ish solver interface: sample plus calibration plus grounding context in, `HumanoidPoseFrame` out.

Godot integration options:

- Preferred runtime path: a `SkeletonModifier3D`-based node that applies the estimated pose to a `Skeleton3D`.
- Optional interop path: publish `HumanoidPoseFrame` to an `XRBodyTracker`, allowing `XRBodyModifier3D` and external systems to consume it.
- Optional VMC path: receive or transmit VMC for tools, tests, recording, and comparison, not as the only runtime API.

## Validation First

The first implementation milestone should be an offline validation harness, not a new solver.

### Dataset Intake

1. Download CMU mocap metadata and motion files into a local cache outside source control.
2. Record exact dataset URLs, checksums, and license notes in a generated manifest.
3. Parse ASF/AMC initially. BVH conversion can be added if it simplifies tooling.
4. Convert the CMU skeleton into a canonical humanoid skeleton:
   - Preserve original joint rotations and root motion.
   - Map to Godot humanoid semantics where possible.
   - Store unresolvable joints as auxiliary data, not as hard failures.
5. Apply the official CMU ASF/AMC meter conversion: multiply ASF lengths and AMC root translations by `(1 / 0.45) * 2.54 / 100 = 0.056444...`.
6. Keep the raw-CMU path independent from AMASS/SMPL tooling so the validation corpus stays commercially usable. Use AMASS-based code as reference material for feature construction and splits, not as a dependency.
7. Account for known CMU quirks:
   - Per-subject ASF skeleton proportions.
   - Euler DOF order from ASF joint definitions.
   - Clips without useful locomotion or with unusual starts/ends.
   - Phantom or zero-length bones in converted skeletons.
   - Frame-rate assumptions; CMU ASF/AMC is commonly 120 Hz.
8. Export normalized clips to a stable local format, probably JSONL or NPZ for Python tooling plus optional Godot resources later.

### Synthetic VR Tracker Generation

For each mocap frame:

- HMD proxy from head/skull plus configurable eye/headset offset.
- Left/right controller proxy from wrist or hand plus configurable controller grip offset.
- Optional tracker proxies from pelvis, chest, feet, knees, and elbows.
- Ground plane from mocap floor estimate, foot contact analysis, or a per-clip calibration frame.
- Character velocity from root/head trajectory.

Generate multiple synthetic modes from the same clip:

- 3-point: HMD plus hands.
- 6-point: HMD, hands, hip, feet.
- 8/10-point: add elbows, knees, or chest.
- Degraded tracking: dropped hand frames, jitter, latency, and stale trackers.

### Reference Pipeline Notes

- HMD-Poser, AvatarPoser, and EgoPoser handle CMU through AMASS-formatted SMPL data. They are useful references for:
  - Subset and split selection, especially CMU/BMLrub/HDM05-style validation groupings.
  - Generating sparse head/hand observations from full-body motion.
  - Representing hand pose and hand position in head space.
  - Deriving velocities, accelerations, and smoothness metrics.
  - Skipping non-motion AMASS files and normalizing frame rates.
- EgoPoser's FOV/dropout handling is a useful model for intermittent observations, even though desktop VR controller tracking is usually not headset-camera FOV limited.
- AMCParser is a compact MIT-licensed raw ASF/AMC parser worth using as a behavior reference, but the Godot validation pipeline should implement its own parser with explicit CMU scale conversion, coordinate conversion, and robust error reporting.
- Humenv has curated CMU-derived hard-pose examples such as crouches, sitting, handstands, and hands-on-ground poses. Its license is not suitable for code reuse here, but the clip/frame choices are useful inspiration for a stress-test list.
- SparsePoser does not appear to be a CMU pipeline; it trains/evaluates BVH files from its own Xsens dataset. Its useful transferable pieces are fixed-skeleton validation, sliding motion windows, sparse-root/feet/head/hands feature selection, BVH round-tripping, skeleton dimension files, and live calibration from initial tracker poses.

### Tracker Attachment And Noise Model

Do not synthesize a single deterministic "HMD equals head joint" stream. Model trackers as noisy devices attached to body-local frames:

- Represent each tracker with:
  - Role: `head`, `left_hand`, `right_hand`, `pelvis`, `chest`, `left_foot`, `right_foot`, `left_elbow`, `right_elbow`, `left_knee`, or `right_knee`.
  - Source bone or joint.
  - Local attachment transform.
  - Confidence or validity.
  - Latency in frames or milliseconds.
  - Noise profile id.
- Split synthetic error into:
  - Per-session calibration bias, drawn once per generated tracking session.
  - Per-frame measurement noise.
  - Low-frequency strap drift/slip for waist, feet, elbows, and knees.
  - Independent tracker latency.
  - Dropout windows with hold-last-value and invalid-confidence variants.
  - Optional smoothing or quantization profiles.
- HMD synthesis should start from the head/skull transform, then apply a calibrated eye/headset offset plus small position and orientation bias.
- Controller synthesis should start from wrist/hand transforms, then apply grip offsets that vary by controller style and how the user holds it.
- Extra tracker synthesis should use bone-local attachment transforms rather than joint centers, since real waist, foot, elbow, and knee trackers are worn with significant placement variance.

### Bootstrap Baseline

Use a deliberately simple baseline to validate the data and runtime harness:

- A trivial baseline: head-driven torso, hands fixed to controller targets, feet projected below hips.
- A simple planted-feet variant: keep feet stable until head/root travel exceeds a step threshold, then move one foot toward a reachable support point.

This should be enough to prove CMU ingestion, synthetic tracker generation, Godot playback, metrics, and visual reporting without spending time fitting RenIK into the new architecture.

### Metrics

Use a mix of target fidelity and plausibility metrics. Exact joint match is not the only goal because 3-point tracking is underconstrained.

Primary metrics:

- Head and hand target error.
- Pelvis position/orientation error.
- Foot position error during planted-contact windows.
- Foot slip distance while a foot is classified as planted.
- Foot penetration and hover distance relative to the ground plane.
- Knee flip count and knee-plane discontinuities.
- Joint velocity and acceleration jitter.
- Root/hip jerk.
- Solve time per frame.

Secondary metrics:

- Full-body joint angle error against ground truth.
- Center-of-mass or pelvis projection relative to support foot/feet.
- Recovery time after tracking loss.
- Sitting/crouching failure classification, even if not MVP behavior.

Outputs:

- Per-clip JSON or CSV metrics.
- Aggregate markdown report.
- Optional frame dumps or short rendered videos for qualitative review.
- A "known hard clips" set that stays stable across algorithm changes.

## Basis-Port Runtime Plan

### Phase 1: Calibration And Canonical Pose

- Build avatar calibration from `Skeleton3D` rest pose and humanoid bone map.
- Compute segment lengths, shoulder width, spine length, leg lengths, arm lengths, controller offsets, and foot dimensions.
- Add a simple standing calibration path for user height and headset-to-neck offset.
- Keep all transforms explicit about coordinate space.

### Phase 2: Virtual Spine And Hips

Port the Basis virtual spine behavior first:

- Estimate neck/chest/spine/hips from head pose, avatar proportions, and optional hip/chest trackers.
- Keep hips yaw coherent with head yaw but damped.
- Clamp hips around the head to avoid spine buckling and excessive stretch.
- Add crouch handling from head height before adding more complex state classification.

### Phase 3: Arms

Port or reimplement a robust two-bone arm solve:

- Hands target controller poses.
- Elbows use stable bend hints and optional elbow trackers.
- Shoulder/chest interaction should avoid obvious chest clipping and elbow inversion.
- Wrist/controller offset should be part of calibration, not hardcoded.

### Phase 4: Sticky Feet And Grounding

Port Basis-style sticky feet with Godot scene access:

- Raycast or shape-cast from predicted foot positions.
- Maintain planted foot anchors.
- Trigger steps from hip/head displacement, yaw change, velocity, and leg reach limits.
- Interpolate lift, swing, contact, and settle phases.
- Solve knees analytically from hip, knee preference, and foot target.
- Support moving platforms later by storing anchor transforms relative to collider/platform.

This phase is why VMC should not be the internal solver API: the estimator needs ground queries and character context that are not naturally represented by a one-way pose stream.

### Phase 5: Optional Trackers

- Blend optional trackers by semantic role and confidence.
- Tracked feet should override procedural foot targets, but still use grounding constraints as a corrective layer if desired.
- Tracked hip/chest should override virtual spine estimates with limits to avoid bad tracker spikes.
- Missing or stale trackers should decay back to estimates smoothly.

### Phase 6: Interop And Recording

- Add VMC receiver for external body streams.
- Add VMC transmitter for debugging and comparison in existing VTuber/VR tools.
- Add a local recording format for `TrackedPoseSample`, `GroundingContext` summaries, and `HumanoidPoseFrame`.
- Use recordings to reproduce live failures offline.

### Phase 7: ML Or Optimizer Backends

Only after validation exists:

- Try a small deterministic optimizer for hips/spine/feet using robotics-style costs:
  - target error,
  - joint limits,
  - smoothness,
  - foot contact,
  - regularization toward rest or previous pose.
- Try an ONNX backend trained on CMU-derived synthetic trackers if deterministic methods plateau.
- Keep ML optional and benchmarked against the deterministic Basis-style solver.
- Do not build the runtime around AMASS/SMPL assets unless license issues are resolved for the exact use case.

## Proposed Repo Layout

Initial planning can stay in this repo. Implementation may later move to a new repo.

Potential future layout:

```text
addons/
  godot_vr_humanoid_pose/
    vr_humanoid_pose.gdextension
src/
  pose/
    tracked_pose_sample.*
    avatar_calibration.*
    humanoid_pose_frame.*
    sparse_pose_estimator.*
    foot_grounding.*
    vmc_adapter.*
validation/
  README.md
  cmu_manifest.json
  scripts/
    fetch_cmu.py
    parse_cmu.py
    synthesize_trackers.py
    run_baselines.py
    metrics.py
  godot_project/
    project.godot
    scenes/
      baseline_runner.tscn
  reports/
```

Generated datasets, downloaded CMU files, rendered videos, and large metric outputs should stay out of git by default.

## Open Questions

- Whether to provide a later compatibility layer for RenIK-style users, or keep this as a separate extension with its own API.
- Whether the first Godot integration should be direct `SkeletonModifier3D` only, or also publish `XRBodyTracker` from the start.
- Which Godot version is the minimum target once the newer humanoid modifier APIs are used.
- How much of the CMU conversion should be Python-only vs. Godot-importable resources.
- Which avatar should be the first visual regression target.
- Whether to include a VMC receiver in this repo or keep it as an optional adapter depending on GodotXRVmcTracker.

## Immediate Next Steps

1. Build the CMU fetch/parse prototype outside the runtime path.
2. Select 10 to 20 CMU clips covering walking, turning, crouching, stepping, idle motion, and arm movement.
3. Generate synthetic 3-point tracker streams from those clips.
4. Implement the trivial/bootstrap baseline in a minimal Godot validation project.
5. Write the first metric and visual report before tuning solver behavior.
6. Start the Basis virtual-spine and foot-grounding port after the data path, playback path, and simple baseline are working.
