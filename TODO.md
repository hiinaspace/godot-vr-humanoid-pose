# TODO

## Coordination

- Another Codex agent may be editing this repo concurrently. Treat this file as
  the shared coordination note.
- Current safe parallel lane: validation metrics and tests that do not modify the
  dirty Godot replay/parity files or the dirty cached-validation runner.
- Avoid touching unrelated untracked Godot scene/debug files unless you own that
  work in the other thread.

## Validation Metrics To Add

The validation harness already reports target errors, reference joint errors,
tail/error-threshold diagnostics, temporal velocity/acceleration/jerk, and
foot-contact slip. Next useful metrics from the HSMR/OpenCapBench/HuMoR-style
research pass:

- Joint-angle diagnostics:
  - Position-derived elbow and knee flexion angle error against reference.
  - Aggregate elbow/knee flexion RMSE or mean absolute error.
  - Excessive-flexion violation rates by joint using conservative limits.
  - Later, replace or augment this with true local-rotation joint-limit checks
    when the canonical skeleton carries reliable joint-local rotations.
- Kinematic waveform diagnostics:
  - Per-joint angle time-series RMSE for hips, knees, ankles, pelvis, and spine.
  - Segment/group summaries for lower body, upper body, and torso.
- Ground/contact diagnostics:
  - Existing contact slip/error should be extended with penetration frequency
    and penetration distance thresholds.
  - Add support-foot or support-polygon distance from pelvis/COM proxy.
- Tracked-vs-inferred diagnostics:
  - Keep HMD/hand target error separate from inferred pelvis/legs/feet.
  - Add explicit tracked/inferred aggregate groups in reports.
- World/root trajectory diagnostics:
  - Root/pelvis yaw error, height error, horizontal drift, and trajectory path
    error per clip.

## Implementation Order

1. Add position-derived elbow/knee flexion metrics in
   `validation/vr_pose_validation/metrics.py`.
2. Cover the angle metrics with focused tests in `tests/test_baseline_metrics.py`.
3. Wire direct ASF/AMC validation if safe.
4. Wire cached validation and plotting after the other agent's dirty cached-runner
   changes are either committed or clearly handed off.
