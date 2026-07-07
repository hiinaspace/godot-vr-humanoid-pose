# Validation

This folder contains the Python-first validation path for the sparse VR humanoid
pose-estimation project.

Generated artifacts stay out of source control:

- `.cache/cmu/` for downloaded CMU archives.
- `validation/cmu_manifest.json` for generated fetch metadata.
- `validation/data/`, `validation/reports/`, and `validation/rendered/` for extracted
  clips, metrics, reports, and qualitative outputs.

The initial source-of-truth dataset is the official CMU ASF/AMC archive from
`http://mocap.cs.cmu.edu/allasfamc.zip`.

## Commands

Download or verify the CMU archive:

```powershell
uv run python -m validation.scripts.fetch_cmu
```

Fetch CMU trial descriptions and derive clip-level environment labels:

```powershell
uv run python -m validation.scripts.fetch_cmu_metadata
```

This refreshes the committed label file at `validation/cmu_clip_metadata.json`.
Normal cache builds read that committed file and do not query CMU's site.
Edit this file directly for manual label corrections.

Run direct ASF/AMC validation without a cache:

```powershell
uv run python -m validation.scripts.run_baseline_validation --max-clips 20
```

Build the flat canonical Parquet cache:

```powershell
uv run python -m validation.scripts.build_cmu_parquet_cache --cache-dir .cache/cmu/parquet/canonical_full
```

The canonical cache contains the validation joints currently exposed by
`canonical_pose_from_cmu`, including head, torso, shoulders, elbows, hands, hips,
knees, and feet. Rebuild the cache after changing that joint set or clip metadata.
When `validation/cmu_clip_metadata.json` exists, cache rows include `environment_split`
labels: `flat`, `nonflat`, or `unknown`, plus deterministic `activity_class`
labels such as `idle_standing`, `sitting`, `stationary_upper_body`, `locomotion`,
`dance_exercise`, `swimming`, `floor_pose`, `sports`, or `unknown`.

Build the SparsePoser/Xsens BVH cache:

```powershell
uv run python -m validation.scripts.build_sparseposer_parquet_cache --cache-dir .cache/sparseposer/parquet/canonical
```

The SparsePoser cache uses the same canonical wide-Parquet columns and adds
`dataset=sparseposer_xsens` plus `dataset_split=train|eval`. Licensing is
settled: J. L. Ponton (UPC-ViRVIG) confirmed over email (2026-07) that the data
is CC BY 4.0 (no ShareAlike) and open-source weights are an approved use, so the
training split is unblocked — keep the CC BY attribution in the license log. For
a fast smoke test, run:

```powershell
uv run python -m validation.scripts.build_sparseposer_parquet_cache --splits eval --max-clips 1 --max-frames-per-clip 120 --cache-dir .cache/sparseposer/parquet/smoke --force
```

If the official SparsePoser reference code has generated BVH outputs in a loose
folder, convert those outputs and compare them against the canonical reference
cache:

```powershell
uv run python -m validation.scripts.build_sparseposer_parquet_cache --input-dir .cache/references/SparsePoser/python/data --dataset-split eval --filename-prefix-to-strip eval_ --splits eval --cache-dir .cache/sparseposer/parquet/model_xsens_eval --force
uv run python -m validation.scripts.run_pose_cache_comparison --reference-cache-dir .cache/sparseposer/parquet/canonical --prediction-cache-dir .cache/sparseposer/parquet/model_xsens_eval --dataset-split eval --report validation/reports/sparseposer_model_xsens_eval_ours.json --csv-report validation/reports/sparseposer_model_xsens_eval_ours.csv
```

Run cached validation across clips with stateful estimator resets per clip:

```powershell
uv run python -m validation.scripts.run_cached_validation --cache-dir .cache/cmu/parquet/canonical_full --workers 16
```

Cached validation reports include aggregate metrics, group summaries by
`environment_split`, `activity_class`, and their combination, plus worst-clip
lists for the main reference, pelvis, torso, limb, foot, tail, temporal, and
contact-slip metrics. Limb diagnostics include paired shoulder, elbow, and knee
errors plus left/right fields in the clip rows. Tail diagnostics include
per-clip reference-error P50/P90/P95/P99 values and percentages over 25 cm,
50 cm, and 100 cm. Region diagnostics report mean reference error for hands,
upper body, lower body, and full body. Temporal diagnostics use finite
differences over each clip to report mean estimated-vs-reference velocity error,
acceleration error, and absolute estimated jerk for all common joints plus
pelvis, torso, shoulders, elbows, knees, and feet. Gait plausibility diagnostics
add output-side low-foot slip, foot-support distance from hips, stance width,
and hip-to-foot reach thresholds, which are useful for spotting skating, lazy
trailing feet, and overstretched legs even when exact GT footfalls are not the
main objective. Use `--top-clips 0` to omit worst-clip lists from the JSON
summary. The dashboard headline uses full-body, upper-body, lower-body, and
joint-specific reference errors; per-joint means for all emitted joints are
available under `mean_joint_errors_m`.

Compare the first Basis-style estimator port against the trivial baseline:

```powershell
uv run python -m validation.scripts.run_cached_validation --cache-dir .cache/cmu/parquet/canonical_full --workers 16 --estimator basis
```

Experimental estimator names are also available for targeted comparisons:
`basis_knee_hint`, `basis_footstep`, `basis_arm_lookup`, `basis_arm_swivel`,
`basis_shrug`, and `basis_research`. The arm variants are research scaffolding for
paper-inspired elbow/shoulder experiments, not promoted defaults.

Research notes for the sparse-VR papers and implementation ideas live in
`validation/RESEARCH_NOTES.md`.

Run only the in-distribution flat-floor split:

```powershell
uv run python -m validation.scripts.run_cached_validation --cache-dir .cache/cmu/parquet/canonical_full --workers 16 --estimator basis --environment-split flat
```

Combine split filters for narrower slices:

```powershell
uv run python -m validation.scripts.run_cached_validation --cache-dir .cache/cmu/parquet/canonical_full --workers 16 --estimator basis --environment-split flat --activity-class idle_standing sitting stationary_upper_body
```

Render a PNG dashboard for any cached validation JSON report:

```powershell
uv run python -m validation.scripts.plot_cached_validation_report validation/reports/cached_baseline_smoke.json --output validation/reports/cached_baseline_smoke.png
```

Analyze GT gait parameters from the cached CMU flat locomotion split and compare
them to the current procedural footstep assumptions:

```powershell
uv run python -m validation.scripts.analyze_cmu_gait --cache-dir .cache/cmu/parquet/canonical_full --report validation/reports/cmu_gait_flat_locomotion.json --steps-csv validation/reports/cmu_gait_flat_locomotion_steps.csv --plot validation/reports/cmu_gait_flat_locomotion.png
```

The gait report extracts approximate foot-contact events, step forward distance,
step width, step interval, cadence, and swing duration. It also runs lightweight
held-out linear predictors from synthetic head/controller features to estimate
how much of those gait parameters is explained by sparse VR-style inputs.

Build causal sparse-window data for upper-body latent ML experiments:

```powershell
uv run --group ml python -m validation.scripts.build_sparse_window_dataset --cache-dir .cache/cmu/parquet/canonical_full --output-dir .cache/ml/upper_body_windows_dense --stride 1
```

Train the GPU-oriented latent model ladder:

```powershell
uv run --group ml python -m validation.scripts.train_upper_body_latents --dataset-dir .cache/ml/upper_body_windows_dense --output-dir .cache/ml/upper_body_models
```

The sparse-window exporter writes ignored Parquet artifacts under `.cache/ml/`.
By default the CLI parallelizes by clip and writes `windows_shards/*.parquet`;
pass `--workers 1 --shard-rows 0` for the older single-file path. The default
worker count is capped at 2 for `--stride 1` dense exports to avoid queuing too
many large clip row batches in RAM, but the expanded FK schema is still safest
with `--workers 1 --shard-rows 5000` for full-cache regeneration. Use
`--stride 1` for dense temporal training; larger strides are useful for quicker
representation smoke tests. By default, rows use `--tracking-space
calibration_yaw`: positions stay relative to the current HMD/head origin, but
the X/Z axes stay fixed to the clip-initial HMD yaw. Use `--tracking-space
head_yaw` for the older moving-head-yaw convention. Numeric training columns
are stored as Float32 Parquet values; dense windows are still large because each
stride-1 row materializes the full causal window. Rows use causal
HMD/controller windows, static calibration features such as height/shoulder
width/arm lengths, and `upper_body_chain_v1` targets: normalized tracking-space
chest, neck, shoulder anchors, plus elbow pole directions masked when the arm is
near straight. Rows also include local reference and heuristic
chest/neck/shoulder/elbow points plus local heuristic base-pose context for
decoded post-IK losses. When that context is present, Torch training decodes
predicted latents through the differentiable runtime arm/torso chain instead of
the older approximate surrogate. Rows also include auxiliary `upper_body_fk_v1`
targets: tracking-space-local hips position/rotation plus upper-body
parent-relative 6D joint rotations. Use
`--target-mode fk --fk-decoded-pose-weight <weight>` to train those rotation
targets with a differentiable FK point objective against the exported reference
chest/neck/shoulder/elbow points. FK mode also reports decoded position,
velocity, acceleration, and jerk metrics, but it is not wired into the runtime
learned-estimator validation path yet. Decoded temporal losses
penalize velocity, acceleration, and jerk,
and the optional temporal excess loss penalizes output jerk only when it exceeds
reference jerk by a configured ratio. Pole temporal losses compare elbow bend
direction velocity/jerk, and the reach hinge penalizes shoulder-hand distances
near full extension before the two-bone IK singularity. `--yaw-augmentation-degrees`
adds train-only random vertical yaw rotations in the tracking frame, with
validation left unaugmented. Torch runs save the best validation checkpoint by
`--checkpoint-metric` rather than blindly keeping the last epoch, and
`--early-stopping-patience` can stop flat runs. Training reports compare the constant
baseline, Torch linear, Torch MLP, Torch TCN, Torch GRU, and native XGBoost CUDA paths.
Torch CUDA is required by default; pass `--allow-cpu` only for smoke tests.
For bounded dense-cache experiments, `--max-clips` and `--max-rows` filter the
existing Parquet cache without rebuilding it. Use `--stream-training` for
larger runs to avoid materializing train rows in CPU RAM. The streaming path reads Parquet shards
with PyArrow, computes normalization over the selected train rows, and keeps a
bounded validation sample controlled by `--stream-validation-rows`. It applies
latent and decoded per-row losses plus batch-local temporal regularization for
contiguous frames that fit inside the same streamed record batch. Temporal
windows do not cross batch or shard boundaries, so large runs should keep
`--batch-size` comfortably above `--temporal-loss-windows` when temporal losses
are important.

For shoulder/elbow experiments, use decoded losses against the exported
reference points instead of only latent MAE:

```powershell
uv run --group ml python -m validation.scripts.train_upper_body_latents --dataset-dir .cache/ml/upper_body_windows_dense --output-dir .cache/ml/upper_body_models_decoded --models torch_mlp torch_tcn torch_gru --latent-loss-weight 0.1 --decoded-pose-weight 25 --decoded-torso-weight 1 --decoded-shoulder-weight 2 --decoded-elbow-weight 2 --decoded-arm-direction-weight 0.5 --decoded-stretch-weight 0.25 --decoded-reach-hinge-weight 0.25 --reach-ratio-limit 0.97 --temporal-velocity-weight 0.2 --temporal-acceleration-weight 0.1 --temporal-jerk-weight 0.05 --temporal-excess-jerk-weight 0.05 --pole-velocity-weight 0.05 --pole-excess-jerk-weight 0.02 --temporal-loss-windows 512 --yaw-augmentation-degrees 180 --early-stopping-patience 12
```

For larger dense-cache runs, stream train batches and keep validation bounded:

```powershell
uv run --group ml python -m validation.scripts.train_upper_body_latents --dataset-dir .cache/ml/upper_body_windows_dense --output-dir .cache/ml/upper_body_models_stream --models torch_mlp torch_tcn torch_gru --stream-training --stream-validation-rows 65536 --latent-loss-weight 0.1 --decoded-pose-weight 10 --decoded-torso-weight 1 --decoded-shoulder-weight 2 --decoded-elbow-weight 2 --decoded-arm-direction-weight 0.5 --decoded-stretch-weight 0.25 --decoded-reach-hinge-weight 0.25 --reach-ratio-limit 0.97 --yaw-augmentation-degrees 180 --early-stopping-patience 12
```

Torch training supports `--tracking-backend mlflow`, `jsonl`, or `none`.
MLflow is the preferred backend for experiment comparisons because it stores
parameters and epoch metrics in a queryable local run store with an actively
maintained UI:

```powershell
uv run --group ml python -m validation.scripts.train_upper_body_latents --dataset-dir .cache/ml/upper_body_windows_dense --output-dir .cache/ml/upper_body_models_decoded --models torch_mlp torch_tcn torch_gru --tracking-backend mlflow --mlflow-tracking-uri sqlite:///.cache/ml/mlflow.db --mlflow-artifact-root .cache/ml/mlflow_artifacts --mlflow-experiment upper_body_dense
uv run --group ml mlflow ui --backend-store-uri sqlite:///.cache/ml/mlflow.db
```

JSONL remains available as a local fallback and writes one history file next to
each `.pt` artifact, with an initial row plus one row per epoch. Render a quick
local dashboard:

```powershell
uv run python -m validation.scripts.plot_training_history .cache/ml/upper_body_models_decoded/torch_mlp_history.jsonl .cache/ml/upper_body_models_decoded/torch_tcn_history.jsonl
```

Run learned upper-body latents through the full cached CMU metric stack:

```powershell
uv run --group ml python -m validation.scripts.run_learned_upper_body_validation --cache-dir .cache/cmu/parquet/canonical_full --dataset-dir .cache/ml/upper_body_windows_dense --artifact .cache/ml/upper_body_models/torch_mlp.pt --normalization .cache/ml/upper_body_models/normalization.json --compare-baseline
```

Compare Basis and learned upper-body output in the frequency domain before
adding runtime smoothing. Pass `--latent-ema-alpha 0.25` to measure a simple
low-pass ablation against the same frequency bands:

```powershell
uv run --group ml python -m validation.scripts.analyze_estimator_frequency --cache-dir .cache/cmu/parquet/canonical_full --dataset-dir .cache/ml/upper_body_windows_dense --artifact .cache/ml/upper_body_models/torch_mlp.pt --normalization .cache/ml/upper_body_models/normalization.json --report validation/reports/estimator_frequency.json --plot validation/reports/estimator_frequency.png
```

Analyze whether the decoded training loss matches the runtime latent decoder,
and whether oracle GT torso/shoulder/elbow groups are independently useful when
decoded on the heuristic base pose:

```powershell
uv run --group ml python -m validation.scripts.analyze_upper_body_decoder --cache-dir .cache/cmu/parquet/canonical_full --dataset-dir .cache/ml/upper_body_windows_dense --report validation/reports/upper_body_decoder_analysis.json --max-clips 30
```

Check the current sparse-window feature and latent target invariance assumptions:

```powershell
uv run python -m validation.scripts.check_upper_body_invariance
```

Run subject-fold cross-validation, optionally with downstream cached validation
per fold:

```powershell
uv run --group ml python -m validation.scripts.run_upper_body_latent_cross_validation --dataset-dir .cache/ml/upper_body_windows_dense --output-dir .cache/ml/upper_body_models/cross_validation --validate-cache-dir .cache/cmu/parquet/canonical_full --decoded-reach-hinge-weight 0.25 --pole-velocity-weight 0.05 --pole-excess-jerk-weight 0.02
```

Build a compact CSV for manual outlier review from a cached validation JSON report:

```powershell
uv run python -m validation.scripts.build_outlier_review validation/reports/outliers_basis_flat_current.json --output validation/reports/outlier_review_flat.csv
```

Run GT-only plausibility checks over cached CMU clips:

```powershell
uv run python -m validation.scripts.analyze_cmu_plausibility --cache-dir .cache/cmu/parquet/canonical_full --output validation/reports/cmu_reference_plausibility.json --csv-output validation/reports/cmu_reference_plausibility.csv --environment-split flat --workers 16
```

The plausibility report flags clips with reference-side issues such as head or
hips below the inferred floor, repeated high-speed discontinuities, high
acceleration spikes, and likely out-of-distribution posture. It can be merged
into outlier review and filtered to keep qualitative videos focused on plausible
GT clips:

```powershell
uv run python -m validation.scripts.build_outlier_review validation/reports/experiment_basis_shrug_flat_upper_clean.json --plausibility-report validation/reports/cmu_reference_plausibility_flat_upper_clean.json --exclude-reference-bucket implausible_reference --exclude-reference-bucket needs_review --output validation/reports/outlier_review_basis_shrug_flat_upper_plausible.csv
```

Summarize a live VR capture session and, for new captures that include calibrated
optional tracker rows, compare tracked elbows/knees/feet/pelvis against the
estimated joints:

```powershell
uv run python -m validation.scripts.analyze_live_capture .cache/live_capture
```

The live recorder keeps running in the background. The Godot overlay sends a
`flush` message when a clip completes, so each finished clip is persisted without
stopping the Python sidecar. The analyzer also reports calibrated tracker role
coverage; a `chest` role in the default seven-tracker setup usually means the
T-pose calibration was captured too early or in the wrong pose.

## Tracker Noise Capture Protocol

PLAN.md milestone 1 fits the synthetic corruption model in
`vr_pose_validation/synthetic_trackers.py` to real device logs instead of the
invented priors. Record a short labeled session with the live overlay (or Brekel
OpenVR Recorder exported into the same `devices.parquet` schema), then fit it.

Capture one clip per condition, wearing the same tracker layout you want to
model (HMD, both controllers, and any strap trackers). Keep the device native
`linear_velocity`/`angular_velocity` channels on — SteamVR poses embed
photon-time prediction overshoot that only the native velocity exposes.

### In-VR capture director (opt-in)

The live overlay ships an opt-in capture director that walks you through the
protocol runs from inside the headset, so you don't have to hand-time each clip.
Enable it with the `capture_director_enabled` export or the `--capture-director`
cmdline flag; when it is off, free-form capture and the SparsePoser flow are
unaffected. The director runs an ordered protocol of named runs (idle,
range_of_motion, sit_stand by default — occlusion is intentionally omitted per
the decision below) and shows each run's instruction plus the remaining seconds
in the head-locked prompt during countdown and recording.

After a run's recording finishes, the director enters a `review` gate instead of
going idle: press the **trigger** to accept the take (advance to the next run) or
the **grip** button to redo the same run (re-record without advancing). The
head-locked prompt shows `"<run> done — [trigger] accept · [grip] redo"`. After
the last run it settles into a terminal `done` state.

Every recorded frame and the per-clip flush message are tagged with the current
run's `run_name` and `run_kind`, persisted as string columns on the frame,
device, and joint parquet tables (and in the flush metadata). Legacy captures
without the director leave both tags empty (`""`), and older parquet without the
columns still loads.

- **Idle standing (~60 s)** — stand still, arms relaxed then loosely folded.
  This clip anchors sub-mm jitter and low-frequency strap drift, so hold each
  pose long enough for the drift estimator to see a stationary window.
- **Dance / fast motion (~60 s)** — large fast arm and torso motion, jumps,
  spins. Drives the excursion and prediction-overshoot terms and the
  velocity-domain error that fails first.
- **Sit / stand transitions (~45 s)** — repeated sit down, settle, stand up at
  a normal desk/chair. Captures occlusion around the hips/legs and the
  reacquisition excursion when trackers come back into view.
- **Controllers occluded (deferred — needs an inside-out rig)** — the >19 cm
  dropout mode that drives dropout run-length, the hold-last vs
  invalid-confidence split, and the reacquisition excursion. On a Lighthouse
  setup with rear basestation coverage this cannot be reproduced representatively
  (blanket-occluding the controllers yields dropout with different geometry and
  duration than the headset-camera occlusion the target audience actually hits).
  **Decision 2026-07: skip this as fit data.** Dropout and excursion parameters
  stay on the literature-informed role priors until captured on an inside-out
  (Quest-class) headset. When that capture happens, record it as a dedicated
  `occlusion`-kind run so the fitter attributes dropout only to that run.

Fit a profile from one or more recorded session directories:

```powershell
uv run python -m validation.scripts.fit_tracker_noise .cache/live_capture --output validation/reports/fitted_tracker_noise.json --frame-rate-hz 90
```

The fitter groups device rows by role, estimates dropout rate and run length,
hold-last fraction, jitter, drift *amplitude*, excursion *magnitude*, and
native-vs-finite-difference velocity error, and writes a
`tracker_noise_profile.v1` JSON document. Magnitude-style terms (excursion,
native-velocity, and angular-velocity noise) are reported as the per-axis
Gaussian standard deviation the generator consumes, recovered from the observed
3D-vector magnitudes. It degrades gracefully on sessions missing some roles,
reports per-role sample counts, and flags the parameters it cannot fit without
ground truth, which fall back to the role priors: calibration bias, latency,
prediction-overshoot gain, excursion run-length (`excursion_frames`), and drift
period (`drift_period_seconds`). Load the result with
`noise_profile_io.config_from_fitted_profile_file` to build a
`SyntheticTrackerConfig` for synthetic tracker generation.

The fitter accepts repeatable `--run NAME` and `--run-kind KIND` filters that
restrict which device rows feed the fit against the new `run_name`/`run_kind`
columns. If a filter is given but the column is absent (legacy parquet), the
selection is empty and the fitter says so.

**Run-kind provenance policy.** Each parameter is fit only from the run whose
motion exercises it representatively:

- **Dropout/excursion** — `dropout_probability`, `dropout_mean_run_frames`,
  `dropout_hold_last_probability`, `invalid_confidence`, `excursion_magnitude_m`
  — fit **only** from `run_kind == "occlusion"` rows.
- **Stationary noise** — `position_std_m` (jitter), `drift_amplitude_m`,
  `velocity_std_mps`, `angular_velocity_std_radps` — fit **only** from
  `run_kind == "idle"` rows. Pooling motion runs folds real body motion into
  these terms and inflates the fitted jitter/velocity by an order of magnitude
  (a mostly-motion session reads ~20 mm "jitter" where the idle run alone reads
  the true sub-mm sensor value).

When a role has no rows of the required kind, those specific fields fall back to
the role prior and the summary flags it (`[from priors: no occlusion rows]` /
`[from priors: no idle rows]`). Parquet with **no** `run_kind` column (legacy
captures) preserves the original behavior and fits every field from all rows.

**Caveat — drift.** The idle protocol above ("arms at sides, then loosely
folded") deliberately repositions the arms, so the fitted `drift_amplitude_m`
for the hands captures that real slow motion, not sensor drift. Isolating true
drift needs a "hold completely still" window in the idle run; until then, treat
the fitted hand drift as an overestimate.

**Base stations excluded.** Rows with role `tracking_reference` are dropped from
the fit: they define the tracking origin rather than reporting a tracked pose, so
they are never a corruption-model attachment and their in-stream jitter/drift is
zero by construction.
