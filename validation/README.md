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
many large clip row batches in RAM; pass a higher `--workers` value only after
watching memory usage. Use `--stride 1` for dense temporal training; larger
strides are useful for quicker representation smoke tests. Rows use causal HMD/controller windows, static
calibration features such as
height/shoulder width/arm lengths, and `upper_body_chain_v1` targets: normalized
head-yaw-relative chest, neck, shoulder anchors, plus elbow pole directions
masked when the arm is near straight. Rows also include local reference and heuristic chest/neck/shoulder/elbow
points plus local heuristic base-pose context for decoded post-IK losses. When
that context is present, Torch training decodes predicted latents through the
differentiable runtime arm/torso chain instead of the older approximate
surrogate. Rows also include auxiliary `upper_body_fk_v1` targets: head-yaw-local
hips position/rotation plus upper-body parent-relative 6D joint rotations. Those
FK targets are not used by the default trainer yet; they are the next
representation path toward a rotation-plus-FK objective. Decoded temporal losses
penalize velocity, acceleration, and jerk,
and the optional temporal excess loss penalizes output jerk only when it exceeds
reference jerk by a configured ratio. Pole temporal losses compare elbow bend
direction velocity/jerk, and the reach hinge penalizes shoulder-hand distances
near full extension before the two-bone IK singularity. Training reports compare the constant
baseline, Torch linear, Torch MLP, Torch TCN, and native XGBoost CUDA paths.
Torch CUDA is required by default; pass `--allow-cpu` only for smoke tests.

For shoulder/elbow experiments, use decoded losses against the exported
reference points instead of only latent MAE:

```powershell
uv run --group ml python -m validation.scripts.train_upper_body_latents --dataset-dir .cache/ml/upper_body_windows_dense --output-dir .cache/ml/upper_body_models_decoded --models torch_mlp torch_tcn --latent-loss-weight 0.1 --decoded-pose-weight 25 --decoded-torso-weight 1 --decoded-shoulder-weight 2 --decoded-elbow-weight 2 --decoded-arm-direction-weight 0.5 --decoded-stretch-weight 0.25 --decoded-reach-hinge-weight 0.25 --reach-ratio-limit 0.97 --temporal-velocity-weight 0.2 --temporal-acceleration-weight 0.1 --temporal-jerk-weight 0.05 --temporal-excess-jerk-weight 0.05 --pole-velocity-weight 0.05 --pole-excess-jerk-weight 0.02 --temporal-loss-windows 512
```

Torch training supports `--tracking-backend aim`, `jsonl`, or `none`. Aim is
the preferred backend for experiment comparisons because it stores parameters
and epoch metrics in a queryable local repo with a UI:

```powershell
uv run --group ml python -m validation.scripts.train_upper_body_latents --dataset-dir .cache/ml/upper_body_windows_dense --output-dir .cache/ml/upper_body_models_decoded --models torch_mlp torch_tcn --tracking-backend aim --aim-repo .cache/ml/aim --aim-experiment upper_body_dense
aim up --repo .cache/ml/aim
```

Native Windows training defaults to `--tracking-backend jsonl` because Aim's
`aimrocks` storage dependency does not currently publish Windows wheels; the
`ml` dependency group installs Aim only on non-Windows platforms. JSONL remains
available as a local fallback and writes one history file next to each `.pt`
artifact, with an initial row plus one row per epoch. Render a quick local
dashboard:

```powershell
uv run python -m validation.scripts.plot_training_history .cache/ml/upper_body_models_decoded/torch_mlp_history.jsonl .cache/ml/upper_body_models_decoded/torch_tcn_history.jsonl
```

Run learned upper-body latents through the full cached CMU metric stack:

```powershell
uv run --group ml python -m validation.scripts.run_learned_upper_body_validation --cache-dir .cache/cmu/parquet/canonical_full --dataset-dir .cache/ml/upper_body_windows_dense --artifact .cache/ml/upper_body_models/torch_mlp.pt --normalization .cache/ml/upper_body_models/normalization.json --compare-baseline
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
