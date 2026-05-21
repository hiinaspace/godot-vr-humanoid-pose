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
`dance_exercise`, `floor_pose`, `sports`, or `unknown`.

Run cached validation across clips with stateful estimator resets per clip:

```powershell
uv run python -m validation.scripts.run_cached_validation --cache-dir .cache/cmu/parquet/canonical_full --workers 16
```

Compare the first Basis-style estimator port against the trivial baseline:

```powershell
uv run python -m validation.scripts.run_cached_validation --cache-dir .cache/cmu/parquet/canonical_full --workers 16 --estimator basis
```

Run only the in-distribution flat-floor split:

```powershell
uv run python -m validation.scripts.run_cached_validation --cache-dir .cache/cmu/parquet/canonical_full --workers 16 --estimator basis --environment-split flat
```

Combine split filters for narrower slices:

```powershell
uv run python -m validation.scripts.run_cached_validation --cache-dir .cache/cmu/parquet/canonical_full --workers 16 --estimator basis --environment-split flat --activity-class idle_standing sitting stationary_upper_body
```
