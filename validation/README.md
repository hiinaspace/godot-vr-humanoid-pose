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

Run direct ASF/AMC validation without a cache:

```powershell
uv run python -m validation.scripts.run_baseline_validation --max-clips 20
```

Build the flat canonical Parquet cache:

```powershell
uv run python -m validation.scripts.build_cmu_parquet_cache --cache-dir .cache/cmu/parquet/canonical_full
```

Run cached validation across clips with stateful estimator resets per clip:

```powershell
uv run python -m validation.scripts.run_cached_validation --cache-dir .cache/cmu/parquet/canonical_full --workers 16
```
