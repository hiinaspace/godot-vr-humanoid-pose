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

