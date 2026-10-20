# Oracle table: `gt_canon` (oracle-v2; v1 columns unchanged, see oracle_v2.py)

Columns as oracle-v1 plus skate (cm/s, foot speed on GT contact frames), pen/float p95 (cm, foot height vs GT-derived floor on GT contact frames), pops per minute (>20 deg per-frame swing change where GT <5 deg), swivel error (deg, GT-gated). Rotation-output methods are placed with the clip's GT bone lengths.

| set | variant | clips | frames | MPJPE-full cm | upper cm | lower cm | root-rel cm | MPJRE deg | bone-dir deg | jitter (GT) | p95 full cm | hand cm | legacy-upper cm | skate cm/s | pen p95 cm | float p95 cm | pops/min | swivel deg |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cmu_test | clean | 118 | 96204 | 0.52 | 0.04 | 1.11 | 0.58 | 0.00 | 0.00 | 2.19 (2.26) | 1.27 | 0.04 | 0.07 | 3.28 | 2.40 | 4.79 | 0.00 | 0.0 |
| cmu_test | corrupted | 118 | 96204 | 0.52 | 0.04 | 1.11 | 0.58 | 0.00 | 0.00 | 2.19 (2.26) | 1.27 | 0.04 | 0.07 | 3.28 | 2.40 | 4.79 | 0.00 | 0.0 |
| xsens_eval | clean | 11 | 126192 | 0.37 | 0.21 | 0.55 | 0.40 | 0.00 | 0.00 | 2.04 (2.04) | 0.83 | 0.29 | 0.22 | 4.65 | 0.02 | 4.86 | 0.00 | 0.0 |
| xsens_eval | corrupted | 11 | 126192 | 0.37 | 0.21 | 0.55 | 0.40 | 0.00 | 0.00 | 2.04 (2.04) | 0.83 | 0.29 | 0.22 | 4.65 | 0.02 | 4.86 | 0.00 | 0.0 |
| xsens_val | clean | 12 | 156879 | 0.37 | 0.18 | 0.60 | 0.41 | 0.00 | 0.00 | 2.22 (2.22) | 0.79 | 0.26 | 0.19 | 5.44 | 0.47 | 4.78 | 0.00 | 0.0 |
| xsens_val | corrupted | 12 | 156879 | 0.37 | 0.18 | 0.60 | 0.41 | 0.00 | 0.00 | 2.22 (2.22) | 0.79 | 0.26 | 0.19 | 5.44 | 0.47 | 4.78 | 0.00 | 0.0 |

wall-clock: cmu_test/clean 0s, cmu_test/corrupted 0s, xsens_eval/clean 2s, xsens_eval/corrupted 2s, xsens_val/clean 1s, xsens_val/corrupted 1s
