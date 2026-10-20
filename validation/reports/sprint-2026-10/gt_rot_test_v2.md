# Oracle table: `gt_rot` (oracle-v2; v1 columns unchanged, see oracle_v2.py)

Columns as oracle-v1 plus skate (cm/s, foot speed on GT contact frames), pen/float p95 (cm, foot height vs GT-derived floor on GT contact frames), pops per minute (>20 deg per-frame swing change where GT <5 deg), swivel error (deg, GT-gated). Rotation-output methods are placed with the clip's GT bone lengths.

| set | variant | clips | frames | MPJPE-full cm | upper cm | lower cm | root-rel cm | MPJRE deg | bone-dir deg | jitter (GT) | p95 full cm | hand cm | legacy-upper cm | skate cm/s | pen p95 cm | float p95 cm | pops/min | swivel deg |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cmu_test | clean | 118 | 96204 | 5.75 | 5.20 | 6.41 | 6.33 | 0.00 | 5.27 | 3.38 (2.26) | 10.42 | 6.06 | 5.44 | 14.85 | 7.30 | 6.93 | 1.53 | 5.0 |
| cmu_test | corrupted | 118 | 96204 | 5.75 | 5.20 | 6.41 | 6.33 | 0.00 | 5.27 | 3.38 (2.26) | 10.42 | 6.06 | 5.44 | 14.85 | 7.30 | 6.93 | 1.53 | 5.0 |
| xsens_eval | clean | 11 | 126192 | 5.64 | 2.82 | 9.03 | 6.21 | 0.00 | 3.35 | 1.92 (2.04) | 10.84 | 3.58 | 2.99 | 7.28 | 3.67 | 7.68 | 0.00 | 4.2 |
| xsens_eval | corrupted | 11 | 126192 | 5.64 | 2.82 | 9.03 | 6.21 | 0.00 | 3.35 | 1.92 (2.04) | 10.84 | 3.58 | 2.99 | 7.28 | 3.67 | 7.68 | 0.00 | 4.2 |
| xsens_val | clean | 12 | 156879 | 6.85 | 3.50 | 10.87 | 7.53 | 0.00 | 3.79 | 2.04 (2.22) | 18.17 | 4.52 | 3.71 | 7.96 | 0.00 | 9.48 | 0.00 | 4.7 |
| xsens_val | corrupted | 12 | 156879 | 6.85 | 3.50 | 10.87 | 7.53 | 0.00 | 3.79 | 2.04 (2.22) | 18.17 | 4.52 | 3.71 | 7.96 | 0.00 | 9.48 | 0.00 | 4.7 |

wall-clock: cmu_test/clean 0s, cmu_test/corrupted 0s, xsens_eval/clean 1s, xsens_eval/corrupted 1s, xsens_val/clean 1s, xsens_val/corrupted 1s
