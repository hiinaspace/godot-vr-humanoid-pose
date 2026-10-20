# Oracle table: `gt` (oracle-v2; v1 columns unchanged, see oracle_v2.py)

Columns as oracle-v1 plus skate (cm/s, foot speed on GT contact frames), pen/float p95 (cm, foot height vs GT-derived floor on GT contact frames), pops per minute (>20 deg per-frame swing change where GT <5 deg), swivel error (deg, GT-gated). Rotation-output methods are placed with the clip's GT bone lengths.

| set | variant | clips | frames | MPJPE-full cm | upper cm | lower cm | root-rel cm | MPJRE deg | bone-dir deg | jitter (GT) | p95 full cm | hand cm | legacy-upper cm | skate cm/s | pen p95 cm | float p95 cm | pops/min | swivel deg |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cmu_test | clean | 6 | 3155 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 2.94 (2.94) | 0.00 | 0.00 | 0.00 | 2.48 | 0.00 | 3.85 | 0.00 | 0.0 |
| cmu_test | corrupted | 6 | 3155 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 2.94 (2.94) | 0.00 | 0.00 | 0.00 | 2.48 | 0.00 | 3.85 | 0.00 | 0.0 |
| xsens_eval | clean | 6 | 10800 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.79 (1.79) | 0.00 | 0.00 | 0.00 | 4.19 | 0.00 | 2.16 | 0.00 | 0.0 |
| xsens_eval | corrupted | 6 | 10800 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.79 (1.79) | 0.00 | 0.00 | 0.00 | 4.19 | 0.00 | 2.16 | 0.00 | 0.0 |
| xsens_val | clean | 6 | 10800 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 2.58 (2.58) | 0.00 | 0.00 | 0.00 | 5.65 | 0.00 | 2.73 | 0.00 | 0.0 |
| xsens_val | corrupted | 6 | 10800 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 2.58 (2.58) | 0.00 | 0.00 | 0.00 | 5.65 | 0.00 | 2.73 | 0.00 | 0.0 |

wall-clock: cmu_test/clean 0s, cmu_test/corrupted 0s, xsens_eval/clean 0s, xsens_eval/corrupted 0s, xsens_val/clean 0s, xsens_val/corrupted 0s
