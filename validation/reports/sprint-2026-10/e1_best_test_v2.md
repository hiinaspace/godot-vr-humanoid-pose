# Oracle table: `hmdposer_e1_best` (oracle-v2; v1 columns unchanged, see oracle_v2.py)

Columns as oracle-v1 plus skate (cm/s, foot speed on GT contact frames), pen/float p95 (cm, foot height vs GT-derived floor on GT contact frames), pops per minute (>20 deg per-frame swing change where GT <5 deg), swivel error (deg, GT-gated). Rotation-output methods are placed with the clip's GT bone lengths.

| set | variant | clips | frames | MPJPE-full cm | upper cm | lower cm | root-rel cm | MPJRE deg | bone-dir deg | jitter (GT) | p95 full cm | hand cm | legacy-upper cm | skate cm/s | pen p95 cm | float p95 cm | pops/min | swivel deg |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cmu_test | clean | 118 | 96204 | 8.74 | 5.41 | 12.72 | 8.44 | 15.25 | 10.34 | 3.37 (2.26) | 18.89 | 3.14 | 4.67 | 28.60 | 3.50 | 6.55 | 1.61 | 16.5 |
| cmu_test | corrupted | 118 | 96204 | 11.50 | 8.75 | 14.79 | 10.04 | 17.78 | 13.67 | 6.16 (2.26) | 20.68 | 13.80 | 9.49 | 32.32 | 3.48 | 7.75 | 8.46 | 20.5 |
| xsens_eval | clean | 11 | 126192 | 7.32 | 3.49 | 11.91 | 7.19 | 9.98 | 8.19 | 3.55 (2.04) | 15.30 | 2.16 | 3.05 | 21.63 | 0.00 | 8.18 | 4.91 | 15.5 |
| xsens_eval | corrupted | 11 | 126192 | 11.84 | 8.13 | 16.30 | 9.58 | 13.61 | 12.92 | 6.56 (2.04) | 21.88 | 13.72 | 9.02 | 26.79 | 0.27 | 9.89 | 26.19 | 20.9 |
| xsens_val | clean | 12 | 156879 | 11.72 | 5.41 | 19.29 | 10.08 | 11.38 | 10.89 | 2.62 (2.22) | 20.33 | 5.13 | 5.22 | 36.17 | 4.48 | 1.24 | 4.11 | 13.1 |
| xsens_val | corrupted | 12 | 156879 | 13.28 | 8.06 | 19.54 | 11.24 | 14.61 | 15.31 | 5.34 (2.22) | 22.84 | 14.08 | 9.11 | 41.26 | 6.10 | 1.94 | 39.58 | 19.0 |

wall-clock: cmu_test/clean 29s, cmu_test/corrupted 30s, xsens_eval/clean 29s, xsens_eval/corrupted 30s, xsens_val/clean 36s, xsens_val/corrupted 37s
