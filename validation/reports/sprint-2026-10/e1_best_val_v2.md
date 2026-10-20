# Oracle table: `hmdposer_e1_best` (oracle-v2; v1 columns unchanged, see oracle_v2.py)

Columns as oracle-v1 plus skate (cm/s, foot speed on GT contact frames), pen/float p95 (cm, foot height vs GT-derived floor on GT contact frames), pops per minute (>20 deg per-frame swing change where GT <5 deg), swivel error (deg, GT-gated). Rotation-output methods are placed with the clip's GT bone lengths.

| set | variant | clips | frames | MPJPE-full cm | upper cm | lower cm | root-rel cm | MPJRE deg | bone-dir deg | jitter (GT) | p95 full cm | hand cm | legacy-upper cm | skate cm/s | pen p95 cm | float p95 cm | pops/min | swivel deg |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cmu_val | clean | 319 | 184488 | 11.81 | 5.36 | 19.55 | 12.37 | 17.09 | 13.04 | 12.42 (8.50) | 36.76 | 3.65 | 4.74 | 37.90 | 7.15 | 6.40 | 10.50 | 16.3 |
| cmu_val | corrupted | 319 | 184488 | 14.38 | 8.70 | 21.20 | 13.89 | 19.74 | 16.43 | 15.31 (8.50) | 38.56 | 14.12 | 9.53 | 42.68 | 7.17 | 7.07 | 21.62 | 19.6 |
| xsens_val | clean | 12 | 156879 | 11.72 | 5.41 | 19.29 | 10.08 | 11.38 | 10.89 | 2.62 (2.22) | 20.33 | 5.13 | 5.22 | 36.17 | 4.48 | 1.24 | 4.11 | 13.1 |
| xsens_val | corrupted | 12 | 156879 | 13.28 | 8.06 | 19.54 | 11.24 | 14.61 | 15.31 | 5.34 (2.22) | 22.84 | 14.08 | 9.11 | 41.26 | 6.10 | 1.94 | 39.58 | 19.0 |

wall-clock: cmu_val/clean 63s, cmu_val/corrupted 65s, xsens_val/clean 36s, xsens_val/corrupted 37s
