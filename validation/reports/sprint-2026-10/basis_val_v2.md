# Oracle table: `basis` (oracle-v2; v1 columns unchanged, see oracle_v2.py)

Columns as oracle-v1 plus skate (cm/s, foot speed on GT contact frames), pen/float p95 (cm, foot height vs GT-derived floor on GT contact frames), pops per minute (>20 deg per-frame swing change where GT <5 deg), swivel error (deg, GT-gated). Rotation-output methods are placed with the clip's GT bone lengths.

| set | variant | clips | frames | MPJPE-full cm | upper cm | lower cm | root-rel cm | MPJRE deg | bone-dir deg | jitter (GT) | p95 full cm | hand cm | legacy-upper cm | skate cm/s | pen p95 cm | float p95 cm | pops/min | swivel deg |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cmu_val | clean | 319 | 184488 | 21.57 | 11.07 | 34.18 | 22.48 | n/a | 25.17 | 8.28 (8.50) | 63.70 | 0.00 | 8.29 | 35.01 | 1.80 | 10.03 | 44.33 | 50.2 |
| cmu_val | corrupted | 319 | 184488 | 22.93 | 13.21 | 34.59 | 23.19 | n/a | 27.73 | 11.27 (8.50) | 64.50 | 14.51 | 13.05 | 37.76 | 1.80 | 10.13 | 103.81 | 52.7 |
| xsens_val | clean | 12 | 156879 | 15.59 | 7.73 | 25.03 | 17.77 | n/a | 20.50 | 2.14 (2.22) | 30.98 | 0.00 | 6.07 | 7.88 | 1.09 | 2.25 | 29.03 | 41.5 |
| xsens_val | corrupted | 12 | 156879 | 16.74 | 9.72 | 25.16 | 18.25 | n/a | 22.89 | 4.66 (2.22) | 32.11 | 14.36 | 10.68 | 8.44 | 1.09 | 3.55 | 114.69 | 43.8 |

wall-clock: cmu_val/clean 4s, cmu_val/corrupted 5s, xsens_val/clean 5s, xsens_val/corrupted 5s
