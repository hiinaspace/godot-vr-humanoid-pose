# Oracle table: `basis` (oracle-v2; v1 columns unchanged, see oracle_v2.py)

Columns as oracle-v1 plus skate (cm/s, foot speed on GT contact frames), pen/float p95 (cm, foot height vs GT-derived floor on GT contact frames), pops per minute (>20 deg per-frame swing change where GT <5 deg), swivel error (deg, GT-gated). Rotation-output methods are placed with the clip's GT bone lengths.

| set | variant | clips | frames | MPJPE-full cm | upper cm | lower cm | root-rel cm | MPJRE deg | bone-dir deg | jitter (GT) | p95 full cm | hand cm | legacy-upper cm | skate cm/s | pen p95 cm | float p95 cm | pops/min | swivel deg |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cmu_test | clean | 118 | 96204 | 17.65 | 10.86 | 25.81 | 18.11 | n/a | 21.23 | 2.72 (2.26) | 31.62 | 0.00 | 8.15 | 26.18 | 0.00 | 12.42 | 18.97 | 60.0 |
| cmu_test | corrupted | 118 | 96204 | 18.69 | 12.52 | 26.10 | 18.95 | n/a | 23.57 | 5.12 (2.26) | 31.69 | 14.36 | 12.55 | 34.68 | 0.00 | 12.50 | 69.64 | 61.5 |
| xsens_eval | clean | 11 | 126192 | 18.26 | 10.37 | 27.72 | 19.25 | n/a | 22.07 | 3.34 (2.04) | 27.78 | 0.00 | 7.83 | 9.03 | 2.25 | 3.35 | 42.79 | 55.4 |
| xsens_eval | corrupted | 11 | 126192 | 19.39 | 12.16 | 28.07 | 19.64 | n/a | 23.96 | 5.76 (2.04) | 29.34 | 14.40 | 12.31 | 9.62 | 2.25 | 3.54 | 105.44 | 57.2 |
| xsens_val | clean | 12 | 156879 | 15.59 | 7.73 | 25.03 | 17.77 | n/a | 20.50 | 2.14 (2.22) | 30.98 | 0.00 | 6.07 | 7.88 | 1.09 | 2.25 | 29.03 | 41.5 |
| xsens_val | corrupted | 12 | 156879 | 16.74 | 9.72 | 25.16 | 18.25 | n/a | 22.89 | 4.66 (2.22) | 32.11 | 14.36 | 10.68 | 8.44 | 1.09 | 3.55 | 114.69 | 43.8 |

wall-clock: cmu_test/clean 2s, cmu_test/corrupted 2s, xsens_eval/clean 7s, xsens_eval/corrupted 7s, xsens_val/clean 3s, xsens_val/corrupted 4s
