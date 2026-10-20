# Oracle table: `hmdposer_e1_last` (oracle-v2; v1 columns unchanged, see oracle_v2.py)

Columns as oracle-v1 plus skate (cm/s, foot speed on GT contact frames), pen/float p95 (cm, foot height vs GT-derived floor on GT contact frames), pops per minute (>20 deg per-frame swing change where GT <5 deg), swivel error (deg, GT-gated). Rotation-output methods are placed with the clip's GT bone lengths.

| set | variant | clips | frames | MPJPE-full cm | upper cm | lower cm | root-rel cm | MPJRE deg | bone-dir deg | jitter (GT) | p95 full cm | hand cm | legacy-upper cm | skate cm/s | pen p95 cm | float p95 cm | pops/min | swivel deg |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cmu_val | clean | 319 | 184488 | 11.85 | 5.35 | 19.65 | 12.38 | 16.28 | 12.84 | 14.67 (8.50) | 38.85 | 3.11 | 4.62 | 37.04 | 8.54 | 5.86 | 11.55 | 15.8 |
| cmu_val | corrupted | 319 | 184488 | 14.80 | 8.99 | 21.79 | 14.25 | 19.72 | 16.77 | 18.19 (8.50) | 40.65 | 14.23 | 9.76 | 44.05 | 8.09 | 6.20 | 27.81 | 19.8 |
| xsens_val | clean | 12 | 156879 | 12.67 | 6.92 | 19.57 | 9.63 | 10.96 | 10.47 | 3.28 (2.22) | 20.81 | 6.78 | 6.79 | 40.31 | 5.48 | 0.00 | 7.89 | 12.9 |
| xsens_val | corrupted | 12 | 156879 | 14.32 | 9.32 | 20.32 | 11.37 | 14.89 | 15.62 | 6.64 (2.22) | 23.68 | 14.66 | 10.23 | 44.27 | 6.92 | 1.32 | 52.62 | 20.8 |

wall-clock: cmu_val/clean 62s, cmu_val/corrupted 65s, xsens_val/clean 36s, xsens_val/corrupted 37s
