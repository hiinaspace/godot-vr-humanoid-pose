# Oracle table: `hmdposer_e1_best` (oracle-v2; v1 columns unchanged, see oracle_v2.py)

Columns as oracle-v1 plus skate (cm/s, foot speed on GT contact frames), pen/float p95 (cm, foot height vs GT-derived floor on GT contact frames), pops per minute (>20 deg per-frame swing change where GT <5 deg), swivel error (deg, GT-gated). Rotation-output methods are placed with the clip's GT bone lengths.

| set | variant | clips | frames | MPJPE-full cm | upper cm | lower cm | root-rel cm | MPJRE deg | bone-dir deg | jitter (GT) | p95 full cm | hand cm | legacy-upper cm | skate cm/s | pen p95 cm | float p95 cm | pops/min | swivel deg |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| xsens_train | clean | 6 | 10800 | 4.19 | 2.36 | 6.39 | 4.13 | 7.50 | 5.43 | 3.62 (3.00) | 7.97 | 1.65 | 2.10 | 18.51 | 0.00 | 6.05 | 4.33 | 9.0 |

wall-clock: xsens_train/clean 4s
