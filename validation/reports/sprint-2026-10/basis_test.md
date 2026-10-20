# Oracle table: `basis` (oracle-v1)

inputs `inputs-v1` seed 20261019; splits `splits-v1`; conventions `conventions-v1`; 60 Hz; global frame (head+hands are inputs); MPJPE over the 11 non-input joints (upper 6, lower 5); root-rel excludes hips; jitter = mean |3rd diff| x fps^3 in 1e2 m/s^3 over all 14 joints, GT in brackets; MPJRE over 13 non-root joints (n/a if the method gives no rotations); bone-dir = angle between predicted and GT parent->child vectors; legacy-upper = pre-sprint 'upper_body' joint set (head, chest, spine, shoulders, elbows, hands).

| set | variant | clips | frames | MPJPE-full cm | upper cm | lower cm | root-rel cm | MPJRE deg | bone-dir deg | jitter (GT) | p95 full cm | hand cm | legacy-upper cm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cmu_test | clean | 118 | 96204 | 17.65 | 10.86 | 25.81 | 18.11 | n/a | 21.23 | 2.72 (2.26) | 31.62 | 0.00 | 8.15 |
| cmu_test | corrupted | 118 | 96204 | 18.69 | 12.52 | 26.10 | 18.95 | n/a | 23.57 | 5.12 (2.26) | 31.69 | 14.36 | 12.55 |
| xsens_eval | clean | 11 | 126192 | 18.26 | 10.37 | 27.72 | 19.25 | n/a | 22.07 | 3.34 (2.04) | 27.78 | 0.00 | 7.83 |
| xsens_eval | corrupted | 11 | 126192 | 19.39 | 12.16 | 28.07 | 19.64 | n/a | 23.96 | 5.76 (2.04) | 29.34 | 14.40 | 12.31 |
| xsens_val | clean | 12 | 156879 | 15.59 | 7.73 | 25.03 | 17.77 | n/a | 20.50 | 2.14 (2.22) | 30.98 | 0.00 | 6.07 |
| xsens_val | corrupted | 12 | 156879 | 16.74 | 9.72 | 25.16 | 18.25 | n/a | 22.89 | 4.66 (2.22) | 32.11 | 14.36 | 10.68 |

## Per-activity (cmu_test, clean)

| activity | clips | frames | full cm | upper cm | lower cm | jitter (GT) |
|---|---|---|---|---|---|---|
| dance_exercise | 6 | 8310 | 16.92 | 12.33 | 22.44 | 2.13 (1.76) |
| floor_pose | 8 | 5417 | 34.67 | 15.80 | 57.31 | 5.13 (4.34) |
| idle_standing | 6 | 11728 | 16.41 | 10.40 | 23.61 | 2.34 (1.61) |
| locomotion | 67 | 61075 | 16.31 | 10.43 | 23.36 | 2.06 (1.60) |
| sitting | 6 | 3452 | 19.39 | 10.17 | 30.46 | 1.47 (0.93) |
| sports | 4 | 660 | 34.07 | 15.39 | 56.47 | 33.21 (31.50) |
| unknown | 21 | 5562 | 16.55 | 9.38 | 25.15 | 6.45 (7.10) |

## Per-activity (cmu_test, corrupted)

| activity | clips | frames | full cm | upper cm | lower cm | jitter (GT) |
|---|---|---|---|---|---|---|
| dance_exercise | 6 | 8310 | 17.45 | 13.87 | 21.75 | 4.03 (1.76) |
| floor_pose | 8 | 5417 | 35.21 | 16.78 | 57.33 | 7.65 (4.34) |
| idle_standing | 6 | 11728 | 17.57 | 12.27 | 23.93 | 4.92 (1.61) |
| locomotion | 67 | 61075 | 17.46 | 12.12 | 23.86 | 4.35 (1.60) |
| sitting | 6 | 3452 | 20.36 | 11.57 | 30.90 | 3.82 (0.93) |
| sports | 4 | 660 | 35.41 | 17.77 | 56.58 | 42.94 (31.50) |
| unknown | 21 | 5562 | 17.32 | 11.18 | 24.69 | 9.56 (7.10) |

Note: official Xsens eval shares subjects ['S01', 'S02', 'S05'] with official train (named clips: unknown subject).

GT discontinuities (a joint moving > 0.5 m in one frame, kept in the data): cmu_test 0, xsens_eval 3, xsens_val 118

wall-clock: cmu_test/clean 1s, cmu_test/corrupted 1s, xsens_eval/clean 6s, xsens_eval/corrupted 6s, xsens_val/clean 3s, xsens_val/corrupted 3s
