# Oracle table: `gt` (oracle-v1)

inputs `inputs-v1` seed 20261019; splits `splits-v1`; conventions `conventions-v1`; 60 Hz; global frame (head+hands are inputs); MPJPE over the 11 non-input joints (upper 6, lower 5); root-rel excludes hips; jitter = mean |3rd diff| x fps^3 in 1e2 m/s^3 over all 14 joints, GT in brackets; MPJRE over 13 non-root joints (n/a if the method gives no rotations); bone-dir = angle between predicted and GT parent->child vectors; legacy-upper = pre-sprint 'upper_body' joint set (head, chest, spine, shoulders, elbows, hands).

| set | variant | clips | frames | MPJPE-full cm | upper cm | lower cm | root-rel cm | MPJRE deg | bone-dir deg | jitter (GT) | p95 full cm | hand cm | legacy-upper cm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cmu_test | clean | 118 | 96204 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 2.26 (2.26) | 0.00 | 0.00 | 0.00 |
| cmu_test | corrupted | 118 | 96204 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 2.26 (2.26) | 0.00 | 0.00 | 0.00 |
| xsens_eval | clean | 11 | 126192 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 2.04 (2.04) | 0.00 | 0.00 | 0.00 |
| xsens_eval | corrupted | 11 | 126192 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 2.04 (2.04) | 0.00 | 0.00 | 0.00 |
| xsens_val | clean | 12 | 156879 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 2.22 (2.22) | 0.00 | 0.00 | 0.00 |
| xsens_val | corrupted | 12 | 156879 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 2.22 (2.22) | 0.00 | 0.00 | 0.00 |

## Per-activity (cmu_test, clean)

| activity | clips | frames | full cm | upper cm | lower cm | jitter (GT) |
|---|---|---|---|---|---|---|
| dance_exercise | 6 | 8310 | 0.00 | 0.00 | 0.00 | 1.76 (1.76) |
| floor_pose | 8 | 5417 | 0.00 | 0.00 | 0.00 | 4.34 (4.34) |
| idle_standing | 6 | 11728 | 0.00 | 0.00 | 0.00 | 1.61 (1.61) |
| locomotion | 67 | 61075 | 0.00 | 0.00 | 0.00 | 1.60 (1.60) |
| sitting | 6 | 3452 | 0.00 | 0.00 | 0.00 | 0.93 (0.93) |
| sports | 4 | 660 | 0.00 | 0.00 | 0.00 | 31.50 (31.50) |
| unknown | 21 | 5562 | 0.00 | 0.00 | 0.00 | 7.10 (7.10) |

## Per-activity (cmu_test, corrupted)

| activity | clips | frames | full cm | upper cm | lower cm | jitter (GT) |
|---|---|---|---|---|---|---|
| dance_exercise | 6 | 8310 | 0.00 | 0.00 | 0.00 | 1.76 (1.76) |
| floor_pose | 8 | 5417 | 0.00 | 0.00 | 0.00 | 4.34 (4.34) |
| idle_standing | 6 | 11728 | 0.00 | 0.00 | 0.00 | 1.61 (1.61) |
| locomotion | 67 | 61075 | 0.00 | 0.00 | 0.00 | 1.60 (1.60) |
| sitting | 6 | 3452 | 0.00 | 0.00 | 0.00 | 0.93 (0.93) |
| sports | 4 | 660 | 0.00 | 0.00 | 0.00 | 31.50 (31.50) |
| unknown | 21 | 5562 | 0.00 | 0.00 | 0.00 | 7.10 (7.10) |

Note: official Xsens eval shares subjects ['S01', 'S02', 'S05'] with official train (named clips: unknown subject).

GT discontinuities (a joint moving > 0.5 m in one frame, kept in the data): cmu_test 0, xsens_eval 3, xsens_val 118

wall-clock: cmu_test/clean 0s, cmu_test/corrupted 0s, xsens_eval/clean 1s, xsens_eval/corrupted 1s, xsens_val/clean 0s, xsens_val/corrupted 1s
