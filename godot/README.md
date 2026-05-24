# Godot Harness

This project hosts the Godot-side sparse VR humanoid pose runtime and debug harness.

The first automated check is parity against a Python-exported fixture:

```powershell
uv run python -m validation.scripts.export_godot_parity_fixture
& 'S:\lib\godot\bin\godot.windows.editor.x86_64.console.exe' --headless --path godot --script res://tests/run_parity_tests.gd
```

The current target is Godot 4.6 compatibility, with local verification using the 4.7 development build in `S:\lib\godot\bin`.

The first CMU replay viewer uses a committed walking fixture generated from the Parquet cache:

```powershell
uv run python -m validation.scripts.export_godot_replay_fixture
& 'S:\lib\godot\bin\godot.windows.editor.x86_64.console.exe' --headless --path godot --script res://tests/run_visualization_tests.gd
New-Item -ItemType Directory -Force -Path .cache\cmu\rendered | Out-Null
$out = (Resolve-Path .cache\cmu\rendered).Path + '\pose_replay_debug.avi'
& 'S:\lib\godot\bin\godot.windows.editor.x86_64.console.exe' --path godot --windowed --resolution 1600x1200 --position 100,100 --scene res://scenes/pose_replay_debug.tscn --write-movie $out --fixed-fps 30 --disable-vsync --quit-after 90
```

## Live Overlay Capture

The first live-capture path keeps visualization in Godot and durable recording in Python:

- Godot scene: `res://scenes/live_overlay_debug.tscn`
- Recorder sidecar: `uv run python -m validation.scripts.run_live_capture_recorder`
- Wire protocol: newline-delimited JSON over localhost, formatted as `godot_vr_humanoid_pose.live_capture.v1`
- Output: `.cache/live_capture/session=<id>/frames.parquet`, `devices.parquet`, and `joints.parquet`
- Clip grouping: repeated calibrations within one Godot run stay in the same
  session directory and are separated by `clip_index` columns in all Parquet
  tables.
- Calibration: look at the floor target until it fills, then use the 3-second
  countdown to stand in a T-pose with trackers visible. The snapshot labels
  anonymous generic trackers by T-pose position and feeds calibrated foot/hip
  trackers into the capture data. The live estimator defaults to 3-point
  HMD/controller input with the procedural footstep planner enabled, so the
  mirror shows the lower-body estimate instead of tracked feet; pass
  `-- --use-calibrated-trackers-for-estimation` to feed calibrated trackers into
  the estimator for debugging.
- Recording: after calibration there is a short neutral-stance settle phase,
  then the scene records a 10-second clip at 30 Hz. The Godot scene sends a
  flush message when the clip completes, so the Python sidecar can stay running
  while completed clips are written to Parquet. Stop the Python recorder with
  Ctrl+C when finished.

For VR overlay mode, make the OpenVR overlay addon available at
`godot/addons/godot-openvr-overlay`. During local development a junction or
symlink to `S:\code\godot-openvr-overlay\project\addons\godot-openvr-overlay`
is enough; that dependency path is ignored by git. If the scene prints
`OpenVR Overlay interface is not available`, open the project once in Godot and
enable the `godot-openvr-overlay` plugin so Godot refreshes its local extension
list.

```powershell
New-Item -ItemType Junction -Path godot\addons\godot-openvr-overlay -Target S:\code\godot-openvr-overlay\project\addons\godot-openvr-overlay
uv run python -m validation.scripts.run_live_capture_recorder
& 'S:\lib\godot\bin\godot.windows.editor.x86_64.console.exe' --path godot --scene res://scenes/live_overlay_debug.tscn
```

Without the OpenVR overlay addon, the scene falls back to synthetic 3-point
input so the renderer and socket stream can still be smoke-tested. Pass
`-- --no-synthetic-live-input` to require live XR poses, or pass
`-- --no-openvr-overlay` for headless synthetic smoke tests where SteamVR should
not be initialized.
