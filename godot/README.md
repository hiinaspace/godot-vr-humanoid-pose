# Godot Harness

This project hosts the Godot-side sparse VR humanoid pose runtime and debug harness.

The Godot live overlay deliberately does not carry a separate estimator port.
Run the Python sidecar while the scene is open; it owns the estimator state,
returns `estimate_response` messages over the same localhost socket used for
capture, and writes completed clips to Parquet.

```powershell
uv run python -m validation.scripts.run_live_capture_recorder --estimator basis_research
& 'S:\lib\godot\bin\godot.windows.editor.x86_64.console.exe' --headless --path godot --script res://tests/run_live_capture_tests.gd
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

The first live-capture path keeps visualization in Godot and estimation plus
durable recording in Python:

- Godot scene: `res://scenes/live_overlay_debug.tscn`
- Estimator/recorder sidecar: `uv run python -m validation.scripts.run_live_capture_recorder --estimator basis_research`
- Wire protocol: bidirectional newline-delimited JSON over localhost, formatted as `godot_vr_humanoid_pose.live_capture.v1`
- Output: `.cache/live_capture/session=<id>/frames.parquet`, `devices.parquet`, and `joints.parquet`
- Clip grouping: repeated calibrations within one Godot run stay in the same
  session directory and are separated by `clip_index` columns in all Parquet
  tables.
- Calibration: look at the floor target until it fills, then use the 3-second
  countdown to stand in a T-pose with trackers visible. The snapshot labels
  anonymous generic trackers by T-pose position and feeds calibrated trackers
  into the capture data. The live estimate defaults to 3-point HMD/controller
  input, so the mirror shows the lower-body estimate instead of tracked feet;
  pass `-- --use-calibrated-trackers-for-estimation` to include calibrated
  trackers in the Python estimator request for debugging.
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
uv run python -m validation.scripts.run_live_capture_recorder --estimator basis_research
& 'S:\lib\godot\bin\godot.windows.editor.x86_64.console.exe' --path godot --scene res://scenes/live_overlay_debug.tscn
```

### SparsePoser Xsens Overlay

The SparsePoser overlay path uses the same Godot scene and socket recorder, but
the Python sidecar bridges to the official SparsePoser runtime in its isolated
reference venv and renders the native 22-joint Xsens skeleton returned by that
runtime. It currently requires the six SparsePoser inputs: pelvis, left foot,
right foot, HMD, left controller, and right controller. Extra calibrated
trackers are still shown as labeled markers and recorded for later analysis.

```powershell
uv run python -m validation.scripts.run_live_capture_recorder --backend sparseposer_xsens --sparseposer-python .cache/references/SparsePoser/python/.venv/Scripts/python.exe --sparseposer-root .cache/references/SparsePoser/python --sparseposer-model .cache/references/SparsePoser/python/models/model_xsens --sparseposer-reference-bvh .cache/references/SparsePoser/SparsePoserUnity/Assets/Data/male_180_94.txt --output-dir .cache/live_capture_sparseposer
& 'S:\lib\godot\bin\godot.windows.editor.x86_64.console.exe' --path godot --scene res://scenes/live_overlay_debug.tscn -- --sparseposer-calibration
```

The calibration flow is two-step when `--sparseposer-calibration` is enabled:
first gaze-trigger the normal T-pose role-detection mode, then press either
controller trigger while holding the T-pose to capture roles immediately. Next,
align yourself with the shown Xsens T-pose skeleton and press either trigger
again. The second snapshot computes tracker-to-Xsens-joint offsets for the six
SparsePoser inputs.

Without the OpenVR overlay addon, the scene falls back to synthetic 3-point
input so the renderer and socket stream can still be smoke-tested. Pass
`-- --no-synthetic-live-input` to require live XR poses, or pass
`-- --no-openvr-overlay` for headless synthetic smoke tests where SteamVR should
not be initialized.
