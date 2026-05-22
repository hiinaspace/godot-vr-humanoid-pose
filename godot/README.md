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
