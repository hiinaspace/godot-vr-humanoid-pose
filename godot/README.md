# Godot Harness

This project hosts the Godot-side sparse VR humanoid pose runtime and debug harness.

The first automated check is parity against a Python-exported fixture:

```powershell
uv run python -m validation.scripts.export_godot_parity_fixture
& 'S:\lib\godot\bin\godot.windows.editor.x86_64.console.exe' --headless --path godot --script res://tests/run_parity_tests.gd
```

The current target is Godot 4.6 compatibility, with local verification using the 4.7 development build in `S:\lib\godot\bin`.
