from __future__ import annotations

import json

from validation.scripts.plot_training_history import render_history


def test_plot_training_history_writes_png(tmp_path) -> None:
    history_path = tmp_path / "torch_mlp_history.jsonl"
    rows = [
        {
            "epoch": 0,
            "validation_mse": 0.2,
            "validation_decoded_position_mse": 0.1,
            "validation_decoded_velocity_mse": 0.05,
        },
        {
            "epoch": 1,
            "train_loss": 0.4,
            "train_latent_mse": 0.08,
            "validation_mse": 0.1,
            "validation_decoded_position_mse": 0.07,
            "validation_decoded_velocity_mse": 0.03,
            "epoch_seconds": 1.5,
        },
    ]
    history_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    output_path = tmp_path / "history.png"

    render_history([history_path], output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
