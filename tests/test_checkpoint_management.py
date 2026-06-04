import json
from pathlib import Path
import sys

import toml
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate import main as evaluate_main
from tracking_train.train import main as train_main


def _write_tiny_config(tmp_path, total_epochs=1):
    config = toml.load(REPO_ROOT / "configs" / "tiny_debug.toml")
    config["data"]["parquet_dir"] = str(REPO_ROOT / config["data"]["parquet_dir"])
    config["model"]["attention_backend"] = "standard"
    config["training"]["total_epochs"] = total_epochs
    config["training"]["start_from_scratch"] = True
    config["training"]["checkpoint_path"] = ""
    config["output"]["base_path"] = str(tmp_path / "outputs")
    config["output"]["base_dir"] = str(tmp_path / "outputs")

    config_path = tmp_path / f"tiny_debug_{total_epochs}_epochs.toml"
    with config_path.open("w", encoding="utf-8") as config_file:
        toml.dump(config, config_file)
    return config_path


def _latest_run_dir(base_dir):
    run_dirs = sorted(base_dir.glob("run_*"))
    assert run_dirs
    return run_dirs[-1]


def test_tiny_debug_checkpoint_save_resume_and_evaluate(tmp_path):
    """Train one epoch, resume from last.pt, and evaluate the resumed checkpoint."""
    config_path = _write_tiny_config(tmp_path, total_epochs=1)

    train_main(str(config_path))

    first_run = _latest_run_dir(tmp_path / "outputs")
    checkpoint_dir = first_run / "checkpoints"
    last_checkpoint = checkpoint_dir / "last.pt"
    assert checkpoint_dir.is_dir()
    assert last_checkpoint.exists()
    assert (checkpoint_dir / "best_val_loss.pt").exists()
    assert (checkpoint_dir / "best_val_trackml.pt").exists()

    checkpoint = torch.load(last_checkpoint, map_location="cpu", weights_only=False)
    expected_keys = {
        "epoch",
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "scaler_state_dict",
        "config",
        "best_val_loss",
        "best_val_trackml",
    }
    assert expected_keys.issubset(checkpoint.keys())
    assert checkpoint["epoch"] == 0
    assert checkpoint["best_val_loss"] >= 0.0
    assert checkpoint["best_val_trackml"] is not None

    resume_config_path = _write_tiny_config(tmp_path, total_epochs=2)
    train_main(str(resume_config_path), resume_checkpoint_path=str(last_checkpoint))

    second_run = _latest_run_dir(tmp_path / "outputs")
    resumed_last = second_run / "checkpoints" / "last.pt"
    assert resumed_last.exists()
    resumed_checkpoint = torch.load(resumed_last, map_location="cpu", weights_only=False)
    assert resumed_checkpoint["epoch"] == 1
    assert resumed_checkpoint["best_val_loss"] <= checkpoint["best_val_loss"]
    assert resumed_checkpoint["best_val_trackml"] is not None

    summary = evaluate_main(str(config_path), str(resumed_last), device_name="cpu")
    eval_run = _latest_run_dir(tmp_path / "outputs")
    saved_summary = json.loads((eval_run / "evaluation_summary.json").read_text())
    assert saved_summary == summary
    assert summary["num_test_batches"] == 1
    assert summary["num_test_tokens"] > 0
    assert summary["test_loss"] >= 0.0
