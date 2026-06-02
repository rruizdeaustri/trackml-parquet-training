import json
from pathlib import Path
import sys

import toml
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate import build_model, main as evaluate_main


def test_evaluate_checkpoint_on_tiny_debug_cpu(tmp_path):
    """Evaluate a tiny non-flash checkpoint on only the held-out Parquet test split."""
    config = toml.load(REPO_ROOT / "configs" / "tiny_debug.toml")
    config["data"]["parquet_dir"] = str(REPO_ROOT / config["data"]["parquet_dir"])
    config["model"]["use_flash_attention"] = False
    config["output"]["base_path"] = str(tmp_path / "outputs")
    config["output"]["base_dir"] = str(tmp_path / "outputs")

    config_path = tmp_path / "tiny_debug.toml"
    with config_path.open("w", encoding="utf-8") as config_file:
        toml.dump(config, config_file)

    model = build_model(config, torch.device("cpu"))
    checkpoint_path = tmp_path / "checkpoint.pth"
    torch.save({"model_state": model.state_dict(), "epoch": 0}, checkpoint_path)

    summary = evaluate_main(str(config_path), str(checkpoint_path), device_name="cpu")

    run_dirs = sorted((tmp_path / "outputs").glob("run_*"))
    assert len(run_dirs) == 1
    summary_path = run_dirs[0] / "evaluation_summary.json"
    assert summary_path.exists()
    saved_summary = json.loads(summary_path.read_text())

    assert saved_summary == summary
    assert summary["num_test_batches"] == 1
    assert summary["num_test_tokens"] > 0
    assert summary["test_loss"] >= 0.0
    assert 0.0 <= summary["test_accuracy"] <= 100.0
    assert 0.0 <= summary["trackml_score"] <= 100.0
    assert 0.0 <= summary["class_trackml_proxy_score"] <= 100.0
    assert summary["used_truth_trackml_score"] is True
