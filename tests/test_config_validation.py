from copy import deepcopy
from pathlib import Path
import re
import sys

import pytest
import toml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tracking_train.train import ConfigError, load_config, validate_config


def tiny_debug_config():
    return toml.load(REPO_ROOT / "configs" / "tiny_debug.toml")


def test_tiny_debug_config_is_valid():
    config = load_config(REPO_ROOT / "configs" / "tiny_debug.toml")

    assert config["data"]["parquet_dir"] == "data_examples"
    assert config["data"]["label_mode"] == "class_id"
    assert config["model"]["use_flash_attention"] is False


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda config: config["data"].pop("label_column"),
            "Missing required config key: [data].label_column",
        ),
        (
            lambda config: config["model"].update(embed_dim=15),
            "[model].embed_dim must be divisible by [model].num_heads",
        ),
        (
            lambda config: config["data"].update(inputfeature_dim=2),
            "[data].inputfeature_dim must match len([data].feature_columns)",
        ),
        (
            lambda config: config["training"].update(start_from_scratch=False, checkpoint_path=""),
            "[training].checkpoint_path is required when [training].start_from_scratch is false",
        ),
        (
            lambda config: config["training"]["scheduler"].update(warmup_epochs=1),
            "[training.scheduler].target_lr when [training.scheduler].warmup_epochs is greater than 0",
        ),
    ],
)
def test_config_validation_fails_early_with_clear_messages(mutate, message):
    config = deepcopy(tiny_debug_config())
    mutate(config)

    with pytest.raises(ConfigError, match=re.escape(message)):
        validate_config(config)
