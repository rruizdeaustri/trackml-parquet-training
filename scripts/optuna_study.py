#!/usr/bin/env python3
"""Run an Optuna hyperparameter study for the TrackML training pipeline.

The study trains on the train/validation split only.  Each trial writes a copied
and overridden TOML config into its own run directory and then delegates model
training/checkpointing to ``tracking_train.train.main`` so checkpoint semantics
match normal training runs.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

import optuna
import toml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tracking_train.train import main as train_main


TRACKML_FALLBACK_THRESHOLD = 0.0


def _positive_unique(values: list[int]) -> list[int]:
    """Return sorted, unique, positive integer values."""
    return sorted({int(value) for value in values if int(value) > 0})


def _default_study_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("runs") / f"optuna_{timestamp}"


def _resolve_study_dir(path: str | None) -> Path:
    return Path(path) if path else _default_study_dir()


def _set_nested(config: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    node = config
    for part in path[:-1]:
        node = node.setdefault(part, {})
    node[path[-1]] = value


def suggest_overrides(trial: optuna.Trial, base_config: dict[str, Any]) -> dict[tuple[str, ...], Any]:
    """Suggest a compact, CPU-safe default hyperparameter search space.

    The choices are centered on the base config so ``configs/tiny_debug.toml``
    remains tiny enough for smoke tests, while larger configs can still explore
    common model/training knobs.
    """
    data_cfg = base_config["data"]
    model_cfg = base_config["model"]
    training_cfg = base_config["training"]
    scheduler_cfg = training_cfg["scheduler"]

    base_max_hits = data_cfg.get("max_hits")
    if base_max_hits is None:
        max_hits = None
    else:
        max_hit_choices = _positive_unique([base_max_hits, min(base_max_hits * 2, 256)])
        max_hits = trial.suggest_categorical("max_hits", max_hit_choices)

    base_embed_dim = int(model_cfg["embed_dim"])
    embed_dim_choices = _positive_unique([base_embed_dim, min(base_embed_dim * 2, 128)])
    embed_dim = trial.suggest_categorical("embed_dim", embed_dim_choices)

    candidate_heads = _positive_unique([1, int(model_cfg["num_heads"]), 2, 4, 8])
    valid_heads = [heads for heads in candidate_heads if embed_dim % heads == 0 and heads <= embed_dim]
    num_heads = trial.suggest_categorical("num_heads", valid_heads)

    num_layers = trial.suggest_int(
        "num_layers",
        int(model_cfg["num_layers"]),
        min(int(model_cfg["num_layers"]) + 1, 4),
    )
    dim_feedforward = trial.suggest_categorical(
        "dim_feedforward",
        _positive_unique([
            int(model_cfg["dim_feedforward"]),
            min(int(model_cfg["dim_feedforward"]) * 2, 512),
        ]),
    )
    dropout = trial.suggest_float("dropout", 0.0, 0.2)
    initial_lr = trial.suggest_float("initial_lr", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    scheduler_factor = trial.suggest_float("scheduler_factor", 0.3, 0.8)
    scheduler_patience = trial.suggest_int(
        "scheduler_patience",
        int(scheduler_cfg.get("patience", 1)),
        min(int(scheduler_cfg.get("patience", 1)) + 2, 5),
    )

    overrides: dict[tuple[str, ...], Any] = {
        ("model", "embed_dim"): embed_dim,
        ("model", "num_heads"): num_heads,
        ("model", "num_layers"): num_layers,
        ("model", "dim_feedforward"): dim_feedforward,
        ("model", "dropout"): dropout,
        ("model", "use_flash_attention"): False,
        ("training", "scheduler", "initial_lr"): initial_lr,
        ("training", "scheduler", "factor"): scheduler_factor,
        ("training", "scheduler", "patience"): scheduler_patience,
        ("training", "weight_decay"): weight_decay,
    }
    if max_hits is not None:
        overrides[("data", "max_hits")] = max_hits
    return overrides


def build_trial_config(
    base_config: dict[str, Any],
    trial: optuna.Trial,
    study_dir: Path,
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    """Copy the base config and apply Optuna-selected trial overrides."""
    trial_config = copy.deepcopy(base_config)
    trial_dir = study_dir / f"trial_{trial.number:04d}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    overrides = suggest_overrides(trial, base_config)
    for path, value in overrides.items():
        _set_nested(trial_config, path, value)

    trial_config.setdefault("output", {})["base_path"] = str(study_dir)
    trial_config["output"]["base_dir"] = str(study_dir)
    trial_config["output"]["run_dir"] = str(trial_dir)
    trial_config["output"]["run_name"] = f"trial_{trial.number:04d}"

    trial_config.setdefault("wandb", {})["enabled"] = False
    trial_config.setdefault("evaluation", {})["run_test"] = False
    trial_config.setdefault("training", {})["start_from_scratch"] = True
    trial_config["training"]["checkpoint_path"] = ""

    config_path = trial_dir / "trial_config.toml"
    with config_path.open("w", encoding="utf-8") as config_file:
        toml.dump(trial_config, config_file)

    flat_overrides = {".".join(path): value for path, value in overrides.items()}
    metadata_path = trial_dir / "trial_overrides.json"
    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        json.dump(flat_overrides, metadata_file, indent=2, sort_keys=True)

    return trial_config, config_path, flat_overrides


def objective_from_summary(summary: dict[str, Any], trial: optuna.Trial) -> float:
    """Return a maximize-compatible Optuna objective value.

    Prefer positive validation TrackML score.  If it is absent or zero, return
    negative validation loss so Optuna's maximize direction still minimizes loss.
    """
    val_trackml = summary.get("best_val_trackml")
    if val_trackml is not None and float(val_trackml) > TRACKML_FALLBACK_THRESHOLD:
        trial.set_user_attr("objective_metric", "best_val_trackml")
        trial.set_user_attr("best_val_trackml", float(val_trackml))
        return float(val_trackml)

    val_loss = summary.get("best_val_loss")
    if val_loss is None or not math.isfinite(float(val_loss)):
        raise RuntimeError("Training did not report a finite validation loss for fallback objective.")

    trial.set_user_attr("objective_metric", "negative_best_val_loss")
    trial.set_user_attr("best_val_loss", float(val_loss))
    return -float(val_loss)


def run_study(
    base_config_path: str | Path,
    study_dir: str | Path | None = None,
    n_trials: int = 2,
    study_name: str = "trackml_optuna",
    storage: str | None = None,
    seed: int | None = 12345,
) -> optuna.Study:
    """Run an Optuna study and return the completed study object."""
    base_config_path = Path(base_config_path)
    study_dir = _resolve_study_dir(str(study_dir) if study_dir is not None else None)
    study_dir.mkdir(parents=True, exist_ok=True)

    base_config = toml.load(base_config_path)
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage,
        load_if_exists=storage is not None,
        sampler=sampler,
    )

    def objective(trial: optuna.Trial) -> float:
        _, config_path, overrides = build_trial_config(base_config, trial, study_dir)
        trial.set_user_attr("trial_dir", str(study_dir / f"trial_{trial.number:04d}"))
        trial.set_user_attr("config_path", str(config_path))
        trial.set_user_attr("overrides", overrides)

        summary = train_main(str(config_path))
        if summary is None:
            raise RuntimeError("Training did not return a metrics summary.")
        trial.set_user_attr("output_dir", summary.get("output_dir"))
        return objective_from_summary(summary, trial)

    study.optimize(objective, n_trials=n_trials)

    summary_path = study_dir / "study_summary.json"
    with summary_path.open("w", encoding="utf-8") as summary_file:
        json.dump(
            {
                "study_name": study.study_name,
                "direction": study.direction.name,
                "best_trial_number": study.best_trial.number,
                "best_value": study.best_value,
                "best_params": study.best_params,
                "best_user_attrs": study.best_trial.user_attrs,
            },
            summary_file,
            indent=2,
            sort_keys=True,
        )

    return study


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an Optuna TrackML hyperparameter study.")
    parser.add_argument("base_config", help="Path to the base TOML training config.")
    parser.add_argument(
        "study_dir",
        nargs="?",
        default=None,
        help="Output study directory. Defaults to runs/optuna_<timestamp>/.",
    )
    parser.add_argument("--n-trials", type=int, default=2, help="Number of Optuna trials to run.")
    parser.add_argument("--study-name", default="trackml_optuna", help="Optuna study name.")
    parser.add_argument(
        "--storage",
        default=None,
        help="Optional Optuna storage URL. Omit for in-memory storage and no generated DB file.",
    )
    parser.add_argument("--seed", type=int, default=12345, help="Sampler seed for reproducible studies.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_study(
        base_config_path=args.base_config,
        study_dir=args.study_dir,
        n_trials=args.n_trials,
        study_name=args.study_name,
        storage=args.storage,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
