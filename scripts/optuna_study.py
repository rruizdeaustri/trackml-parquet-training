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
SuggestionPath = tuple[str, ...]
OptunaConfig = dict[str, Any]


DEFAULT_STUDY_SETTINGS: dict[str, Any] = {
    "study_name": "trackml_optuna",
    "n_trials": 2,
    "seed": 12345,
    "direction": "maximize",
    "storage": None,
}


DEFAULT_OPTUNA_CONFIG: OptunaConfig = {
    "fixed": {
        "model": {"attention_backend": "standard"},
    },
    "search": {
        "data": {
            "max_hits": {"type": "default_max_hits"},
        },
        "model": {
            "embed_dim": {"type": "default_embed_dim"},
            "num_heads": {"type": "default_num_heads"},
            "num_layers": {"type": "default_num_layers"},
            "dim_feedforward": {"type": "default_dim_feedforward"},
            "dropout": {"type": "float", "low": 0.0, "high": 0.2},
        },
        "training": {
            "weight_decay": {"type": "log_float", "low": 1e-6, "high": 1e-2},
            "scheduler": {
                "initial_lr": {"type": "log_float", "low": 1e-5, "high": 1e-3},
                "factor": {"type": "float", "low": 0.3, "high": 0.8},
                "patience": {"type": "default_scheduler_patience"},
            },
        },
    },
    "constraints": {
        "embed_dim_divisible_by_num_heads": True,
    },
}


_BUILTIN_SUGGESTION_TYPES = {
    "default_max_hits",
    "default_embed_dim",
    "default_num_heads",
    "default_num_layers",
    "default_dim_feedforward",
    "default_scheduler_patience",
}
_SUGGESTION_TYPES = {"categorical", "int", "float", "log_float", "log-float", *_BUILTIN_SUGGESTION_TYPES}


def _positive_unique(values: list[int]) -> list[int]:
    """Return sorted, unique, positive integer values."""
    return sorted({int(value) for value in values if int(value) > 0})


def _default_study_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("runs") / f"optuna_{timestamp}"


def _resolve_study_dir(path: str | None) -> Path:
    return Path(path) if path else _default_study_dir()


def _get_nested(config: dict[str, Any], path: SuggestionPath, default: Any = None) -> Any:
    node: Any = config
    for part in path:
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def _set_nested(config: dict[str, Any], path: SuggestionPath, value: Any) -> None:
    node = config
    for part in path[:-1]:
        node = node.setdefault(part, {})
    node[path[-1]] = value


def _is_suggestion_spec(value: Any) -> bool:
    return isinstance(value, dict) and isinstance(value.get("type"), str) and value["type"] in _SUGGESTION_TYPES


def _walk_suggestions(node: dict[str, Any], prefix: SuggestionPath = ()) -> list[tuple[SuggestionPath, dict[str, Any]]]:
    suggestions: list[tuple[SuggestionPath, dict[str, Any]]] = []
    for key, value in node.items():
        path = (*prefix, str(key))
        if _is_suggestion_spec(value):
            suggestions.append((path, value))
        elif isinstance(value, dict):
            suggestions.extend(_walk_suggestions(value, path))
        else:
            raise ValueError(
                f"Invalid Optuna search entry at {'.'.join(path)}: expected a suggestion table with a type."
            )
    return suggestions


def _walk_fixed_overrides(node: dict[str, Any], prefix: SuggestionPath = ()) -> list[tuple[SuggestionPath, Any]]:
    overrides: list[tuple[SuggestionPath, Any]] = []
    for key, value in node.items():
        key_path = tuple(str(key).split("."))
        path = (*prefix, *key_path)
        if isinstance(value, dict):
            overrides.extend(_walk_fixed_overrides(value, path))
        else:
            overrides.append((path, value))
    return overrides


def _trial_param_name(path: SuggestionPath, spec: dict[str, Any]) -> str:
    return str(spec.get("name", ".".join(path)))


def _suggest_default(
    trial: optuna.Trial,
    path: SuggestionPath,
    spec: dict[str, Any],
    base_config: dict[str, Any],
    selected: dict[SuggestionPath, Any],
) -> Any:
    suggestion_type = spec["type"]
    name = _trial_param_name(path, spec)

    if suggestion_type == "default_max_hits":
        base_max_hits = _get_nested(base_config, path)
        if base_max_hits is None:
            return None
        choices = _positive_unique([int(base_max_hits), min(int(base_max_hits) * 2, 256)])
        return trial.suggest_categorical(name, choices)

    if suggestion_type == "default_embed_dim":
        base_embed_dim = int(_get_nested(base_config, path))
        choices = _positive_unique([base_embed_dim, min(base_embed_dim * 2, 128)])
        return trial.suggest_categorical(name, choices)

    if suggestion_type == "default_num_heads":
        embed_dim = int(selected.get(("model", "embed_dim"), _get_nested(base_config, ("model", "embed_dim"))))
        base_heads = int(_get_nested(base_config, path))
        candidate_heads = _positive_unique([1, base_heads, 2, 4, 8])
        valid_heads = [heads for heads in candidate_heads if embed_dim % heads == 0 and heads <= embed_dim]
        if not valid_heads:
            raise ValueError(f"No valid num_heads choices divide embed_dim={embed_dim}.")
        return trial.suggest_categorical(name, valid_heads)

    if suggestion_type == "default_num_layers":
        base_layers = int(_get_nested(base_config, path))
        return trial.suggest_int(name, base_layers, min(base_layers + 1, 4))

    if suggestion_type == "default_dim_feedforward":
        base_dim = int(_get_nested(base_config, path))
        choices = _positive_unique([base_dim, min(base_dim * 2, 512)])
        return trial.suggest_categorical(name, choices)

    if suggestion_type == "default_scheduler_patience":
        base_patience = int(_get_nested(base_config, path, 1))
        return trial.suggest_int(name, base_patience, min(base_patience + 2, 5))

    raise ValueError(f"Unknown built-in Optuna suggestion type: {suggestion_type}")


def _filter_num_heads_choices(
    path: SuggestionPath,
    spec: dict[str, Any],
    base_config: dict[str, Any],
    selected: dict[SuggestionPath, Any],
) -> dict[str, Any]:
    if path != ("model", "num_heads") or spec.get("type") != "categorical" or "choices" not in spec:
        return spec

    embed_dim = selected.get(("model", "embed_dim"), _get_nested(base_config, ("model", "embed_dim")))
    if embed_dim is None:
        return spec

    valid_choices = [choice for choice in spec["choices"] if int(embed_dim) % int(choice) == 0 and int(choice) <= int(embed_dim)]
    if not valid_choices:
        raise optuna.TrialPruned(f"No configured num_heads choices divide embed_dim={embed_dim}.")

    filtered = dict(spec)
    filtered["choices"] = valid_choices
    return filtered


def _suggest_from_spec(
    trial: optuna.Trial,
    path: SuggestionPath,
    spec: dict[str, Any],
    base_config: dict[str, Any],
    selected: dict[SuggestionPath, Any],
) -> Any:
    suggestion_type = spec["type"]
    if suggestion_type in _BUILTIN_SUGGESTION_TYPES:
        return _suggest_default(trial, path, spec, base_config, selected)

    spec = _filter_num_heads_choices(path, spec, base_config, selected)
    name = _trial_param_name(path, spec)

    if suggestion_type == "categorical":
        if "choices" not in spec or not isinstance(spec["choices"], list) or not spec["choices"]:
            raise ValueError(f"Categorical suggestion {name} requires a non-empty choices list.")
        return trial.suggest_categorical(name, spec["choices"])

    if suggestion_type == "int":
        kwargs: dict[str, Any] = {}
        if "step" in spec:
            kwargs["step"] = int(spec["step"])
        return trial.suggest_int(name, int(spec["low"]), int(spec["high"]), **kwargs)

    if suggestion_type in {"float", "log_float", "log-float"}:
        kwargs: dict[str, Any] = {"log": suggestion_type in {"log_float", "log-float"}}
        if "step" in spec:
            kwargs["step"] = float(spec["step"])
        return trial.suggest_float(name, float(spec["low"]), float(spec["high"]), **kwargs)

    raise ValueError(f"Unsupported Optuna suggestion type for {name}: {suggestion_type}")


def _assert_compatible(overrides: dict[SuggestionPath, Any], base_config: dict[str, Any], optuna_config: OptunaConfig) -> None:
    constraints = optuna_config.get("constraints", {})
    if constraints.get("embed_dim_divisible_by_num_heads", True):
        embed_dim = int(overrides.get(("model", "embed_dim"), _get_nested(base_config, ("model", "embed_dim"))))
        num_heads = int(overrides.get(("model", "num_heads"), _get_nested(base_config, ("model", "num_heads"))))
        if embed_dim % num_heads != 0:
            raise optuna.TrialPruned(f"Incompatible trial: model.embed_dim={embed_dim} is not divisible by model.num_heads={num_heads}.")


def load_optuna_config(optuna_config_path: str | Path | None) -> OptunaConfig:
    """Load an Optuna search-space config or return the backward-compatible default."""
    if optuna_config_path is None:
        config = copy.deepcopy(DEFAULT_OPTUNA_CONFIG)
    else:
        config = toml.load(Path(optuna_config_path))
    config.setdefault("study", {})
    config.setdefault("fixed", {})
    config.setdefault("search", {})
    config.setdefault("constraints", {})
    config["constraints"].setdefault("embed_dim_divisible_by_num_heads", True)
    return config


def _resolve_study_settings(
    optuna_config: OptunaConfig,
    *,
    n_trials: int | None = None,
    study_name: str | None = None,
    storage: str | None = None,
    seed: int | None = None,
    direction: str | None = None,
) -> dict[str, Any]:
    """Merge defaults, TOML [study] values, and explicit CLI/API overrides."""
    study_config = optuna_config.get("study", {})
    if study_config is None:
        study_config = {}
    if not isinstance(study_config, dict):
        raise ValueError("Optuna config [study] section must be a table.")

    settings = copy.deepcopy(DEFAULT_STUDY_SETTINGS)
    toml_study_name = study_config.get("study_name", study_config.get("name"))
    if toml_study_name is not None:
        settings["study_name"] = str(toml_study_name)
    for key in ("n_trials", "seed", "direction", "storage"):
        if key in study_config:
            settings[key] = study_config[key]

    explicit_overrides = {
        "study_name": study_name,
        "n_trials": n_trials,
        "seed": seed,
        "direction": direction,
        "storage": storage,
    }
    for key, value in explicit_overrides.items():
        if value is not None:
            settings[key] = value

    settings["n_trials"] = int(settings["n_trials"])
    if settings["n_trials"] < 1:
        raise ValueError("Study n_trials must be at least 1.")

    settings["study_name"] = str(settings["study_name"])
    settings["seed"] = None if settings.get("seed") is None else int(settings["seed"])
    settings["storage"] = None if settings.get("storage") in {None, ""} else str(settings["storage"])
    settings["direction"] = str(settings["direction"]).lower()
    if settings["direction"] not in {"maximize", "minimize"}:
        raise ValueError("Study direction must be either 'maximize' or 'minimize'.")
    return settings


def suggest_overrides(
    trial: optuna.Trial,
    base_config: dict[str, Any],
    optuna_config: OptunaConfig | None = None,
) -> dict[SuggestionPath, Any]:
    """Suggest trial overrides from TOML-driven Optuna configuration.

    If ``optuna_config`` is omitted, the historical compact CPU-safe search
    space is used for backward compatibility.
    """
    optuna_config = copy.deepcopy(optuna_config or DEFAULT_OPTUNA_CONFIG)
    overrides: dict[SuggestionPath, Any] = {}

    for path, value in _walk_fixed_overrides(optuna_config.get("fixed", {})):
        overrides[path] = value

    for path, spec in _walk_suggestions(optuna_config.get("search", {})):
        value = _suggest_from_spec(trial, path, spec, base_config, overrides)
        if value is not None:
            overrides[path] = value

    _assert_compatible(overrides, base_config, optuna_config)
    return overrides


def build_trial_config(
    base_config: dict[str, Any],
    trial: optuna.Trial,
    study_dir: Path,
    optuna_config: OptunaConfig | None = None,
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    """Copy the base config and apply Optuna-selected trial overrides."""
    trial_config = copy.deepcopy(base_config)
    trial_dir = study_dir / f"trial_{trial.number:04d}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    overrides = suggest_overrides(trial, base_config, optuna_config)
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


def _save_optuna_config(study_dir: Path, optuna_config: OptunaConfig, study_settings: dict[str, Any]) -> Path:
    """Save the effective Optuna config, including resolved study settings."""
    destination = study_dir / "optuna_config.toml"
    effective_config = copy.deepcopy(optuna_config)
    effective_config["study"] = copy.deepcopy(study_settings)
    with destination.open("w", encoding="utf-8") as config_file:
        toml.dump(effective_config, config_file)
    return destination


def run_study(
    base_config_path: str | Path,
    study_dir: str | Path | None = None,
    n_trials: int | None = None,
    study_name: str | None = None,
    storage: str | None = None,
    seed: int | None = None,
    optuna_config_path: str | Path | None = None,
    direction: str | None = None,
) -> optuna.Study:
    """Run an Optuna study and return the completed study object."""
    base_config_path = Path(base_config_path)
    study_dir = _resolve_study_dir(str(study_dir) if study_dir is not None else None)
    study_dir.mkdir(parents=True, exist_ok=True)

    base_config = toml.load(base_config_path)
    optuna_config = load_optuna_config(optuna_config_path)
    study_settings = _resolve_study_settings(
        optuna_config,
        n_trials=n_trials,
        study_name=study_name,
        storage=storage,
        seed=seed,
        direction=direction,
    )
    saved_optuna_config_path = _save_optuna_config(study_dir, optuna_config, study_settings)

    sampler = optuna.samplers.TPESampler(seed=study_settings["seed"])
    study = optuna.create_study(
        direction=study_settings["direction"],
        study_name=study_settings["study_name"],
        storage=study_settings["storage"],
        load_if_exists=study_settings["storage"] is not None,
        sampler=sampler,
    )

    def objective(trial: optuna.Trial) -> float:
        _, config_path, overrides = build_trial_config(base_config, trial, study_dir, optuna_config)
        trial.set_user_attr("trial_dir", str(study_dir / f"trial_{trial.number:04d}"))
        trial.set_user_attr("config_path", str(config_path))
        trial.set_user_attr("optuna_config_path", str(saved_optuna_config_path))
        trial.set_user_attr("overrides", overrides)

        summary = train_main(str(config_path))
        if summary is None:
            raise RuntimeError("Training did not return a metrics summary.")
        trial.set_user_attr("output_dir", summary.get("output_dir"))
        return objective_from_summary(summary, trial)

    study.optimize(objective, n_trials=study_settings["n_trials"])

    summary_path = study_dir / "study_summary.json"
    with summary_path.open("w", encoding="utf-8") as summary_file:
        json.dump(
            {
                "study_name": study.study_name,
                "direction": study.direction.name,
                "study_settings": study_settings,
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
    parser.add_argument("--n-trials", type=int, default=None, help="Number of Optuna trials to run.")
    parser.add_argument("--study-name", default=None, help="Optuna study name.")
    parser.add_argument(
        "--storage",
        default=None,
        help="Optional Optuna storage URL. Omit for in-memory storage and no generated DB file.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Sampler seed for reproducible studies.")
    parser.add_argument(
        "--direction",
        choices=("maximize", "minimize"),
        default=None,
        help="Optuna study direction.",
    )
    parser.add_argument(
        "--optuna-config",
        default=None,
        help="Optional TOML config that defines fixed trial overrides and the Optuna search space.",
    )
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
        optuna_config_path=args.optuna_config,
        direction=args.direction,
    )


if __name__ == "__main__":
    main()
