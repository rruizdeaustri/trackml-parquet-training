import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import os
import sys
import toml
import logging
import wandb
import argparse
import copy
from tracking_train.models.cla_model import ClaModel, TransformerClassifier, generate_flex_padding_mask, generate_cluster_padding_mask, generate_sliding_window_padding_mask, token_pad_mask_from_seq_lengths, generate_distance_mask, generate_deltar_cone_mask
import tracking_train.metrics.trackml as metrics_calculator
import tracking_train.utils.training as training_utils
import tracking_train.data.datasets as data_utils
import tracking_train.utils.output as output_utils
import tracking_train.utils.wandb_logger as wandb_utils
from torch.amp import autocast, GradScaler

import math
import matplotlib.pyplot as plt
import pandas as pd

# import torch.nn.functional as F

torch.set_float32_matmul_precision('high')
torch.set_default_dtype(torch.float32)

# import random
# random.seed(42)
# torch.manual_seed(42)
# torch.cuda.manual_seed(42)
# np.random.seed(42)



class ConfigError(ValueError):
    """Raised when a TOML training configuration is missing or inconsistent."""


def _toml_path(path):
    if len(path) == 1:
        return f"[{path[0]}]"
    return f"[{'.'.join(path[:-1])}].{path[-1]}"


def _get_config_value(config, path, errors, expected_type=None):
    node = config
    for part in path:
        if not isinstance(node, dict) or part not in node:
            errors.append(f"Missing required config key: {_toml_path(path)}")
            return None
        node = node[part]

    if expected_type is not None and not isinstance(node, expected_type):
        type_name = (
            expected_type.__name__
            if isinstance(expected_type, type)
            else " or ".join(t.__name__ for t in expected_type)
        )
        errors.append(
            f"Invalid config key {_toml_path(path)}: expected {type_name}, "
            f"got {type(node).__name__}"
        )
    return node


def _require_positive_int(config, path, errors):
    value = _get_config_value(config, path, errors, int)
    if value is not None and (isinstance(value, bool) or value <= 0):
        errors.append(f"Invalid config key {_toml_path(path)}: expected a positive integer")
    return value


def _require_nonnegative_int(config, path, errors):
    value = _get_config_value(config, path, errors, int)
    if value is not None and (isinstance(value, bool) or value < 0):
        errors.append(f"Invalid config key {_toml_path(path)}: expected a non-negative integer")
    return value


def _require_positive_number(config, path, errors):
    value = _get_config_value(config, path, errors, (int, float))
    if value is not None and (isinstance(value, bool) or value <= 0):
        errors.append(f"Invalid config key {_toml_path(path)}: expected a positive number")
    return value


def _require_bool(config, path, errors):
    value = _get_config_value(config, path, errors, bool)
    return value


def _require_nonempty_string(config, path, errors):
    value = _get_config_value(config, path, errors, str)
    if value is not None and not value:
        errors.append(f"Invalid config key {_toml_path(path)}: expected a non-empty string")
    return value


def validate_config(config):
    """Validate training config before filesystem setup, data loading, or model creation.

    Raises
    ------
    ConfigError
        If required TOML keys are missing or internally inconsistent.
    """
    errors = []

    for section in ("data", "model", "training", "logging", "output", "wandb"):
        _get_config_value(config, (section,), errors, dict)

    data_format = config.get("data", {}).get("format", "pt")
    if data_format not in {"pt", "parquet"}:
        errors.append(
            "Invalid config key [data].format: expected 'pt' or 'parquet'"
        )

    _require_positive_int(config, ("data", "num_classes"), errors)
    data_input_dim = None

    if data_format == "parquet":
        feature_columns = _get_config_value(config, ("data", "feature_columns"), errors, list)
        data_input_dim = _require_positive_int(config, ("data", "inputfeature_dim"), errors)

        if feature_columns is not None:
            if not feature_columns or not all(isinstance(col, str) and col for col in feature_columns):
                errors.append(
                    "Invalid config key [data].feature_columns: expected a non-empty list of strings"
                )
            elif data_input_dim is not None and len(feature_columns) != data_input_dim:
                errors.append(
                    "Inconsistent config: [data].inputfeature_dim must match "
                    "len([data].feature_columns)"
                )

        _require_nonempty_string(config, ("data", "parquet_dir"), errors)
        _require_nonempty_string(config, ("data", "parquet_glob"), errors)
        label_mode = _require_nonempty_string(config, ("data", "label_mode"), errors)
        if label_mode not in {None, "signal_vs_noise", "high_pt_signal_vs_rest", "class_id"}:
            errors.append(
                "Invalid config key [data].label_mode: expected one of "
                "'signal_vs_noise', 'high_pt_signal_vs_rest', or 'class_id'"
            )
        if label_mode == "class_id":
            _require_nonempty_string(config, ("data", "label_column"), errors)

        max_hits = config.get("data", {}).get("max_hits")
        if max_hits is not None and (not isinstance(max_hits, int) or isinstance(max_hits, bool) or max_hits <= 0):
            errors.append(
                "Invalid config key [data].max_hits: expected null or a positive integer"
            )

        for fraction_key in ("train_fraction", "val_fraction"):
            if fraction_key in config.get("data", {}):
                value = config["data"][fraction_key]
                if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 <= value <= 1:
                    errors.append(
                        f"Invalid config key [data].{fraction_key}: expected a number in [0, 1]"
                    )
        train_fraction = config.get("data", {}).get("train_fraction", 0.8)
        val_fraction = config.get("data", {}).get("val_fraction", 0.1)
        if isinstance(train_fraction, (int, float)) and isinstance(val_fraction, (int, float)):
            if train_fraction + val_fraction > 1:
                errors.append(
                    "Inconsistent config: [data].train_fraction + [data].val_fraction must be <= 1"
                )
    else:
        _require_nonempty_string(config, ("data", "data_dir"), errors)
        _require_nonempty_string(config, ("data", "train_file"), errors)
        _require_nonempty_string(config, ("data", "val_file"), errors)
        _require_nonempty_string(config, ("data", "test_file"), errors)
        _require_nonempty_string(config, ("data", "test_helperfile"), errors)

    dataloader_workers = config.get("data", {}).get("dataloader_num_workers", 0)
    if not isinstance(dataloader_workers, int) or isinstance(dataloader_workers, bool) or dataloader_workers < 0:
        errors.append(
            "Invalid config key [data].dataloader_num_workers: expected a non-negative integer"
        )

    model_input_dim = _require_positive_int(config, ("model", "inputfeature_dim"), errors)
    if data_input_dim is not None and model_input_dim is not None and data_input_dim != model_input_dim:
        errors.append(
            "Inconsistent config: [model].inputfeature_dim must match [data].inputfeature_dim"
        )
    num_heads = _require_positive_int(config, ("model", "num_heads"), errors)
    embed_dim = _require_positive_int(config, ("model", "embed_dim"), errors)
    _require_positive_int(config, ("model", "num_layers"), errors)
    _require_positive_int(config, ("model", "dim_feedforward"), errors)
    dropout = _get_config_value(config, ("model", "dropout"), errors, (int, float))
    _require_bool(config, ("model", "use_flash_attention"), errors)
    if embed_dim is not None and num_heads is not None and embed_dim % num_heads != 0:
        errors.append(
            "Inconsistent config: [model].embed_dim must be divisible by [model].num_heads"
        )
    if dropout is not None and (isinstance(dropout, bool) or not 0 <= dropout < 1):
        errors.append(
            "Invalid config key [model].dropout: expected a number in [0, 1)"
        )

    _require_positive_int(config, ("training", "batch_size"), errors)
    _require_bool(config, ("training", "shuffle"), errors)
    _require_positive_int(config, ("training", "total_epochs"), errors)
    if "weight_decay" in config.get("training", {}):
        weight_decay = _get_config_value(config, ("training", "weight_decay"), errors, (int, float))
        if weight_decay is not None and (isinstance(weight_decay, bool) or weight_decay < 0):
            errors.append(
                "Invalid config key [training].weight_decay: expected a non-negative number"
            )
    start_from_scratch = _require_bool(config, ("training", "start_from_scratch"), errors)
    if start_from_scratch is False and not config.get("training", {}).get("checkpoint_path"):
        errors.append(
            "Inconsistent config: [training].checkpoint_path is required when "
            "[training].start_from_scratch is false"
        )

    _get_config_value(config, ("training", "scheduler"), errors, dict)
    _require_positive_number(config, ("training", "scheduler", "initial_lr"), errors)
    scheduler_mode = _require_nonempty_string(config, ("training", "scheduler", "mode"), errors)
    if scheduler_mode not in {None, "min", "max"}:
        errors.append("Invalid config key [training.scheduler].mode: expected 'min' or 'max'")
    scheduler_factor = _get_config_value(config, ("training", "scheduler", "factor"), errors, (int, float))
    if scheduler_factor is not None and (isinstance(scheduler_factor, bool) or not 0 < scheduler_factor < 1):
        errors.append(
            "Invalid config key [training.scheduler].factor: expected a number in (0, 1)"
        )
    _require_nonnegative_int(config, ("training", "scheduler", "patience"), errors)
    _require_bool(config, ("training", "scheduler", "verbose"), errors)
    warmup_epochs = _require_nonnegative_int(config, ("training", "scheduler", "warmup_epochs"), errors)
    if warmup_epochs and "target_lr" not in config.get("training", {}).get("scheduler", {}):
        errors.append(
            "Missing required config key: [training.scheduler].target_lr when "
            "[training.scheduler].warmup_epochs is greater than 0"
        )
    elif "target_lr" in config.get("training", {}).get("scheduler", {}):
        target_lr = config["training"]["scheduler"]["target_lr"]
        if not isinstance(target_lr, (int, float)) or isinstance(target_lr, bool) or target_lr <= 0:
            errors.append(
                "Invalid config key [training.scheduler].target_lr: expected a positive number"
            )

    _get_config_value(config, ("training", "early_stopping"), errors, dict)
    _require_positive_int(config, ("training", "early_stopping", "patience"), errors)
    _require_bool(config, ("training", "early_stopping", "verbose"), errors)

    _require_nonempty_string(config, ("logging", "level"), errors)
    _require_positive_int(config, ("logging", "epoch_log_interval"), errors)
    _require_positive_int(config, ("logging", "model_save_interval"), errors)
    _require_nonempty_string(config, ("output", "base_path"), errors)
    _require_bool(config, ("wandb", "enabled"), errors)

    if errors:
        raise ConfigError("Invalid training config:\n- " + "\n- ".join(errors))
    return config


class DummyWandbLogger:
    """No-op logger used when WandB is disabled."""

    def __init__(self):
        self.run = self

    def initialize(self):
        pass

    def log(self, *args, **kwargs):
        pass

    def log_gradient_norm(self, *args, **kwargs):
        pass

    def save_model(self, *args, **kwargs):
        pass

    def watch(self, *args, **kwargs):
        pass

    def finish(self):
        pass


def compute_losses(cla_criterion, logits, labels, pad_mask):
    """Compute classification loss over non-padded tokens.

    Parameters
    ----------
    cla_criterion:
        Loss function.
    logits:
        Tensor of shape (B, L, C).
    labels:
        Tensor of shape (B, L) with integer class labels, where -1 or
        padded positions are ignored.
    pad_mask:
        Boolean tensor of shape (B, L) where True indicates padded
        tokens to be excluded from loss averaging.

    Returns
    -------
    torch.Tensor
        Scalar tensor representing the mean loss over non-padded tokens.
    """
    B, L, C = logits.size(0), logits.size(1), logits.size(-1)
    labels_flat = labels.view(-1)
    pad_flat = pad_mask.view(-1)
    invalid = ((labels_flat >= C) | (labels_flat < -1)) & (~pad_flat)
    if invalid.any():
        bad_vals = labels_flat[invalid].unique().tolist()
        raise RuntimeError(
            f"Invalid labels for CrossEntropy: found values {bad_vals} "
            f"outside [-1 or 0..{C-1}] on non-padding tokens. "
            f"Check dataset mapping / num_classes."
        )
    ce_per_tok = cla_criterion(logits.view(-1, C), labels_flat)
    ce_mean = ce_per_tok[~pad_flat].mean()
    return ce_mean

def load_config(config_path):
    """
    Load the TOML configuration file and return a dictionary.
    """
    with open(config_path, "r") as config_file:
        config = toml.load(config_file)
    return validate_config(config)


def setup_logging(config, output_dir):
    """Configure root logger to log to both file and stdout."""
    level = getattr(logging, config["logging"]["level"].upper(), logging.INFO)
    log_file = os.path.join(output_dir, "training.log")
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        force=True,
    )


def checkpoint_dir(output_dir):
    """Return the run-local checkpoint directory, creating it if needed."""
    path = os.path.join(output_dir, "checkpoints")
    os.makedirs(path, exist_ok=True)
    return path


def _checkpoint_state_dict(checkpoint, *names):
    """Return the first matching state-dict entry from a checkpoint."""
    if not isinstance(checkpoint, dict):
        return None
    for name in names:
        state = checkpoint.get(name)
        if state is not None:
            return state
    return None


def load_training_checkpoint(
    checkpoint_path,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    device=None,
):
    """Load a training checkpoint, supporting both new and legacy key names."""
    checkpoint = torch.load(
        checkpoint_path, map_location=device or "cpu", weights_only=False
    )
    model_state = (
        _checkpoint_state_dict(checkpoint, "model_state_dict", "model_state")
        if isinstance(checkpoint, dict)
        else checkpoint
    )
    if model_state is None:
        raise KeyError(
            f"Checkpoint {checkpoint_path!r} does not contain model weights."
        )
    model.load_state_dict(model_state)

    optimizer_state = _checkpoint_state_dict(
        checkpoint, "optimizer_state_dict", "optimizer_state"
    )
    if optimizer is not None and optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    scheduler_state = _checkpoint_state_dict(
        checkpoint, "scheduler_state_dict", "scheduler_state"
    )
    if scheduler is not None and scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)

    scaler_state = _checkpoint_state_dict(
        checkpoint, "scaler_state_dict", "scaler_state"
    )
    if scaler is not None and scaler_state is not None:
        scaler.load_state_dict(scaler_state)

    start_epoch = (
        checkpoint.get("epoch", -1) + 1 if isinstance(checkpoint, dict) else 0
    )
    best_val_loss = (
        checkpoint.get("best_val_loss", math.inf)
        if isinstance(checkpoint, dict)
        else math.inf
    )
    best_val_trackml = (
        checkpoint.get("best_val_trackml", None)
        if isinstance(checkpoint, dict)
        else None
    )
    return checkpoint, start_epoch, best_val_loss, best_val_trackml


def build_checkpoint(
    epoch,
    model,
    optimizer,
    scheduler,
    scaler,
    config,
    best_val_loss,
    best_val_trackml,
):
    """Build a complete, self-contained training checkpoint dictionary."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": copy.deepcopy(config),
        "best_val_loss": best_val_loss,
        "best_val_trackml": best_val_trackml,
    }
    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()
    return checkpoint


def save_checkpoint(
    path,
    epoch,
    model,
    optimizer,
    scheduler,
    scaler,
    config,
    best_val_loss,
    best_val_trackml,
):
    """Save a complete training checkpoint to ``path``."""
    torch.save(
        build_checkpoint(
            epoch,
            model,
            optimizer,
            scheduler,
            scaler,
            config,
            best_val_loss,
            best_val_trackml,
        ),
        path,
    )

def initialize_wandb(config, output_dir):
    """Initialize Weights & Biases logging via WandbLogger, if enabled."""
    wandb_config = config.get("wandb", {})

    if not wandb_config.get("enabled", False):
        return DummyWandbLogger()

    wandb_logger = wandb_utils.WandbLogger(
        config={
            "project_name": wandb_config.get("project", "trackml"),
            "entity": wandb_config.get("entity", ""),
            "run_name": wandb_config.get("name", "tracking_run"),
            "watch_interval": wandb_config.get("watch_interval", 100),
        },
        output_dir=output_dir,
        job_type="training",
    )
    wandb_logger.initialize()
    return wandb_logger


def setup_training(config, device, scaler=None, resume_checkpoint_path=None):
    """Construct model, optimizer, scheduler, loss, and optional resume state.

    Parameters
    ----------
    config:
        Full configuration dictionary loaded from TOML.
    device:
        Target device.
    scaler:
        Optional AMP gradient scaler to restore when checkpoint state is available.
    resume_checkpoint_path:
        Optional CLI-provided checkpoint path. When omitted,
        config["training"]["checkpoint_path"] is used if start_from_scratch is false.

    Returns
    -------
    model, optimizer, lr_scheduler, cla_criterion, start_epoch, best_val_loss, best_val_trackml
        Training components and checkpoint bookkeeping.
    """
    model = ClaModel(
        TransformerClassifier(
            inputfeature_dim=config["model"]["inputfeature_dim"],
            num_classes=config["data"]["num_classes"],
            num_heads=config["model"]["num_heads"],
            embed_dim=config["model"]["embed_dim"],
            num_layers=config["model"]["num_layers"],
            dim_feedforward = config["model"]["dim_feedforward"],
            dropout=config["model"]["dropout"],
            use_flash_attention=config["model"]["use_flash_attention"],
        )
    ).to(device)

    # optimizer
    initial_lr = config["training"]["scheduler"]["initial_lr"]
    weight_decay = config["training"].get("weight_decay", 0.0)
    optimizer = optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=weight_decay)

    # scheduler
    mode = config["training"]["scheduler"]["mode"]
    factor = config["training"]["scheduler"]["factor"]
    patience = config["training"]["scheduler"]["patience"]
    lr_scheduler = ReduceLROnPlateau(
        optimizer, mode=mode, factor=factor, patience=patience
    )

    # criterion
    cla_criterion  = nn.CrossEntropyLoss(ignore_index=-1, reduction="none")

    # check whether to load from checkpoint
    checkpoint_path = resume_checkpoint_path or config["training"].get("checkpoint_path")
    should_resume = bool(checkpoint_path) and not config["training"].get(
        "start_from_scratch", True
    )
    start_epoch = 0
    best_val_loss = math.inf
    best_val_trackml = None

    if should_resume:
        if not os.path.exists(checkpoint_path):
            logging.error("Checkpoint file not found: %s", checkpoint_path)
            sys.exit("Error: Checkpoint file does not exist.")
        _, start_epoch, best_val_loss, best_val_trackml = load_training_checkpoint(
            checkpoint_path,
            model,
            optimizer=optimizer,
            scheduler=lr_scheduler,
            scaler=scaler,
            device=device,
        )
        logging.info(
            "Resuming training from checkpoint %s at epoch %s.",
            checkpoint_path,
            start_epoch,
        )
    elif not config["training"].get("start_from_scratch", True):
        logging.error(
            "Checkpoint path must be provided when resuming from a checkpoint."
        )
        sys.exit(
            "Error: Checkpoint path not provided but required for resuming training."
        )
    elif checkpoint_path:
        logging.warning(
            "Checkpoint path provided but will not be used since training starts from scratch."
        )

    return (
        model,
        optimizer,
        lr_scheduler,
        cla_criterion,
        start_epoch,
        best_val_loss,
        best_val_trackml,
    )


def train_epoch(
    model,
    trainloader,
    optimizer,
    scaler,
    cla_criterion,
    device,
    config,
    epoch,
    metrics_calculator,
    wandb_logger,
    output_dir,
    ):
    """Run a single training epoch over the train loader."""
    model.train()  # Set model to training mode

    running_tot = 0.

    for i, (coords, _, labels, pos_enc, seq_lengths) in enumerate(trainloader):
        optimizer.zero_grad(set_to_none=True)
        coords, labels, seq_lengths, pos_enc = coords.to(device), labels.to(device), seq_lengths.to(device), pos_enc.to(device)

        # flex_padding_mask = generate_flex_padding_mask(seq_lengths)
        # flex_padding_mask = generate_distance_mask(seq_lengths, pos_enc[:,:,1])
        flex_padding_mask = generate_deltar_cone_mask(seq_lengths, pos_enc[:,:,3], pos_enc[:,:,4])

        with autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            logits, pad_mask = model(coords, f'train_{i}', flex_padding_mask, seq_lengths, pos_enc)

            tot_loss = compute_losses(cla_criterion, logits, labels, pad_mask)

        scaler.scale(tot_loss).backward()

        if config["logging"]["level"] == "DEBUG":
            wandb_logger.log_gradient_norm(model)

        scaler.step(optimizer)
        scaler.update()

        
        metrics_calculator.update(
            logits.view(-1, model.num_classes),
            labels.view(-1),
            loss=tot_loss.item()
        )
        
        
        running_tot += tot_loss.item()      

    n_batches   = len(trainloader)
    epoch_loss  = running_tot / n_batches
    epoch_accuracy   = metrics_calculator.calculate_accuracy()

    if epoch % config["logging"]["epoch_log_interval"] == 0:
        logging.info(f"Epoch {epoch+1} | "
                     f"train loss: {epoch_loss:.4f} ")
        logging.info(f"train accuracy: {epoch_accuracy:.2f}%")

    wandb_logger.log(
        {
            "train_loss": epoch_loss,
            "train_accuracy": epoch_accuracy,
            "epoch": epoch,
        }
    )

    if epoch % 10 == 0:
        epoch_score = metrics_calculator.calculate_trackml_score()
        logging.info(f"Training TrackML score: {epoch_score:.2f}%")
        wandb_logger.log({"train_score": epoch_score, "epoch": epoch})

    if epoch == 0:
        training_utils.log_memory_usage()


def validate_epoch(
    model,
    valloader,
    cla_criterion,
    device,
    config,
    epoch,
    metrics_calculator,
    wandb_logger
    ):
    """Run a single validation epoch and return the mean validation loss."""
    model.eval()

    running_tot = 0.

    with torch.no_grad():
      for i, (coords, _, labels, pos_enc, seq_lengths) in enumerate(valloader):
        coords, labels, seq_lengths, pos_enc = coords.to(device), labels.to(device), seq_lengths.to(device), pos_enc.to(device)

        # flex_padding_mask = generate_flex_padding_mask(seq_lengths)
        # flex_padding_mask = generate_distance_mask(seq_lengths, pos_enc[:,:,1])
        flex_padding_mask = generate_deltar_cone_mask(seq_lengths, pos_enc[:,:,3], pos_enc[:,:,4])

        with autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            logits, pad_mask = model(coords, f'validation_{i}', flex_padding_mask, seq_lengths, pos_enc)

            tot_loss = compute_losses(cla_criterion, logits, labels, pad_mask)

        metrics_calculator.update(
            logits.view(-1, model.num_classes),
            labels.view(-1),
            loss=tot_loss.item()
        )
        
        running_tot += tot_loss.item()         

    n_batches = len(valloader)
    epoch_loss = running_tot / n_batches
    epoch_accuracy = metrics_calculator.calculate_accuracy()

    if epoch % config["logging"]["epoch_log_interval"] == 0:
        logging.info(f"Epoch {epoch+1} | "
                     f"val loss: {epoch_loss:.4f} ")
        logging.info(f"val accuracy: {epoch_accuracy:.2f}%")

    wandb_logger.log(
        {
            "val_loss": epoch_loss,
            "val_accuracy": epoch_accuracy,
            "epoch": epoch,
        }
    )

    epoch_score = metrics_calculator.calculate_trackml_score()
    if epoch % 10 == 0:
        logging.info(f"Val TrackML score: {epoch_score:.2f}%")
        wandb_logger.log({"val_score": epoch_score, "epoch": epoch})

    return epoch_loss, epoch_score

def binary_roc_curve(y_true, y_score):
    desc_score_indices = np.argsort(-y_score)
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    tps = np.cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps

    tps = np.r_[0, tps]
    fps = np.r_[0, fps]

    fpr = fps / fps[-1]
    tpr = tps / tps[-1]

    return fpr, tpr


def binary_auc(fpr, tpr):
    return np.trapz(tpr, fpr)

def test(
    model,
    testloader,
    helperloader,
    truths_df,
    device,
    wandb_logger
    ):
    """Evaluate the model on the test set and log accuracy and TrackML scores."""
    model.eval()

    test_metrics_calculator = metrics_calculator.MetricsCalculator(model.num_classes)

    # roc stuff
    # all_probs = []
    # all_labels = []
    
    with torch.no_grad():
        for i, ((coords, _, labels, pos_enc, seq_lengths), (hit_ids, event_ids, _)) in enumerate(zip(testloader, helperloader)):
            coords, labels, seq_lengths, pos_enc = coords.to(device), labels.to(device), seq_lengths.to(device), pos_enc.to(device)

            # flex_padding_mask = generate_flex_padding_mask(seq_lengths)
            # flex_padding_mask = generate_distance_mask(seq_lengths, pos_enc[:,:,1])
            flex_padding_mask = generate_deltar_cone_mask(seq_lengths, pos_enc[:,:,3], pos_enc[:,:,4])

            with autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                logits, _ = model(coords, f'test_{i}', flex_padding_mask, seq_lengths, pos_enc)

            # Build token pad mask (True = padded)
            B, L = labels.size(0), coords.size(1)
            pad_mask = token_pad_mask_from_seq_lengths(seq_lengths, L)

            # Flatten predictions and labels
            outputs = logits.view(-1, model.num_classes)
            flat_logits = outputs.argmax(dim=-1)
            flat_labels = labels.view(-1)
            event_ids_flat = event_ids.view(-1)
            hit_ids_flat = hit_ids.view(-1)

            # roc stuff
            # probs = F.softmax(outputs, dim=-1)
            # positive_probs = probs[:, 1]

            # Keep only non-background and non-padding tokens
            mask_real = (flat_labels != 0) & (~pad_mask.view(-1))
            flat_predicted_real = flat_logits[mask_real]
            event_ids_real = event_ids_flat[mask_real.cpu()]
            hit_ids_real = hit_ids_flat[mask_real.cpu()]

            test_metrics_calculator.update(outputs.detach().cpu(), flat_labels.detach().cpu())
            # test_metrics_calculator.update(outputs[mask_real].detach().cpu(), flat_labels[mask_real].detach().cpu())
            test_metrics_calculator.add_true_score(
                hit_ids_real, event_ids_real, flat_predicted_real, truths_df
            )

            # roc stuff
            # valid_mask = ~pad_mask.view(-1)
            # labels_valid = flat_labels[valid_mask]
            # probs_valid = positive_probs[valid_mask]
            # all_labels.append(labels_valid.float().detach().cpu())
            # all_probs.append(probs_valid.float().detach().cpu())

    accuracy = test_metrics_calculator.calculate_accuracy()
    score = test_metrics_calculator.calculate_trackml_score()
    all_true_scores = test_metrics_calculator.get_all_true_scores()
    true_score = np.mean(all_true_scores) if all_true_scores else 0

    # roc stuff
    # all_probs = torch.cat(all_probs).numpy()
    # all_labels = torch.cat(all_labels).numpy()
    # fpr, tpr = binary_roc_curve(all_labels, all_probs)
    # roc_auc = binary_auc(fpr, tpr)
    # plt.figure(figsize=(8, 6))
    # plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    # plt.plot([0, 1], [0, 1], linestyle="--")
    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive Rate")
    # plt.title("ROC Curve")
    # plt.legend(loc="lower right")
    # wandb_logger.log({
    #     "roc_auc": roc_auc,
    #     "roc_curve": wandb.Image(plt)
    # })
    # plt.close()

    logging.info(f"Test accuracy: {accuracy:.2f}%")
    logging.info(f"Test TrackML score: {score:.2f}%")
    logging.info(f"Test true score: {true_score:.2f}%")
    wandb_logger.log(
        {"test_accuracy": accuracy, "test_score": score, "true_score": true_score}
    )


def main(config_path, resume_checkpoint_path=None):
    just_eval = False
    config = load_config(config_path)
    if resume_checkpoint_path:
        config["training"]["start_from_scratch"] = False
        config["training"]["checkpoint_path"] = resume_checkpoint_path
    output_dir = output_utils.unique_output_dir(config)  # with time stamp
    output_utils.copy_config_to_output(config_path, output_dir)
    setup_logging(config, output_dir)
    wandb_logger = initialize_wandb(config, output_dir)
    checkpoint_output_dir = checkpoint_dir(output_dir)
    logging.info(f"output_dir: {output_dir}")
    logging.info(f"checkpoint_dir: {checkpoint_output_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    if not just_eval:
        early_stopper = training_utils.EarlyStopping(
            config["training"]["early_stopping"], output_dir
        )
        scaler = GradScaler(enabled=device.type == "cuda")
        (
            model,
            optimizer,
            lr_scheduler,
            cla_criterion,
            start_epoch,
            best_val_loss,
            best_val_trackml,
        ) = setup_training(config, device, scaler=scaler)
    
    run_test = config.get("evaluation", {}).get("run_test", False)
    loader_mode = "all" if run_test else "train"
    loaders = data_utils.load_dataloader(config, device, mode=loader_mode)
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    test_loader = loaders.get("test")
    helper_loader = loaders.get("test_helper")

    if not just_eval:
        train_metrics_calculator = metrics_calculator.MetricsCalculator(model.num_classes)
        val_metrics_calculator = metrics_calculator.MetricsCalculator(model.num_classes)

        logging.info("Started training and validation")
        training_utils.log_memory_usage()
        if "watch_interval" in config["wandb"]:
            watch_interval = config["wandb"]["watch_interval"]
            wandb_logger.run.watch(model, log_freq=watch_interval)
            logging.info(f"wandb started watching at interval {watch_interval} ")
        for epoch in range(start_epoch, config["training"]["total_epochs"]):
            # resetting values used for calculating epoch metrics
            train_metrics_calculator.reset()
            val_metrics_calculator.reset()

            train_epoch(
                model,
                train_loader,
                optimizer,
                scaler,
                cla_criterion,
                device,
                config,
                epoch,
                train_metrics_calculator,
                wandb_logger,
                output_dir,
            )

            val_loss, val_trackml = validate_epoch(
                model,
                val_loader,
                cla_criterion,
                device,
                config,
                epoch,
                val_metrics_calculator,
                wandb_logger,
            )
            # adjust learning rate based on validation loss
            lr_scheduler.step(val_loss)
            if config["training"]["scheduler"]["verbose"]:
                current_lr = optimizer.param_groups[0]["lr"]  # get last lr
                logging.info(f"lr: {current_lr}")
                wandb_logger.log({"lr": current_lr})

            val_loss_improved = val_loss < best_val_loss
            val_trackml_improved = val_trackml is not None and (
                best_val_trackml is None or val_trackml > best_val_trackml
            )
            if val_loss_improved:
                best_val_loss = val_loss
            if val_trackml_improved:
                best_val_trackml = val_trackml

            if val_loss_improved:
                save_checkpoint(
                    os.path.join(checkpoint_output_dir, "best_val_loss.pt"),
                    epoch,
                    model,
                    optimizer,
                    lr_scheduler,
                    scaler,
                    config,
                    best_val_loss,
                    best_val_trackml,
                )
                logging.info("Saved new best validation-loss checkpoint.")

            if val_trackml_improved:
                save_checkpoint(
                    os.path.join(checkpoint_output_dir, "best_val_trackml.pt"),
                    epoch,
                    model,
                    optimizer,
                    lr_scheduler,
                    scaler,
                    config,
                    best_val_loss,
                    best_val_trackml,
                )
                logging.info("Saved new best validation-TrackML checkpoint.")

            save_checkpoint(
                os.path.join(checkpoint_output_dir, "last.pt"),
                epoch,
                model,
                optimizer,
                lr_scheduler,
                scaler,
                config,
                best_val_loss,
                best_val_trackml,
            )
            logging.info("Saved last checkpoint for epoch %s.", epoch)

            # stop training if val loss stops improving
            early_stopper(val_loss)
            if early_stopper.should_stop():
                logging.info("Early stopping triggered.")
                break
            # learning rate warm-up
            training_utils.adjust_learning_rate(optimizer, epoch, config)

        logging.info("Finished training.")
    else:
        model = ClaModel(
            TransformerClassifier(
                inputfeature_dim=config["model"]["inputfeature_dim"],
                num_classes=config["data"]["num_classes"],
                num_heads=config["model"]["num_heads"],
                embed_dim=config["model"]["embed_dim"],
                num_layers=config["model"]["num_layers"],
                dim_feedforward = config["model"]["dim_feedforward"],
                dropout=config["model"]["dropout"],
                use_flash_attention=config["model"]["use_flash_attention"],
            )
        ).to(device)
        checkpoint = torch.load(config["model"]["checkpoint_path"])
        model.load_state_dict(checkpoint["model_state"])
        epoch = checkpoint["epoch"] + 1
        logging.info(
            f"Loaded model_state of epoch {epoch}. Ignoring optimizer_state and scheduler_state. Starting evaluation from checkpoint."
        )
        model.eval()

    if run_test and test_loader is not None and helper_loader is not None:
        truths_df = data_utils.load_truths(config)
        if truths_df is None:
            logging.warning("Skipping TrackML test scoring because no test_truthfile is configured.")
        else:
            test(model, test_loader, helper_loader, truths_df, device, wandb_logger)
            logging.info("Finished testing")
    else:
        logging.info("Skipping final test stage. Set [evaluation].run_test = true and provide truth data to enable it.")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total Trainable Parameters: {total_params}")
    wandb_logger.finish()
    return {
        "output_dir": output_dir,
        "checkpoint_dir": checkpoint_output_dir,
        "best_val_loss": best_val_loss if not just_eval else None,
        "best_val_trackml": best_val_trackml if not just_eval else None,
        "total_params": total_params,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a model with a given config file."
    )
    parser.add_argument(
        "config_path", type=str, help="Path to the configuration TOML file."
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to a checkpoint to resume from. Overrides [training].checkpoint_path.",
    )

    args = parser.parse_args()
    main(args.config_path, resume_checkpoint_path=args.resume)
