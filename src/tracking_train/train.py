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
    return config


def setup_logging(config, output_dir):
    """Configure root logger to log to both file and stdout."""
    level = getattr(logging, config["logging"]["level"].upper(), logging.INFO)
    log_file = os.path.join(output_dir, "training.log")
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
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


def setup_training(config, device):
    """Construct model, optimizer, scheduler, and loss; optionally load checkpoint.

    Parameters
    ----------
    config:
        Full configuration dictionary loaded from TOML.
    device:
        Target device.

    Returns
    -------
    model:
        Classification-only model on the given device.
    optimizer:
        AdamW optimizer over model parameters.
    lr_scheduler:
        ReduceLROnPlateau scheduler configured from config["training"]["scheduler"].
    cla_criterion:
        Classification loss.
    start_epoch:
        Epoch index to start from, either 0 or checkpoint_epoch + 1.
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
    optimizer = optim.AdamW(model.parameters(), lr=initial_lr)

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
    if not config["training"]["start_from_scratch"]:
        if (
            "checkpoint_path" not in config["training"]
            or not config["training"]["checkpoint_path"]
        ):
            logging.error(
                "Checkpoint path must be provided when resuming from a checkpoint."
            )
            sys.exit(
                "Error: Checkpoint path not provided but required for resuming training."
            )
        elif not os.path.exists(config["training"]["checkpoint_path"]):
            logging.error(
                f"Checkpoint file not found: {config['training']['checkpoint_path']}"
            )
            sys.exit("Error: Checkpoint file does not exist.")
        else:
            checkpoint = torch.load(config["training"]["checkpoint_path"])
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            lr_scheduler.load_state_dict(checkpoint["scheduler_state"])
            start_epoch = checkpoint["epoch"] + 1
            logging.info("Resuming training from checkpoint.")
    else:
        start_epoch = 0
        if (
            "checkpoint_path" in config["training"]
            and config["training"]["checkpoint_path"]
        ):
            logging.warning(
                "Checkpoint path provided but will not be used since training starts from scratch."
            )

    return model, optimizer, lr_scheduler, cla_criterion, start_epoch


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

        with autocast(device_type="cuda", dtype=torch.bfloat16):
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

        with autocast(device_type="cuda", dtype=torch.bfloat16):
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

    if epoch % 10 == 0:
        epoch_score = metrics_calculator.calculate_trackml_score()
        logging.info(f"Val TrackML score: {epoch_score:.2f}%")
        wandb_logger.log({"val_score": epoch_score, "epoch": epoch})

    return epoch_loss

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

            with autocast(device_type="cuda", dtype=torch.bfloat16):
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


def main(config_path):
    just_eval = False
    config = load_config(config_path)
    output_dir = output_utils.unique_output_dir(config)  # with time stamp
    output_utils.copy_config_to_output(config_path, output_dir)
    setup_logging(config, output_dir)
    wandb_logger = initialize_wandb(config, output_dir)
    logging.info(f"output_dir: {output_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    if not just_eval:
        early_stopper = training_utils.EarlyStopping(
            config["training"]["early_stopping"], output_dir
        )
        scaler = GradScaler()
        model, optimizer, lr_scheduler, cla_criterion, start_epoch = setup_training(
            config, device
        )
    
    loaders = data_utils.load_dataloader(config, device, mode="all")
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

            val_loss = validate_epoch(
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

            # stop training and checkpoint the model if val loss stops improving
            early_stopper(val_loss)
            if early_stopper.should_stop():
                logging.info("Early stopping triggered. Saving checkpoint.")
                wandb_logger.save_model(
                    model,
                    f"model_earlystop_epoch_{epoch}.pth",
                    optimizer,
                    lr_scheduler,
                    epoch,
                    output_dir,
                )
                logging.info("Checkpoint saved to output_dir.")
                break
            # learning rate warm-up
            training_utils.adjust_learning_rate(optimizer, epoch, config)

            if epoch % config["logging"]["model_save_interval"] == 0:
                wandb_logger.save_model(
                    model,
                    f"model_epoch_{epoch}.pth",
                    optimizer,
                    lr_scheduler,
                    epoch,
                    output_dir,
                )

        logging.info("Finished training.")
        wandb_logger.save_model(
            model, "model_final.pth", optimizer, lr_scheduler, epoch, output_dir
        )
        logging.info("Checkpoint saved to output_dir.")
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

    run_test = config.get("evaluation", {}).get("run_test", False)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a model with a given config file."
    )
    parser.add_argument(
        "config_path", type=str, help="Path to the configuration TOML file."
    )

    args = parser.parse_args()
    main(args.config_path)
