#!/usr/bin/env python3
"""Evaluate a saved TrackML classifier checkpoint on the held-out test split."""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch import nn

from tracking_train.data import datasets as data_utils
from tracking_train.metrics.trackml import MetricsCalculator
from tracking_train.models.cla_model import (
    ClaModel,
    TransformerClassifier,
    generate_deltar_cone_mask,
)
from tracking_train.train import compute_losses, load_config, setup_logging
from tracking_train.utils import output as output_utils


def build_model(config, device):
    """Build the classifier architecture described by ``config``."""
    model = ClaModel(
        TransformerClassifier(
            inputfeature_dim=config["model"]["inputfeature_dim"],
            num_classes=config["data"]["num_classes"],
            num_heads=config["model"]["num_heads"],
            embed_dim=config["model"]["embed_dim"],
            num_layers=config["model"]["num_layers"],
            dim_feedforward=config["model"]["dim_feedforward"],
            dropout=config["model"]["dropout"],
            use_flash_attention=config["model"]["use_flash_attention"],
        )
    )
    return model.to(device)


def load_checkpoint(model, checkpoint_path, device):
    """Load model weights from a training checkpoint or raw state dict."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = (
        checkpoint.get("model_state", checkpoint)
        if isinstance(checkpoint, dict)
        else checkpoint
    )
    model.load_state_dict(state_dict)
    return checkpoint


def evaluate_test_split(
    model, test_loader, criterion, device, config, truths_df=None, helper_loader=None
):
    """Run evaluation over only the test split and return JSON-serializable metrics."""
    model.eval()
    metrics = MetricsCalculator(model.num_classes)
    running_loss = 0.0
    num_batches = 0

    test_iter = (
        enumerate(zip(test_loader, helper_loader))
        if helper_loader is not None
        else enumerate(test_loader)
    )

    with torch.no_grad():
        for batch_idx, batch in test_iter:
            if helper_loader is not None:
                batch, helper_batch = batch
            else:
                helper_batch = None
            coords, _, labels, pos_enc, seq_lengths = batch
            coords = coords.to(device)
            labels = labels.to(device)
            pos_enc = pos_enc.to(device)
            seq_lengths = seq_lengths.to(device)

            flex_padding_mask = None
            if config["model"].get("use_flash_attention", False):
                flex_padding_mask = generate_deltar_cone_mask(
                    seq_lengths, pos_enc[:, :, 3], pos_enc[:, :, 4]
                )

            logits, pad_mask = model(
                coords,
                batch_name=f"evaluate_test_{batch_idx}",
                flex_padding_mask=flex_padding_mask,
                seq_lengths=seq_lengths,
                pos_enc=pos_enc,
            )
            loss = compute_losses(criterion, logits, labels, pad_mask)
            loss_value = float(loss.item())
            running_loss += loss_value
            num_batches += 1

            metrics.update(
                logits.reshape(-1, model.num_classes).detach().cpu(),
                labels.reshape(-1).detach().cpu(),
                loss=loss_value,
            )

            if truths_df is not None and helper_batch is not None:
                hit_ids, event_ids = helper_batch
                predictions = logits.argmax(dim=-1).detach().cpu()
                real_mask = (~pad_mask.detach().cpu()) & (labels.detach().cpu() != 0)
                metrics.add_true_score(
                    hit_ids[real_mask],
                    event_ids[real_mask],
                    predictions[real_mask],
                    truths_df,
                )

    if num_batches == 0:
        raise RuntimeError(
            "The held-out test split is empty; no evaluation batches were produced."
        )

    class_trackml_proxy_score = metrics.calculate_trackml_score()
    true_scores = metrics.get_all_true_scores()
    trackml_score = (
        float(np.mean(true_scores)) if true_scores else class_trackml_proxy_score
    )
    summary = {
        "test_loss": running_loss / num_batches,
        "test_accuracy": (
            metrics.calculate_accuracy() if metrics.total_predictions else 0.0
        ),
        "trackml_score": trackml_score,
        "class_trackml_proxy_score": class_trackml_proxy_score,
        "num_test_batches": num_batches,
        "num_test_tokens": metrics.total_predictions,
        "used_truth_trackml_score": bool(true_scores),
    }
    return summary


def save_summary(summary, output_dir):
    """Write evaluation metrics to ``evaluation_summary.json``."""
    summary_path = Path(output_dir) / "evaluation_summary.json"
    with summary_path.open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2, sort_keys=True)
        summary_file.write("\n")
    return summary_path


def main(config_path, checkpoint_path, device_name=None):
    config = load_config(config_path)
    output_dir = output_utils.unique_output_dir(config)
    output_utils.copy_config_to_output(config_path, output_dir)
    setup_logging(config, output_dir)

    device = torch.device(
        device_name or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    logging.info("output_dir: %s", output_dir)
    logging.info("Device: %s", device)
    logging.info("Checkpoint: %s", checkpoint_path)

    model = build_model(config, device)
    checkpoint = load_checkpoint(model, checkpoint_path, device)
    if isinstance(checkpoint, dict) and "epoch" in checkpoint:
        logging.info("Loaded model_state from checkpoint epoch %s", checkpoint["epoch"])

    criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction="none")
    loaders = data_utils.load_dataloader(config, device, mode="eval")
    test_loader = loaders.get("test")
    if test_loader is None:
        raise RuntimeError("No held-out test loader was created for this config.")

    helper_loader = loaders.get("test_helper")
    truths_df = data_utils.load_truths(config)
    summary = evaluate_test_split(
        model, test_loader, criterion, device, config, truths_df, helper_loader
    )
    summary_path = save_summary(summary, output_dir)

    logging.info("Test loss: %.6f", summary["test_loss"])
    logging.info("Test accuracy: %.2f%%", summary["test_accuracy"])
    logging.info("TrackML score: %.6f", summary["trackml_score"])
    logging.info(
        "Class TrackML proxy score: %.2f%%", summary["class_trackml_proxy_score"]
    )
    logging.info("Saved evaluation summary to %s", summary_path)

    print(
        json.dumps(
            {"output_dir": output_dir, "summary_path": str(summary_path), **summary},
            indent=2,
            sort_keys=True,
        )
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a checkpoint on the held-out test split."
    )
    parser.add_argument("config", help="Path to TOML config")
    parser.add_argument("checkpoint", help="Path to a saved checkpoint")
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default=None,
        help="Evaluation device. Defaults to CUDA when available, otherwise CPU.",
    )
    args = parser.parse_args()
    main(args.config, args.checkpoint, args.device)
