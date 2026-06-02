from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd
import toml
import torch
from torch import nn
from torch.utils.data import DataLoader

from tracking_train.data.datasets import ParquetEventDataset, collate_fn
from tracking_train.models.cla_model import ClaModel, TransformerClassifier


def test_parquet_dataset_one_batch_cpu_training_smoke():
    """Exercise the tiny Parquet dataset and one CPU training batch."""
    config = toml.load(REPO_ROOT / "configs" / "tiny_debug.toml")
    data_cfg = config["data"]

    parquet_files = sorted((REPO_ROOT / data_cfg["parquet_dir"]).glob(data_cfg["parquet_glob"]))
    assert parquet_files, "Expected data_examples/*.parquet smoke-test inputs."

    dataset = ParquetEventDataset(
        [str(path) for path in parquet_files],
        feature_columns=data_cfg["feature_columns"],
        label_mode=data_cfg["label_mode"],
        label_column=data_cfg["label_column"],
        max_hits=data_cfg["max_hits"],
        sort_by=data_cfg["sort_by"],
    )
    coords, _, labels, pos_enc = dataset[0]
    expected_labels = pd.read_parquet(
        parquet_files[0], columns=[data_cfg["label_column"]]
    )[data_cfg["label_column"]].iloc[: data_cfg["max_hits"]]
    assert coords.shape == (data_cfg["max_hits"], data_cfg["inputfeature_dim"])
    assert labels.dtype == torch.long
    assert labels.numel() == data_cfg["max_hits"]
    assert labels.tolist() == expected_labels.astype("int64").tolist()
    assert torch.count_nonzero(labels > 0).item() > 0
    assert pos_enc.shape[0] == data_cfg["max_hits"]

    loader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=config["training"]["shuffle"],
        num_workers=data_cfg["dataloader_num_workers"],
        collate_fn=collate_fn,
    )
    batch_coords, _, batch_labels, batch_pos_enc, seq_lengths = next(iter(loader))
    assert batch_coords.device.type == "cpu"
    assert batch_labels.device.type == "cpu"
    assert batch_labels.shape[:2] == batch_coords.shape[:2]
    assert torch.all(seq_lengths == data_cfg["max_hits"])

    model_cfg = config["model"]
    model = ClaModel(
        TransformerClassifier(
            inputfeature_dim=model_cfg["inputfeature_dim"],
            num_classes=data_cfg["num_classes"],
            num_heads=model_cfg["num_heads"],
            embed_dim=model_cfg["embed_dim"],
            num_layers=model_cfg["num_layers"],
            dim_feedforward=model_cfg["dim_feedforward"],
            dropout=model_cfg["dropout"],
            use_flash_attention=model_cfg["use_flash_attention"],
        )
    )
    assert not model_cfg["use_flash_attention"]
    assert next(model.parameters()).device.type == "cpu"

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["scheduler"]["initial_lr"])
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits, pad_mask = model(
        batch_coords,
        batch_name="tiny_debug_cpu",
        flex_padding_mask=None,
        seq_lengths=seq_lengths,
        pos_enc=batch_pos_enc,
    )
    assert logits.shape == (*batch_labels.shape, data_cfg["num_classes"])
    assert pad_mask.shape == batch_labels.shape

    loss = criterion(logits.reshape(-1, logits.size(-1)), batch_labels.reshape(-1))
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
