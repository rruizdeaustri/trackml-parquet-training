import glob
import logging
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Compatibility for old torch pickles created with newer numpy layouts.
import sys
sys.modules['numpy._core'] = np.core
sys.modules['numpy._core.multiarray'] = np.core.multiarray

PAD_TOKEN = 0


def _cfg_get(section, key, default=None):
    return section[key] if key in section else default


def _split_files(files, train_fraction=0.8, val_fraction=0.1, seed=12345):
    files = list(sorted(files))
    rng = random.Random(seed)
    rng.shuffle(files)
    n = len(files)
    n_train = int(n * train_fraction)
    n_val = int(n * val_fraction)
    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]
    if not val_files and len(train_files) > 1:
        val_files = [train_files.pop()]
    return train_files, val_files, test_files


def load_dataloader(config, device=None, mode="all"):
    """Create train/val/test dataloaders.

    Supported data formats:
      - data.format = "pt": original tensor files.
      - data.format = "parquet": one Parquet file per full TrackML event/batch.
    """
    data_cfg = config["data"]
    fmt = _cfg_get(data_cfg, "format", "pt")
    batch_size = config["training"]["batch_size"]
    num_workers = _cfg_get(data_cfg, "dataloader_num_workers", 0)
    loaders = {}

    if fmt == "parquet":
        parquet_dir = data_cfg["parquet_dir"]
        pattern = _cfg_get(data_cfg, "parquet_glob", "*.parquet")
        files = sorted(glob.glob(os.path.join(parquet_dir, pattern)))
        if not files:
            raise FileNotFoundError(f"No Parquet files found in {parquet_dir!r} with pattern {pattern!r}")

        train_files, val_files, test_files = _split_files(
            files,
            train_fraction=_cfg_get(data_cfg, "train_fraction", 0.8),
            val_fraction=_cfg_get(data_cfg, "val_fraction", 0.1),
            seed=_cfg_get(data_cfg, "split_seed", 12345),
        )
        logging.info(
            "Parquet split: %d train, %d val, %d test files",
            len(train_files), len(val_files), len(test_files)
        )

        common_kwargs = {
            "feature_columns": data_cfg.get("feature_columns", ["x", "y", "z"]),
            "label_mode": data_cfg.get("label_mode", "signal_vs_noise"),
            "label_column": data_cfg.get("label_column", "class_id"),
            "max_hits": data_cfg.get("max_hits"),
            "sort_by": data_cfg.get("sort_by", "none"),
        }

        if mode in ("train", "all"):
            loaders["train"] = DataLoader(
                ParquetEventDataset(train_files, **common_kwargs),
                batch_size=batch_size,
                shuffle=config["training"]["shuffle"],
                num_workers=num_workers,
                collate_fn=collate_fn,
                pin_memory=torch.cuda.is_available(),
            )
            loaders["val"] = DataLoader(
                ParquetEventDataset(val_files, **common_kwargs),
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=collate_fn,
                pin_memory=torch.cuda.is_available(),
            )

        if mode in ("eval", "all") and test_files:
            loaders["test"] = DataLoader(
                ParquetEventDataset(test_files, **common_kwargs),
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=collate_fn,
                pin_memory=torch.cuda.is_available(),
            )
            loaders["test_helper"] = DataLoader(
                ParquetScoringHelperDataset(
                    test_files,
                    max_hits=data_cfg.get("max_hits"),
                    sort_by=data_cfg.get("sort_by", "none"),
                ),
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=collate_fn_scoringhelper,
                pin_memory=torch.cuda.is_available(),
            )
        return loaders

    # Original .pt tensor workflow.
    data_dir = data_cfg["data_dir"]
    if mode == "train" or mode == "all":
        logging.info("Loading train data with original tensor DataLoader")
        loaders["train"] = DataLoader(
            ForwardPassDataset(data_dir, data_cfg["train_file"]),
            batch_size=batch_size,
            shuffle=config["training"]["shuffle"],
            num_workers=num_workers,
            collate_fn=collate_fn,
        )
        loaders["val"] = DataLoader(
            ForwardPassDataset(data_dir, data_cfg["val_file"]),
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )

    if mode == "eval" or mode == "all":
        loaders["test"] = DataLoader(
            ForwardPassDataset(data_dir, data_cfg["test_file"]),
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )
        loaders["test_helper"] = DataLoader(
            ScoringHelperDataset(data_dir, data_cfg["test_helperfile"]),
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn_scoringhelper,
        )
    return loaders


def load_truths(config):
    data_cfg = config["data"]
    if _cfg_get(data_cfg, "format", "pt") == "parquet":
        truth_file = _cfg_get(data_cfg, "test_truthfile", "")
        if truth_file:
            return pd.read_csv(truth_file)

        parquet_dir = data_cfg["parquet_dir"]
        pattern = _cfg_get(data_cfg, "parquet_glob", "*.parquet")
        files = sorted(glob.glob(os.path.join(parquet_dir, pattern)))
        _, _, test_files = _split_files(
            files,
            train_fraction=_cfg_get(data_cfg, "train_fraction", 0.8),
            val_fraction=_cfg_get(data_cfg, "val_fraction", 0.1),
            seed=_cfg_get(data_cfg, "split_seed", 12345),
        )
        truth_frames = []
        for file_path in test_files:
            df = pd.read_parquet(file_path)
            if (
                _cfg_get(data_cfg, "sort_by", "none") != "none"
                and data_cfg["sort_by"] in df.columns
            ):
                df = df.sort_values(data_cfg["sort_by"], kind="mergesort")
            max_hits = _cfg_get(data_cfg, "max_hits")
            if max_hits is not None and len(df) > max_hits:
                df = df.iloc[:max_hits]
            required = ["hit_id", "particle_id", "weight", "event_id"]
            missing = [column for column in required if column not in df.columns]
            if missing:
                logging.warning(
                    "Skipping inferred Parquet truths for %s because columns are missing: %s",
                    file_path,
                    missing,
                )
                return None
            truth_frames.append(df[required])
        if not truth_frames:
            return None
        return pd.concat(truth_frames, ignore_index=True)
    return pd.read_csv(os.path.join(data_cfg["data_dir"], data_cfg["test_truthfile"]))


def round_up_to_multiple(x, multiple=128):
    return ((x + multiple - 1) // multiple) * multiple


def collate_fn(batch):
    batch_dim = len(batch[0])
    if batch_dim >= 3:
        dat1, dat2, dat3, dat4 = zip(*batch)
        seq_lengths = torch.tensor([x.size(0) for x in dat1], dtype=torch.long)
        max_seq_len = max(seq_lengths).item()
        padded_len = round_up_to_multiple(max_seq_len)

        def pad_to_len(tensors, target_len, pad_value=PAD_TOKEN):
            padded = []
            for t in tensors:
                pad_len = target_len - t.size(0)
                if pad_len == 0:
                    padded.append(t)
                elif t.dim() == 1:
                    padded.append(F.pad(t, (0, pad_len), value=pad_value))
                elif t.dim() == 2:
                    padded.append(F.pad(t, (0, 0, 0, pad_len), value=pad_value))
                else:
                    raise ValueError(f"Unexpected tensor rank {t.dim()} in collate_fn.")
            return torch.stack(padded)

        return (
            pad_to_len(dat1, padded_len),
            pad_to_len(dat2, padded_len),
            pad_to_len(dat3, padded_len),
            pad_to_len(dat4, padded_len),
            seq_lengths,
        )

    if batch_dim == 2:
        dat1, dat2 = zip(*batch)
        seq_lengths = torch.tensor([x.size(0) for x in dat1], dtype=torch.long)
        max_seq_len = max(seq_lengths).item()
        padded_len = round_up_to_multiple(max_seq_len)

        def pad_to_len(tensors, target_len, pad_value=PAD_TOKEN):
            padded = []
            for t in tensors:
                pad_len = target_len - t.size(0)
                if pad_len == 0:
                    padded.append(t)
                elif t.dim() == 1:
                    padded.append(F.pad(t, (0, pad_len), value=pad_value))
                elif t.dim() == 2:
                    padded.append(F.pad(t, (0, 0, 0, pad_len), value=pad_value))
                else:
                    raise ValueError(f"Unexpected tensor rank {t.dim()} in collate_fn.")
            return torch.stack(padded)

        return pad_to_len(dat1, padded_len), pad_to_len(dat2, padded_len), seq_lengths

    raise ValueError(f"Unexpected sample length {batch_dim}. Expected 2 or 4 elements.")


def collate_fn_scoringhelper(batch):
    dat1, dat2 = zip(*batch)
    max_seq_len = max(x.size(0) for x in dat1)
    padded_len = round_up_to_multiple(max_seq_len)

    def pad_to_len(tensors, target_len):
        padded = []
        for tensor in tensors:
            pad_len = target_len - tensor.size(0)
            padded.append(F.pad(tensor, (0, pad_len), value=PAD_TOKEN))
        return torch.stack(padded)

    dat1_padded = pad_to_len(dat1, padded_len)
    dat2_padded = pad_to_len(dat2, padded_len)
    return dat1_padded, dat2_padded


class ParquetEventDataset(Dataset):
    """Dataset that reads one full TrackML event/batch from one Parquet file.

    Expected columns from your converter:
      x, y, z, px, py, pz, cos_theta, sin_phi, cos_phi, q, pt, eta,
      particle_id, weight, event_id, volume_id, layer_id, module_id

    Returned sample:
      coords:  (N, len(feature_columns)) float32
      params:  (N,) float32 dummy/compatibility tensor
      labels:  (N,) int64, currently binary by default
      pos_enc: (N, 5) long/float-compatible tensor with
               [volume_id, layer_id, module_id, eta, phi]
    """
    def __init__(
        self,
        parquet_files,
        feature_columns=None,
        label_mode="signal_vs_noise",
        label_column="class_id",
        max_hits=None,
        sort_by="none",
    ):
        self.files = list(parquet_files)
        self.feature_columns = feature_columns or ["x", "y", "z"]
        self.label_mode = label_mode
        self.label_column = label_column
        self.max_hits = max_hits
        self.sort_by = sort_by

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        df = pd.read_parquet(self.files[idx])

        if self.sort_by and self.sort_by != "none" and self.sort_by in df.columns:
            df = df.sort_values(self.sort_by, kind="mergesort")

        if self.max_hits is not None and len(df) > self.max_hits:
            # Deterministic truncation after optional sorting. For unbiased random
            # truncation, replace this with df.sample(..., random_state=idx).
            df = df.iloc[: self.max_hits]

        missing = [c for c in self.feature_columns if c not in df.columns]
        if missing:
            raise KeyError(f"Missing feature columns in {self.files[idx]}: {missing}")

        coords = torch.tensor(df[self.feature_columns].to_numpy(np.float32), dtype=torch.float32)

        if self.label_mode == "signal_vs_noise":
            labels_np = ((df["particle_id"].to_numpy(np.int64) > 0) &
                         (df["weight"].to_numpy(np.float32) > 0)).astype(np.int64)
        elif self.label_mode == "high_pt_signal_vs_rest":
            labels_np = ((df["particle_id"].to_numpy(np.int64) > 0) &
                         (df["weight"].to_numpy(np.float32) > 0) &
                         (df["pt"].to_numpy(np.float32) > 0.9)).astype(np.int64)
        elif self.label_mode == "class_id":
            if self.label_column not in df.columns:
                raise ValueError(
                    f"label_column={self.label_column} not found in parquet file. "
                f"Available columns: {list(df.columns)}"
            )
            labels_np = df[self.label_column].to_numpy(dtype=np.int64)
        else:
            raise ValueError(f"Unknown label_mode: {self.label_mode}")
        labels = torch.tensor(labels_np, dtype=torch.long)

        x = df["x"].to_numpy(np.float32)
        y = df["y"].to_numpy(np.float32)
        phi = np.arctan2(y, x).astype(np.float32)
        eta = df["eta"].to_numpy(np.float32)
        pos_np = np.stack([
            df["volume_id"].to_numpy(np.float32),
            df["layer_id"].to_numpy(np.float32),
            df["module_id"].to_numpy(np.float32),
            eta,
            phi,
        ], axis=1)
        pos_enc = torch.tensor(pos_np, dtype=torch.float32)

        params = coords[:, 0]
        return coords, params, labels, pos_enc

    def get_hits_xyz(self, idx):
        df = pd.read_parquet(self.files[idx], columns=["x", "y", "z"])
        return torch.tensor(df[["x", "y", "z"]].to_numpy(np.float32), dtype=torch.float32)


class ParquetScoringHelperDataset(Dataset):
    def __init__(self, parquet_files, max_hits=None, sort_by="none"):
        self.files = list(parquet_files)
        self.max_hits = max_hits
        self.sort_by = sort_by

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        df = pd.read_parquet(self.files[idx])
        if self.sort_by and self.sort_by != "none" and self.sort_by in df.columns:
            df = df.sort_values(self.sort_by, kind="mergesort")
        if self.max_hits is not None and len(df) > self.max_hits:
            df = df.iloc[: self.max_hits]
        if "hit_id" in df.columns:
            hit_ids = df["hit_id"].to_numpy(np.int64)
        else:
            hit_ids = np.arange(len(df), dtype=np.int64)
        event_ids = df["event_id"].to_numpy(np.int64)
        return torch.tensor(hit_ids, dtype=torch.long), torch.tensor(event_ids, dtype=torch.long)


class ForwardPassDataset(Dataset):
    def __init__(self, data_dir, file_name):
        file_path = os.path.join(data_dir, file_name)
        self.coords, self.labels, self.pos_data = torch.load(file_path, weights_only=False)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        coords_tensor = torch.tensor(self.coords[idx][:, :3], dtype=torch.float)
        params_tensor = torch.tensor(self.coords[idx][:, 0], dtype=torch.float)
        labels_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
        posenc_tensor = torch.tensor(self.pos_data[idx], dtype=torch.long)
        return coords_tensor, params_tensor, labels_tensor, posenc_tensor

    def get_hits_xyz(self, idx):
        return torch.tensor(self.coords[idx][:, :3], dtype=torch.float)


class ScoringHelperDataset(Dataset):
    def __init__(self, data_dir, file_name):
        file_path = os.path.join(data_dir, file_name)
        self.hit_ids, self.event_ids, _ = torch.load(file_path, weights_only=False)

    def __len__(self):
        return len(self.hit_ids)

    def __getitem__(self, idx):
        hit_ids_tensor = torch.tensor(self.hit_ids[idx], dtype=torch.long)
        event_ids_tensor = torch.tensor(self.event_ids[idx], dtype=torch.long)
        return hit_ids_tensor, event_ids_tensor
