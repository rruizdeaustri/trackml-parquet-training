# Tracking training project

Compact package layout for TrackML-style tracking training.

## Layout

```text
tracking_train_project/
  configs/
    default.toml      # original .pt-tensor workflow
    parquet.toml      # direct Parquet workflow
  scripts/
    train.py
  src/tracking_train/
    train.py
    models/
    data/
    metrics/
    utils/
```

## Install

From this folder:

```bash
python -m pip install -e .
```

For direct Parquet training you also need a Parquet engine, normally `pyarrow`:

```bash
python -m pip install pyarrow
```

## Train directly from Parquet files

Edit:

```bash
configs/parquet.toml
```

Set:

```toml
[data]
parquet_dir = "/home/nadia/batches_parquet"
parquet_glob = "batch_*.parquet"
```

Then run:

```bash
python scripts/train.py configs/parquet.toml
```

The Parquet dataset assumes one file is one full event/batch and expects columns like:

```text
x, y, z, px, py, pz, cos_theta, sin_phi, cos_phi, q, pt, eta,
particle_id, weight, event_id, volume_id, layer_id, module_id
```

By default the model receives `[x, y, z]` as features and uses a binary target:

```text
label = 1 if particle_id > 0 and weight > 0 else 0
```

For a first smoke test on very large full events, set `[data].max_hits = 20000` in `configs/parquet.toml`. Remove or comment it later to train on complete events.

## Original `.pt` workflow

The original preprocessed tensor workflow is still available with:

```bash
python scripts/train.py configs/default.toml
```
