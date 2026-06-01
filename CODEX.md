# Codex instructions

This repository trains a TrackML hit-level transformer classifier from per-event Parquet files.

Important constraints:
- Do not commit large data, checkpoints, run outputs, or full Parquet datasets.
- Only use data_examples/*.parquet for tests.
- Real data lives outside the repo and is configured through configs/*.toml.
- Keep the package importable with `python -m pip install -e .`.
- Prefer small, testable changes.
- For CPU tests, do not require CUDA/flex attention.
- For GPU training, flex attention may require CUDA/ROCm compatibility checks.

Primary objective:
- Maintain a clean Parquet-based training pipeline with class_id labels.

Useful commands:
- Install package: `python -m pip install -e .`
- Tiny/debug run: `python scripts/train.py configs/debug.toml`
- Real run: `python scripts/train.py configs/parquet.toml`
