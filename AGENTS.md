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

## Scientific objective

The final goal is to train hit-level models on full TrackML events, ideally using all hits in each event, not only small sampled subsets. The current `max_hits` option is a debugging/scaling control and should not be treated as the final training target.

The primary final metric is the TrackML score on the held-out test dataset. Validation loss/accuracy are useful for debugging and model selection during development, but final comparisons should report TrackML score on the test split.

## Model development goals

The code should support systematic hyperparameter studies, including:
- `max_hits`
- `embed_dim`
- `num_layers`
- `num_heads`
- `dim_feedforward`
- `dropout`
- learning rate
- weight decay
- scheduler settings
- attention/mask settings

Eventually, the project should support Optuna optimization over these hyperparameters, with the objective based on validation TrackML score or a proxy metric, and final reporting on test TrackML score.

## Important constraint

Do not optimize only for cross-entropy loss or binned-class accuracy. These are useful diagnostics, but the downstream target is track reconstruction quality as measured by the TrackML score.
