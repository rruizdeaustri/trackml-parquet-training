from pathlib import Path
import sys
import types

import toml

try:
    import optuna  # noqa: F401
except ModuleNotFoundError:
    fake_optuna = types.ModuleType("optuna")

    class _FakeDirection:
        name = "MAXIMIZE"

    class _FakeTrial:
        def __init__(self, number):
            self.number = number
            self.params = {}
            self.user_attrs = {}

        def set_user_attr(self, key, value):
            self.user_attrs[key] = value

        def suggest_categorical(self, name, choices):
            value = choices[self.number % len(choices)]
            self.params[name] = value
            return value

        def suggest_int(self, name, low, high):
            value = low if low == high else low + (self.number % (high - low + 1))
            self.params[name] = value
            return value

        def suggest_float(self, name, low, high, log=False):
            value = low if self.number == 0 else high
            self.params[name] = value
            return value

    class _FakeStudy:
        def __init__(self, study_name):
            self.study_name = study_name
            self.direction = _FakeDirection()
            self.trials = []
            self.best_trial = None
            self.best_value = None
            self.best_params = {}

        def optimize(self, objective, n_trials):
            for number in range(n_trials):
                trial = _FakeTrial(number)
                value = objective(trial)
                trial.value = value
                self.trials.append(trial)
                if self.best_value is None or value > self.best_value:
                    self.best_value = value
                    self.best_trial = trial
                    self.best_params = trial.params

    class _FakeTPESampler:
        def __init__(self, seed=None):
            self.seed = seed

    fake_optuna.Trial = _FakeTrial
    fake_optuna.Study = _FakeStudy
    fake_optuna.samplers = types.SimpleNamespace(TPESampler=_FakeTPESampler)
    fake_optuna.create_study = lambda direction, study_name, storage, load_if_exists, sampler: _FakeStudy(study_name)
    sys.modules["optuna"] = fake_optuna

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from scripts.optuna_study import run_study, suggest_overrides


def _write_tiny_config(tmp_path):
    config = toml.load(REPO_ROOT / "configs" / "tiny_debug.toml")
    config["data"]["parquet_dir"] = str(REPO_ROOT / config["data"]["parquet_dir"])
    config["data"]["max_hits"] = 16
    config["model"]["attention_backend"] = "standard"
    config["training"]["batch_size"] = 2
    config["training"]["total_epochs"] = 1
    config["training"]["start_from_scratch"] = True
    config["training"]["checkpoint_path"] = ""
    config["training"]["scheduler"]["verbose"] = False
    config["logging"]["level"] = "INFO"
    config["wandb"]["enabled"] = False
    config.setdefault("evaluation", {})["run_test"] = False

    config_path = tmp_path / "tiny_debug_optuna.toml"
    with config_path.open("w", encoding="utf-8") as config_file:
        toml.dump(config, config_file)
    return config_path


def test_optuna_study_runs_two_tiny_cpu_trials_from_toml_without_cuda(tmp_path):
    config_path = _write_tiny_config(tmp_path)
    study_dir = tmp_path / "runs" / "optuna_test"

    optuna_config_path = REPO_ROOT / "configs" / "optuna_tiny.toml"

    study = run_study(
        base_config_path=config_path,
        study_dir=study_dir,
        n_trials=2,
        study_name="tiny_cpu_test",
        seed=7,
        optuna_config_path=optuna_config_path,
    )

    assert len(study.trials) == 2
    assert study.best_trial.value is not None
    assert (study_dir / "study_summary.json").exists()
    assert (study_dir / "optuna_config.toml").exists()

    for trial_number in range(2):
        trial_dir = study_dir / f"trial_{trial_number:04d}"
        assert trial_dir.is_dir()
        assert (trial_dir / "trial_config.toml").exists()
        assert (trial_dir / "trial_overrides.json").exists()
        assert (trial_dir / "training.log").exists()
        assert (trial_dir / "checkpoints" / "last.pt").exists()
        assert not list(trial_dir.glob("run_*"))

    first_trial_config = toml.load(study_dir / "trial_0000" / "trial_config.toml")
    assert first_trial_config["evaluation"]["run_test"] is False
    assert first_trial_config["model"]["attention_backend"] == "standard"
    assert first_trial_config["data"]["max_hits"] == 16
    assert first_trial_config["training"]["total_epochs"] == 1
    assert first_trial_config["model"]["embed_dim"] % first_trial_config["model"]["num_heads"] == 0


def test_optuna_study_keeps_backward_compatible_default_search_space(tmp_path):
    config_path = _write_tiny_config(tmp_path)
    base_config = toml.load(config_path)

    class FixedTrial:
        def __init__(self):
            self.params = {}

        def suggest_categorical(self, name, choices):
            self.params[name] = choices[0]
            return choices[0]

        def suggest_int(self, name, low, high, **kwargs):
            self.params[name] = low
            return low

        def suggest_float(self, name, low, high, **kwargs):
            self.params[name] = low
            return low

    overrides = suggest_overrides(FixedTrial(), base_config)
    assert overrides[("model", "attention_backend")] == "standard"
    assert overrides[("model", "embed_dim")] % overrides[("model", "num_heads")] == 0
    assert ("training", "scheduler", "initial_lr") in overrides
