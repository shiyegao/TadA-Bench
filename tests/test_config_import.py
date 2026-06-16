from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from src.constants import OFFICIAL_HF_DATASET_REVISION


ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str):
    spec = spec_from_file_location(Path(path).stem, ROOT / path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_full_configs_write_outputs_and_pin_revision():
    for path in [
        "config/TadABench_future_round_MLP_ESM2-35M.py",
        "config/TadABench_future_round_MLP_ESMC-300M.py",
        "config/TadABench_future_round_MLP_NT-50M.py",
        "config/TadABench_future_round_MLP_Carbon-500M.py",
        "config/TadABench_future_round_MLP_Carbon-3B.py",
        "config/TadABench_future_round_MLP_Carbon-8B.py",
        "config/TadABench_future_round_FT_Carbon-500M.py",
        "config/TadABench_future_round_FT_Carbon-3B.py",
        "config/TadABench_future_round_FT_Carbon-8B.py",
    ]:
        cfg = load_config(path)
        assert cfg.huggingface_revision == OFFICIAL_HF_DATASET_REVISION
        assert cfg.save_dir == "ckpt/future_round"
        assert cfg.prediction_dir == "predictions/future_round"
        assert cfg.metric_dir == "results/metrics/future_round"
        assert cfg.eval_after_train is True
        assert cfg.run_id.startswith(("TadABench_future_round_MLP_", "TadABench_future_round_FT_"))


def test_esmc_configs_use_cosine_scheduler():
    for path in [
        "config/NB1M_ood_MLP_ESMC-300M.py",
        "config/smoke/NB1M_ood_MLP_ESMC-300M_smoke.py",
    ]:
        cfg = load_config(path)
        assert cfg.scheduler_kwargs["type"] == "CosineAnnealingLR"
        assert cfg.scheduler_kwargs["num_training_steps"] == cfg.num_epochs


def test_carbon_configs_use_expected_dimensions():
    expected = {
        "config/TadABench_future_round_MLP_Carbon-500M.py": (
            "HuggingFaceBio/Carbon-500M",
            84,
            1024,
        ),
        "config/TadABench_future_round_MLP_Carbon-3B.py": (
            "HuggingFaceBio/Carbon-3B",
            84,
            3072,
        ),
        "config/TadABench_future_round_MLP_Carbon-8B.py": (
            "HuggingFaceBio/Carbon-8B",
            84,
            4096,
        ),
        "config/smoke/TadABench_future_round_MLP_Carbon-500M_smoke.py": (
            "HuggingFaceBio/Carbon-500M",
            84,
            1024,
        ),
        "config/smoke/TadABench_future_round_MLP_Carbon-3B_smoke.py": (
            "HuggingFaceBio/Carbon-3B",
            84,
            3072,
        ),
        "config/probe/TadABench_future_round_MLP_Carbon-8B_profile.py": (
            "HuggingFaceBio/Carbon-8B",
            84,
            4096,
        ),
    }
    for path, (model_name, num_tokens, embed_dim) in expected.items():
        cfg = load_config(path)
        assert cfg.embed_name == model_name
        assert cfg.seq_type == "DNA"
        assert cfg.del_special_tokens is True
        assert cfg.num_tokens == num_tokens
        assert cfg.embed_dim == embed_dim
        assert cfg.dtype["backbone"] == "bf16"


def test_carbon_generated_mlp_seed_configs():
    for model_name in ["Carbon-500M", "Carbon-3B", "Carbon-8B"]:
        for seed in [1, 2, 3]:
            path = (
                "config/generated_carbon/"
                f"TadABench_future_round_MLP_{model_name}_seed{seed}.py"
            )
            cfg = load_config(path)
            assert cfg.seed == seed
            assert cfg.run_id == f"TadABench_future_round_MLP_{model_name}_seed{seed}"
            assert cfg.prediction_dir == "predictions/future_round"
            assert cfg.metric_dir == "results/metrics/future_round"
            assert cfg.eval_after_train is True
            assert not hasattr(cfg, "max_samples") or cfg.max_samples is None


def test_carbon_generated_ft_seed_configs():
    for model_name in ["Carbon-500M", "Carbon-3B", "Carbon-8B"]:
        for seed in [1, 2, 3]:
            path = (
                "config/generated_carbon/"
                f"TadABench_future_round_FT_{model_name}_seed{seed}.py"
            )
            cfg = load_config(path)
            assert cfg.seed == seed
            assert cfg.run_id == f"TadABench_future_round_FT_{model_name}_seed{seed}"
            assert cfg.frozen_backbone is False
            assert cfg.protocol == "supervised MLP full fine-tuning"
            assert cfg.prediction_dir == "predictions/future_round"
            assert cfg.metric_dir == "results/metrics/future_round"
            assert cfg.eval_after_train is True
            assert not hasattr(cfg, "max_samples") or cfg.max_samples is None


def test_carbon_ft_probe_configs_unfreeze_backbone():
    for model_name in ["Carbon-500M", "Carbon-3B", "Carbon-8B"]:
        cfg = load_config(f"config/probe/TadABench_future_round_FT_{model_name}_probe.py")
        assert cfg.frozen_backbone is False
        assert cfg.max_train_samples == 2
        assert cfg.batch_size == 1
        assert cfg.learning_rate["backbone"] == 1e-6
        assert cfg.weight_decay["backbone"] == 1e-4


def test_carbon_mlp_profile_config_uses_subset_probe():
    cfg = load_config("config/probe/TadABench_future_round_MLP_Carbon-500M_profile.py")
    assert cfg.frozen_backbone is True
    assert cfg.max_train_samples == 4096
    assert cfg.max_val_samples == 2048
    assert cfg.max_test_samples == 2048
    assert cfg.num_epochs == 1
    assert cfg.learning_rate["head"] == 1e-4
