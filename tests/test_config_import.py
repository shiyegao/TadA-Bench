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
    ]:
        cfg = load_config(path)
        assert cfg.huggingface_revision == OFFICIAL_HF_DATASET_REVISION
        assert cfg.save_dir == "ckpt/future_round"
        assert cfg.prediction_dir == "predictions/future_round"
        assert cfg.metric_dir == "results/metrics/future_round"
        assert cfg.eval_after_train is True
        assert cfg.run_id.startswith("TadABench_future_round_MLP_")


def test_esmc_configs_use_cosine_scheduler():
    for path in [
        "config/NB1M_ood_MLP_ESMC-300M.py",
        "config/smoke/NB1M_ood_MLP_ESMC-300M_smoke.py",
    ]:
        cfg = load_config(path)
        assert cfg.scheduler_kwargs["type"] == "CosineAnnealingLR"
        assert cfg.scheduler_kwargs["num_training_steps"] == cfg.num_epochs
