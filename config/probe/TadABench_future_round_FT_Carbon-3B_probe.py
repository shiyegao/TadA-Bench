from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_source = Path(__file__).parents[1] / "NB1M_ood_FT_Carbon-3B.py"
_spec = spec_from_file_location("_tadabench_carbon_3b_ft", _source)
assert _spec is not None and _spec.loader is not None
_module = module_from_spec(_spec)
_spec.loader.exec_module(_module)

for _name in dir(_module):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_module, _name)

max_train_samples = 2
max_val_samples = 2
max_test_samples = 2
num_epochs = 1
batch_size = 1
test_batch_size = 1
learning_rate = {"backbone": 1e-6, "head": 1e-4}
weight_decay = {"backbone": 1e-4, "head": 1e-4}
save_dir = "ckpt/probe"
prediction_dir = "predictions/probe"
metric_dir = "results/metrics/probe"
run_id = "TadABench_future_round_FT_Carbon-3B_probe"
tag = "FT_Carbon-3B_probe"
protocol = "supervised MLP full fine-tuning probe"
