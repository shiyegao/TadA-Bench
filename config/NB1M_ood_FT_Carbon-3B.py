from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_source = Path(__file__).with_name("NB1M_ood_MLP_Carbon-3B.py")
_spec = spec_from_file_location("_tadabench_carbon_3b_mlp", _source)
assert _spec is not None and _spec.loader is not None
_module = module_from_spec(_spec)
_spec.loader.exec_module(_module)

for _name in dir(_module):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_module, _name)

frozen_backbone = False
batch_size = 8
test_batch_size = 16
run_id = "TadABench_future_round_FT_Carbon-3B"
tag = "FT_Carbon-3B"
protocol = "supervised MLP full fine-tuning"

learning_rate = [
    {"backbone": 3e-5, "head": 3e-5},
    {"backbone": 1e-4, "head": 1e-4},
    {"backbone": 3e-4, "head": 3e-4},
]
weight_decay = {"backbone": 1e-4, "head": 1e-4}
