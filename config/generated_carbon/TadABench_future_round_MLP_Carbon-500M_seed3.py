from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_source = Path(__file__).parents[1] / "NB1M_ood_MLP_Carbon-500M.py"
_spec = spec_from_file_location("_tadabench_carbon_500m_future", _source)
assert _spec is not None and _spec.loader is not None
_module = module_from_spec(_spec)
_spec.loader.exec_module(_module)

for _name in dir(_module):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_module, _name)

seed = 3
run_id = "TadABench_future_round_MLP_Carbon-500M_seed3"
tag = "MLP_Carbon-500M_seed3"
prediction_dir = "predictions/future_round"
metric_dir = "results/metrics/future_round"
eval_after_train = True
