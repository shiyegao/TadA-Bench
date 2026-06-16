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

seed = 2
run_id = "TadABench_future_round_FT_Carbon-3B_seed2"
tag = "FT_Carbon-3B_seed2"
prediction_dir = "predictions/future_round"
metric_dir = "results/metrics/future_round"
eval_after_train = True
