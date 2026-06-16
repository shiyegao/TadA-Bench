from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_source = Path(__file__).with_name("NB1M_ood_MLP_Carbon-8B.py")
_spec = spec_from_file_location("_tadabench_carbon_8b_mlp", _source)
assert _spec is not None and _spec.loader is not None
_module = module_from_spec(_spec)
_spec.loader.exec_module(_module)

for _name in dir(_module):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_module, _name)

run_id = "TadABench_future_round_MLP_Carbon-8B"
tag = "MLP_Carbon-8B"
