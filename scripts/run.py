#!/usr/bin/env python3

from itertools import product
import sys

sys.path.append(".")

from src.tools import NoWandb
from src.util import (
    load_config,
    train_and_test,
    make_hyperparameter_id,
    make_run_id,
    normalize_evaluation,
    set_seed,
)


def run(cfg):
    # All hyperparameters with lists will be ablated
    all_params = {}
    param_dict = {}
    for k in dir(cfg):
        v = getattr(cfg, k)
        if not k.startswith("_"):
            all_params[k] = v
        if isinstance(v, list):
            param_dict[k] = v

    keys = param_dict.keys()
    values = param_dict.values()
    multi_keys = [key for key, value in param_dict.items() if len(value) > 1]
    base_run_id = getattr(cfg, "run_id", getattr(cfg, "tag", "run"))
    param_combinations = [
        dict(zip(keys, combination)) for combination in product(*values)
    ]

    for i, ablation in enumerate(param_combinations):
        print(f"All params: {all_params}")
        print(
            f"\nNow running {i + 1} / {len(param_combinations)} combinations: {ablation}"
        )

        for k, v in ablation.items():
            setattr(cfg, k, v)
        cfg.evaluation = normalize_evaluation(cfg.evaluation)
        cfg.ablation = ablation
        cfg.hyperparameter_id = make_hyperparameter_id(ablation, multi_keys)
        cfg.run_id = make_run_id(base_run_id, ablation, multi_keys)
        set_seed(getattr(cfg, "seed", 42))
        if not cfg.use_wandb:
            wandb = NoWandb()
        else:
            import wandb

            hyperparams = {
                attr: getattr(cfg, attr)
                for attr in dir(cfg)
                if not attr.startswith("__")
            }
            run_name = f"{cfg.tag}-" + "_".join(
                [k for k in keys if len(param_dict[k]) > 1]
            )
            print(f"run_name: {run_name}")
            wandb_run = wandb.init(
                project=cfg.wandb_proj_name,
                entity=cfg.entity,
                config=hyperparams,
                name=run_name,
            )

        print("[epoch-based]")
        train_and_test(cfg, wandb)

        if cfg.use_wandb:
            wandb_run.finish()


if __name__ == "__main__":
    cfg = load_config()
    func = globals().get(cfg.task)
    if func and callable(func):
        func(cfg)
    else:
        print(f"No function found for task: {cfg.task}")
