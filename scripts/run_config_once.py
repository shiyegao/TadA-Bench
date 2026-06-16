#!/usr/bin/env python3

import argparse
import sys

sys.path.append(".")

from src.tools import NoWandb
from src.util import (
    load_config,
    make_hyperparameter_id,
    make_run_id,
    normalize_evaluation,
    set_seed,
    train_and_test,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run one resolved config with optional learning-rate override."
    )
    parser.add_argument("--cfg_path", required=True)
    parser.add_argument("--lr", type=float)
    parser.add_argument(
        "--lr_modules",
        nargs="+",
        default=["backbone", "head"],
        help="Optimizer module names that should receive --lr.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.cfg_path)
    for attr in dir(cfg):
        if attr.startswith("_"):
            continue
        value = getattr(cfg, attr)
        if isinstance(value, list) and len(value) == 1:
            setattr(cfg, attr, value[0])

    base_run_id = getattr(cfg, "run_id", getattr(cfg, "tag", "run"))
    ablation = {}
    multi_keys = []

    if args.lr is not None:
        cfg.learning_rate = {module: args.lr for module in args.lr_modules}
        ablation["learning_rate"] = cfg.learning_rate
        multi_keys.append("learning_rate")

    cfg.ablation = ablation
    cfg.hyperparameter_id = make_hyperparameter_id(ablation, multi_keys)
    cfg.run_id = make_run_id(base_run_id, ablation, multi_keys)
    cfg.evaluation = normalize_evaluation(cfg.evaluation)
    cfg.use_wandb = False
    set_seed(getattr(cfg, "seed", 42))

    train_and_test(cfg, NoWandb())


if __name__ == "__main__":
    main()
