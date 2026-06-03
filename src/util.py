import os
import re
import random
import argparse
import subprocess
import numpy as np
import importlib.util
from tqdm import tqdm
from pathlib import Path
from typing import Any
from datetime import datetime

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

import src.dataset as dataset_utils
import src.model as model_utils
from src.tools import (
    get_loss_func,
    get_optimizer,
    get_scheduler,
    test_model,
    best_eval_metric,
)


def load_args():
    parser = argparse.ArgumentParser(
        description="Load config and print seq_path variable"
    )
    parser.add_argument(
        "--cfg_path", type=str, required=True, help="Path to the config file"
    )
    args = parser.parse_args()
    return args


def import_config(config_path):
    spec = importlib.util.spec_from_file_location("config", config_path)
    assert spec is not None and spec.loader is not None, (
        f"Failed to load config from {config_path}"
    )
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config


def load_config(config_path: Any = None):
    if config_path is None:
        args = load_args()
        config_path = args.cfg_path

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file {config_path} does not exist.")

    config = import_config(config_path)
    config.cfg_path = str(config_path)
    if not hasattr(config, "run_id"):
        config.run_id = Path(str(config_path)).stem
    config.git_commit = get_git_commit()
    set_seed(getattr(config, "seed", 42))
    return config


def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def normalize_evaluation(evaluation):
    if isinstance(evaluation, str):
        return [evaluation]
    if (
        isinstance(evaluation, (list, tuple))
        and len(evaluation) == 1
        and isinstance(evaluation[0], (list, tuple))
    ):
        return list(evaluation[0])
    return list(evaluation)


def normalize_loss_info(loss_info):
    if isinstance(loss_info, (list, tuple)) and len(loss_info) == 1:
        return loss_info[0]
    return loss_info


def sanitize_run_id(value: Any) -> str:
    if isinstance(value, dict):
        text = "__".join(
            f"{sanitize_run_id(key)}={sanitize_run_id(value[key])}"
            for key in sorted(value, key=str)
        )
        return text or "empty"
    if isinstance(value, (list, tuple)):
        text = "__".join(sanitize_run_id(item) for item in value)
        return text or "empty"
    text = str(value)
    text = text.replace("{", "").replace("}", "").replace("'", "")
    text = text.replace('"', "").replace(" ", "")
    text = text.replace(":", "").replace("/", "-")
    return re.sub(r"[^A-Za-z0-9_.=-]+", "-", text).strip("-")


def make_run_id(base_run_id: str, ablation: dict, multi_keys) -> str:
    parts = [sanitize_run_id(base_run_id)]
    for key in multi_keys:
        if key not in ablation:
            continue
        parts.append(f"{sanitize_run_id(key)}={sanitize_run_id(ablation[key])}")
    return "__".join(part for part in parts if part)


def make_hyperparameter_id(ablation: dict, multi_keys) -> str:
    parts = []
    for key in multi_keys:
        if key not in ablation:
            continue
        parts.append(f"{sanitize_run_id(key)}={sanitize_run_id(ablation[key])}")
    return "__".join(parts) if parts else "default"


def can_train_backbone(model) -> bool:
    return hasattr(model, "backbone") and hasattr(model.backbone, "model")


def set_backbone_train_mode(model):
    if not can_train_backbone(model):
        return
    if getattr(model, "frozen_backbone", False):
        model.backbone.model.eval()
    else:
        model.backbone.model.train()


def set_seed(seed: int):
    print(f"Setting seed to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_model(head_model_type, device, cfg):
    assert hasattr(model_utils, head_model_type), (
        f"Invalid head model type: {head_model_type}"
    )
    cfg.device = device
    model = getattr(model_utils, head_model_type)(cfg=cfg).to(device)
    return model


def seed_worker(worker_id):
    # Seed for each worker, not the main process
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def configure_dataloader_sharing():
    try:
        mp.set_sharing_strategy("file_system")
    except RuntimeError:
        pass


def get_dataset(cfg, has_val):
    assert hasattr(dataset_utils, cfg.dataset_type), (
        f"Invalid dataset type: {cfg.dataset_type}"
    )
    configure_dataloader_sharing()

    def build_dataset(split: str, return_metadata: bool):
        sentinel = object()
        previous = getattr(cfg, "return_metadata", sentinel)
        cfg.return_metadata = return_metadata
        try:
            return getattr(dataset_utils, cfg.dataset_type)(cfg, split=split)
        finally:
            if previous is sentinel:
                delattr(cfg, "return_metadata")
            else:
                cfg.return_metadata = previous

    trainset = build_dataset("train", return_metadata=False)
    testset = build_dataset("test", return_metadata=True)
    if has_val:
        valset = build_dataset("val", return_metadata=True)

    num_workers = getattr(cfg, "num_workers", 0)
    testloader = DataLoader(
        testset,
        batch_size=cfg.test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
    )
    trainloader = DataLoader(
        trainset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
    )
    if has_val:
        valloader = DataLoader(
            valset,
            batch_size=cfg.test_batch_size,
            shuffle=False,
            num_workers=num_workers,
            worker_init_fn=seed_worker,
        )
        return trainset, valset, testset, trainloader, valloader, testloader
    else:
        # if only train and test, regard valset as testset
        return trainset, testset, None, trainloader, testloader, None


def train_and_test(cfg, wandb):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    cfg.device = device
    cfg.evaluation = normalize_evaluation(cfg.evaluation)
    cfg.loss_type = normalize_loss_info(cfg.loss_type)
    has_val = getattr(cfg, "train_val_test", False)
    use_embed_mapper = getattr(cfg, "use_embed_mapper", False)

    # if only train and test, valset = testset, testset = None
    trainset, valset, testset, trainloader, valloader, testloader = get_dataset(
        cfg, has_val
    )

    if has_val:
        wandb.config.update(
            {
                "train_num": len(trainset),
                "val_num": len(valset),
                "test_num": len(testset),
            }
        )
    else:
        wandb.config.update({"train_num": len(trainset), "test_num": len(valset)})

    model = get_model(cfg.head_model_type, device, cfg)
    loss_func = (
        get_loss_func(cfg.loss_type)
        if isinstance(cfg.loss_type, str)
        else get_loss_func(normalize_loss_info(cfg.loss_type))
    )

    optimizer = get_optimizer(
        cfg.optimizer_type,
        model,
        getattr(cfg, "learning_rate", 0),
        getattr(cfg, "weight_decay", 0),
    )
    sched_cfg = getattr(cfg, "scheduler_kwargs", {"type": "NoScheduler"})
    scheduler, is_epoch_scheduler = get_scheduler(optimizer, sched_cfg)

    print("Training...")
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(cfg.save_dir, time_str)
    best_metric_val = {eval_metric: -1 for eval_metric in cfg.evaluation}
    if getattr(cfg, "eval_before_train", True):
        print("Evaluating before training...")
        val_scores, loss_eval = test_model(
            model,
            valloader,
            wandb,
            0,
            "val",
            cfg.evaluation,
            loss_func,
            cfg=cfg,
        )
        test_scores, loss_eval = test_model(
            model,
            testloader,
            wandb,
            0,
            "test",
            cfg.evaluation,
            loss_func,
            cfg=cfg,
        ) if testloader is not None else ({}, None)
        for eval_metric in cfg.evaluation:
            best_metric_val[eval_metric] = best_eval_metric(
                eval_metric, best_metric_val[eval_metric], val_scores[eval_metric]
            )

    model.train()
    if not use_embed_mapper:
        set_backbone_train_mode(model)
    for epoch in range(1, cfg.num_epochs + 1):
        epoch_loss = 0.0
        for data, labels in tqdm(
            trainloader, desc=f"Epoch {epoch}, Training", dynamic_ncols=True
        ):
            if not isinstance(labels, torch.Tensor):
                labels = torch.stack(labels).T

            # data/label: shape = [k, batch_size]
            optimizer.zero_grad()

            predicted_scores = model(data)

            labels = labels.to(device).to(dtype=predicted_scores.dtype)
            loss = loss_func(labels, predicted_scores)

            cfg_tmp = {"epoch": epoch, "train loss": loss.item()}
            for i, group in enumerate(optimizer.param_groups):
                cfg_tmp[f"lr_{i}"] = group["lr"]
            wandb.log(cfg_tmp)

            loss.backward()
            optimizer.step()
            if not is_epoch_scheduler:
                scheduler.step()
            epoch_loss += loss.item()

        print(
            f"\nEpoch {epoch}/{cfg.num_epochs}, Train Loss: {epoch_loss / len(trainloader)}"
        )

        if epoch % cfg.eval_interval == 0 or (epoch == cfg.num_epochs):
            scores, loss_eval = test_model(
                model,
                valloader,
                wandb,
                epoch,
                "val",
                cfg.evaluation,
                loss_func,
                cfg=cfg,
            )
            for eval_metric in cfg.evaluation:
                best_metric_val[eval_metric] = best_eval_metric(
                    eval_metric, best_metric_val[eval_metric], scores[eval_metric]
                )
            if is_epoch_scheduler:
                if sched_cfg["type"] == "ReduceLROnPlateau":
                    scheduler.step(loss_eval)
                else:
                    scheduler.step()

            model.train()
            if not use_embed_mapper:
                set_backbone_train_mode(model)

        if cfg.save_interval > -1 and (
            epoch % cfg.save_interval == 0 or (epoch == cfg.num_epochs)
        ):
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(save_dir, f"epoch_{epoch}.pth"))

    if getattr(cfg, "eval_after_train", True) and testloader is not None:
        scores = test_model(
            model,
            testloader,
            wandb,
            epoch,
            "test",
            cfg.evaluation,
            loss_func,
            cfg=cfg,
        )

    wandb.log(
        {
            f"best {eval_metric} score": best_metric_val[eval_metric]
            for eval_metric in cfg.evaluation
        }
    )
