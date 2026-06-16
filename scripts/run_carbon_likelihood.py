#!/usr/bin/env python3

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

sys.path.append(".")

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.constants import OFFICIAL_HF_DATASET, OFFICIAL_HF_DATASET_REVISION
from src.tools.evaluation import (
    PREDICTION_FIELDS,
    get_ndcg_at_10pct_score,
    get_recall_at_10pct_score,
    get_sp_score,
)
from src.util import get_git_commit


METRICS = {
    "sp": get_sp_score,
    "recall_at_10pct": get_recall_at_10pct_score,
    "ndcg_at_10pct": get_ndcg_at_10pct_score,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run zero-shot Carbon DNA likelihood scoring on TadA-Bench."
    )
    parser.add_argument("--model_name", default="HuggingFaceBio/Carbon-500M")
    parser.add_argument("--revision", default="fns")
    parser.add_argument("--dataset", default=OFFICIAL_HF_DATASET)
    parser.add_argument("--dataset_revision", default=OFFICIAL_HF_DATASET_REVISION)
    parser.add_argument("--seq_type", default="DNA", choices=["DNA"])
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--prediction_dir", default="predictions/future_round")
    parser.add_argument("--metric_dir", default="results/metrics/future_round")
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--repeat", type=int, default=None)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--disable_tqdm", action="store_true")
    return parser.parse_args()


def split_name(seq_type: str, split: str, max_samples: int | None):
    name = f"all.{seq_type}.{split}"
    if max_samples is not None:
        if max_samples < 0:
            raise ValueError(f"max_samples must be non-negative, got {max_samples}")
        name = f"{name}[:{max_samples}]"
    return name


def load_carbon(model_name: str, revision: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=revision,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device)
    if hasattr(model, "setup_tokenizer"):
        model.setup_tokenizer(tokenizer)
    model.eval()
    if not hasattr(model, "score_sequence"):
        raise AttributeError(
            "Carbon likelihood scoring requires the FNS revision with "
            "`score_sequence()` support."
        )
    return tokenizer, model


def score_batch(model, seqs):
    with torch.no_grad():
        _, actual_probs = model.score_sequence(list(seqs))
    return [
        torch.log(prob.clamp_min(1e-12)).mean().detach().cpu().item()
        for prob in actual_probs
    ]


def compute_metrics(labels, preds):
    labels = np.asarray(labels, dtype=float)
    preds = np.asarray(preds, dtype=float)
    return {name: float(func(labels, preds)) for name, func in METRICS.items()}


def write_prediction_csv(
    path,
    rows,
    run_id,
    model_name,
    seq_type,
    revision,
    repeat,
    max_samples,
    script_path,
    git_commit,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PREDICTION_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "sequence": row["sequence"],
                    "y_true": float(row["label"]),
                    "y_pred": float(row["prediction"]),
                    "split": row["split"],
                    "domain": row["domain"],
                    "run_id": run_id,
                    "model": model_name,
                    "modality": seq_type,
                    "example_index": row["example_index"],
                    "config_path": script_path,
                    "git_commit": git_commit,
                    "seed": "" if repeat is None else int(repeat),
                    "repeat": "" if repeat is None else int(repeat),
                    "revision": revision,
                    "protocol": "zero-shot base-pair log-likelihood",
                    "max_samples": "" if max_samples is None else int(max_samples),
                    "is_subset": max_samples is not None,
                }
            )


def write_metric_json(
    path,
    run_id,
    model_name,
    seq_type,
    revision,
    repeat,
    max_samples,
    split,
    num_examples,
    metrics,
    script_path,
    git_commit,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "run_id": run_id,
        "model": model_name,
        "modality": seq_type,
        "hyperparameter_id": f"revision={revision}__protocol=likelihood",
        "hyperparameters": {
            "revision": revision,
            "protocol": "zero-shot base-pair log-likelihood",
            "repeat": repeat,
            "max_samples": max_samples,
            "is_subset": max_samples is not None,
        },
        "split": split,
        "num_examples": int(num_examples),
        "metrics": metrics,
        "config_path": script_path,
        "git_commit": git_commit,
        "epoch": 0,
        "max_samples": max_samples,
        "is_subset": max_samples is not None,
    }
    if repeat is not None:
        payload["repeat"] = int(repeat)
        payload["seed"] = int(repeat)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def run_split(args, model, split, script_path, git_commit):
    data = load_dataset(
        args.dataset,
        split=split_name(args.seq_type, split, args.max_samples),
        revision=args.dataset_revision,
    )
    seqs = [str(seq) for seq in data["Sequence"]]
    labels = [float(label) for label in data["Value"]]
    domains = (
        [str(domain) for domain in data["Domain"]]
        if "Domain" in data.column_names
        else ["unknown"] * len(seqs)
    )
    preds = []
    num_batches = math.ceil(len(seqs) / args.batch_size)
    run_id = resolve_run_id(args)
    prediction_path = os.path.join(args.prediction_dir, f"{run_id}_{split}.csv")
    metric_path = os.path.join(args.metric_dir, f"{run_id}_{split}.json")
    if args.skip_existing and os.path.exists(prediction_path) and os.path.exists(metric_path):
        print(f"{split}: skipping existing outputs for {run_id}")
        return

    for start in tqdm(
        range(0, len(seqs), args.batch_size),
        total=num_batches,
        desc=f"Scoring {split}",
        dynamic_ncols=True,
        disable=args.disable_tqdm,
    ):
        batch = seqs[start : start + args.batch_size]
        preds.extend(score_batch(model, batch))

    rows = [
        {
            "sequence": seq,
            "label": label,
            "prediction": pred,
            "split": split,
            "domain": domain,
            "example_index": idx,
        }
        for idx, (seq, label, pred, domain) in enumerate(
            zip(seqs, labels, preds, domains, strict=True)
        )
    ]
    metrics = compute_metrics(labels, preds)
    write_prediction_csv(
        prediction_path,
        rows,
        run_id,
        args.model_name,
        args.seq_type,
        args.revision,
        args.repeat,
        args.max_samples,
        script_path,
        git_commit,
    )
    write_metric_json(
        metric_path,
        run_id,
        args.model_name,
        args.seq_type,
        args.revision,
        args.repeat,
        args.max_samples,
        split,
        len(labels),
        metrics,
        script_path,
        git_commit,
    )
    print(f"{split}: {metrics}")


def resolve_run_id(args):
    if args.run_id:
        return args.run_id
    run_id = f"TadABench_future_round_Carbon_likelihood_{Path(args.model_name).name}"
    if args.repeat is not None:
        run_id = f"{run_id}_repeat{args.repeat}"
    return run_id


def main():
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {args.batch_size}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    _, model = load_carbon(args.model_name, args.revision, device)
    script_path = str(Path(__file__))
    git_commit = get_git_commit()
    for split in args.splits:
        run_split(args, model, split, script_path, git_commit)


if __name__ == "__main__":
    main()
