#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path

import pandas as pd


METRIC_COLUMNS = ["sp", "recall_at_10pct", "ndcg_at_10pct"]
CARBON_PATTERN = re.compile(r"Carbon-(?:500M|3B|8B)")
SEED_PATTERN = re.compile(r"_seed(\d+)")
REPEAT_PATTERN = re.compile(r"_repeat(\d+)")
EXPECTED_MLP_HYPERPARAMETERS = {
    "supervised MLP frozen backbone": [
        "learning_rate=head=3e-05",
        "learning_rate=head=0.0001",
        "learning_rate=head=0.0003",
    ],
    "supervised MLP full fine-tuning": [
        "learning_rate=backbone=3e-05__head=3e-05",
        "learning_rate=backbone=0.0001__head=0.0001",
        "learning_rate=backbone=0.0003__head=0.0003",
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize Carbon TadA-Bench metric JSON files."
    )
    parser.add_argument("--metric_dir", default="results/metrics/future_round")
    parser.add_argument("--out_dir", default="results/carbon_summary")
    parser.add_argument("--selection_metric", default="sp")
    parser.add_argument("--expected_repeats", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--expected_seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--expected_model_sizes", nargs="+", default=None)
    parser.add_argument("--expected_protocols", nargs="+", default=None)
    parser.add_argument(
        "--expected_model_protocols",
        nargs="+",
        default=None,
        help=(
            "Expected model/protocol pairs formatted as model_size::protocol. "
            "Use this when not every model is expected to have every protocol."
        ),
    )
    parser.add_argument(
        "--expected_split_examples",
        nargs="+",
        default=None,
        help="Expected full-split sizes, formatted as split=count, e.g. val=148014 test=149884.",
    )
    parser.add_argument("--require_non_subset", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def parse_expected_split_examples(values):
    expected = {}
    if not values:
        return expected
    for value in values:
        split, sep, count = str(value).partition("=")
        if not sep:
            raise ValueError(
                f"Invalid --expected_split_examples value {value!r}; expected split=count"
            )
        expected[split] = int(count)
    return expected


def parse_expected_model_protocols(values):
    expected = []
    if not values:
        return expected
    for value in values:
        model_size, sep, protocol = str(value).partition("::")
        if not sep:
            raise ValueError(
                f"Invalid --expected_model_protocols value {value!r}; "
                "expected model_size::protocol"
            )
        expected.append((model_size, protocol))
    return expected


def is_truthy(value):
    if value is None or pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def infer_protocol(payload):
    hparams = payload.get("hyperparameters", {})
    if hparams.get("protocol"):
        return str(hparams["protocol"])
    run_id = str(payload.get("run_id", ""))
    if "FT" in run_id:
        return "supervised MLP full fine-tuning"
    if "MLP" in run_id:
        frozen = payload.get("frozen_backbone")
        if frozen is False:
            return "supervised MLP full fine-tuning"
        return "supervised MLP frozen backbone"
    return "unknown"


def infer_repeat(payload):
    if payload.get("repeat") is not None:
        return int(payload["repeat"])
    hparams = payload.get("hyperparameters", {})
    if hparams.get("repeat") is not None:
        return int(hparams["repeat"])
    match = REPEAT_PATTERN.search(str(payload.get("run_id", "")))
    return int(match.group(1)) if match else None


def infer_seed(payload):
    if payload.get("seed") is not None:
        return int(payload["seed"])
    match = SEED_PATTERN.search(str(payload.get("run_id", "")))
    return int(match.group(1)) if match else infer_repeat(payload)


def flatten_metric_file(path):
    payload = json.loads(path.read_text())
    run_id = str(payload.get("run_id", ""))
    model = str(payload.get("model", ""))
    if "Carbon" not in run_id and "Carbon" not in model:
        return None

    hparams = payload.get("hyperparameters", {})
    metrics = payload.get("metrics", {})
    row = {
        "source_file": str(path),
        "run_id": run_id,
        "model": model,
        "model_size": CARBON_PATTERN.search(run_id + " " + model).group(0)
        if CARBON_PATTERN.search(run_id + " " + model)
        else "Carbon",
        "modality": payload.get("modality"),
        "protocol": infer_protocol(payload),
        "split": payload.get("split"),
        "epoch": payload.get("epoch"),
        "num_examples": payload.get("num_examples"),
        "hyperparameter_id": str(payload.get("hyperparameter_id", "default")),
        "seed": infer_seed(payload),
        "repeat": infer_repeat(payload),
        "is_subset": payload.get("is_subset", hparams.get("is_subset")),
        "max_samples": payload.get("max_samples", hparams.get("max_samples")),
        "config_path": payload.get("config_path"),
        "git_commit": payload.get("git_commit"),
    }
    for metric in METRIC_COLUMNS:
        row[metric] = metrics.get(metric)
    return row


def load_metrics(metric_dir):
    rows = []
    for path in sorted(Path(metric_dir).rglob("*.json")):
        row = flatten_metric_file(path)
        if row is not None:
            rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No Carbon metric JSON files found in {metric_dir}")
    return pd.DataFrame(rows)


def assert_complete(df, args):
    errors = []
    expected_split_examples = parse_expected_split_examples(args.expected_split_examples)
    expected_model_protocols = parse_expected_model_protocols(
        args.expected_model_protocols
    )
    if args.expected_model_sizes:
        observed = sorted(str(v) for v in df["model_size"].dropna().unique())
        expected = sorted(args.expected_model_sizes)
        if observed != expected:
            errors.append(f"Model sizes observed={observed}, expected={expected}")

    if expected_model_protocols:
        for model_size, protocol in expected_model_protocols:
            group = df[
                (df["model_size"] == model_size)
                & (df["protocol"] == protocol)
            ]
            if group.empty:
                errors.append(f"Missing results for {model_size} / {protocol}")
    elif args.expected_model_sizes and args.expected_protocols:
        for model_size in args.expected_model_sizes:
            for protocol in args.expected_protocols:
                group = df[
                    (df["model_size"] == model_size)
                    & (df["protocol"] == protocol)
                ]
                if group.empty:
                    errors.append(f"Missing results for {model_size} / {protocol}")

    if expected_split_examples:
        for split, expected_count in expected_split_examples.items():
            split_df = df[df["split"] == split]
            if split_df.empty:
                errors.append(f"No rows found for expected split {split}")
                continue
            observed_counts = sorted(
                int(v) for v in split_df["num_examples"].dropna().unique()
            )
            if observed_counts != [expected_count]:
                errors.append(
                    f"Split {split} num_examples observed={observed_counts}, "
                    f"expected={[expected_count]}"
                )

    if args.require_non_subset:
        subset_rows = df[df["is_subset"].map(is_truthy)]
        if not subset_rows.empty:
            offenders = sorted(subset_rows["run_id"].dropna().unique())
            errors.append(f"Subset rows present in formal summary: {offenders}")
        max_sample_rows = df[df["max_samples"].notna()]
        if not max_sample_rows.empty:
            offenders = sorted(max_sample_rows["run_id"].dropna().unique())
            errors.append(f"max_samples rows present in formal summary: {offenders}")

    zshot = df[df["protocol"].str.contains("likelihood", case=False, na=False)]
    for (model_size, split), group in zshot.groupby(["model_size", "split"], dropna=False):
        observed = sorted(int(v) for v in group["repeat"].dropna().unique())
        if observed and observed != args.expected_repeats:
            errors.append(
                f"Zero-shot {model_size} {split} repeats observed={observed}, "
                f"expected={args.expected_repeats}"
            )

    mlp = df[df["protocol"].str.contains("MLP", case=False, na=False)]
    if not mlp.empty:
        expected_splits = ["test", "val"]
        for (model_size, protocol), protocol_group in mlp.groupby(
            ["model_size", "protocol"], dropna=False
        ):
            expected_hparams = EXPECTED_MLP_HYPERPARAMETERS.get(str(protocol))
            if expected_hparams:
                observed_hparams = sorted(
                    str(v) for v in protocol_group["hyperparameter_id"].dropna().unique()
                )
                if sorted(expected_hparams) != observed_hparams:
                    errors.append(
                        f"{protocol} {model_size} hyperparameters observed="
                        f"{observed_hparams}, expected={sorted(expected_hparams)}"
                    )
                observed_splits = sorted(
                    str(v) for v in protocol_group["split"].dropna().unique()
                )
                if observed_splits != expected_splits:
                    errors.append(
                        f"{protocol} {model_size} splits observed={observed_splits}, "
                        f"expected={expected_splits}"
                    )

        grouped = mlp.groupby(["model_size", "hyperparameter_id", "split"], dropna=False)
        for keys, group in grouped:
            observed = sorted(int(v) for v in group["seed"].dropna().unique())
            if observed and observed != args.expected_seeds:
                errors.append(
                    f"MLP {keys} seeds observed={observed}, expected={args.expected_seeds}"
                )
    if errors and args.strict:
        raise ValueError("Incomplete Carbon result matrix:\n" + "\n".join(errors))
    return errors


def select_mlp(df, selection_metric):
    mlp = df[
        df["protocol"].str.contains("MLP", case=False, na=False)
        & (df["epoch"] == 20)
    ].copy()
    if mlp.empty:
        return pd.DataFrame(), pd.DataFrame()

    val = mlp[mlp["split"] == "val"].copy()
    test = mlp[mlp["split"] == "test"].copy()
    if val.empty or test.empty:
        return pd.DataFrame(), pd.DataFrame()

    hp_scores = (
        val.groupby(["model_size", "model", "modality", "protocol", "hyperparameter_id"])
        .agg(
            val_selection_mean=(selection_metric, "mean"),
            val_selection_std=(selection_metric, "std"),
            val_selection_num_seeds=("seed", "nunique"),
        )
        .reset_index()
    )
    hp_scores["val_selection_std"] = hp_scores["val_selection_std"].fillna(0.0)
    selected_idx = hp_scores.groupby(["model_size", "model", "modality", "protocol"])[
        "val_selection_mean"
    ].idxmax()
    selected_hp = hp_scores.loc[selected_idx].copy()
    selected_test = test.merge(
        selected_hp,
        on=["model_size", "model", "modality", "protocol", "hyperparameter_id"],
        how="inner",
    )
    return selected_hp, selected_test


def summarize_repeats(df):
    rows = []
    for keys, group in df.groupby(
        ["model_size", "model", "modality", "protocol", "split", "hyperparameter_id"],
        dropna=False,
    ):
        row = {
            "model_size": keys[0],
            "model": keys[1],
            "modality": keys[2],
            "protocol": keys[3],
            "split": keys[4],
            "hyperparameter_id": keys[5],
            "num_runs": len(group),
            "seeds": ",".join(str(int(v)) for v in sorted(group["seed"].dropna().unique())),
            "repeats": ",".join(
                str(int(v)) for v in sorted(group["repeat"].dropna().unique())
            ),
            "num_examples": ",".join(
                str(int(v)) for v in sorted(group["num_examples"].dropna().unique())
            ),
        }
        for metric in METRIC_COLUMNS:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = values.mean() if not values.empty else None
            row[f"{metric}_std"] = values.std(ddof=1) if len(values) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def write_table(df, csv_path, md_path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    df.to_markdown(md_path, index=False)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    df = load_metrics(args.metric_dir)
    errors = assert_complete(df, args)

    all_csv = out_dir / "carbon_all_metrics.csv"
    all_md = out_dir / "carbon_all_metrics.md"
    write_table(df.sort_values(["protocol", "model_size", "run_id", "split"]), all_csv, all_md)

    repeat_summary = summarize_repeats(df)
    write_table(
        repeat_summary.sort_values(["protocol", "model_size", "split", "hyperparameter_id"]),
        out_dir / "carbon_repeat_summary.csv",
        out_dir / "carbon_repeat_summary.md",
    )

    selected_hp, selected_test = select_mlp(df, args.selection_metric)
    if not selected_hp.empty:
        write_table(
            selected_hp.sort_values(["model_size", "hyperparameter_id"]),
            out_dir / "carbon_mlp_selected_hyperparameters.csv",
            out_dir / "carbon_mlp_selected_hyperparameters.md",
        )
        write_table(
            selected_test.sort_values(["model_size", "seed"]),
            out_dir / "carbon_mlp_selected_test_rows.csv",
            out_dir / "carbon_mlp_selected_test_rows.md",
        )

    status = {"complete": not errors, "errors": errors}
    (out_dir / "carbon_summary_status.json").write_text(json.dumps(status, indent=2))
    print(f"Wrote Carbon summaries under {out_dir}")
    if errors:
        print("Completeness warnings:")
        for error in errors:
            print(f"- {error}")


if __name__ == "__main__":
    main()
