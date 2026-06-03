#!/usr/bin/env python3
"""Validate TadA-Bench leaderboard prediction CSVs."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.constants import (  # noqa: E402
    OFFICIAL_HF_DATASET,
    OFFICIAL_HF_DATASET_REVISION,
    OFFICIAL_SEQ_TYPES,
    OFFICIAL_SPLITS,
)


REQUIRED_COLUMNS = {"sequence"}
EMPTY_EXTERNAL_DATA = {"", "none", "n/a", "na", "null"}


def _matches_type(value: object, expected: str) -> bool:
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    return True


def validate_metadata(
    metadata_path: str,
    schema_path: str,
    *,
    submission_path: str | None = None,
    seq_type: str | None = None,
    split: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    required = schema.get("required", [])
    properties = schema.get("properties", {})
    errors: list[str] = []

    for key in required:
        if key not in metadata:
            errors.append(f"Missing required metadata field: {key}")

    for key, spec in properties.items():
        if key not in metadata:
            continue
        expected_type = spec.get("type")
        if expected_type and not _matches_type(metadata[key], expected_type):
            errors.append(
                f"Metadata field `{key}` has type {type(metadata[key]).__name__}, "
                f"expected {expected_type}"
            )
        if "enum" in spec and metadata[key] not in spec["enum"]:
            errors.append(f"Metadata field `{key}` must be one of {spec['enum']}")

    if seq_type and metadata.get("modality") not in (None, seq_type):
        errors.append(
            f"Metadata modality `{metadata.get('modality')}` does not match --seq_type `{seq_type}`"
        )
    if split and metadata.get("split") not in (None, split):
        errors.append(f"Metadata split `{metadata.get('split')}` does not match --split `{split}`")

    if metadata.get("test_labels_used_for_training") is True:
        errors.append("Submissions that used test labels for training are not allowed.")

    if metadata.get("uses_tadabench_train_only") is False:
        external_data = str(metadata.get("external_data", "")).strip().lower()
        if external_data in EMPTY_EXTERNAL_DATA:
            errors.append(
                "`external_data` must describe data sources when "
                "`uses_tadabench_train_only` is false."
            )

    if submission_path and metadata.get("prediction_csv"):
        metadata_name = Path(str(metadata["prediction_csv"])).name
        submission_name = Path(submission_path).name
        if metadata_name != submission_name:
            errors.append(
                f"Metadata prediction_csv `{metadata['prediction_csv']}` does not match "
                f"submission file `{submission_path}` by file name."
            )

    if errors:
        raise ValueError("Metadata validation failed:\n" + "\n".join(f"- {e}" for e in errors))

    summary = {
        "metadata_json": metadata_path,
        "schema_json": schema_path,
        "validated": True,
        "method_name": metadata.get("method_name"),
        "modality": metadata.get("modality"),
        "split": metadata.get("split"),
        "prediction_csv": metadata.get("prediction_csv"),
    }
    return summary, metadata


def recall_at_10pct(labels, preds) -> float:
    import numpy as np

    n = len(labels)
    k = max(1, int(n * 0.1))
    threshold = np.percentile(labels, 90)
    positive = set(np.where(labels >= threshold)[0])
    top = np.argsort(preds)[::-1][:k]
    if not positive:
        return 0.0
    return len(set(top).intersection(positive)) / len(positive)


def ndcg_at_10pct(labels, preds) -> float:
    import numpy as np

    n = len(labels)
    k = max(1, int(n * 0.1))
    gains = (labels - labels.min()) / (labels.max() - labels.min() + 1e-8)
    order = np.argsort(preds)[::-1][:k]
    dcg = np.sum(gains[order] / np.log2(np.arange(2, k + 2)))
    ideal = np.argsort(gains)[::-1][:k]
    idcg = np.sum(gains[ideal] / np.log2(np.arange(2, k + 2)))
    return float(dcg / idcg) if idcg > 0 else 0.0


def read_submission(path: str):
    import numpy as np
    import pandas as pd

    submission = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(submission.columns)
    if missing:
        raise ValueError(f"Submission missing columns: {sorted(missing)}")

    submission = submission.copy()
    if "prediction" not in submission.columns:
        if "y_pred" in submission.columns:
            submission["prediction"] = submission["y_pred"]
        else:
            raise ValueError("Submission must include either `prediction` or `y_pred`.")

    if "method_name" not in submission.columns:
        if "run_id" in submission.columns:
            submission["method_name"] = submission["run_id"].astype(str)
        else:
            submission["method_name"] = Path(path).stem

    duplicate = submission["sequence"].duplicated()
    if duplicate.any():
        raise ValueError(f"Submission contains duplicated sequences: {int(duplicate.sum())}")

    try:
        submission["prediction"] = submission["prediction"].astype(float)
    except ValueError as exc:
        raise ValueError("Submission prediction column must be numeric.") from exc

    if not np.isfinite(submission["prediction"].to_numpy()).all():
        raise ValueError("Submission prediction column contains non-finite values.")

    return submission


def check_sequence_set(labels, submission) -> None:
    expected = set(labels["sequence"])
    submitted = set(submission["sequence"])
    missing = expected - submitted
    extra = submitted - expected

    if missing:
        raise ValueError(f"Missing {len(missing)} required sequences.")
    if extra:
        raise ValueError(f"Submission contains {len(extra)} extra sequences not in the fixed split.")
    if len(submission) != len(labels):
        raise ValueError(
            f"Submission row count {len(submission)} does not match fixed split size {len(labels)}."
        )


def validate_format(args) -> dict[str, Any]:
    submission = read_submission(args.submission)
    metadata_result = None
    if args.metadata_json:
        metadata_result, _ = validate_metadata(
            args.metadata_json,
            args.schema_json,
            submission_path=args.submission,
            seq_type=args.seq_type,
            split=args.split,
        )

    result = {
        "source_file": str(args.submission),
        "format_only": True,
        "num_rows": int(len(submission)),
        "validated": True,
    }
    if metadata_result:
        result["metadata_validation"] = metadata_result
    return result


def validate_full(args) -> dict[str, Any]:
    import pandas as pd
    from datasets import load_dataset
    from scipy.stats import spearmanr

    if not args.seq_type:
        raise ValueError("--seq_type is required unless --format_only is used.")

    submission = read_submission(args.submission)
    metadata_result = None
    metadata = {}
    if args.metadata_json:
        metadata_result, metadata = validate_metadata(
            args.metadata_json,
            args.schema_json,
            submission_path=args.submission,
            seq_type=args.seq_type,
            split=args.split,
        )

    split_name = f"all.{args.seq_type}.{args.split}"
    ds = load_dataset(args.dataset, split=split_name, revision=args.dataset_revision)
    labels = pd.DataFrame({"sequence": ds["Sequence"], "label": ds["Value"]})
    check_sequence_set(labels, submission)

    merged = labels.merge(submission[["sequence", "prediction"]], on="sequence", how="left")
    missing_pred = int(merged["prediction"].isna().sum())
    if missing_pred:
        raise ValueError(f"Submission missing predictions for {missing_pred} sequences in {split_name}")

    y = merged["label"].astype(float).to_numpy()
    p = merged["prediction"].astype(float).to_numpy()
    sp = spearmanr(y, p).correlation
    if math.isnan(sp):
        sp = 0.0

    result = {
        "method_name": str(metadata.get("method_name", submission["method_name"].iloc[0])),
        "source_file": str(args.submission),
        "dataset": args.dataset,
        "dataset_revision": args.dataset_revision,
        "seq_type": args.seq_type,
        "split": args.split,
        "num_examples": int(len(y)),
        "sp": float(sp),
        "recall_at_10pct": float(recall_at_10pct(y, p)),
        "ndcg_at_10pct": float(ndcg_at_10pct(y, p)),
    }
    if metadata_result:
        result["metadata_validation"] = metadata_result
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", required=True)
    parser.add_argument("--dataset", default=OFFICIAL_HF_DATASET)
    parser.add_argument("--dataset_revision", default=OFFICIAL_HF_DATASET_REVISION)
    parser.add_argument("--seq_type", choices=OFFICIAL_SEQ_TYPES)
    parser.add_argument("--split", choices=OFFICIAL_SPLITS, default="test")
    parser.add_argument("--metadata_json", default=None)
    parser.add_argument(
        "--schema_json",
        default="schemas/leaderboard_submission_schema.json",
        help="JSON schema used when --metadata_json is provided.",
    )
    parser.add_argument("--out_json", default=None)
    parser.add_argument(
        "--format_only",
        action="store_true",
        help="Validate CSV/metadata format without downloading fixed-split labels.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = validate_format(args) if args.format_only else validate_full(args)

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
