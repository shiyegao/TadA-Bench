import argparse
import json

import pandas as pd
import pytest

from scripts.validate_leaderboard_submission import (
    check_sequence_set,
    validate_format,
    validate_metadata,
)


def write_metadata(path, **overrides):
    payload = {
        "method_name": "ExampleMethod",
        "model_family": "ExampleFamily",
        "modality": "AA",
        "split": "test",
        "prediction_csv": "submission.csv",
        "code_url": "https://example.com/repo",
        "uses_tadabench_train_only": True,
        "external_data": "None",
        "test_labels_used_for_training": False,
        "contact": "maintainer@example.com",
    }
    payload.update(overrides)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_format_only_accepts_example_files():
    args = argparse.Namespace(
        submission="examples/example_leaderboard_submission.csv",
        metadata_json="examples/example_leaderboard_metadata.json",
        schema_json="schemas/leaderboard_submission_schema.json",
        seq_type=None,
        split="test",
    )

    result = validate_format(args)

    assert result["format_only"] is True
    assert result["validated"] is True


def test_duplicate_sequences_are_rejected(tmp_path):
    submission = tmp_path / "submission.csv"
    submission.write_text("sequence,prediction\nA,0.1\nA,0.2\n", encoding="utf-8")
    metadata = tmp_path / "metadata.json"
    write_metadata(metadata)
    args = argparse.Namespace(
        submission=str(submission),
        metadata_json=str(metadata),
        schema_json="schemas/leaderboard_submission_schema.json",
        seq_type="AA",
        split="test",
    )

    with pytest.raises(ValueError, match="duplicated sequences"):
        validate_format(args)


def test_test_label_leakage_metadata_is_rejected(tmp_path):
    metadata = tmp_path / "metadata.json"
    write_metadata(metadata, test_labels_used_for_training=True)

    with pytest.raises(ValueError, match="test labels"):
        validate_metadata(
            str(metadata),
            "schemas/leaderboard_submission_schema.json",
            submission_path="submission.csv",
            seq_type="AA",
            split="test",
        )


def test_external_data_must_be_described_when_not_train_only(tmp_path):
    metadata = tmp_path / "metadata.json"
    write_metadata(metadata, uses_tadabench_train_only=False, external_data="None")

    with pytest.raises(ValueError, match="external_data"):
        validate_metadata(
            str(metadata),
            "schemas/leaderboard_submission_schema.json",
            submission_path="submission.csv",
            seq_type="AA",
            split="test",
        )


def test_exact_sequence_set_rejects_missing_and_extra():
    labels = pd.DataFrame({"sequence": ["A", "B"], "label": [1.0, 2.0]})
    missing = pd.DataFrame({"sequence": ["A"], "prediction": [0.1]})
    extra = pd.DataFrame({"sequence": ["A", "B", "C"], "prediction": [0.1, 0.2, 0.3]})

    with pytest.raises(ValueError, match="Missing 1 required sequences"):
        check_sequence_set(labels, missing)
    with pytest.raises(ValueError, match="extra sequences"):
        check_sequence_set(labels, extra)
