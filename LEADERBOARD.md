# Leaderboard

Official TadA-Bench leaderboard submissions are evaluated on the fixed Hugging
Face splits from `JinGao/TadA-Bench` at revision
`07168448caaafab4efb26eca04ec3e503edf1c04`.

## Submission Files

Submit one prediction CSV per modality and split. Required columns:

```text
sequence,prediction,method_name
```

Baseline CSVs produced by this repository are also accepted because the
validator maps `y_pred` to `prediction` and `run_id` to `method_name`.

The submitted sequence set must exactly match the selected fixed split. Duplicate
sequences, missing sequences, and extra sequences are rejected.

## Metadata

Metadata JSON must follow `schemas/leaderboard_submission_schema.json`. The
required fields are:

```text
method_name
modality
split
prediction_csv
code_url
uses_tadabench_train_only
external_data
test_labels_used_for_training
contact
```

Submissions that used test labels for training are not allowed. If
`uses_tadabench_train_only` is `false`, `external_data` must describe the data
sources used.

## Validation

Format-only validation checks CSV columns, numeric predictions, duplicated
sequences, and metadata policy:

```bash
uv run python scripts/validate_leaderboard_submission.py \
  --submission examples/example_leaderboard_submission.csv \
  --metadata_json examples/example_leaderboard_metadata.json \
  --schema_json schemas/leaderboard_submission_schema.json \
  --format_only
```

Full validation downloads the selected fixed split, checks exact sequence-set
coverage, and reports Spearman, Recall@10%, and nDCG@10%:

```bash
uv run python scripts/validate_leaderboard_submission.py \
  --submission path/to/submission.csv \
  --metadata_json path/to/metadata.json \
  --schema_json schemas/leaderboard_submission_schema.json \
  --dataset JinGao/TadA-Bench \
  --dataset_revision 07168448caaafab4efb26eca04ec3e503edf1c04 \
  --seq_type AA \
  --split test \
  --out_json results/leaderboard_validation/example.json
```
