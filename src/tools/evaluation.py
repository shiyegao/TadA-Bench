import csv
import json
import math
import os
import torch
import numpy as np
from collections.abc import Mapping
from typing import List
from tqdm import tqdm
from scipy.stats import spearmanr
from sklearn.metrics import (
    ndcg_score as get_ndcg_ranking_score,
    roc_auc_score,
    f1_score,
    log_loss,
)


PREDICTION_FIELDS = [
    "sequence",
    "y_true",
    "y_pred",
    "split",
    "domain",
    "run_id",
    "model",
    "modality",
    "example_index",
    "config_path",
    "git_commit",
]


def best_eval_metric(eval_metric: str, best_metric: float, new_metric: float) -> float:
    if eval_metric in ["mse", "ece", "nll", "mae", "rmse", "mape"]:
        return min(1000 if best_metric < 0 else best_metric, new_metric)
    elif eval_metric in [
        "mrr",
        "ndcg",
        "sp",
        "erank",
        "mrr_ranking",
        "ndcg_ranking",
        "sp_ranking",
        "auroc",
        "acc",
        "f1",
        "explained_variance",
        "r2",
        "pearson",
        "recall_at_10pct",
        "ndcg_at_10pct",
    ]:
        return max(best_metric, new_metric)
    elif eval_metric in ["per_class_precision", "per_class_recall"]:
        sum_new = sum(new_metric.values())
        sum_best = (
            sum(best_metric.values()) if isinstance(best_metric, dict) else best_metric
        )
        return new_metric if sum_new > sum_best else best_metric
    else:
        raise ValueError(f"Unknown evaluation metric: {eval_metric}")


def get_mrr_score(labels: np.ndarray, predicted: np.ndarray) -> float:
    index_in_labels = np.argmax(labels)
    predicted_value = predicted[index_in_labels]
    predicted_rank = 0

    for i in range(len(predicted)):
        # The higher the predicted score, the more top the ranking is
        if predicted[i] >= predicted_value:
            predicted_rank += 1

    return 1 / predicted_rank


def get_mrr_ranking_score(labels: np.ndarray, predicted: np.ndarray) -> float:
    batch_size = predicted.shape[0]
    total_mrr = 0.0

    for i in range(batch_size):
        total_mrr += get_mrr_score(labels[i], predicted[i])

    return total_mrr / batch_size


def get_acc_score(labels: np.ndarray, predicted: np.ndarray) -> float:
    predicted = np.argmax(predicted, axis=1)
    acc = np.mean(labels == predicted)
    return acc


def get_per_class_precision_score(labels: np.ndarray, predicted: np.ndarray) -> dict:
    """
    Compute per-class precision.

    Precision = TP / (TP + FP)

    Args:
        labels: np.ndarray of shape (N,), true class labels.
        predicted: np.ndarray of shape (N, C), predicted logits or probabilities.

    Returns:
        Dictionary mapping class index to accuracy.
    """
    predicted_classes = np.argmax(predicted, axis=1)
    unique_classes = np.unique(labels)
    per_class_precision = {}

    for cls in unique_classes:
        cls_mask = predicted_classes == cls
        cls_total = np.sum(cls_mask)
        cls_correct = np.sum(labels[cls_mask] == cls)
        precision = cls_correct / cls_total if cls_total > 0 else 0.0
        per_class_precision[int(cls)] = precision

    return per_class_precision


def get_recall_at_10pct_score(labels: np.ndarray, predicted: np.ndarray) -> float:
    n = len(labels)
    k = max(1, int(n * 0.1))  # top 10% 样本数

    # 真实 top 10% positive 定义
    label_threshold = np.percentile(labels, 90)
    positive_idx = np.where(labels >= label_threshold)[0]

    # 按 predicted 排序，取 top-k 样本
    order = np.argsort(predicted)[::-1]
    topk_idx = order[:k]

    # top-k里命中了多少positive
    true_positives = np.intersect1d(topk_idx, positive_idx).size

    if positive_idx.size == 0:
        return 0.0
    return true_positives / positive_idx.size


def get_per_class_recall_score(labels: np.ndarray, predicted: np.ndarray) -> dict:
    """
    Compute per-class recall.

    Recall = TP / (TP + FN)

    Args:
        labels: np.ndarray of shape (N,), true class labels.
        predicted: np.ndarray of shape (N, C), predicted logits or probabilities.

    Returns:
        Dictionary mapping class index to recall.
    """
    predicted_classes = np.argmax(predicted, axis=1)
    unique_classes = np.unique(labels)
    per_class_recall = {}

    for cls in unique_classes:
        cls_mask = labels == cls
        cls_total = np.sum(cls_mask)
        cls_tp = np.sum(predicted_classes[cls_mask] == cls)
        recall = cls_tp / cls_total if cls_total > 0 else 0.0
        per_class_recall[int(cls)] = recall

    return per_class_recall


def get_ece_score(labels: np.ndarray, predicted: np.ndarray, n_bins: int = 15) -> float:
    """
    Compute Expected Calibration Error (ECE)

    Args:
        labels (np.ndarray): True labels, shape (N,)
        predicted (np.ndarray): Predicted probabilities, shape (N, C)
        n_bins (int): Number of bins to use for calibration

    Returns:
        float: Expected Calibration Error
    """
    confidences = np.max(predicted, axis=1)
    predictions = np.argmax(predicted, axis=1)
    accuracies = predictions == labels

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            acc_in_bin = np.mean(accuracies[in_bin])
            avg_conf_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(acc_in_bin - avg_conf_in_bin) * prop_in_bin

    return ece


def get_f1_score(labels: np.ndarray, predicted: np.ndarray) -> float:
    """
    Compute macro-averaged F1 score for multi-class classification.

    Args:
        labels (np.ndarray): True labels, shape (N,)
        predicted (np.ndarray): Predicted probabilities, shape (N, C)

    Returns:
        float: macro F1-score
    """
    predictions = np.argmax(predicted, axis=1)
    return f1_score(labels, predictions, average="macro")


def get_nll_score(labels: np.ndarray, predicted: np.ndarray) -> float:
    """
    Compute Negative Log Likelihood (NLL) via log loss.

    Args:
        labels (np.ndarray): True labels, shape (N,)
        predicted (np.ndarray): Predicted probabilities, shape (N, C)

    Returns:
        float: Negative Log Likelihood
    """
    return log_loss(labels, predicted)


def get_ndcg_score(labels: np.ndarray, predicted: np.ndarray) -> float:
    return get_ndcg_ranking_score([labels], [predicted])


def get_ndcg_at_10pct_score(labels: np.ndarray, predicted: np.ndarray) -> float:
    n = len(labels)
    k = max(1, int(n * 0.1))  # top 10% 样本数，至少是1个

    # min-max normalize labels（作为 continuous gain）
    gains = (labels - labels.min()) / (labels.max() - labels.min() + 1e-8)

    # 按 predicted 排序，取前k个样本
    order = np.argsort(predicted)[::-1]
    topk_idx = order[:k]

    dcg = np.sum(gains[topk_idx] / np.log2(np.arange(2, k + 2)))

    # 理想排序（按真实 gains 排）
    ideal_order = np.argsort(gains)[::-1]
    ideal_topk_idx = ideal_order[:k]
    idcg = np.sum(gains[ideal_topk_idx] / np.log2(np.arange(2, k + 2)))

    if idcg == 0:
        return 0.0
    return dcg / idcg


def get_mse_score(labels: np.ndarray, predicted: np.ndarray) -> float:
    return np.mean((labels - predicted) ** 2)


def get_rmse_score(labels: np.ndarray, predicted: np.ndarray) -> float:
    mse = get_mse_score(labels, predicted)
    return np.sqrt(mse)


def get_mae_score(labels: np.ndarray, predicted: np.ndarray) -> float:
    return np.mean(np.abs(labels - predicted))


def get_r2_score(labels: np.ndarray, predicted: np.ndarray) -> float:
    total_variance = np.var(labels)
    residual_variance = np.var(labels - predicted)
    return 1 - (residual_variance / total_variance)


def get_medae_score(labels: np.ndarray, predicted: np.ndarray) -> float:
    return np.median(np.abs(labels - predicted))


def get_mape_score(labels: np.ndarray, predicted: np.ndarray) -> float:
    epsilon = 1e-10
    return np.mean(np.abs((labels - predicted) / (labels + epsilon))) * 100


def get_explained_variance_score(labels: np.ndarray, predicted: np.ndarray) -> float:
    variance_labels = np.var(labels)
    variance_residuals = np.var(labels - predicted)
    return 1 - (variance_residuals / variance_labels)


def get_pearson_score(labels: np.ndarray, predicted: np.ndarray) -> float:
    pearson = np.corrcoef(labels, predicted)[0, 1]
    if math.isnan(pearson) or np.isnan(pearson):
        pearson = np.float64(0.0)
    return pearson


def get_sp_score(labels: np.ndarray, predicted: np.ndarray) -> float:
    spc, p = spearmanr(labels, predicted)
    if math.isnan(spc) or np.isnan(spc):
        spc = np.float64(0.0)
    return spc


def get_auroc_score(labels: np.ndarray, predicted: np.ndarray) -> float:
    return roc_auc_score(labels, predicted)


def get_sp_ranking_score(
    labels_batch: np.ndarray, predicted_batch: np.ndarray
) -> float:
    batch_size = predicted_batch.shape[0]
    total_sp = 0.0

    for i in range(batch_size):
        spc, p = spearmanr(labels_batch[i], predicted_batch[i])
        if math.isnan(spc) or np.isnan(spc):
            spc = np.float64(0.0)
        total_sp += spc

    return total_sp / batch_size


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


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return json_safe(value.detach().cpu().numpy())
    return value


def scalar_float(value):
    arr = np.asarray(value).reshape(-1)
    if len(arr) == 0:
        return float("nan")
    return float(arr[0])


def split_model_input(batch_data):
    if isinstance(batch_data, Mapping) and "input" in batch_data:
        return batch_data["input"], batch_data.get("metadata", {})
    return batch_data, {}


def to_list(value, n: int, default):
    if value is None:
        return [default] * n
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        value = value.reshape(-1).tolist()
    elif isinstance(value, (list, tuple)):
        value = list(value)
    else:
        return [value] * n

    if len(value) == n:
        return value
    if len(value) == 1:
        return value * n
    if len(value) < n:
        return value + [default] * (n - len(value))
    return value[:n]


def metadata_rows(metadata, n: int, mode: str, offset: int):
    if not isinstance(metadata, Mapping):
        metadata = {}
    sequences = to_list(metadata.get("sequence"), n, "")
    domains = to_list(metadata.get("domain"), n, "unknown")
    indices = to_list(metadata.get("example_index"), n, None)
    splits = to_list(metadata.get("split"), n, mode)
    rows = []
    for i in range(n):
        index = indices[i]
        if index is None:
            index = offset + i
        rows.append(
            {
                "sequence": str(sequences[i]),
                "domain": str(domains[i]),
                "example_index": int(index),
                "split": str(splits[i] or mode),
            }
        )
    return rows


def cfg_value(cfg, attr: str, default):
    return getattr(cfg, attr, default) if cfg is not None else default


def run_id_for(cfg):
    return str(cfg_value(cfg, "run_id", cfg_value(cfg, "tag", "run")))


def is_regression_output(model, raw_pred: torch.Tensor) -> bool:
    return (
        getattr(model, "regression", True)
        or raw_pred.ndim == 1
        or (raw_pred.ndim > 1 and raw_pred.shape[-1] == 1)
    )


def save_prediction_csv(cfg, mode: str, rows, labels_all, preds_all):
    prediction_dir = cfg_value(cfg, "prediction_dir", None)
    if not prediction_dir:
        return

    os.makedirs(prediction_dir, exist_ok=True)
    run_id = run_id_for(cfg)
    path = os.path.join(prediction_dir, f"{run_id}_{mode}.csv")
    labels = np.asarray(labels_all)
    preds = np.asarray(preds_all)
    flat_labels = labels.reshape(labels.shape[0], -1)
    flat_preds = preds.reshape(preds.shape[0], -1)

    model_name = cfg_value(cfg, "embed_name", cfg_value(cfg, "tag", "unknown"))
    modality = cfg_value(cfg, "seq_type", "unknown")
    config_path = cfg_value(cfg, "cfg_path", "unknown")
    git_commit = cfg_value(cfg, "git_commit", "unknown")

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PREDICTION_FIELDS)
        writer.writeheader()
        for i, row in enumerate(rows[: len(flat_labels)]):
            y_true = (
                float(flat_labels[i, 0])
                if flat_labels.shape[1] == 1
                else json.dumps([float(v) for v in flat_labels[i]])
            )
            y_pred = (
                float(flat_preds[i, 0])
                if flat_preds.shape[1] == 1
                else json.dumps([float(v) for v in flat_preds[i]])
            )
            writer.writerow(
                {
                    "sequence": row.get("sequence", ""),
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "split": row.get("split", mode),
                    "domain": row.get("domain", "unknown"),
                    "run_id": run_id,
                    "model": model_name,
                    "modality": modality,
                    "example_index": row.get("example_index", i),
                    "config_path": config_path,
                    "git_commit": git_commit,
                }
            )


def save_metric_json(cfg, mode: str, epoch: int, num_examples: int, total_score):
    metric_dir = cfg_value(cfg, "metric_dir", None)
    if not metric_dir:
        return

    os.makedirs(metric_dir, exist_ok=True)
    run_id = run_id_for(cfg)
    payload = {
        "run_id": run_id,
        "model": cfg_value(cfg, "embed_name", cfg_value(cfg, "tag", "unknown")),
        "modality": cfg_value(cfg, "seq_type", "unknown"),
        "hyperparameter_id": cfg_value(cfg, "hyperparameter_id", "default"),
        "hyperparameters": json_safe(cfg_value(cfg, "ablation", {})),
        "split": mode,
        "num_examples": int(num_examples),
        "metrics": json_safe(total_score),
        "config_path": cfg_value(cfg, "cfg_path", "unknown"),
        "git_commit": cfg_value(cfg, "git_commit", "unknown"),
        "epoch": int(epoch),
    }
    with open(os.path.join(metric_dir, f"{run_id}_{mode}.json"), "w") as f:
        json.dump(payload, f, indent=2)


def test_model(
    model,
    dataloader,
    wandb,
    epoch: int,
    mode: str,
    evaluation: List[str],
    loss_func,
    cfg=None,
):
    evaluation = normalize_evaluation(evaluation)

    model.eval()
    preds_all, labels_all, all_rows = [], [], []
    row_offset = 0
    for test_data, test_labels in tqdm(
        dataloader, desc=f"Epoch {epoch}, {mode}ing", dynamic_ncols=True
    ):
        model_input, metadata = split_model_input(test_data)
        with torch.no_grad():
            raw_pred = model(model_input).detach().to("cpu").to(dtype=torch.float32)
            if is_regression_output(model, raw_pred):
                predicted_scores = raw_pred.reshape(-1).numpy()
            else:
                predicted_scores = raw_pred.softmax(dim=-1).numpy()
            if not isinstance(test_labels, torch.Tensor):
                labels = torch.stack(test_labels).T.to("cpu").numpy()
            else:
                labels = test_labels.to("cpu").numpy()
            if predicted_scores.ndim == 1:
                labels = np.asarray(labels).reshape(-1)

        n_rows = len(np.asarray(labels).reshape(labels.shape[0], -1))
        all_rows.extend(metadata_rows(metadata, n_rows, mode, row_offset))
        row_offset += n_rows
        preds_all.append(predicted_scores)
        labels_all.append(labels)

    preds_all = np.concatenate(preds_all)
    labels_all = np.concatenate(labels_all)
    total_score = {}
    for eval_metric in evaluation:
        total_score[eval_metric] = globals()[f"get_{eval_metric}_score"](
            labels_all, preds_all
        )

    loss = loss_func(torch.tensor(labels_all), torch.tensor(preds_all))

    total_score = {
        eval_metric: json_safe(total_score[eval_metric])
        if isinstance(total_score[eval_metric], dict)
        else scalar_float(total_score[eval_metric])
        for eval_metric in evaluation
    }

    save_prediction_csv(cfg, mode, all_rows, labels_all, preds_all)
    save_metric_json(cfg, mode, epoch, len(labels_all), total_score)

    print(f"Epoch {epoch}, {mode}ing {len(labels_all)} samples: {total_score}")
    wandb.log(
        {
            "epoch": epoch,
            f"{mode}_loss": loss.item() if hasattr(loss, "item") else loss,
            **{
                f"{mode}_{eval_metric}": total_score[eval_metric]
                for eval_metric in evaluation
            },
        }
    )

    return total_score, loss
