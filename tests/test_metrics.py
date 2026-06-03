import numpy as np

from scripts.validate_leaderboard_submission import ndcg_at_10pct, recall_at_10pct


def test_top_10_metrics_are_perfect_for_perfect_ranking():
    labels = np.arange(20, dtype=float)
    preds = labels.copy()

    assert recall_at_10pct(labels, preds) == 1.0
    assert ndcg_at_10pct(labels, preds) == 1.0
