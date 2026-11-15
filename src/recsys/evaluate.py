"""Ranking metrics and popularity-bias diagnostics."""
from __future__ import annotations

import numpy as np


def rmse(y, yhat) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2)))


def precision_recall_at_k(recs: dict[int, list[int]], relevant: dict[int, set[int]], k: int = 10):
    ps, rs = [], []
    for user, rel in relevant.items():
        if not rel:
            continue
        hits = len(set(recs.get(user, [])[:k]) & rel)
        ps.append(hits / k)
        rs.append(hits / len(rel))
    return float(np.mean(ps)), float(np.mean(rs))


def coverage(recs: dict[int, list[int]], n_items: int) -> float:
    """Share of the catalogue that appears in at least one user's recommendations."""
    return len({i for r in recs.values() for i in r}) / n_items


def avg_popularity_percentile(recs: dict[int, list[int]], item_counts: np.ndarray) -> float:
    """Mean popularity percentile of recommended items (100 = the most-rated movie)."""
    pct = item_counts.argsort().argsort() / (len(item_counts) - 1) * 100
    return float(np.mean([pct[i] for r in recs.values() for i in r]))
