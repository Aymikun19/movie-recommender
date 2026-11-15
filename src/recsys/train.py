"""Train baselines, ALS and a trending-aware hybrid; evaluate accuracy, top-10 ranking and popularity bias.

Usage:  python -m src.recsys.train
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import k_core, load_movies, load_ratings, temporal_split
from .evaluate import avg_popularity_percentile, coverage, precision_recall_at_k, rmse
from .models import BiasedALS, Biases, GlobalMean, Index

ROOT = Path(__file__).resolve().parents[2]
K = 10
RELEVANT = 8          # a "liked" movie: rated 8/10 or higher
TRENDING_DAYS = 30


def z(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-9)


def top_k(scores: np.ndarray, seen: set[int], k: int = K) -> list[int]:
    s = scores.astype(float).copy()
    if seen:
        s[list(seen)] = -np.inf
    part = np.argpartition(-s, k)[:k]
    return [int(x) for x in part[np.argsort(-s[part])]]


class Setup:
    """Everything needed to train and score on one train/test split."""

    def __init__(self, train: pd.DataFrame, test: pd.DataFrame):
        self.idx = Index(train)
        self.u, self.i, self.r, _ = self.idx.encode(train)
        self.tu, self.ti, self.tr, _ = self.idx.encode(test)
        self.n_u, self.n_i = len(self.idx.users), len(self.idx.items)
        self.counts = np.bincount(self.i, minlength=self.n_i)
        recent = train[train["ts"] >= train["ts"].max() - TRENDING_DAYS * 86400]
        ri = recent["item"].map(self.idx.items).dropna().astype(int)
        self.trending = np.log1p(np.bincount(ri, minlength=self.n_i))
        self.seen: dict[int, set[int]] = {}
        for a, b in zip(self.u, self.i):
            self.seen.setdefault(int(a), set()).add(int(b))
        self.relevant: dict[int, set[int]] = {}
        for a, b, y in zip(self.tu, self.ti, self.tr):
            if y >= RELEVANT:
                self.relevant.setdefault(int(a), set()).add(int(b))

    def rank(self, score_fn) -> dict[int, list[int]]:
        return {x: top_k(score_fn(x), self.seen.get(x, set())) for x in self.relevant}

    def ranking_metrics(self, recs) -> dict:
        p, r = precision_recall_at_k(recs, self.relevant, K)
        return {f"precision@{K}": p, f"recall@{K}": r, "coverage": coverage(recs, self.n_i),
                "avg_popularity_pct": avg_popularity_percentile(recs, self.counts)}


def hybrid_scorer(s: Setup, als: BiasedALS, alpha: float):
    items = np.arange(s.n_i)
    trend = z(s.trending)
    return lambda x: z(als.predict(np.full(s.n_i, x), items)) + alpha * trend


def tune_alpha(train: pd.DataFrame) -> tuple[float, list[tuple[float, float]]]:
    """Pick the trending weight on a validation split carved from the training data only."""
    fit, val = temporal_split(train, 0.2)
    s = Setup(fit, val)
    als = BiasedALS().fit(s.u, s.i, s.r, s.n_u, s.n_i)
    curve = []
    for a in [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]:
        p, _ = precision_recall_at_k(s.rank(hybrid_scorer(s, als, a)), s.relevant, K)
        curve.append((a, p))
    return max(curve, key=lambda t: t[1])[0], curve


def main() -> None:
    raw = load_ratings()
    data = k_core(raw, 5)
    train, test = temporal_split(data, 0.2)
    alpha, curve = tune_alpha(train)
    s = Setup(train, test)
    print(f"{len(raw):,} ratings → {len(data):,} after 5-core ({s.n_u:,} users, {s.n_i:,} movies). Hybrid alpha={alpha}")

    items = np.arange(s.n_i)
    gm = GlobalMean().fit(s.u, s.i, s.r, s.n_u, s.n_i)
    bias = Biases().fit(s.u, s.i, s.r, s.n_u, s.n_i)
    als = BiasedALS().fit(s.u, s.i, s.r, s.n_u, s.n_i)

    rows = [
        {"model": "Global mean", "rmse": rmse(s.tr, gm.predict(s.tu, s.ti))},
        {"model": "Most popular (all time)", **s.ranking_metrics(s.rank(lambda x: s.counts))},
        {"model": f"Trending (last {TRENDING_DAYS} days)", **s.ranking_metrics(s.rank(lambda x: s.trending))},
        {"model": "User + item biases", "rmse": rmse(s.tr, bias.predict(s.tu, s.ti)),
         **s.ranking_metrics(s.rank(lambda x: bias.predict(np.full(s.n_i, x), items)))},
        {"model": "Matrix factorisation (ALS)", "rmse": rmse(s.tr, als.predict(s.tu, s.ti)),
         **s.ranking_metrics(s.rank(lambda x: als.predict(np.full(s.n_i, x), items)))},
        {"model": f"Hybrid: ALS + trending (α={alpha})", **s.ranking_metrics(s.rank(hybrid_scorer(s, als, alpha)))},
    ]
    for row in rows:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})

    reports = ROOT / "reports"
    (reports / "figures").mkdir(parents=True, exist_ok=True)
    (reports / "metrics.json").write_text(json.dumps({
        "data": {"raw_ratings": len(raw), "kcore_ratings": len(data), "users": s.n_u, "movies": s.n_i,
                 "test_ratings": int(len(s.tr)), "test_users_with_liked_movies": len(s.relevant)},
        "hybrid_alpha_validation_curve": curve,
        "results": rows,
    }, indent=2))

    ranked = [r for r in rows if f"precision@{K}" in r]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    names = [r["model"].split(" (")[0].replace("Matrix factorisation", "ALS").replace("Hybrid: ALS + trending", "Hybrid") for r in ranked]
    axes[0].barh(names, [r[f"recall@{K}"] for r in ranked], color="#e11d48")
    axes[0].set_title(f"Recall@{K} on future liked movies")
    axes[1].scatter([r["avg_popularity_pct"] for r in ranked], [r["coverage"] * 100 for r in ranked], s=60, color="#0f766e")
    for n, r in zip(names, ranked):
        axes[1].annotate(n, (r["avg_popularity_pct"], r["coverage"] * 100), fontsize=8, xytext=(4, 4), textcoords="offset points")
    axes[1].set_xlabel("Avg popularity percentile of recs"); axes[1].set_ylabel("Catalogue coverage (%)")
    axes[1].set_title("Popularity bias vs. diversity")
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(reports / "figures" / "model_comparison.png", dpi=150)

    movies = load_movies().set_index("item")
    titles = [movies["title"].get(m, str(m)) for m in s.idx.item_ids]
    (ROOT / "models").mkdir(exist_ok=True)
    np.savez_compressed(ROOT / "models" / "als.npz", mu=als.mu, bi=als.bi, Q=als.Q, reg=als.reg, f_reg=als.f_reg,
                        item_ids=s.idx.item_ids, titles=np.array(titles), counts=s.counts, trending=s.trending, alpha=alpha)
    print("Saved model + reports.")


if __name__ == "__main__":
    main()
