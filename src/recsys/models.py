"""Rating-prediction models, from simple baselines to biased matrix factorisation (ALS)."""
from __future__ import annotations

import numpy as np
import pandas as pd


class Index:
    """Maps raw user/item ids to contiguous integer indices."""

    def __init__(self, train: pd.DataFrame):
        self.users = {u: i for i, u in enumerate(train["user"].unique())}
        self.items = {m: i for i, m in enumerate(train["item"].unique())}
        self.item_ids = np.array(list(self.items))

    def encode(self, df: pd.DataFrame):
        u = df["user"].map(self.users)
        i = df["item"].map(self.items)
        ok = u.notna() & i.notna()
        return u[ok].astype(int).to_numpy(), i[ok].astype(int).to_numpy(), df.loc[ok, "rating"].to_numpy(float), ok.to_numpy()


class GlobalMean:
    name = "Global mean"

    def fit(self, u, i, r, n_users, n_items):
        self.mu = r.mean()
        return self

    def predict(self, u, i):
        return np.full(len(u), self.mu)


class Biases:
    """r̂ = μ + b_u + b_i, with L2-regularised biases fitted by alternating closed-form updates."""

    name = "User + item biases"

    def __init__(self, reg: float = 3.0, iters: int = 10):
        self.reg, self.iters = reg, iters

    def fit(self, u, i, r, n_users, n_items):
        self.mu = r.mean()
        self.bu, self.bi = np.zeros(n_users), np.zeros(n_items)
        for _ in range(self.iters):
            res = r - self.mu - self.bu[u]
            self.bi = np.bincount(i, res, n_items) / (np.bincount(i, minlength=n_items) + self.reg)
            res = r - self.mu - self.bi[i]
            self.bu = np.bincount(u, res, n_users) / (np.bincount(u, minlength=n_users) + self.reg)
        return self

    def predict(self, u, i):
        return self.mu + self.bu[u] + self.bi[i]


class BiasedALS(Biases):
    """Biases plus low-rank user/item factors on the residuals, fitted with Alternating Least Squares."""

    name = "Matrix factorisation (ALS)"

    def __init__(self, k: int = 16, reg: float = 20.0, bias_reg: float = 3.0, iters: int = 12, seed: int = 0):
        super().__init__(bias_reg)
        self.k, self.f_reg, self.f_iters, self.seed = k, reg, iters, seed

    def fit(self, u, i, r, n_users, n_items):
        super().fit(u, i, r, n_users, n_items)
        res = r - super().predict(u, i)
        rng = np.random.default_rng(self.seed)
        self.P = rng.normal(0, 0.1, (n_users, self.k))
        self.Q = rng.normal(0, 0.1, (n_items, self.k))
        by_user = [np.flatnonzero(u == x) for x in range(n_users)]
        by_item = [np.flatnonzero(i == x) for x in range(n_items)]
        eye = self.f_reg * np.eye(self.k)
        self.history = []
        for _ in range(self.f_iters):
            for x, idx in enumerate(by_user):
                Qx = self.Q[i[idx]]
                self.P[x] = np.linalg.solve(Qx.T @ Qx + eye, Qx.T @ res[idx])
            for x, idx in enumerate(by_item):
                Px = self.P[u[idx]]
                self.Q[x] = np.linalg.solve(Px.T @ Px + eye, Px.T @ res[idx])
            self.history.append(float(np.sqrt(np.mean((res - np.sum(self.P[u] * self.Q[i], 1)) ** 2))))
        return self

    def predict(self, u, i):
        return super().predict(u, i) + np.sum(self.P[u] * self.Q[i], axis=1)

    def fold_in(self, item_idx: np.ndarray, ratings: np.ndarray) -> np.ndarray:
        """Score all items for a brand-new user from a few ratings (used by the app)."""
        bu = np.sum(ratings - self.mu - self.bi[item_idx]) / (len(ratings) + self.reg)
        res = ratings - self.mu - bu - self.bi[item_idx]
        Qx = self.Q[item_idx]
        p = np.linalg.solve(Qx.T @ Qx + self.f_reg * np.eye(self.k), Qx.T @ res)
        return self.mu + bu + self.bi + self.Q @ p
