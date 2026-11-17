"""Inference helpers for the Streamlit app: fold in a new user, explain and filter recommendations."""
from __future__ import annotations

import re

import numpy as np


def fold_in(ratings: dict[int, float], Q: np.ndarray, bi: np.ndarray, mu: float, reg: float, f_reg: float) -> tuple[np.ndarray, float]:
    """Solve for a new user's bias and factor vector from a handful of ratings (one ALS half-step)."""
    idx = np.array(list(ratings))
    r = np.array(list(ratings.values()), float)
    bu = float(np.sum(r - mu - bi[idx]) / (len(r) + reg))
    res = r - mu - bu - bi[idx]
    p = np.linalg.solve(Q[idx].T @ Q[idx] + f_reg * np.eye(Q.shape[1]), Q[idx].T @ res)
    return p, bu


def zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-9)


def cosine_neighbours(Q: np.ndarray, k: int, n: int = 10) -> list[tuple[int, float]]:
    """Movies whose factor vectors point the same way as movie k."""
    norms = np.linalg.norm(Q, axis=1) + 1e-9
    sims = (Q @ Q[k]) / (norms * norms[k])
    sims[k] = -np.inf
    top = np.argsort(-sims)[:n]
    return [(int(j), float(sims[j])) for j in top]


def because(Q: np.ndarray, rec: int, liked: list[int]) -> tuple[int, float]:
    """The liked movie most similar to a recommendation, used as a one-line explanation."""
    norms = np.linalg.norm(Q, axis=1) + 1e-9
    sims = [(j, float(Q[rec] @ Q[j] / (norms[rec] * norms[j]))) for j in liked]
    return max(sims, key=lambda t: t[1])


def year_of(title: str) -> int | None:
    m = re.search(r"\((\d{4})\)\s*$", title)
    return int(m.group(1)) if m else None
