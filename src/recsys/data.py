"""Load MovieTweetings (100K snapshot), apply a k-core filter and make a per-user temporal split."""
from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

BASE = "https://raw.githubusercontent.com/sidooms/MovieTweetings/master/snapshots/100K/"
DATA = Path(__file__).resolve().parents[2] / "data"


def _get(name: str) -> Path:
    p = DATA / name
    if not p.exists():
        DATA.mkdir(exist_ok=True)
        urlretrieve(BASE + name, p)
    return p


def load_ratings() -> pd.DataFrame:
    return pd.read_csv(_get("ratings.dat"), sep="::", engine="python", names=["user", "item", "rating", "ts"])


def load_movies() -> pd.DataFrame:
    m = pd.read_csv(_get("movies.dat"), sep="::", engine="python", names=["item", "title", "genres"], encoding="utf-8")
    m["genres"] = m["genres"].fillna("")
    return m


def k_core(df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """Repeatedly drop users and items with fewer than k ratings until the data is stable."""
    while True:
        u = df["user"].map(df["user"].value_counts())
        i = df["item"].map(df["item"].value_counts())
        kept = df[(u >= k) & (i >= k)]
        if len(kept) == len(df):
            return kept.reset_index(drop=True)
        df = kept


def temporal_split(df: pd.DataFrame, test_frac: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """For each user, hold out their most recent ratings: it simulates predicting the future."""
    df = df.sort_values(["user", "ts"])
    rank = df.groupby("user").cumcount(ascending=False)  # 0 = newest
    n = df.groupby("user")["item"].transform("size")
    test_mask = rank < (n * test_frac).round().clip(lower=1)
    return df[~test_mask].reset_index(drop=True), df[test_mask].reset_index(drop=True)
