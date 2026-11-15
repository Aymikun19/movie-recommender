import numpy as np
import pandas as pd

from src.recsys.data import k_core, temporal_split
from src.recsys.evaluate import coverage, precision_recall_at_k, rmse
from src.recsys.models import BiasedALS, Biases


def test_k_core_is_stable():
    df = pd.DataFrame({"user": [1, 1, 2, 2, 3], "item": [10, 11, 10, 11, 12], "rating": 5, "ts": 0})
    out = k_core(df, k=2)
    assert set(out["item"]) == {10, 11} and set(out["user"]) == {1, 2}


def test_temporal_split_holds_out_latest():
    df = pd.DataFrame({"user": [1] * 5, "item": range(5), "rating": 7, "ts": [5, 1, 4, 2, 3]})
    train, test = temporal_split(df, 0.2)
    assert list(test["ts"]) == [5] and len(train) == 4


def test_metrics():
    assert rmse([1, 2], [1, 4]) == np.sqrt(2)
    p, r = precision_recall_at_k({0: [1, 2, 3]}, {0: {2, 9}}, k=3)
    assert p == 1 / 3 and r == 0.5
    assert coverage({0: [1, 2], 1: [2]}, 4) == 0.5


def test_als_fits_a_low_rank_signal():
    rng = np.random.default_rng(1)
    P, Q = rng.normal(size=(40, 3)), rng.normal(size=(30, 3))
    u, i = np.divmod(np.arange(1200), 30)
    r = 5 + (P[u] * Q[i]).sum(1)
    base = Biases().fit(u, i, r, 40, 30)
    mf = BiasedALS(k=3, reg=0.1, iters=15).fit(u, i, r, 40, 30)
    assert rmse(r, mf.predict(u, i)) < 0.2 * rmse(r, base.predict(u, i))
