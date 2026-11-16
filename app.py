"""Streamlit demo: rate a few movies, get recommendations.  Run with:  streamlit run app.py"""
from pathlib import Path

import numpy as np
import streamlit as st

LOGO = Path(__file__).parent / "assets" / "logo.png"

MODEL = Path(__file__).parent / "models" / "als.npz"
st.set_page_config(page_title="Movie Recommender", page_icon=str(LOGO))
st.logo(str(LOGO), size="large")
st.title("Movie Recommender")
st.caption("Matrix factorisation (ALS) on MovieTweetings, blended with what's trending. See the README for the evaluation.")

if not MODEL.exists():
    st.error("Run `python -m src.recsys.train` first.")
    st.stop()
m = np.load(MODEL, allow_pickle=False)
titles, Q, bi, mu = m["titles"], m["Q"], m["bi"], float(m["mu"])
by_title = {t: k for k, t in enumerate(titles)}
popular = [titles[k] for k in np.argsort(-m["counts"])[:400]]

picked = st.multiselect("Pick movies you've seen", sorted(popular), max_selections=15)
ratings = {t: st.slider(t, 0, 10, 8, key=t) for t in picked}
alpha = st.slider("Trending boost", 0.0, 8.0, float(m["alpha"]), help="0 = pure personal taste (ALS); higher = favour movies people are watching right now")

if st.button("Recommend", type="primary", disabled=len(ratings) < 2):
    idx = np.array([by_title[t] for t in ratings])
    r = np.array(list(ratings.values()), float)
    reg, f_reg = float(m["reg"]), float(m["f_reg"])
    bu = np.sum(r - mu - bi[idx]) / (len(r) + reg)
    res = r - mu - bu - bi[idx]
    p = np.linalg.solve(Q[idx].T @ Q[idx] + f_reg * np.eye(Q.shape[1]), Q[idx].T @ res)
    pred = mu + bu + bi + Q @ p
    z = lambda x: (x - x.mean()) / (x.std() + 1e-9)
    score = z(pred) + alpha * z(m["trending"])
    score[idx] = -np.inf
    st.subheader("You might like")
    for rank, k in enumerate(np.argsort(-score)[:10], 1):
        st.write(f"**{rank}. {titles[k]}** · predicted rating {np.clip(pred[k], 0, 10):.1f}/10")
