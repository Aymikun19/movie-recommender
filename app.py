"""Streamlit demo: rate a few movies, get explained recommendations.  Run with:  streamlit run app.py"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.recsys.serve import because, cosine_neighbours, fold_in, year_of, zscore

ROOT = Path(__file__).parent
LOGO = ROOT / "assets" / "logo.png"
MODEL = ROOT / "models" / "als.npz"

st.set_page_config(page_title="Movie Recommender", page_icon=str(LOGO), layout="wide")
st.logo(str(LOGO), size="large")

if not MODEL.exists():
    st.error("Run `python -m src.recsys.train` first.")
    st.stop()


@st.cache_resource
def model():
    m = np.load(MODEL, allow_pickle=False)
    return {k: m[k] for k in m.files}


@st.cache_data
def catalogue(item_ids: tuple[int, ...]) -> pd.DataFrame:
    movies = pd.read_csv(ROOT / "data" / "movies.dat", sep="::", engine="python", names=["item", "title", "genres"], encoding="utf-8")
    movies["genres"] = movies["genres"].fillna("")
    g = movies.set_index("item")["genres"].to_dict()
    return pd.DataFrame({"item": item_ids, "genres": [g.get(i, "") for i in item_ids]})


M = model()
titles, Q, bi, mu = M["titles"], M["Q"], M["bi"], float(M["mu"])
cat = catalogue(tuple(int(x) for x in M["item_ids"]))
genres_of = [set(filter(None, s.split("|"))) for s in cat["genres"]]
all_genres = sorted(set().union(*genres_of) - {"Short", "News", "Reality-TV", "Talk-Show", "Game-Show"})
years = np.array([year_of(t) or 0 for t in titles])
by_title = {t: k for k, t in enumerate(titles)}
popular = [titles[k] for k in np.argsort(-M["counts"])[:500]]

st.title("Movie Recommender")
st.caption(
    f"Matrix factorisation (ALS, {Q.shape[1]} factors) trained on MovieTweetings and blended with what's trending. "
    f"{len(titles):,} movies in the model."
)

watch: dict[str, dict] = st.session_state.setdefault("watchlist", {})


def toggle_watch(title: str, predicted: float | None = None, reason: str = "") -> None:
    if title in watch:
        del watch[title]
    else:
        watch[title] = {"predicted": predicted, "reason": reason, "watched": False}


tab_rec, tab_similar, tab_watch, tab_about = st.tabs(["Get recommendations", "Find similar movies", f"Watchlist ({len(watch)})", "How it works"])

with tab_rec:
    left, right = st.columns([1, 1.4], gap="large")
    with left:
        st.subheader("1. Rate a few movies")
        picked = st.multiselect("Movies you've seen", sorted(popular), max_selections=15, placeholder="Start typing a title")
        ratings = {t: st.slider(t, 0, 10, 8, key=f"r_{t}") for t in picked}
        st.subheader("2. Tune the results")
        alpha = st.slider("Trending boost", 0.0, 8.0, float(M["alpha"]), help="0 = pure personal taste; higher favours movies people are watching right now")
        want = st.multiselect("Only these genres", all_genres, placeholder="Any genre")
        avoid = st.multiselect("Never show these genres", all_genres, default=["Horror"] if "Horror" in all_genres else None, placeholder="None")
        lo, hi = int(years[years > 0].min()), int(years.max())
        yr = st.slider("Release year", lo, hi, (max(lo, 1980), hi))
        n = st.select_slider("How many", [5, 10, 15, 20], 10)

    with right:
        if len(ratings) < 2:
            st.info("Rate at least two movies on the left to get recommendations.")
        else:
            idx = {by_title[t]: v for t, v in ratings.items()}
            p, bu = fold_in(idx, Q, bi, mu, float(M["reg"]), float(M["f_reg"]))
            pred = mu + bu + bi + Q @ p
            score = zscore(pred) + alpha * zscore(M["trending"])
            mask = (years >= yr[0]) & (years <= yr[1])
            if want:
                mask &= np.array([bool(g & set(want)) for g in genres_of])
            if avoid:
                mask &= np.array([not (g & set(avoid)) for g in genres_of])
            mask[list(idx)] = False
            score[~mask] = -np.inf
            liked = [k for k, v in idx.items() if v >= 7] or list(idx)
            recs = [k for k in np.argsort(-score)[:n] if np.isfinite(score[k])]

            st.subheader("You might like")
            if not recs:
                st.warning("Nothing matches those filters. Try widening the year range or genres.")
            for rank, k in enumerate(recs, 1):
                src, sim = because(Q, k, liked)
                with st.container(border=True):
                    a, b = st.columns([4, 1])
                    why = f"Because you liked {titles[src]}" if sim > 0.2 else "Popular with people who rate like you"
                    a.markdown(f"**{rank}. {titles[k]}**")
                    a.caption(" · ".join(sorted(genres_of[k])) or "Genre unknown")
                    a.markdown(f"Because you liked *{titles[src]}*" if sim > 0.2 else why)
                    b.metric("Predicted", f"{np.clip(pred[k], 0, 10):.1f}/10")
                    saved = titles[k] in watch
                    b.button("Saved" if saved else "Save", key=f"w_{k}", icon=":material/bookmark_added:" if saved else ":material/bookmark_add:",
                             on_click=toggle_watch, args=(titles[k], round(float(np.clip(pred[k], 0, 10)), 1), why), width="stretch")

            if recs:
                st.subheader("Your taste profile")
                rated = pd.Series([g for k in idx for g in genres_of[k]]).value_counts(normalize=True)
                recd = pd.Series([g for k in recs for g in genres_of[k]]).value_counts(normalize=True)
                prof = pd.DataFrame({"Movies you rated": rated, "Recommendations": recd}).fillna(0).sort_values("Movies you rated", ascending=False).head(8)
                st.bar_chart(prof, horizontal=True, stack=False, height=300)
                st.download_button(
                    "Download this list (CSV)",
                    pd.DataFrame({"rank": range(1, len(recs) + 1), "title": [titles[k] for k in recs], "predicted_rating": [round(float(np.clip(pred[k], 0, 10)), 1) for k in recs]}).to_csv(index=False).encode(),
                    "recommendations.csv",
                    "text/csv",
                )

with tab_similar:
    st.subheader("Movies like…")
    st.caption("Nearest neighbours in the learned taste space: movies the same people tend to rate the same way.")
    seed = st.selectbox("Pick a movie", sorted(popular), index=None, placeholder="Start typing a title")
    if seed:
        k = by_title[seed]
        rows = [{"Movie": titles[j], "Genres": ", ".join(sorted(genres_of[j])), "Similarity": s, "Ratings in data": int(M["counts"][j])} for j, s in cosine_neighbours(Q, k, 12)]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                     column_config={"Similarity": st.column_config.ProgressColumn("Similarity", min_value=0, max_value=1, format="%.2f")})

with tab_watch:
    st.subheader("Your watchlist")
    st.caption("Save recommendations to watch later. The list lives in this browser tab, so download it to keep it, and upload it next time to pick up where you left off.")
    up = st.file_uploader("Restore a saved watchlist (CSV)", type="csv", key="watch_up")
    if up is not None and st.session_state.get("watch_loaded") != up.file_id:
        try:
            df_up = pd.read_csv(up)
            for _, r in df_up.iterrows():
                t = str(r.get("title", "")).strip()
                if t:
                    watch[t] = {"predicted": r.get("predicted_rating"), "reason": str(r.get("reason", "") if pd.notna(r.get("reason", "")) else ""),
                                "watched": str(r.get("watched", "")).lower() in ("true", "yes", "1")}
            st.session_state["watch_loaded"] = up.file_id
            st.success(f"Loaded {len(df_up)} movies.")
        except Exception:  # noqa: BLE001 - show a friendly message for any malformed upload
            st.error("That file doesn't look like a watchlist export.")
    if not watch:
        st.info("Nothing saved yet. Press Save on any recommendation.")
    else:
        df = pd.DataFrame([{"title": t, "predicted_rating": v["predicted"], "reason": v["reason"], "watched": v["watched"]} for t, v in watch.items()])
        edited = st.data_editor(df, hide_index=True, width="stretch", disabled=["title", "predicted_rating", "reason"], key="watch_editor",
                                column_config={"title": "Movie", "predicted_rating": st.column_config.NumberColumn("Predicted", format="%.1f"),
                                               "reason": "Why", "watched": st.column_config.CheckboxColumn("Watched")})
        for t, w in zip(edited["title"], edited["watched"]):
            watch[t]["watched"] = bool(w)
        done = int(edited["watched"].sum())
        c1, c2, c3 = st.columns(3)
        c1.metric("Saved", len(edited))
        c2.metric("Watched", done)
        c3.metric("Still to watch", len(edited) - done)
        d1, d2 = st.columns(2)
        d1.download_button("Download watchlist (CSV)", edited.to_csv(index=False).encode(), "watchlist.csv", "text/csv", type="primary", width="stretch")
        if d2.button("Remove watched", width="stretch", disabled=done == 0):
            for t in [t for t, v in watch.items() if v["watched"]]:
                del watch[t]
            st.rerun()

with tab_about:
    st.subheader("How it works")
    st.markdown(
        """
1. **Learning taste.** Alternating least squares factorises the ratings matrix into a short vector for every movie. Movies with similar vectors are liked by the same people.
2. **Folding you in.** Your ratings are turned into your own vector with one regularised least-squares step, without retraining the model.
3. **Blending with trends.** Predicted ratings are standardised and mixed with a recent-popularity score. The trending boost slider sets the mix.
4. **Explaining picks.** Each recommendation names the movie you liked whose vector is closest to it.

The evaluation (held-out *future* ratings) is in the [README](https://github.com/Aymikun19/movie-recommender#results-held-out-future-ratings): low RMSE alone does not mean good top-10 lists, and the trending-aware hybrid beats every baseline on recall@10.
"""
    )
