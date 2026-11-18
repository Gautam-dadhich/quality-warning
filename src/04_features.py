"""Step 4 - build model inputs for every labelled product-month.

Adds to the structured features from step 2:
  * complaint-aspect shares in the past window and their change vs the
    preceding window (asp_<aspect>_share, asp_<aspect>_change)
  * low-rating share in the preceding window and its change
  * CUSUM statistics on monthly low-rating share (one column per slack k)
  * one text document per window (for TF-IDF in step 6)

All inputs use only months strictly before the decision month.

Outputs (data/processed/):
  features.*      labelled windows: ids, split, label, all numeric inputs
  window_docs.*   product_id, decision_date, doc

Usage:
    python src/04_features.py
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from aspects import ASPECTS, tag_aspects
from utils import load_config, load_table, resolve, save_table

ASP_COLS = [f"asp_{a}" for a in ASPECTS] + ["asp_any"]


def month_index(dates: pd.Series) -> np.ndarray:
    # int64 everywhere: pandas may give int32 year/month, and merge_asof needs identical key dtypes
    return (dates.dt.year * 12 + dates.dt.month - 1).to_numpy().astype("int64")


# ------------------------------------------------------------------ CUSUM
def cusum_scores(pm: pd.DataFrame, ks: list[float]) -> pd.DataFrame:
    """One-sided upper CUSUM on each product's monthly low-rating share.

    For month t (months with >=1 review), x_t = low share in t and
    mu_t = pooled low share over all of the product's months BEFORE t.
    S_t = max(0, S_{t-1} + x_t - mu_t - k), S = 0 in the first month.
    Returns pm rows with columns cusum_k<k> = S_t (to be looked up at m-1).
    """
    pm = pm.sort_values(["product_id", "mi"]).reset_index(drop=True)
    x = (pm["n_low"] / pm["n"]).to_numpy()
    g = pm.groupby("product_id", sort=False)
    cum_low_before = g["n_low"].cumsum() - pm["n_low"]
    cum_n_before = g["n"].cumsum() - pm["n"]
    mu = (cum_low_before / cum_n_before.replace(0, np.nan)).to_numpy()
    first = (cum_n_before == 0).to_numpy()
    pid_codes = pm["product_id"].astype("category").cat.codes.to_numpy()
    new_prod = np.r_[True, pid_codes[1:] != pid_codes[:-1]]

    out = pm[["product_id", "mi"]].copy()
    for k in ks:
        S = np.zeros(len(pm))
        s = 0.0
        for i in range(len(pm)):
            if new_prod[i] or first[i]:
                s = 0.0
            else:
                s = max(0.0, s + x[i] - mu[i] - k)
            S[i] = s
        out[f"cusum_k{k:g}"] = S
    return out


# ------------------------------------------------------------------ main build
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    P = int(cfg["windows"]["past_months"])
    low_max = cfg["label"]["low_rating_max"]
    fcfg = cfg["features"]
    t0 = time.time()

    print("Loading windows and clean reviews ...")
    win = load_table(cfg["data"]["windows"], parse_dates=["decision_date"])
    win = win[win["label"].notna()].reset_index(drop=True)
    win["mi"] = month_index(win["decision_date"])
    rev = load_table(cfg["data"]["clean_reviews"], parse_dates=["date"])
    rev["title"] = rev["title"].fillna("").astype(str)   # empty strings come back as NaN from CSV
    rev["text"] = rev["text"].fillna("").astype(str)
    rev["mi"] = month_index(rev["date"])
    rev["low"] = (rev["rating"] <= low_max).astype("int8")
    print(f"  labelled windows: {len(win):,}   reviews: {len(rev):,}")

    print("Tagging complaint aspects (regex lexicon) ...")
    asp = tag_aspects(rev["title"], rev["text"])
    rev = pd.concat([rev, asp], axis=1)
    rates = asp.mean().mul(100).round(2)
    resolve(cfg.get("results_dir", "results")).mkdir(parents=True, exist_ok=True)
    rates.rename("pct_reviews_tagged").to_csv(resolve(cfg.get("results_dir", "results")) / "aspect_prevalence.csv")
    print("  % of reviews tagged per aspect:\n" + rates.to_string())

    # product-month table
    pm = (rev.groupby(["product_id", "mi"])
             .agg(n=("rating", "size"), n_low=("low", "sum"), **{c: (c, "sum") for c in ASP_COLS})
             .reset_index())

    # ---- past / previous window sums by merging individual months
    def window_sum(lags: range) -> pd.DataFrame:
        acc = None
        for k in lags:
            m = win[["product_id", "mi"]].assign(mi=win["mi"] - k).merge(
                pm, on=["product_id", "mi"], how="left")[["n", "n_low"] + ASP_COLS].fillna(0)
            acc = m if acc is None else acc + m.values
        return acc.reset_index(drop=True)

    past = window_sum(range(1, P + 1))
    prev = window_sum(range(P + 1, 2 * P + 1))
    # consistency check with step 2
    assert np.allclose(past["n"].to_numpy(), win["n_past"].to_numpy()), "past-window counts disagree with step 2"

    feats = win.drop(columns=["mi"]).copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        prev_low_share = prev["n_low"] / prev["n"]
        feats["low_share_prev"] = prev_low_share.fillna(feats["low_share_past"]).to_numpy()
        feats["low_share_change"] = feats["low_share_past"] - feats["low_share_prev"]
        for c in ASP_COLS:
            share_past = (past[c] / past["n"]).to_numpy()
            share_prev = (prev[c] / prev["n"]).to_numpy()
            share_prev = np.where(np.isnan(share_prev), share_past, share_prev)
            feats[f"{c}_share"] = share_past
            feats[f"{c}_change"] = share_past - share_prev
            # aspect share among LOW-rated reviews only (what low raters complain about)
            feats[f"{c}_share_low"] = 0.0  # filled below
    # aspect counts among low-rated reviews in the past window
    pm_low = (rev[rev["low"] == 1].groupby(["product_id", "mi"])[ASP_COLS].sum().reset_index())
    acc = None
    for k in range(1, P + 1):
        m = win[["product_id", "mi"]].assign(mi=win["mi"] - k).merge(
            pm_low, on=["product_id", "mi"], how="left")[ASP_COLS].fillna(0)
        acc = m if acc is None else acc + m.values
    nlow = feats["n_low_past"].replace(0, np.nan).to_numpy()
    for c in ASP_COLS:
        feats[f"{c}_share_low"] = np.nan_to_num(acc[c].to_numpy() / nlow)

    # ---- CUSUM, looked up at the last review month strictly before m
    print("Computing CUSUM statistics ...")
    cs = cusum_scores(pm[["product_id", "mi", "n", "n_low"]], cfg["baselines"]["cusum_k"])
    cs_cols = [c for c in cs.columns if c.startswith("cusum_")]
    look = win[["product_id", "mi"]].reset_index().rename(columns={"index": "row"})
    look["mi_lookup"] = (look["mi"] - 1).astype("int64")
    cs = cs.rename(columns={"mi": "mi_lookup"})
    cs["mi_lookup"] = cs["mi_lookup"].astype("int64")
    merged = pd.merge_asof(look.sort_values("mi_lookup"), cs.sort_values("mi_lookup"),
                           on="mi_lookup", by="product_id", direction="backward")
    merged = merged.set_index("row").sort_index()
    for c in cs_cols:
        feats[c] = merged[c].fillna(0).to_numpy()

    p = save_table(feats, fcfg["features_table"])
    print(f"Saved features -> {p}  ({len(feats):,} rows, {feats.shape[1]} columns)")

    # ---- text documents per window
    print("Building text documents per window ...")
    need = set()
    for k in range(1, P + 1):
        need.update(zip(win["product_id"], win["mi"] - k))
    key = pd.MultiIndex.from_arrays([rev["product_id"], rev["mi"]])
    r = rev.loc[key.isin(list(need)), ["product_id", "mi", "rating", "date", "title", "text"]].copy()
    r = r.sort_values(["product_id", "mi", "rating", "date"], ascending=[True, True, True, False])
    r = r.groupby(["product_id", "mi"]).head(int(fcfg["max_reviews_per_month_doc"]))
    mc = int(fcfg["max_chars_per_review"])
    r["snip"] = (r["title"].str.strip() + ". " + r["text"].str.strip()).str.slice(0, mc)
    month_docs = r.groupby(["product_id", "mi"])["snip"].agg(" ".join).rename("doc").reset_index()
    docs = None
    for k in range(1, P + 1):
        m = win[["product_id", "mi"]].assign(mi=win["mi"] - k).merge(
            month_docs, on=["product_id", "mi"], how="left")["doc"].fillna("")
        docs = m if docs is None else docs + " " + m
    docs_df = pd.DataFrame({"product_id": win["product_id"], "decision_date": win["decision_date"],
                            "doc": docs.str.strip()})
    p = save_table(docs_df, fcfg["docs_table"])
    print(f"Saved documents -> {p}  (median length {int(docs_df['doc'].str.len().median()):,} chars)")
    print(f"Done in {time.time() - t0:.0f}s. Next: python src/05_baselines.py")


if __name__ == "__main__":
    main()
