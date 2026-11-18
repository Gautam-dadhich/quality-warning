"""Brute-force checks for step 4 features (aspect shares, low-share change, CUSUM).

    python tests/test_features.py [--config tests/config_synthetic.yaml] [--n 300]

CUSUM is recomputed for each sampled row from ONLY the reviews dated before the
decision date, so a match proves the feature uses no current/future information.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aspects import tag_aspects  # noqa: E402
from utils import load_config, load_table  # noqa: E402


def cusum_from_reviews(r: pd.DataFrame, low_max: int, k: float) -> float:
    if r.empty:
        return 0.0
    m = r.assign(mi=r["date"].dt.year * 12 + r["date"].dt.month - 1,
                 low=(r["rating"] <= low_max).astype(int))
    pm = m.groupby("mi").agg(n=("low", "size"), n_low=("low", "sum")).sort_index()
    s, cl, cn = 0.0, 0, 0
    for _, row in pm.iterrows():
        x = row.n_low / row.n
        s = 0.0 if cn == 0 else max(0.0, s + x - cl / cn - k)
        cl += row.n_low
        cn += row.n
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--n", type=int, default=300)
    args = ap.parse_args()
    cfg = load_config(args.config)
    P = cfg["windows"]["past_months"]
    low_max = cfg["label"]["low_rating_max"]

    rev = load_table(cfg["data"]["clean_reviews"], parse_dates=["date"])
    rev["title"], rev["text"] = rev["title"].fillna(""), rev["text"].fillna("")
    feats = load_table(cfg["features"]["features_table"], parse_dates=["decision_date"])
    by_prod = {k: v for k, v in rev.groupby("product_id")}
    ks = cfg["baselines"]["cusum_k"]

    sample = feats.sample(min(args.n, len(feats)), random_state=1)
    fails = 0
    for _, row in sample.iterrows():
        d, r = row["decision_date"], by_prod[row["product_id"]]
        before = r[r["date"] < d]
        past = before[before["date"] >= d - pd.DateOffset(months=P)]
        prev = before[(before["date"] >= d - pd.DateOffset(months=2 * P)) & (before["date"] < d - pd.DateOffset(months=P))]
        a = tag_aspects(past["title"], past["text"]).mean()
        checks = {f"{c}_share": (a[c], row[f"{c}_share"]) for c in ["asp_any", "asp_leak", "asp_stopped_working"]}
        lp = (past["rating"] <= low_max).mean()
        lprev = (prev["rating"] <= low_max).mean() if len(prev) else lp
        checks["low_share_change"] = (lp - lprev, row["low_share_change"])
        for k in ks:
            checks[f"cusum_k{k:g}"] = (cusum_from_reviews(before, low_max, k), row[f"cusum_k{k:g}"])
        for name, (want, got) in checks.items():
            if not np.isclose(want, got, atol=1e-6):
                fails += 1
                print(f"MISMATCH {row['product_id']} {d.date()} {name}: brute={want:.6f} table={got:.6f}")

    banned = [c for c in feats.columns if c.startswith("fut_")]
    print(f"Target-only columns present (must be excluded from models): {banned}")
    print(f"Brute-force checked {len(sample)} rows: {fails} mismatches.")
    if fails:
        sys.exit(1)
    print("ALL FEATURE CHECKS PASSED")


if __name__ == "__main__":
    main()
