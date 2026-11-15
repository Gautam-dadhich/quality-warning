"""Independent checks of the window table against a brute-force recount
from the cleaned reviews. Run after the pipeline on synthetic data:

    python tests/make_synthetic.py
    python src/02_build_windows.py --config tests/config_synthetic.yaml
    python tests/test_windows.py --config tests/config_synthetic.yaml

Also works on the real data (default config) - it just samples rows.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from utils import load_config, load_table, resolve  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--n", type=int, default=500, help="rows to brute-force check")
    args = ap.parse_args()
    cfg = load_config(args.config)
    P, F = cfg["windows"]["past_months"], cfg["windows"]["future_months"]
    low_max = cfg["label"]["low_rating_max"]

    rev = load_table(cfg["data"]["clean_reviews"], parse_dates=["date"])
    win = load_table(cfg["data"]["windows"], parse_dates=["decision_date"])
    by_prod = {k: v for k, v in rev.groupby("product_id")}

    sample = win.sample(min(args.n, len(win)), random_state=0)
    fails = 0
    for _, row in sample.iterrows():
        d = row["decision_date"]
        r = by_prod[row["product_id"]]
        past = r[(r["date"] >= d - pd.DateOffset(months=P)) & (r["date"] < d)]
        fut = r[(r["date"] >= d) & (r["date"] < d + pd.DateOffset(months=F))]
        checks = {
            "n_past": (len(past), row["n_past"]),
            "n_low_past": (int((past["rating"] <= low_max).sum()), row["n_low_past"]),
            "mean_rating_past": (past["rating"].mean(), row["mean_rating_past"]),
            "std_rating_past": (past["rating"].std(ddof=0), row["std_rating_past"]),
            "fut_n": (len(fut), row["fut_n"]),
            "fut_n_low": (int((fut["rating"] <= low_max).sum()), row["fut_n_low"]),
        }
        for k, (want, got) in checks.items():
            if not np.isclose(want, got, equal_nan=True, atol=1e-6):
                fails += 1
                print(f"MISMATCH {row['product_id']} {d.date()} {k}: brute={want} table={got}")
        # leakage guard: every past review strictly before decision date
        assert (past["date"] < d).all()

    # Structural checks
    assert win["decision_date"].dt.day.eq(1).all(), "decision dates must be month starts"
    assert not win.duplicated(["product_id", "decision_date"]).any(), "duplicate product-months"
    lab = win.dropna(subset=["label"])
    assert (lab["n_past"] >= cfg["label"]["min_past_reviews"]).all()
    assert (lab["fut_n"] >= cfg["label"]["min_future_reviews"]).all()
    for s1, s2 in [("train", "val"), ("val", "test")]:
        a, b = win.loc[win.split == s1, "decision_date"], win.loc[win.split == s2, "decision_date"]
        if len(a) and len(b):
            assert a.max() < b.min(), f"{s1} overlaps {s2}"
    # The target-defining future window of the last train month must not reach into val
    # (it does by F-1 months if F > 1 -> report it so a gap can be added)
    last_train = win.loc[win.split == "train", "decision_date"].max()
    if F > 1:
        print(f"NOTE: F={F}; leave a {F - 1}-month gap after {last_train.date()} before val starts.")

    print(f"Brute-force checked {len(sample)} rows: {fails} mismatches.")

    # Synthetic only: are planted episodes labelled positive?
    ep_path = resolve("tests/tmp/episodes.csv")
    if "synthetic" in str(cfg["data"]["raw_reviews"]) and ep_path.exists():
        ep = pd.read_csv(ep_path)
        ep["decision_date"] = pd.PeriodIndex(ep["episode_start_month"], freq="M").to_timestamp()
        m = ep.merge(win[["product_id", "decision_date", "label"]], on=["product_id", "decision_date"], how="left")
        lab = m.dropna(subset=["label"])
        print(f"Planted episodes with a labelled start month: {len(lab)}/{len(ep)}; "
              f"labelled positive: {int(lab['label'].sum())} ({100 * lab['label'].mean():.0f}%)")

    if fails:
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
