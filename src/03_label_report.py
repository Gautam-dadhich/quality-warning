"""Step 3 - label feasibility report.

Answers: how many scorable/labelled product-months and positive events do we
have, per year and per split, and how sensitive is that to the thresholds?

Outputs (results/):
  label_counts_by_year.csv
  label_counts_by_split.csv
  label_sensitivity.csv
  dataset_profile.csv        (numbers for the paper's dataset table)

Usage:
    python src/03_label_report.py
"""
from __future__ import annotations

import argparse
import itertools

import pandas as pd

from labels import make_label
from utils import load_config, load_table, resolve


def summarise(df: pd.DataFrame, y: pd.Series, by: pd.Series) -> pd.DataFrame:
    t = pd.DataFrame({"by": by, "y": y, "product_id": df["product_id"]})
    g = t.groupby("by")
    out = pd.DataFrame({
        "windows_with_past_activity": g.size(),
        "labelled": g["y"].count(),
        "positives": g["y"].sum().astype(int),
        "products_labelled": t.dropna(subset=["y"]).groupby("by")["product_id"].nunique(),
    }).fillna(0)
    out["positive_rate_%"] = (100 * out["positives"] / out["labelled"].where(out["labelled"] > 0)).round(2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    lab = cfg["label"]
    res = resolve(cfg.get("results_dir", "results"))
    res.mkdir(parents=True, exist_ok=True)

    win = load_table(cfg["data"]["windows"], parse_dates=["decision_date"])
    y = win["label"]

    print("=" * 70)
    print("Main label definition:",
          f"past>={lab['min_past_reviews']}, future>={lab['min_future_reviews']},",
          f"fut_low_share>={lab['min_future_low_share']}, increase>={lab['min_share_increase']}")
    print("=" * 70)

    by_year = summarise(win, y, win["decision_date"].dt.year.rename("year"))
    by_year.index.name = "year"
    by_year.to_csv(res / "label_counts_by_year.csv")
    print("\nBy decision year:\n", by_year.to_string())

    by_split = summarise(win, y, win["split"]).reindex(["train", "val", "test"])
    by_split.index.name = "split"
    by_split.to_csv(res / "label_counts_by_split.csv")
    print("\nBy split:\n", by_split.to_string())

    # ---------------- sensitivity grid
    s = cfg["sensitivity"]
    rows = []
    for inc, mfut, mpast in itertools.product(s["min_share_increase"], s["min_future_reviews"], s["min_past_reviews"]):
        yy = make_label(win, mpast, mfut, lab["min_future_low_share"], inc)
        sp = win["split"]
        row = {"min_share_increase": inc, "min_future_reviews": mfut, "min_past_reviews": mpast,
               "labelled": int(yy.notna().sum()), "positives": int((yy == 1).sum())}
        for split in ["train", "val", "test"]:
            row[f"pos_{split}"] = int(((yy == 1) & (sp == split)).sum())
        row["positive_rate_%"] = round(100 * row["positives"] / max(row["labelled"], 1), 2)
        rows.append(row)
    sens = pd.DataFrame(rows)
    sens.to_csv(res / "label_sensitivity.csv", index=False)
    print("\nSensitivity grid:\n", sens.to_string(index=False))

    # ---------------- dataset profile for the paper
    rev_log = pd.read_csv(res / "cleaning_log.csv") if (res / "cleaning_log.csv").exists() else None
    labelled = win[y.notna()]
    profile = {
        "decision months (first)": str(win["decision_date"].min().date()),
        "decision months (last)": str(win["decision_date"].max().date()),
        "product-months with past activity": len(win),
        "labelled product-months": len(labelled),
        "distinct products (labelled)": labelled["product_id"].nunique(),
        "positive events": int((y == 1).sum()),
        "positive rate (%)": round(100 * (y == 1).sum() / max(len(labelled), 1), 2),
        "train ends": cfg["split"]["train_end"],
        "val ends": cfg["split"]["val_end"],
    }
    if rev_log is not None:
        d = dict(zip(rev_log["step"], rev_log["count"]))
        profile = {"clean reviews": d.get("clean reviews"), "products (any review)": d.get("products"), **profile}
    pd.Series(profile, name="value").to_csv(res / "dataset_profile.csv", header=True)
    print("\nDataset profile:")
    for k, v in profile.items():
        print(f"  {k:<38s} {v}")

    # ---------------- feasibility warning
    print("\nFeasibility check:")
    for split in ["train", "val", "test"]:
        npos = int(by_split.loc[split, "positives"]) if split in by_split.index else 0
        flag = "OK" if npos >= 100 else ("LOW - consider loosening thresholds" if npos >= 30 else "TOO FEW")
        print(f"  {split:<5s} positives = {npos:>6,}  -> {flag}")


if __name__ == "__main__":
    main()
