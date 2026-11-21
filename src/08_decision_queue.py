"""Evaluate ranked monthly queues before looking up next-month outcomes."""
import argparse
from importlib import import_module

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

from utils import load_config, load_table, resolve

models = import_module("06_models")
KEY = ["decision_date", "product_id"]


def evaluate_queue(scores, outcomes, score, cap):
    # Product ID breaks ties deterministically, without reference to the outcome.
    ranked = scores[KEY + [score]].sort_values(
        ["decision_date", score, "product_id"], ascending=[True, False, True])
    alerts = ranked.groupby("decision_date", sort=False).head(cap).copy()
    alerts["rank"] = alerts.groupby("decision_date").cumcount() + 1
    alerts = alerts.merge(outcomes, on=KEY, validate="one_to_one", how="left")
    alerts["method"] = score
    alerts = alerts.rename(columns={score: "score"})
    alerts["outcome"] = alerts.label.map({1: "event", 0: "non_event"}).fillna("unknown")
    return alerts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    cfg = load_config(parser.parse_args().config)
    df = load_table(cfg["features"]["features_table"] + "_queue", parse_dates=["decision_date"])
    if df.duplicated(KEY).any() or (df.n_past < cfg["label"]["min_past_reviews"]).any():
        raise ValueError("Invalid decision-time cohort")
    tr = df.split.eq("train") & df.label.notna()
    va = df.split.eq("val") & df.label.notna()
    te = df.split.eq("test")
    cusum = sorted(c for c in df if c.startswith("cusum_k"))
    # Helpfulness votes lack an as-of timestamp in the public release.
    structured = [c for c in models.STRUCTURED if c != "helpful_mean_past"] + cusum
    scores = df.loc[te, KEY].copy()
    y = df.loc[va, "label"]
    best = max(cusum, key=lambda c: average_precision_score(y, df.loc[va, c]))
    scores["CUSUM"] = df.loc[te, best]
    scores["Volume"] = df.loc[te, "n_past"]
    # This direction can exploit the headroom inherent in a change-from-baseline label.
    scores["Low prior share"] = -df.loc[te, "low_share_past"]
    features = structured + sorted(c for c in df if c.startswith("asp_"))
    X = models.numeric_matrix(df, features)
    best_ap = -np.inf
    for params in cfg["models"]["hgb_grid"]:
        model = HistGradientBoostingClassifier(**params, l2_regularization=1,
                    early_stopping=False, random_state=cfg["seed"]).fit(X[tr], df.loc[tr, "label"])
        ap = average_precision_score(y, model.predict_proba(X[va])[:, 1])
        if ap > best_ap:
            best_ap, chosen = ap, model
    scores["Boosting"] = chosen.predict_proba(X[te])[:, 1]
    # Probability of a verifiable event = P(observable | history) times
    # P(event | observable, history). Freeze this exploratory policy before comparison.
    all_train = df.split.eq("train")
    coverage = HistGradientBoostingClassifier(**chosen.get_params()).fit(
        X[all_train], df.loc[all_train, "label"].notna().astype(int))
    scores["Coverage-adjusted"] = scores["Boosting"] * coverage.predict_proba(X[te])[:, 1]
    outcomes = df.loc[te, KEY + ["label"]]
    dest = resolve(cfg.get("results_dir", "results"))
    outputs = []
    monthly = []
    for method in ["Boosting", "Coverage-adjusted", "CUSUM", "Volume", "Low prior share"]:
        for cap in [10, 50, 100, 200]:
            alerts = evaluate_queue(scores, outcomes, method, cap)
            counts = pd.crosstab(alerts.decision_date, alerts.outcome).reindex(
                columns=["event", "non_event", "unknown"], fill_value=0)
            monthly.append(counts.assign(method=method, cap=cap).reset_index())
            if cap == 50:
                outputs.append(alerts)
    monthly = pd.concat(monthly, ignore_index=True)
    monthly.to_csv(dest / "queue_monthly.csv", index=False)
    pd.concat(outputs, ignore_index=True).to_csv(dest / "queue_top50.csv.gz", index=False)
    sums = monthly.groupby(["method", "cap"])[["event", "non_event", "unknown"]].sum()
    sums["issued"] = sums.sum(axis=1)
    sums["verified_yield"] = sums.event / sums.issued
    sums["precision_observed"] = sums.event / (sums.event + sums.non_event)
    sums.to_csv(dest / "queue_summary.csv")
    print(f"Validation AP {best_ap:.4f}, leaves {chosen.max_leaf_nodes}, comparator {best}")
    print(sums.xs(50, level="cap"))
    print(f"Eligible test windows: {te.sum()}, labelled: {outcomes.label.notna().sum()}")


if __name__ == "__main__":
    main()
