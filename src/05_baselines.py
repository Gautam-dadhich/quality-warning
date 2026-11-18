"""Step 5 - non-ML baselines.

  rule_low_share    rank products by past-window low-rating share
  rule_share_change rank by change in low-rating share vs the preceding window
  cusum             one-sided CUSUM on monthly low-rating share; slack k chosen on validation

Each baseline gets a threshold tuned on validation (max F1), then is scored once on test.

Output: results/baselines.csv  (also printed)

Usage:
    python src/05_baselines.py
"""
from __future__ import annotations

import argparse

import pandas as pd
from sklearn.metrics import average_precision_score

from evaluation import evaluate_method
from utils import load_config, load_table, resolve

SHOW = ["method", "split", "positives", "prevalence", "pr_auc", "pr_auc_lift",
        "roc_auc", "precision", "recall", "f1"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    res = resolve(cfg.get("results_dir", "results"))
    res.mkdir(parents=True, exist_ok=True)

    df = load_table(cfg["features"]["features_table"], parse_dates=["decision_date"])
    df = df[df["split"].isin(["val", "test"])].copy()

    rows = []
    rows += evaluate_method("rule_low_share", df, "low_share_past", cfg)
    rows += evaluate_method("rule_share_change", df, "low_share_change", cfg)

    # choose CUSUM slack on validation PR-AUC
    val = df[df["split"] == "val"]
    ks = [c for c in df.columns if c.startswith("cusum_k")]
    k_scores = {c: average_precision_score(val["label"].astype(int), val[c]) for c in ks}
    best = max(k_scores, key=k_scores.get)
    print("CUSUM validation PR-AUC by slack:", {k: round(v, 4) for k, v in k_scores.items()}, "-> using", best)
    rows += evaluate_method(f"cusum ({best.replace('cusum_', '')})", df, best, cfg)

    out = pd.DataFrame(rows)
    out.to_csv(res / "baselines.csv", index=False)
    pd.set_option("display.width", 200)
    print("\n", out[SHOW + [c for c in out.columns if c.startswith(("p_at", "r_at"))]]
          .round(4).to_string(index=False))
    t = out[out["split"] == "test"][["method", "pr_auc", "pr_auc_ci_low", "pr_auc_ci_high"]].round(4)
    print("\nTest PR-AUC with 95% month-bootstrap CI:\n", t.to_string(index=False))
    print(f"\nSaved -> {res / 'baselines.csv'}")


if __name__ == "__main__":
    main()
