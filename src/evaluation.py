"""Shared evaluation so every method is scored identically.

Metrics (on labelled windows of one split):
  prevalence      share of positives (= PR-AUC of a random ranking)
  pr_auc          average precision
  pr_auc_lift     pr_auc / prevalence
  roc_auc         secondary
  p_at_top, r_at_top   precision / recall among the top `top_frac` of windows
  p_at_cap, r_at_cap   precision / recall when only `monthly_capacity` alerts
                       per decision month can be investigated
  precision, recall, f1   at a threshold chosen on VALIDATION (max F1), applied unchanged
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def best_f1_threshold(y: np.ndarray, s: np.ndarray) -> float:
    p, r, t = precision_recall_curve(y, s)
    f1 = 2 * p * r / np.clip(p + r, 1e-12, None)
    i = int(np.nanargmax(f1[:-1])) if len(t) else 0
    return float(t[i]) if len(t) else float(np.median(s))


def top_frac_metrics(y: np.ndarray, s: np.ndarray, frac: float) -> tuple[float, float]:
    k = max(1, int(round(frac * len(y))))
    idx = np.argsort(-s, kind="stable")[:k]
    tp = y[idx].sum()
    return tp / k, tp / max(y.sum(), 1)


def capacity_metrics(y: np.ndarray, s: np.ndarray, month: np.ndarray, cap: int) -> tuple[float, float]:
    df = pd.DataFrame({"y": y, "s": s, "m": month})
    df["rank"] = df.groupby("m")["s"].rank(ascending=False, method="first")
    flagged = df[df["rank"] <= cap]
    return flagged["y"].mean(), flagged["y"].sum() / max(df["y"].sum(), 1)


def evaluate(y, s, month, threshold: float, top_frac: float, cap: int) -> dict:
    y, s, month = np.asarray(y, int), np.asarray(s, float), np.asarray(month)
    prev = y.mean()
    pr = average_precision_score(y, s)
    pt, rt = top_frac_metrics(y, s, top_frac)
    pc, rc = capacity_metrics(y, s, month, cap)
    pred = s >= threshold
    tp = (pred & (y == 1)).sum()
    prec = tp / max(pred.sum(), 1)
    rec = tp / max(y.sum(), 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    return {
        "n": len(y), "positives": int(y.sum()), "prevalence": prev,
        "pr_auc": pr, "pr_auc_lift": pr / prev if prev > 0 else np.nan,
        "roc_auc": roc_auc_score(y, s) if 0 < y.sum() < len(y) else np.nan,
        f"p_at_top{int(top_frac * 100)}": pt, f"r_at_top{int(top_frac * 100)}": rt,
        f"p_at_cap{cap}": pc, f"r_at_cap{cap}": rc,
        "threshold": threshold, "precision": prec, "recall": rec, "f1": f1,
    }


def bootstrap_pr_auc_by_month(y, s, month, n_boot: int, seed: int) -> tuple[float, float]:
    """95% CI of PR-AUC, resampling decision MONTHS (keeps within-month correlation)."""
    return _ci(np.asarray(y, int), np.asarray(s, float), np.asarray(month), n_boot, seed)


def _ci(y, s, month, n_boot, seed):
    rng = np.random.default_rng(seed)
    vals = []
    if month is None:
        n = len(y)
        for _ in range(n_boot):
            i = rng.integers(0, n, n)
            if 0 < y[i].sum() < n:
                vals.append(average_precision_score(y[i], s[i]))
    else:
        months = np.unique(month)
        groups = {m: np.where(month == m)[0] for m in months}
        for _ in range(n_boot):
            pick = rng.choice(months, len(months), replace=True)
            i = np.concatenate([groups[m] for m in pick])
            if 0 < y[i].sum() < len(i):
                vals.append(average_precision_score(y[i], s[i]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def evaluate_method(name: str, df: pd.DataFrame, score_col: str, cfg: dict) -> list[dict]:
    """Tune threshold on val, report val and test rows for one score column."""
    ev = cfg["evaluation"]
    val, test = df[df["split"] == "val"], df[df["split"] == "test"]
    thr = best_f1_threshold(val["label"].to_numpy(int), val[score_col].to_numpy(float))
    rows = []
    for split, d in [("val", val), ("test", test)]:
        m = evaluate(d["label"], d[score_col], d["decision_date"], thr, ev["top_frac"], ev["monthly_capacity"])
        if split == "test":
            lo, hi = bootstrap_pr_auc_by_month(d["label"], d[score_col], d["decision_date"],
                                               ev["n_bootstrap"], cfg["seed"])
            m["pr_auc_ci_low"], m["pr_auc_ci_high"] = lo, hi
        rows.append({"method": name, "split": split, **m})
    return rows
