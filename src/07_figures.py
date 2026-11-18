"""Step 7 - paper figures (PDF for LaTeX + PNG for preview) in results/figures/.

  fig_pr_curves        test precision-recall curves: baselines vs best models
  fig_pr_auc_bars      test PR-AUC with 95% month-bootstrap CI, all methods (the ablation)
  fig_capacity         precision and recall vs monthly alert capacity (business view)
  fig_importance       top-15 permutation importances of the best boosting model
  fig_aspects          share of reviews mentioning each complaint aspect
  fig_case_study       one flagged product: monthly low-rating share with alert month

Sized for an IEEE single column (3.5 in wide).

Usage:
    python src/07_figures.py
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

from utils import load_config, load_table, resolve  # noqa: E402

# Validated categorical slots (light mode) + recessive neutrals
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#9a9893", "#e6e5e1"

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.6,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})
W = 3.5


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}")
    plt.close(fig)
    print(f"  saved {name}.pdf/.png")


FEATURE_NAMES = {
    "mean_rating_past": "Mean rating (past)", "low_share_change": "Low-share change",
    "low_share_prev": "Low share (prior window)", "low_share_past": "Low share (past)",
    "n_past": "Review count (past)", "n_prev": "Review count (prior)", "n_low_past": "Low-rated count (past)",
    "std_rating_past": "Rating std. dev. (past)", "product_age_months": "Product age (months)",
    "volume_change": "Volume change", "helpful_mean_past": "Helpful votes (mean)",
    "verified_share_past": "Verified share",
}


def feature_label(f: str) -> str:
    if f in FEATURE_NAMES:
        return FEATURE_NAMES[f]
    if f.startswith("cusum_k"):
        return f"CUSUM (k={f[7:]})"
    if f.startswith("asp_"):
        core = f[4:]
        for suf, txt in (("_share_low", " (share, 1-2*)"), ("_change", " (change)"), ("_share", " (share)")):
            if core.endswith(suf):
                return "Aspect: " + core[: -len(suf)].replace("_", " ") + txt
    if f.startswith("text_svd_"):
        return f"Text component {f[9:]}"
    return f


def pretty(m: str) -> str:
    return (m.replace("logreg", "LogReg").replace("boosting", "GBM").replace("rule_low_share", "Rule: low share")
             .replace("rule_share_change", "Rule: share change").replace("cusum", "CUSUM"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    res = resolve(cfg.get("results_dir", "results"))
    out = res / "figures"
    out.mkdir(parents=True, exist_ok=True)

    results = pd.read_csv(res / "main_results.csv")
    preds = pd.read_csv(resolve(cfg["models"]["predictions"]), parse_dates=["decision_date"])
    test = preds[preds["split"] == "test"]
    val_res = results[results["split"] == "val"].set_index("method")["pr_auc"]
    model_cols = [c for c in preds.columns if c.startswith(("logreg", "boosting"))]
    best_lr = max((c for c in model_cols if c.startswith("logreg")), key=lambda c: val_res[c])
    best_gbm = max((c for c in model_cols if c.startswith("boosting")), key=lambda c: val_res[c])
    best_model = max([best_lr, best_gbm], key=lambda c: val_res[c])
    prev = test["label"].mean()
    print(f"Best by validation PR-AUC: {best_lr}, {best_gbm}  -> overall {best_model}")

    # 1. PR curves -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(W, 2.6))
    curves = [("rule_low_share", MUTED, (0, (4, 2))), ("cusum", INK2, (0, (1, 1.5))),
              ("logreg [S]", AQUA, "-"), (best_lr, BLUE, "-"), (best_gbm, ORANGE, "-")]
    seen = set()
    for col, color, ls in curves:
        if col in seen:
            continue
        seen.add(col)
        p, r, _ = precision_recall_curve(test["label"], test[col])
        tr_ = results[results["split"] == "test"]
        match = tr_[tr_["method"] == col] if col != "cusum" else tr_[tr_["method"].str.startswith("cusum")]
        ap_ = match["pr_auc"].iloc[0]
        ax.plot(r, p, color=color, ls=ls, label=f"{pretty(col)} ({ap_:.3f})")
    ax.axhline(prev, color=MUTED, lw=0.8)
    ax.text(0.99, prev - 0.015, f"random ({prev:.3f})", ha="right", va="top", color=INK2, fontsize=6.5)
    ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1))
    ax.legend(title="Method (test PR-AUC)", title_fontsize=6.5, frameon=False, loc="upper right")
    save(fig, out, "fig_pr_curves")

    # 2. PR-AUC bars with CI (ablation) ----------------------------------------
    t = results[results["split"] == "test"].copy()
    t = t.iloc[::-1]
    fig, ax = plt.subplots(figsize=(W, 0.22 * len(t) + 0.6))
    colors = [BLUE if m.startswith(("logreg", "boosting")) else MUTED for m in t["method"]]
    y = np.arange(len(t))
    ax.barh(y, t["pr_auc"], color=colors, height=0.62, edgecolor="white", linewidth=1)
    ax.errorbar(t["pr_auc"], y, xerr=[t["pr_auc"] - t["pr_auc_ci_low"], t["pr_auc_ci_high"] - t["pr_auc"]],
                fmt="none", ecolor=INK, elinewidth=0.8, capsize=2)
    xmax = max(t["pr_auc_ci_high"])
    for yi, v, hi in zip(y, t["pr_auc"], t["pr_auc_ci_high"]):
        ax.text(hi + 0.01 * xmax, yi, f"{v:.3f}", va="center", fontsize=6.5, color=INK2)
    ax.axvline(prev, color=INK2, lw=0.8, ls=(0, (3, 2)))
    ax.text(prev, len(t) - 0.3, " random", color=INK2, fontsize=6.5, va="bottom")
    ax.set_yticks(y, [pretty(m) for m in t["method"]])
    ax.set_xlabel("Test PR-AUC (95% month-bootstrap CI)")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, xmax * 1.18)
    save(fig, out, "fig_pr_auc_bars")

    # 3. capacity curve --------------------------------------------------------
    caps = [5, 10, 20, 30, 50, 75, 100, 150, 200]
    fig, axes = plt.subplots(1, 2, figsize=(W, 1.8), sharex=True)
    for col, color, ls in [("rule_low_share", MUTED, (0, (4, 2))), ("cusum", INK2, (0, (1, 1.5))), (best_model, BLUE, "-")]:
        P, R = [], []
        rk = test.groupby("decision_date")[col].rank(ascending=False, method="first")
        for k in caps:
            f = test[rk <= k]
            P.append(f["label"].mean())
            R.append(f["label"].sum() / test["label"].sum())
        axes[0].plot(caps, P, color=color, ls=ls, marker="o", ms=2.5, label=pretty(col))
        axes[1].plot(caps, R, color=color, ls=ls, marker="o", ms=2.5)
    axes[0].set(xlabel="Alerts per month", ylabel="Precision", ylim=(0, 1))
    axes[1].set(xlabel="Alerts per month", ylabel="Recall", ylim=(0, 1))
    axes[0].legend(frameon=False, loc="upper right")
    fig.tight_layout()
    save(fig, out, "fig_capacity")

    # 4. permutation importance ------------------------------------------------
    imp = pd.read_csv(res / "hgb_permutation_importance.csv").head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(W, 2.6))
    ax.barh([feature_label(f) for f in imp["feature"]], imp["importance"], xerr=imp["std"], color=BLUE, height=0.62,
            edgecolor="white", linewidth=1, error_kw={"elinewidth": 0.7, "ecolor": INK2})
    ax.set_xlabel("Drop in validation PR-AUC when permuted")
    ax.grid(axis="y", visible=False)
    save(fig, out, "fig_importance")

    # 5. aspect prevalence -----------------------------------------------------
    ap_path = res / "aspect_prevalence.csv"
    if ap_path.exists():
        a = pd.read_csv(ap_path, index_col=0).iloc[:, 0].drop("asp_any", errors="ignore").sort_values()
        fig, ax = plt.subplots(figsize=(W, 1.9))
        labels = [i.replace("asp_", "").replace("_", " ") for i in a.index]
        ax.barh(labels, a.values, color=BLUE, height=0.62, edgecolor="white", linewidth=1)
        for i, v in enumerate(a.values):
            ax.text(v + 0.15, i, f"{v:.1f}%", va="center", fontsize=6.5, color=INK2)
        ax.set_xlabel("% of reviews mentioning aspect")
        ax.grid(axis="y", visible=False)
        save(fig, out, "fig_aspects")

    # 6. case study ------------------------------------------------------------
    # Pick a TRUE event that looked healthy in its feature window (low past
    # low-rating share) but was ranked highly by the model: the early-warning story.
    tp = test[(test["label"] == 1)].copy()
    healthy = tp[tp["rule_low_share"] <= 0.15]
    pool = healthy if len(healthy) else tp
    if len(pool):
        row = pool.sort_values(best_model, ascending=False).iloc[0]
        rank_in_month = int(test[test["decision_date"] == row["decision_date"]][best_model]
                            .rank(ascending=False).loc[row.name])
        rev = load_table(cfg["data"]["clean_reviews"], parse_dates=["date"])
        r = rev[rev["product_id"] == row["product_id"]].copy()
        r["month"] = r["date"].dt.to_period("M").dt.to_timestamp()
        low_max = cfg["label"]["low_rating_max"]
        mth = r.groupby("month").agg(n=("rating", "size"), low=("rating", lambda s: (s <= low_max).mean()))
        d = row["decision_date"]
        mth = mth[(mth.index >= d - pd.DateOffset(months=9)) & (mth.index <= d + pd.DateOffset(months=4))]
        mth = mth[mth["n"] >= 3]  # hide months too thin to read
        mid = mth.index + pd.Timedelta(days=14)  # plot each month at its middle
        fig, ax = plt.subplots(figsize=(W, 1.9))
        P_, F_ = cfg["windows"]["past_months"], cfg["windows"]["future_months"]
        ax.axvspan(d - pd.DateOffset(months=P_), d, color=GRID, alpha=0.8, lw=0)
        ax.axvspan(d, d + pd.DateOffset(months=F_), color=ORANGE, alpha=0.18, lw=0)
        ax.plot(mid, mth["low"] * 100, color=BLUE, marker="o", ms=3)
        ax.axvline(d, color=ORANGE, lw=1.2)
        ax.set_ylim(0, 105)
        ax.text(d - pd.Timedelta(days=4), 101, f"alert: rank {rank_in_month} of month ",
                ha="right", va="top", color=INK, fontsize=6.5)
        ax.set_title("grey: feature window   orange: target month", loc="left", color=INK2, fontsize=6.5)
        ax.set(ylabel="Low-rating share (%)", xlabel="Month (months with >= 3 reviews)")
        fig.autofmt_xdate()
        save(fig, out, "fig_case_study")
        info = row.to_dict()
        info["rank_in_month"] = rank_in_month
        info["monthly_reviews"] = mth["n"].to_dict()
        pd.DataFrame([info]).to_csv(res / "case_study_row.csv", index=False)
        print(f"  case study: {row['product_id']} on {d.date()}, past low share {row['rule_low_share']:.2f}, "
              f"rank {rank_in_month} in its month")

    print(f"Figures in {out}")


if __name__ == "__main__":
    main()
