"""Step 6 - machine-learning models with a feature-set ablation.

Feature sets (all from the past window only):
  S    structured  : volume, rating level/dispersion, low-share and its change,
                     verified share, helpfulness, product age, CUSUM statistics
  S+A  + complaint aspects (shares, changes, shares among 1-2 star reviews)
  S+A+T + review text (TF-IDF for logistic regression; TF-IDF->SVD for boosting)

Models: L2 logistic regression (class-balanced) and histogram gradient boosting.
Protocol: fit on TRAIN; choose hyper-parameters on VALIDATION PR-AUC; choose the
alert threshold on VALIDATION (max F1); score TEST once.

Outputs (results/):
  main_results.csv            baselines + all models, val and test rows
  predictions.csv.gz          test/val scores of every method (for figures)
  lr_coefficients.csv         standardised LR coefficients (S+A model)
  top_text_terms.csv          strongest positive/negative TF-IDF terms (S+A+T LR)
  hgb_permutation_importance.csv   permutation importance on validation (best HGB)

Usage:
    python src/06_models.py
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

from evaluation import evaluate_method
from utils import load_config, load_table, resolve

STRUCTURED = [
    "n_past", "n_low_past", "low_share_past", "mean_rating_past", "std_rating_past",
    "verified_share_past", "helpful_mean_past", "n_prev", "volume_change",
    "product_age_months", "low_share_prev", "low_share_change",
]
LOG_COLS = ["n_past", "n_low_past", "n_prev", "product_age_months", "helpful_mean_past"]


def numeric_matrix(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    X = df[cols].astype(float).copy()
    for c in LOG_COLS:
        if c in X:
            X[c] = np.log1p(X[c].clip(lower=0))
    if "volume_change" in X:
        X["volume_change"] = X["volume_change"].clip(-5, 20)
    return X.fillna(0).to_numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    mcfg = cfg["models"]
    seed = cfg["seed"]
    res = resolve(cfg.get("results_dir", "results"))
    res.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("Loading features and documents ...")
    df = load_table(cfg["features"]["features_table"], parse_dates=["decision_date"])
    docs = load_table(cfg["features"]["docs_table"], parse_dates=["decision_date"])
    df = df.merge(docs, on=["product_id", "decision_date"], how="left", validate="1:1")
    df["doc"] = df["doc"].fillna("").astype(str)
    df["label"] = df["label"].astype(int)

    cusum_cols = sorted(c for c in df.columns if c.startswith("cusum_k"))
    aspect_cols = sorted(c for c in df.columns if c.startswith("asp_"))
    S = STRUCTURED + cusum_cols
    SA = S + aspect_cols
    for cols in (S, SA):
        assert not any(c.startswith("fut_") or c in ("label", "split") for c in cols), "target leak in inputs"
    print(f"  rows: {len(df):,}  structured: {len(S)}  aspect: {len(aspect_cols)}")

    tr, va, te = (df["split"] == s for s in ("train", "val", "test"))
    y = df["label"].to_numpy()

    # ---- numeric blocks, scaler fitted on train only
    blocks = {}
    for name, cols in [("S", S), ("S+A", SA)]:
        X = numeric_matrix(df, cols)
        sc = StandardScaler().fit(X[tr])
        blocks[name] = sc.transform(X)

    # ---- text: TF-IDF fitted on train documents only
    print("Fitting TF-IDF on training documents ...")
    tcfg = mcfg["tfidf"]
    tfidf = TfidfVectorizer(max_features=tcfg["max_features"], ngram_range=tuple(tcfg["ngram_range"]),
                            min_df=tcfg["min_df"], sublinear_tf=True, stop_words="english",
                            dtype=np.float32)
    tfidf.fit(df.loc[tr, "doc"])
    T = tfidf.transform(df["doc"])
    n_comp = min(mcfg["svd_components"], T.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_comp, random_state=seed).fit(T[tr.to_numpy()])
    T_svd = svd.transform(T)
    print(f"  vocabulary: {T.shape[1]:,} terms; SVD explained variance: {svd.explained_variance_ratio_.sum():.2f}")

    X_lr = {
        "S": blocks["S"], "S+A": blocks["S+A"],
        "S+A+T": sparse.hstack([sparse.csr_matrix(blocks["S+A"]), T]).tocsr(),
    }
    X_hgb = {
        "S": numeric_matrix(df, S), "S+A": numeric_matrix(df, SA),
        "S+A+T": np.hstack([numeric_matrix(df, SA), T_svd]),
    }
    hgb_names = {"S": S, "S+A": SA, "S+A+T": SA + [f"text_svd_{i}" for i in range(n_comp)]}

    trn, van = tr.to_numpy(), va.to_numpy()
    preds = df.loc[va | te, ["product_id", "decision_date", "split", "label"]].copy()
    rows = []
    fitted = {}

    for fs in ["S", "S+A", "S+A+T"]:
        # logistic regression
        best = None
        for C in mcfg["logreg_C"]:
            m = LogisticRegression(C=C, class_weight="balanced", max_iter=2000, solver="liblinear")
            m.fit(X_lr[fs][trn], y[trn])
            ap_ = average_precision_score(y[van], m.predict_proba(X_lr[fs][van])[:, 1])
            if best is None or ap_ > best[0]:
                best = (ap_, C, m)
        name = f"logreg [{fs}]"
        print(f"  {name:<22s} best C={best[1]}  val PR-AUC={best[0]:.4f}")
        df[name] = best[2].predict_proba(X_lr[fs])[:, 1]
        fitted[name] = best[2]
        rows += evaluate_method(name, df, name, cfg)
        preds[name] = df.loc[va | te, name]

        # gradient boosting
        best = None
        for params in mcfg["hgb_grid"]:
            m = HistGradientBoostingClassifier(**params, l2_regularization=1.0, early_stopping=False,
                                               random_state=seed)
            m.fit(X_hgb[fs][trn], y[trn])
            ap_ = average_precision_score(y[van], m.predict_proba(X_hgb[fs][van])[:, 1])
            if best is None or ap_ > best[0]:
                best = (ap_, params, m)
        name = f"boosting [{fs}]"
        print(f"  {name:<22s} best {best[1]}  val PR-AUC={best[0]:.4f}")
        df[name] = best[2].predict_proba(X_hgb[fs])[:, 1]
        fitted[name] = (best[2], best[0])
        rows += evaluate_method(name, df, name, cfg)
        preds[name] = df.loc[va | te, name]

    # ---- add baseline scores to predictions for the figures
    preds["rule_low_share"] = df.loc[va | te, "low_share_past"]
    preds["rule_share_change"] = df.loc[va | te, "low_share_change"]
    val_ap = {c: average_precision_score(y[van], df.loc[va, c]) for c in cusum_cols}
    best_k = max(val_ap, key=val_ap.get)
    preds["cusum"] = df.loc[va | te, best_k]
    preds.to_csv(resolve(mcfg["predictions"]), index=False, compression="gzip")

    # ---- combined results table
    out = pd.DataFrame(rows)
    base_path = res / "baselines.csv"
    if base_path.exists():
        out = pd.concat([pd.read_csv(base_path), out], ignore_index=True)
    out.to_csv(res / "main_results.csv", index=False)

    # ---- interpretability
    lr_sa = fitted["logreg [S+A]"]
    pd.DataFrame({"feature": SA, "coef": lr_sa.coef_[0]}).assign(abs=lambda d: d["coef"].abs()) \
        .sort_values("abs", ascending=False).drop(columns="abs").to_csv(res / "lr_coefficients.csv", index=False)

    lr_t = fitted["logreg [S+A+T]"]
    coefs = lr_t.coef_[0][len(SA):]
    vocab = np.array(tfidf.get_feature_names_out())
    order = np.argsort(coefs)
    pd.DataFrame({
        "positive_term": vocab[order[::-1][:30]], "positive_coef": coefs[order[::-1][:30]],
        "negative_term": vocab[order[:30]], "negative_coef": coefs[order[:30]],
    }).to_csv(res / "top_text_terms.csv", index=False)

    best_hgb = max((k for k in fitted if k.startswith("boosting")), key=lambda k: fitted[k][1])
    model = fitted[best_hgb][0]
    fs = best_hgb.split("[")[1].rstrip("]")
    rng_idx = np.where(van)[0]
    if len(rng_idx) > 8000:
        rng_idx = np.random.default_rng(seed).choice(rng_idx, 8000, replace=False)
    pi = permutation_importance(model, X_hgb[fs][rng_idx], y[rng_idx], scoring="average_precision",
                                n_repeats=3, random_state=seed)
    imp = pd.DataFrame({"feature": hgb_names[fs], "importance": pi.importances_mean,
                        "std": pi.importances_std}).sort_values("importance", ascending=False)
    imp.to_csv(res / "hgb_permutation_importance.csv", index=False)

    # ---- print summary
    pd.set_option("display.width", 220)
    cap = cfg["evaluation"]["monthly_capacity"]
    cols = ["method", "pr_auc", "pr_auc_ci_low", "pr_auc_ci_high", "pr_auc_lift", "roc_auc",
            f"p_at_cap{cap}", f"r_at_cap{cap}", "precision", "recall", "f1"]
    test = out[out["split"] == "test"][cols].round(4)
    prev = out.loc[out["split"] == "test", "prevalence"].iloc[0]
    print(f"\n=== TEST RESULTS (prevalence = {prev:.4f}, i.e. random PR-AUC) ===")
    print(test.to_string(index=False))
    print("\nTop 10 LR [S+A] coefficients (standardised):")
    print(pd.read_csv(res / "lr_coefficients.csv").head(10).round(3).to_string(index=False))
    print(f"\nTop 10 permutation importances ({best_hgb}, validation):")
    print(imp.head(10).round(4).to_string(index=False))
    print("\nTop complaint terms (S+A+T LR):", ", ".join(vocab[order[::-1][:15]]))
    print(f"\nDone in {time.time() - t0:.0f}s. Next: python src/07_figures.py")


if __name__ == "__main__":
    main()
