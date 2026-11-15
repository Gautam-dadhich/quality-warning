"""Step 2 - clean reviews and build the product x decision-month table.

Output (data/processed/):
  reviews_clean.*     one row per cleaned review (used later for text features)
  product_windows.*   one row per (product, decision month) with
                      - past-window features  (inputs; computed only from months < decision month)
                      - fut_* columns          (TARGET-ONLY; never use as model inputs)
                      - label                  (from config thresholds; NaN = not scorable/labelable)
                      - split                  (train / val / test by decision month)
  results/cleaning_log.csv

Window logic (P = past_months, F = future_months, m = decision month):
  past window   = months [m-P, m-1]
  future window = months [m, m+F-1]
so the two windows never overlap and features never see the target period.

Usage:
    python src/02_build_windows.py
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from labels import label_from_config
from utils import assign_split, load_config, resolve, save_table

REVIEW_COLS = [
    "rating", "title", "text", "asin", "parent_asin", "user_id",
    "timestamp", "helpful_vote", "verified_purchase",
]


# ------------------------------------------------------------------ cleaning
def load_and_clean(cfg: dict) -> tuple[pd.DataFrame, list[tuple[str, int]]]:
    path = resolve(cfg["data"]["raw_reviews"])
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run src/01_download.py first.")

    log: list[tuple[str, int]] = []
    chunks = []
    n_raw = 0
    # convert_dates/keep_default_dates=False: stop pandas from auto-parsing the
    # "timestamp" column (it guesses the unit); we convert it explicitly below.
    reader = pd.read_json(path, lines=True, chunksize=250_000, compression="infer",
                          dtype={"asin": str, "parent_asin": str, "user_id": str},
                          convert_dates=False, keep_default_dates=False)
    for i, ch in enumerate(reader):
        n_raw += len(ch)
        for c in REVIEW_COLS:
            if c not in ch.columns:
                ch[c] = np.nan
        chunks.append(ch[REVIEW_COLS])
        print(f"  read chunk {i + 1}: {n_raw:,} rows", end="\r")
    print()
    df = pd.concat(chunks, ignore_index=True)
    log.append(("raw reviews", n_raw))

    # product id: prefer parent_asin (groups colour/size variants)
    df["product_id"] = df["parent_asin"].where(df["parent_asin"].notna() & (df["parent_asin"] != ""), df["asin"])

    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["product_id", "timestamp", "rating"])
    log.append(("dropped: missing product/timestamp/rating", before - len(df)))

    before = len(df)
    df = df[df["rating"].between(1, 5)]
    log.append(("dropped: rating outside 1-5", before - len(df)))

    # Reviews'23 timestamps are Unix milliseconds; detect the unit defensively
    # in case a different pandas version or file variant gives seconds/nanoseconds.
    med = float(df["timestamp"].median())
    unit = "ns" if med > 1e17 else ("us" if med > 1e14 else ("ms" if med > 1e11 else "s"))
    print(f"  timestamp unit detected: {unit} (median value {med:.3e})")
    df["date"] = pd.to_datetime(df["timestamp"].astype("int64"), unit=unit, utc=True).dt.tz_localize(None)
    start, end = pd.Timestamp(cfg["data"]["start_date"]), pd.Timestamp(cfg["data"]["end_date"])
    before = len(df)
    df = df[(df["date"] >= start) & (df["date"] < end + pd.Timedelta(days=1))]
    log.append(("dropped: outside study period", before - len(df)))

    df["title"] = df["title"].fillna("").astype(str)
    df["text"] = df["text"].fillna("").astype(str)
    df["helpful_vote"] = pd.to_numeric(df["helpful_vote"], errors="coerce").fillna(0).astype(int)
    df["verified_purchase"] = df["verified_purchase"].fillna(False).astype(bool)

    before = len(df)
    df = df.drop_duplicates(subset=["user_id", "product_id", "timestamp", "rating", "text"])
    log.append(("dropped: exact duplicates", before - len(df)))

    n_empty = int(((df["text"].str.strip() == "") & (df["title"].str.strip() == "")).sum())
    log.append(("kept but empty title+text", n_empty))

    df = df[["product_id", "user_id", "date", "rating", "title", "text",
             "helpful_vote", "verified_purchase"]].sort_values(["product_id", "date"]).reset_index(drop=True)
    log.append(("clean reviews", len(df)))
    log.append(("products", df["product_id"].nunique()))
    return df, log


# ------------------------------------------------------------------ windows
def month_index(dates: pd.Series) -> np.ndarray:
    return (dates.dt.year * 12 + dates.dt.month - 1).to_numpy()


def build_windows(rev: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    P = int(cfg["windows"]["past_months"])
    F = int(cfg["windows"]["future_months"])
    low_max = cfg["label"]["low_rating_max"]

    start_mi = month_index(pd.Series([pd.Timestamp(cfg["data"]["start_date"])]))[0]
    end_mi = month_index(pd.Series([pd.Timestamp(cfg["data"]["end_date"])]))[0]

    r = rev[["product_id", "date", "rating", "helpful_vote", "verified_purchase"]].copy()
    r["mi"] = month_index(r["date"])
    r["low"] = (r["rating"] <= low_max).astype(int)
    r["r2"] = r["rating"] ** 2

    # 1) product-month counts
    pm = (r.groupby(["product_id", "mi"], sort=True)
            .agg(n=("rating", "size"), n_low=("low", "sum"), s_rating=("rating", "sum"),
                 s_rating2=("r2", "sum"), n_verified=("verified_purchase", "sum"),
                 s_helpful=("helpful_vote", "sum"))
            .reset_index())

    # 2) contiguous monthly grid per product.
    #    Decision months run from first review month to last review month + P
    #    (after that the past window is empty). We extend by F-1 months so the
    #    future window of the last decision month can be summed, then drop those rows.
    span = pm.groupby("product_id")["mi"].agg(["min", "max"])
    lo = span["min"].to_numpy()
    hi_dec = np.minimum(span["max"].to_numpy() + P, end_mi - F + 1)  # last decision month
    hi_ext = hi_dec + F - 1
    lengths = np.clip(hi_ext - lo + 1, 0, None)
    keep = lengths > 0
    prods = span.index.to_numpy()[keep]
    lo, hi_dec, lengths = lo[keep], hi_dec[keep], lengths[keep]

    grid_pid = np.repeat(prods, lengths)
    offsets = np.arange(lengths.sum()) - np.repeat(np.cumsum(lengths) - lengths, lengths)
    grid_mi = np.repeat(lo, lengths) + offsets
    grid = pd.DataFrame({"product_id": grid_pid, "mi": grid_mi,
                         "hi_dec": np.repeat(hi_dec, lengths),
                         "first_mi": np.repeat(lo, lengths)})
    grid = grid.merge(pm, on=["product_id", "mi"], how="left")
    count_cols = ["n", "n_low", "s_rating", "s_rating2", "n_verified", "s_helpful"]
    grid[count_cols] = grid[count_cols].fillna(0)

    # 3) window sums via within-product cumulative sums.
    #    C[m] = sum of months <= m.  past(m) = C[m-1] - C[m-P-1];  future(m) = C[m+F-1] - C[m-1]
    g = grid.groupby("product_id", sort=False)
    C = g[count_cols].cumsum()
    C["product_id"] = grid["product_id"]
    gc = C.groupby("product_id", sort=False)

    def lag(k: int) -> pd.DataFrame:
        return gc[count_cols].shift(k).fillna(0)

    past = lag(1) - lag(P + 1)
    prev = lag(P + 1) - lag(2 * P + 1)          # the P months before the past window (volume trend)
    fut = gc[count_cols].shift(-(F - 1)) - lag(1)  # NaN at the end of the extended grid (dropped below)

    out = pd.DataFrame({"product_id": grid["product_id"], "mi": grid["mi"]})
    n_p = past["n"]
    out["n_past"] = n_p
    out["n_low_past"] = past["n_low"]
    with np.errstate(divide="ignore", invalid="ignore"):
        out["low_share_past"] = past["n_low"] / n_p
        out["mean_rating_past"] = past["s_rating"] / n_p
        var = past["s_rating2"] / n_p - out["mean_rating_past"] ** 2
        out["std_rating_past"] = np.sqrt(np.clip(var, 0, None))
        out["verified_share_past"] = past["n_verified"] / n_p
        out["helpful_mean_past"] = past["s_helpful"] / n_p
    out["n_prev"] = prev["n"]
    out["volume_change"] = (n_p - prev["n"]) / (prev["n"] + 1)
    out["product_age_months"] = grid["mi"] - grid["first_mi"]

    # TARGET-ONLY columns
    out["fut_n"] = fut["n"]
    out["fut_n_low"] = fut["n_low"]
    with np.errstate(divide="ignore", invalid="ignore"):
        out["fut_low_share"] = fut["n_low"] / fut["n"]

    # 4) keep valid decision months with any past activity
    valid = (
        (grid["mi"] <= grid["hi_dec"])          # future window fully inside the grid
        & (grid["mi"] >= start_mi + P)          # full past window inside study period
        & (out["n_past"] > 0)
    )
    out = out[valid].copy()
    out["decision_date"] = pd.to_datetime(
        {"year": out["mi"] // 12, "month": out["mi"] % 12 + 1, "day": 1})
    out = out.drop(columns="mi")

    int_cols = ["n_past", "n_low_past", "n_prev", "product_age_months", "fut_n", "fut_n_low"]
    out[int_cols] = out[int_cols].astype(int)
    out["label"] = label_from_config(out, cfg)
    out["split"] = assign_split(out["decision_date"], cfg)
    cols = ["product_id", "decision_date", "split", "label"] + [
        c for c in out.columns if c not in ("product_id", "decision_date", "split", "label")]
    return out[cols].sort_values(["decision_date", "product_id"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)

    t0 = time.time()
    print("Loading and cleaning reviews ...")
    rev, log = load_and_clean(cfg)
    for k, v in log:
        print(f"  {k:<45s} {v:>12,}")
    p = save_table(rev, cfg["data"]["clean_reviews"])
    print(f"Saved clean reviews -> {p}")

    print("Building product x month windows ...")
    win = build_windows(rev, cfg)
    p = save_table(win, cfg["data"]["windows"])
    print(f"Saved windows -> {p}  ({len(win):,} rows)")

    log += [("windows (n_past > 0)", len(win)),
            ("windows scorable+labelled", int(win["label"].notna().sum())),
            ("positive windows", int((win["label"] == 1).sum()))]
    res = resolve(cfg.get("results_dir", "results"))
    res.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(log, columns=["step", "count"]).to_csv(res / "cleaning_log.csv", index=False)
    print(f"Done in {time.time() - t0:.1f}s. Next: python src/03_label_report.py")


if __name__ == "__main__":
    main()
