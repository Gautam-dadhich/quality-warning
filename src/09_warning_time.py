"""Calendar days from an alert to the first sustained, observable ratings signal."""
import argparse
import numpy as np
import pandas as pd
from utils import load_config, load_table, resolve

KEY = ["product_id", "decision_date"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    cfg = load_config(parser.parse_args().config)
    dest = resolve(cfg.get("results_dir", "results"))
    alerts = pd.read_csv(dest / "queue_top50.csv.gz", parse_dates=["decision_date"])
    hits = alerts[alerts.outcome.eq("event")]
    windows = load_table(cfg["data"]["windows"], parse_dates=["decision_date"])
    selected = hits[KEY].drop_duplicates().merge(windows[KEY + ["low_share_past"]],
                                                  on=KEY, validate="one_to_one")
    rev = load_table(cfg["data"]["clean_reviews"], parse_dates=["date"])[["product_id", "date", "rating"]]
    rev["decision_date"] = rev.date.dt.to_period("M").dt.to_timestamp()
    rows = rev.merge(selected, on=KEY, validate="many_to_one")
    rows["day"] = rows.date.dt.normalize()
    lab = cfg["label"]
    rows["low"] = rows.rating.le(lab["low_rating_max"])
    daily = rows.groupby(KEY + ["day"], sort=True).agg(n=("rating", "size"),
                    low=("low", "sum"), past=("low_share_past", "first")).reset_index()
    totals = daily.groupby(KEY)[["n", "low"]].cumsum()
    share = totals.low / totals.n
    meets = (totals.n.ge(lab["min_future_reviews"]) & share.ge(lab["min_future_low_share"])
             & (share - daily.past).ge(lab["min_share_increase"]))
    # Require the criterion to stay true through the last review of the month.
    daily["meets"] = meets
    sustained = daily.iloc[::-1].groupby(KEY).meets.cummin().reindex(daily.index)
    first = daily[sustained].groupby(KEY).day.min().rename("crossing").reset_index()
    first["days"] = (first.crossing - first.decision_date).dt.days
    timed = hits.merge(first, on=KEY, how="left", validate="many_to_one")
    if timed.days.isna().any():
        raise ValueError("A confirmed event lacks a sustained crossing")
    timed[KEY + ["method", "crossing", "days"]].to_csv(dest / "warning_days.csv", index=False)
    print(timed.groupby("method").days.agg(["count", "median", "mean"]))


if __name__ == "__main__":
    main()
