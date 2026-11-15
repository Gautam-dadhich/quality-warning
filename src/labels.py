"""Target definition, kept in one place so the label report and the models
use exactly the same rule."""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_label(
    df: pd.DataFrame,
    min_past_reviews: int,
    min_future_reviews: int,
    min_future_low_share: float,
    min_share_increase: float,
) -> pd.Series:
    """Return a float Series: 1 = quality-risk event, 0 = no event,
    NaN = not scorable (too few past reviews) or not labelable (too few future reviews).

    Uses fut_* columns, which are TARGET-ONLY and must never be model inputs.
    """
    scorable = df["n_past"] >= min_past_reviews
    labelable = df["fut_n"] >= min_future_reviews
    event = (df["fut_low_share"] >= min_future_low_share) & (
        (df["fut_low_share"] - df["low_share_past"]) >= min_share_increase
    )
    y = pd.Series(np.nan, index=df.index)
    ok = scorable & labelable
    y[ok] = event[ok].astype(float)
    return y


def label_from_config(df: pd.DataFrame, cfg: dict) -> pd.Series:
    lab = cfg["label"]
    return make_label(
        df,
        min_past_reviews=lab["min_past_reviews"],
        min_future_reviews=lab["min_future_reviews"],
        min_future_low_share=lab["min_future_low_share"],
        min_share_increase=lab["min_share_increase"],
    )
