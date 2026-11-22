# Data dictionary

## `data/processed/reviews_clean.*` — one row per cleaned review

| Column | Type | Description |
|---|---|---|
| product_id | str | `parent_asin` (falls back to `asin` if missing) |
| user_id | str | reviewer id |
| date | datetime | review time (UTC), from Unix-ms `timestamp` |
| rating | float | star rating 1–5 |
| title, text | str | review title and body (empty string if missing) |
| helpful_vote | int | helpful votes as recorded in the 2023 dataset snapshot (includes votes cast after the review date) |
| verified_purchase | bool | verified purchase flag |

Cleaning rules (counts logged in `results/cleaning_log.csv`): drop rows missing
product id / timestamp / rating; drop ratings outside 1–5; keep only reviews in the
study period; drop exact duplicates on (user, product, timestamp, rating, text).

## `data/processed/product_windows.*` — one row per (product, decision month)

Decision month *m*; feature window = months [*m*−P, *m*−1]; target window = months [*m*, *m*+F−1].
Defaults P = 2, F = 1. Only rows with ≥ 1 review in the feature window are kept.

### Identifiers
| Column | Description |
|---|---|
| product_id | product |
| decision_date | first day of decision month *m* |
| split | `train` / `val` / `test` by decision_date (see config) |
| label | 1 = quality-risk event, 0 = no event, empty = not scorable or not labelled |

### Features (inputs — feature window only)
| Column | Description |
|---|---|
| n_past | number of reviews |
| n_low_past | number of 1–2★ reviews |
| low_share_past | n_low_past / n_past |
| mean_rating_past | mean star rating |
| std_rating_past | std. dev. of star rating (population) |
| verified_share_past | share of verified purchases |
| helpful_mean_past | mean helpful votes per review (snapshot counts, so not strictly available on the decision date; used in the Table II benchmark only, excluded from the decision-date queue) |
| n_prev | reviews in the P months *before* the feature window |
| volume_change | (n_past − n_prev) / (n_prev + 1) |
| product_age_months | months since the product's first review |

### Target-only (NEVER use as model inputs)
| Column | Description |
|---|---|
| fut_n | reviews in target window |
| fut_n_low | 1–2★ reviews in target window |
| fut_low_share | fut_n_low / fut_n |

## `data/processed/features.*` — labelled windows only (model input table)

All columns of `product_windows` (labelled rows), plus:

| Column | Description |
|---|---|
| low_share_prev | low-rating share in the P months before the feature window (= low_share_past if no reviews) |
| low_share_change | low_share_past − low_share_prev |
| asp_<a>_share | share of feature-window reviews matching aspect *a* (see `src/aspects.py`) |
| asp_<a>_change | asp_<a>_share minus the same share in the preceding window |
| asp_<a>_share_low | share of feature-window **1–2★** reviews matching aspect *a* |
| cusum_k<k> | one-sided CUSUM of monthly low-rating share vs the product's own prior pooled share, slack k, taken at the last month before the decision month |

Aspects *a*: stopped_working, leak, noise, broken_damaged, missing_parts, fit_compat,
heat_electrical, return_refund, any.

## `data/processed/window_docs.*` — one text document per labelled window

| Column | Description |
|---|---|
| product_id, decision_date | join keys to `features` |
| doc | concatenated title + text of feature-window reviews; per month up to 25 reviews, lowest-rated first then most recent, each truncated to 400 characters |
