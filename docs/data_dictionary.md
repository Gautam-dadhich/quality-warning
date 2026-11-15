# Data dictionary

## `data/processed/reviews_clean.*` — one row per cleaned review

| Column | Type | Description |
|---|---|---|
| product_id | str | `parent_asin` (falls back to `asin` if missing) |
| user_id | str | reviewer id |
| date | datetime | review time (UTC), from Unix-ms `timestamp` |
| rating | float | star rating 1–5 |
| title, text | str | review title and body (empty string if missing) |
| helpful_vote | int | helpful votes |
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
| helpful_mean_past | mean helpful votes per review |
| n_prev | reviews in the P months *before* the feature window |
| volume_change | (n_past − n_prev) / (n_prev + 1) |
| product_age_months | months since the product's first review |

### Target-only (NEVER use as model inputs)
| Column | Description |
|---|---|
| fut_n | reviews in target window |
| fut_n_low | 1–2★ reviews in target window |
| fut_low_share | fut_n_low / fut_n |
