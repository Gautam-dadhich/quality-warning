# Early Detection of Product Quality Problems from Online Reviews

AMLBT course project — Goa Institute of Management, Big Data Analytics.
Parth Arya, Vijay Krishna, Gautam Dadhich.

A time-aware early-warning framework that predicts, for each product and month,
whether low-rating activity will rise in the following month, using only
review information available before the decision date.

## Data

Amazon Reviews'23, **Appliances** category (Hou et al., 2024).
Source: https://amazon-reviews-2023.github.io/ — raw files are **not** redistributed
in this repository; `src/01_download.py` fetches them from the official source.

## Setup

```bash
git clone <this-repo-url>
cd amlbt-quality-warning
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Tested with Python 3.11–3.13.

## Reproduce

```bash
python src/01_download.py          # ~downloads Appliances.jsonl.gz into data/raw/
python src/02_build_windows.py     # cleaning + product x month windows + labels
python src/03_label_report.py      # label counts, sensitivity grid, dataset profile
python tests/test_windows.py       # brute-force check of window sums / leakage guards
```

All thresholds, window lengths, split dates and the random seed are in
[`config.yaml`](config.yaml). Outputs go to `data/processed/` and `results/`.

### Smoke test without downloading (synthetic data)

```bash
python tests/make_synthetic.py
python src/02_build_windows.py --config tests/config_synthetic.yaml
python src/03_label_report.py  --config tests/config_synthetic.yaml
python tests/test_windows.py   --config tests/config_synthetic.yaml
```

## Method summary

| Element | Definition (defaults in `config.yaml`) |
|---|---|
| Unit | product (`parent_asin`) × decision month *m* (first day of month) |
| Feature window | the 2 full calendar months before *m* |
| Target window | calendar month *m* |
| Low rating | 1–2 stars |
| Scorable | ≥ 5 reviews in the feature window |
| Labelled | ≥ 5 reviews in the target window (otherwise excluded) |
| Positive event | target low-rating share ≥ 30 % **and** ≥ 10 pp above feature-window share |
| Split | chronological by decision month: train ≤ 2020-12, val 2021, test ≥ 2022-01 |

Columns prefixed `fut_` are used **only** to build the label and are never model inputs.
If `future_months > 1`, leave a gap of `future_months − 1` months between train and val
(the test script prints a reminder).

## Repository layout

```
config.yaml              all parameters
src/
  01_download.py         fetch raw data
  02_build_windows.py    clean reviews, build product-month windows, labels, split
  03_label_report.py     label feasibility, sensitivity grid, dataset profile
  labels.py              single definition of the target
  utils.py               config + I/O helpers
tests/
  make_synthetic.py      synthetic data in the Reviews'23 schema with planted episodes
  test_windows.py        brute-force recount + leakage/structure assertions
docs/data_dictionary.md  every column explained
results/                 generated tables (CSV) and figures
```

## AI use

AI assistance (Claude) was used to help draft parts of the pipeline code and
tests; all code was reviewed, run and validated by the team. See the AI Use
Declaration in the report.
