# Early Warning of Product Quality Problems from Customer Reviews: Which Products Should a Quality Team Investigate First?

AMLBT course project — Goa Institute of Management, Big Data Analytics.
Parth Arya, Vijay Krishna, Gautam Dadhich.

Code and archived results for the paper in [`paper/main.tex`](paper/main.tex).
The project ranks Amazon Appliances products each month for investigation by a
quality team, using only review information available on the decision date, and
counts every alert slot, including alerts whose outcome cannot be observed
because too few reviews arrive in the following month.

## Data

Amazon Reviews'23, **Appliances** category (Hou et al., 2024).
Source: https://amazon-reviews-2023.github.io/ — raw files are **not** redistributed
in this repository; `src/01_download.py` fetches them from the official source.
The raw Appliances archive used for the paper has SHA-256
`150f209befceaa6f837abc997065b2d251034bbbda19bebc4ad56dac779730c2`.

## Setup

```bash
git clone https://github.com/Gautam-dadhich/quality-warning.git
cd quality-warning
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Tested with Python 3.11–3.13. All thresholds, window lengths, split dates and the
random seed are in [`config.yaml`](config.yaml). Outputs go to `data/processed/`
and `results/`.

## Reproduce the paper

### 1. Decision-date queue (Tables III–V, Figs. 1–3)

```bash
python src/01_download.py
python src/02_build_windows.py
python src/04_features.py --decision-queue
python src/08_decision_queue.py
python src/09_warning_time.py
```

The risk model is fitted on labelled training windows and the coverage model on
all past-eligible training windows. Every eligible test product is ranked before
next-month outcomes are joined. The snapshot helpful-vote feature is excluded
from these models.

### 2. Original benchmark (Table II)

```bash
python src/03_label_report.py      # label counts, sensitivity grid, dataset profile
python tests/test_windows.py       # brute-force check of window sums / leakage guards
python src/04_features.py          # complaint aspects, share changes, CUSUM, text documents
python tests/test_features.py      # brute-force check of step-4 features (incl. CUSUM leakage)
python src/05_baselines.py         # rule-based and CUSUM baselines -> results/baselines.csv
python src/06_models.py            # LogReg + gradient boosting, ablation S / S+A / S+A+T
```

**Note:** the Table II benchmark includes `helpful_mean_past`, built from each
review's helpful-vote count as recorded in the 2023 snapshot. Those counts include
votes cast after the decision date, so Table II may slightly overstate what was
achievable on the decision date. On the validation year, permutation importance
for this feature is about 0.003 AP (rank 13 of 43,
`results/hgb_permutation_importance.csv`). The paper states this; the queue
models in step 1 do not use the feature.

### 3. Build the paper

`paper/main.tex` is self-contained (all numbers and figures are written into the
file). Compile it with pdfLaTeX and the IEEEtran class, e.g. in Overleaf.

### Smoke test without downloading (synthetic data)

```bash
python tests/make_synthetic.py
python src/02_build_windows.py --config tests/config_synthetic.yaml
python src/03_label_report.py  --config tests/config_synthetic.yaml
python tests/test_windows.py   --config tests/config_synthetic.yaml
```

## Where each result in the paper comes from

| Paper item | File in `results/` |
|---|---|
| Table I (cohort by period) | `label_counts_by_split.csv` |
| Table II (archived benchmark) | `main_results.csv` |
| Table III (50-slot replay) | `queue_summary.csv`, rows with `cap = 50` |
| Table IV (capacity) | `queue_summary.csv`, all caps |
| Table V (1 January 2022 ledger) | `queue_top50.csv.gz` |
| Fig. 1 (monthly outcomes) | `queue_monthly.csv`, `cap = 50` |
| Fig. 2 (yield vs. precision) | `queue_summary.csv`, `cap = 50` |
| Fig. 3 (time to confirmation) | `warning_days.csv`, method `Coverage-adjusted` |
| 388 / 170 distinct products | `queue_top50.csv.gz` |

**Checks** (Section VII-A of the paper): each policy in `queue_top50.csv.gz` has
50 distinct product IDs on each of 21 decision dates; grouping the ledger by
`outcome` reproduces Table III; `warning_days.csv` has 264 `Coverage-adjusted`
rows with a median of 17 days.

**Time to confirmation** (`days` in `warning_days.csv`) is the number of days
from the decision date to the first day on which the event criteria were met and
stayed met through month-end. Because the outcome window is the month that begins
on the decision date, it cannot exceed about 30 days; it measures how quickly
confirming reviews arrived, not how far ahead of a problem the alert came.

## Method summary

| Element | Definition (defaults in `config.yaml`) |
|---|---|
| Unit | product (`parent_asin`) × decision month *m* (first day of month) |
| Feature window | the 2 full calendar months before *m* |
| Target window | calendar month *m* |
| Low rating | 1–2 stars |
| Scorable | ≥ 5 reviews in the feature window |
| Labelled | ≥ 5 reviews in the target window (otherwise unresolved in the decision-date replay) |
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
  04_features.py         aspect / change / CUSUM features + per-window text documents
                         (--decision-queue: all past-eligible windows, incl. unknown outcomes)
  05_baselines.py        rolling-rating rules and CUSUM baseline
  06_models.py           ML models + feature-set ablation, importances, predictions
  07_figures.py          figures for the earlier report version
  08_decision_queue.py   decision-date replay: risk, coverage-adjusted, CUSUM, volume, low prior share
  09_warning_time.py     time to confirmation for confirmed alerts
  aspects.py             complaint-aspect regex lexicon (8 aspects)
  evaluation.py          shared metrics (PR-AUC, P@top10%, capacity, val-tuned F1, bootstrap CI)
  labels.py              single definition of the target
  utils.py               config + I/O helpers
tests/
  make_synthetic.py      synthetic data in the Reviews'23 schema with planted episodes
  test_windows.py        brute-force recount + leakage/structure assertions
  test_features.py       brute-force check of aspect, change and CUSUM features
docs/data_dictionary.md  every column explained
paper/main.tex           final paper source (self-contained)
results/                 generated tables (CSV) and figures
```

## AI use

AI tools (Anthropic's Claude and OpenAI's Codex) were used in this project. For
the original pipeline, AI helped draft parts of the code and tests. For the
decision-date extension, AI assisted in designing, coding, running and writing
it. All code was run and checked by the team, and every number in the paper was
verified against the files in `results/`. See the AI Use Declaration in the paper.
