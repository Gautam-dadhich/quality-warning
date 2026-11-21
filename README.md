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
git clone https://github.com/Gautam-dadhich/quality-warning.git
cd quality-warning
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
python src/04_features.py          # complaint aspects, share changes, CUSUM, text documents
python tests/test_features.py      # brute-force check of step-4 features (incl. CUSUM leakage)
python src/05_baselines.py         # rule-based and CUSUM baselines -> results/baselines.csv
python src/06_models.py            # LogReg + gradient boosting, ablation S / S+A / S+A+T
python src/07_figures.py           # paper figures -> results/figures/
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
  05_baselines.py        rolling-rating rules and CUSUM baseline
  06_models.py           ML models + feature-set ablation, importances, predictions
  07_figures.py          PR curves, PR-AUC bars, capacity curves, importance, case study
  aspects.py             complaint-aspect regex lexicon (8 aspects)
  evaluation.py          shared metrics (PR-AUC, P@top10%, capacity, val-tuned F1, bootstrap CI)
  labels.py              single definition of the target
  utils.py               config + I/O helpers
tests/
  make_synthetic.py      synthetic data in the Reviews'23 schema with planted episodes
  test_windows.py        brute-force recount + leakage/structure assertions
  test_features.py       brute-force check of aspect, change and CUSUM features
docs/data_dictionary.md  every column explained
results/                 generated tables (CSV) and figures
```

## AI use

AI assistance (Claude) was used to help draft parts of the pipeline code and
tests; all code was reviewed, run and validated by the team. See the AI Use
Declaration in the report.

## Decision-date investigation queue

For the revised paper, fit the risk model on labelled training windows and the coverage model on all past-eligible training windows. Rank every eligible test product before joining next-month outcomes. Snapshot helpful votes are excluded in this replay. In the existing environment, run:

    python src/01_download.py
    python src/02_build_windows.py
    python src/04_features.py --decision-queue
    python src/08_decision_queue.py
    python src/09_warning_time.py
    python paper/render_results.py
    cd paper
    pdflatex main.tex
    pdflatex main.tex

The raw Appliances archive used for this revision has SHA-256 150f209befceaa6f837abc997065b2d251034bbbda19bebc4ad56dac779730c2. The historical benchmark reads the pre-existing results/main_results.csv. The queue-summary and monthly CSVs count confirmed spikes, observable non-events and unknowns. The compressed top-50 ledger identifies every alert. The warning-days CSV records dated sustained crossings for confirmed hits. The paper source also compiles in Overleaf using pdfLaTeX and IEEEtran.

AI assisted in designing, coding, running and writing the decision-date extension. The original AI-use statement above describes the earlier pipeline.
