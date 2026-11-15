"""Generate a small synthetic file in the Amazon Reviews'23 review schema,
for smoke-testing the pipeline without downloading the real data.

Includes deliberate dirt (duplicates, missing ids, bad ratings, empty text)
and planted "defect episodes" where a product's low-rating share jumps.
The ground-truth episodes are written to tests/tmp/episodes.csv.

Usage:
    python tests/make_synthetic.py
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "tmp"
rng = np.random.default_rng(0)


def main(n_products: int = 400) -> None:
    OUT.mkdir(exist_ok=True)
    start, end = pd.Timestamp("2014-01-01"), pd.Timestamp("2023-09-30")
    months = pd.period_range(start, end, freq="M")
    rows, episodes = [], []
    good_txt = ["Works great", "Very happy with this", "Does the job", "Solid build, recommend"]
    bad_txt = ["Stopped working after two weeks", "Leaks water everywhere",
               "Very loud noise, returned it", "Arrived cracked, part missing"]

    for p in range(n_products):
        pid = f"P{p:05d}"
        launch = rng.integers(0, len(months) - 12)
        life = rng.integers(12, 80)
        base_rate = rng.uniform(1, 12)            # reviews per month
        base_low = rng.uniform(0.05, 0.2)
        has_ep = rng.random() < 0.35
        ep_start = launch + rng.integers(3, max(4, life - 3)) if has_ep else -1
        ep_len = rng.integers(1, 4)
        if has_ep:
            episodes.append((pid, str(months[min(ep_start, len(months) - 1)]), int(ep_len)))
        for m in range(launch, min(launch + life, len(months))):
            in_ep = has_ep and ep_start <= m < ep_start + ep_len
            # complaints start creeping up one month before the episode (an early signal)
            pre_ep = has_ep and m == ep_start - 1
            low_p = 0.6 if in_ep else (0.3 if pre_ep else base_low)
            lam = base_rate * (1.5 if in_ep else 1.0)
            for _ in range(rng.poisson(lam)):
                day = months[m].start_time + pd.Timedelta(days=int(rng.integers(0, 28)))
                low = rng.random() < low_p
                rating = int(rng.choice([1, 2])) if low else int(rng.choice([3, 4, 5], p=[.15, .35, .5]))
                rows.append({
                    "rating": float(rating),
                    "title": rng.choice(bad_txt if low else good_txt),
                    "text": rng.choice(bad_txt if low else good_txt) + ".",
                    "images": [],
                    "asin": f"{pid}-v{rng.integers(0, 3)}",
                    "parent_asin": pid,
                    "user_id": f"U{rng.integers(0, 10**7):07d}",
                    "timestamp": int(day.value // 10**6),   # Unix ms like Reviews'23
                    "helpful_vote": int(rng.poisson(0.5)),
                    "verified_purchase": bool(rng.random() < 0.85),
                })

    # dirt
    n = len(rows)
    rows += [dict(rows[i]) for i in rng.integers(0, n, 200)]                       # exact dups
    for i in rng.integers(0, n, 50):
        rows[i] = {**rows[i], "parent_asin": None, "asin": None}                    # missing id
    for i in rng.integers(0, n, 30):
        rows[i] = {**rows[i], "rating": 7.0}                                        # bad rating
    for i in rng.integers(0, n, 30):
        rows[i] = {**rows[i], "text": "", "title": ""}                              # empty text
    rows.append({**rows[0], "timestamp": int(pd.Timestamp("2010-05-01").value // 10**6)})  # out of period

    with gzip.open(OUT / "Appliances_synthetic.jsonl.gz", "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    pd.DataFrame(episodes, columns=["product_id", "episode_start_month", "length_months"]).to_csv(
        OUT / "episodes.csv", index=False)
    print(f"wrote {len(rows):,} synthetic reviews, {len(episodes)} planted episodes -> {OUT}")


if __name__ == "__main__":
    main()
