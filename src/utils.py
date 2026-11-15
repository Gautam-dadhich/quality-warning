"""Shared helpers: config loading and table I/O."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | os.PathLike | None = None) -> dict:
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(p: str) -> Path:
    """Resolve a config path relative to the repo root."""
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def _has_parquet() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        return False


def save_table(df: pd.DataFrame, stem: str) -> Path:
    """Save as parquet if pyarrow is installed, otherwise gzip CSV."""
    stem_path = resolve(stem)
    stem_path.parent.mkdir(parents=True, exist_ok=True)
    if _has_parquet():
        out = stem_path.with_suffix(".parquet")
        df.to_parquet(out, index=False)
    else:
        out = stem_path.with_suffix(".csv.gz")
        df.to_csv(out, index=False, compression="gzip")
    return out


def load_table(stem: str, parse_dates: list[str] | None = None) -> pd.DataFrame:
    stem_path = resolve(stem)
    pq, csv = stem_path.with_suffix(".parquet"), stem_path.with_suffix(".csv.gz")
    if pq.exists():
        return pd.read_parquet(pq)
    if csv.exists():
        return pd.read_csv(csv, parse_dates=parse_dates or [])
    raise FileNotFoundError(f"Neither {pq} nor {csv} exists. Run the previous pipeline step first.")


def assign_split(dates: pd.Series, cfg: dict) -> pd.Series:
    train_end = pd.Timestamp(cfg["split"]["train_end"])
    val_end = pd.Timestamp(cfg["split"]["val_end"])
    out = pd.Series("test", index=dates.index)
    out[dates <= val_end] = "val"
    out[dates <= train_end] = "train"
    return out
