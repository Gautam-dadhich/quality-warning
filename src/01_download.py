"""Step 1 - download the Amazon Reviews'23 Appliances review (and metadata) files.

Usage:
    python src/01_download.py            # reviews only
    python src/01_download.py --meta     # also product metadata

Skips files that already exist. If the URL in config.yaml stops working,
download manually from https://amazon-reviews-2023.github.io/ and place the
file at the path given by data.raw_reviews.
"""
from __future__ import annotations

import argparse
import sys
import urllib.request

from utils import load_config, resolve


def download(url: str, dest) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] {dest} already exists ({dest.stat().st_size / 1e6:.1f} MB)")
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"[get ] {url}")

    def hook(blocks, block_size, total):
        done = blocks * block_size
        if total > 0:
            pct = min(100, done * 100 / total)
            sys.stdout.write(f"\r       {done / 1e6:8.1f} / {total / 1e6:.1f} MB ({pct:5.1f}%)")
        else:
            sys.stdout.write(f"\r       {done / 1e6:8.1f} MB")
        sys.stdout.flush()

    urllib.request.urlretrieve(url, tmp, reporthook=hook)
    tmp.rename(dest)
    print(f"\n[done] saved to {dest}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--meta", action="store_true", help="also download product metadata")
    args = ap.parse_args()
    cfg = load_config(args.config)
    d = cfg["data"]
    download(d["reviews_url"], resolve(d["raw_reviews"]))
    if args.meta:
        download(d["meta_url"], resolve(d["raw_meta"]))


if __name__ == "__main__":
    main()
