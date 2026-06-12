"""
Sprint 6+ — v7 dataset: drop ALL rows where target = Cropwise estimate.

Audit revealed that target_source='estimate_normalized' was used for 78% of
sunflower rows. That's the Cropwise final prediction, not factual yield.
Training/evaluating on it makes the comparison with Cropwise circular.

v7 = v6 filtered to target_source ∈ {physical_harvested_weight, productivity_normalized}.

This costs us rows but gives a clean factual target.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
os.chdir(ROOT_DIR)

DATA_PROCESSED = Path("data_processed")
ASOF_TAGS = ["07_01", "08_01", "09_01"]
KEEP_SOURCES = {"physical_harvested_weight", "productivity_normalized"}


def main():
    print("=" * 80)
    print("BUILD v7 — factual-only targets")
    print("=" * 80)
    for tag in ASOF_TAGS:
        in_path = DATA_PROCESSED / f"ml_dataset_clean_v6_asof_{tag}.csv"
        out_path = DATA_PROCESSED / f"ml_dataset_clean_v7_asof_{tag}.csv"
        df = pd.read_csv(in_path)
        n0 = len(df)
        out = df[df["target_source"].isin(KEEP_SOURCES)].copy().reset_index(drop=True)
        n1 = len(out)
        out.to_csv(out_path, index=False)
        breakdown = out["target_source"].value_counts().to_dict()
        print(f"  {tag}: {n0} → {n1} rows (-{n0-n1})  {breakdown}")
    print("\nDone.")


if __name__ == "__main__":
    main()
