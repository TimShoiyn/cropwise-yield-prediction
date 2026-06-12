"""
Quick utility: train one CatBoost on v3 (sunflower, as-of 1 Aug) using all data
and dump feature importance to see whether Open-Meteo features actually got picked up.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

ROOT_DIR = Path(__file__).resolve().parents[2]
os.chdir(ROOT_DIR)

DATA_PROCESSED = Path("data_processed")
DATA_RAW = Path("data_raw")

CAT_FEATURES = ["crop_id", "prev_crop_id", "field_id", "rotation_pair"]
NON_FEATURE = ["field_id", "year", "target_yield_t_ha", "target_source", "unit_fix_applied", "standard_name"]

CROPS_DF = pd.read_csv(DATA_RAW / "crops.csv")[["id", "standard_name"]].rename(columns={"id": "crop_id"})


def load(scenario: str, tag: str, version: str) -> pd.DataFrame:
    p = DATA_PROCESSED / f"ml_dataset_clean_{version}_asof_{tag}.csv"
    df = pd.read_csv(p).merge(CROPS_DF, on="crop_id", how="left")
    if scenario == "sunflower":
        df = df[df["standard_name"] == "sunflower"]
    elif scenario == "wheat_combined":
        df = df[df["standard_name"].isin(["wheat_spring", "wheat_winter"])]
    return df.dropna(subset=["target_yield_t_ha"]).reset_index(drop=True)


def importance(df: pd.DataFrame, version: str, scenario: str, tag: str):
    feat_cols = [c for c in df.columns if c not in NON_FEATURE]
    feat_cols = list(dict.fromkeys(feat_cols + ["field_id"]))
    X = df[feat_cols].copy()
    cat_in_X = [c for c in CAT_FEATURES if c in X.columns]
    for c in cat_in_X:
        if c == "rotation_pair":
            X[c] = X[c].fillna("missing").astype(str)
        else:
            X[c] = X[c].apply(lambda v: "missing" if pd.isna(v) else str(int(float(v))))
    for c in [c for c in feat_cols if c not in cat_in_X]:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    cat_idx = [feat_cols.index(c) for c in cat_in_X]
    y = df["target_yield_t_ha"].astype(float)

    m = CatBoostRegressor(iterations=600, learning_rate=0.05, depth=5, random_seed=42, verbose=False, allow_writing_files=False)
    m.fit(Pool(X, y, cat_features=cat_idx))
    imp = pd.DataFrame({"feature": feat_cols, "importance": m.get_feature_importance()})
    imp = imp.sort_values("importance", ascending=False).reset_index(drop=True)
    print(f"\n=== {scenario} | {version} | as-of {tag} | top 20 ===")
    print(imp.head(20).to_string(index=False))
    return imp


def main():
    print("Comparing top features v2b vs v3 (sunflower, 1 Aug)\n")
    for version in ("v2b", "v3"):
        df = load("sunflower", "08_01", version)
        importance(df, version, "sunflower", "08_01")
    print("\n\nWheat combined, 1 Aug:")
    for version in ("v2b", "v3"):
        df = load("wheat_combined", "08_01", version)
        importance(df, version, "wheat_combined", "08_01")


if __name__ == "__main__":
    main()
