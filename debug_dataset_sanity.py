"""
Sanity checks for the crop-window dataset.

Read-only script: loads a prepared ML dataset, runs diagnostics, and saves plots/tables
into reports/debug/.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error


ROOT_DIR = Path(__file__).resolve().parent
os.chdir(ROOT_DIR)

DATASET_PATH = Path("data_processed/ml_dataset_full_extended_cropwindow_v1_may_sep_oct.csv")
RAW_CROPS_PATH = Path("data_raw/crops.csv")
DEBUG_DIR = Path("reports/debug")

# Same definition as in train_baseline_model.py
LEAKY_FEATURES = ["ops_yield_t_ha"]
EXCLUDE_COLS = ["field_id", "year", "target_yield_t_ha", "crop_name", "ops_season_start", "ops_season_end"]

# Same standard_name values as in train_baseline_model.py
STANDARD_NAME_WHEAT_SPRING = "wheat_spring"
STANDARD_NAME_SUNFLOWER = "sunflower"


def _load_crops_df() -> pd.DataFrame:
    if not RAW_CROPS_PATH.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(RAW_CROPS_PATH)
    except Exception:
        return pd.DataFrame()
    cols = [c for c in ["id", "name", "standard_name"] if c in df.columns]
    if not cols:
        return pd.DataFrame()
    out = df[cols].copy()
    if "id" in out.columns:
        out["id"] = pd.to_numeric(out["id"], errors="coerce")
    if "standard_name" in out.columns:
        out["standard_name"] = out["standard_name"].astype(str).str.strip()
    return out


def _standard_name_to_crop_ids(crops_df: pd.DataFrame) -> dict[str, set[int]]:
    if crops_df.empty or "id" not in crops_df.columns or "standard_name" not in crops_df.columns:
        return {}
    out: dict[str, set[int]] = {}
    tmp = crops_df.dropna(subset=["id"]).copy()
    for _, r in tmp.iterrows():
        std = str(r.get("standard_name") or "").strip()
        if not std:
            continue
        try:
            cid_i = int(float(r.get("id")))
        except Exception:
            continue
        out.setdefault(std, set()).add(cid_i)
    return out


def filter_by_standard_name(df: pd.DataFrame, *, crops_df: pd.DataFrame, standard_name: Optional[str]) -> pd.DataFrame:
    if standard_name is None:
        return df
    if "crop_id" not in df.columns:
        return df.iloc[0:0].copy()
    std2ids = _standard_name_to_crop_ids(crops_df)
    ids = std2ids.get(str(standard_name).strip(), set())
    if not ids:
        return df.iloc[0:0].copy()
    crop_ids = pd.to_numeric(df["crop_id"], errors="coerce")
    return df[crop_ids.isin([float(x) for x in sorted(ids)])].copy()


def features_all_no_leak(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    cols = [c for c in cols if c not in LEAKY_FEATURES]
    X = df[cols].select_dtypes(include=[np.number])
    return list(X.columns)


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), 1e-6)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)


def _safe_year_col(df: pd.DataFrame) -> str:
    for c in ["year", "season_year", "season", "harvest_year"]:
        if c in df.columns:
            return c
    return "year"


def _describe_series(s: pd.Series) -> dict[str, float]:
    s = pd.to_numeric(s, errors="coerce")
    if not s.notna().any():
        return {"min": float("nan"), "max": float("nan"), "mean": float("nan"), "std": float("nan")}
    return {
        "min": float(np.nanmin(s.values)),
        "max": float(np.nanmax(s.values)),
        "mean": float(np.nanmean(s.values)),
        "std": float(np.nanstd(s.values)),
    }


def _save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _plot_target_hist(y: pd.Series, *, title: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    y_num = pd.to_numeric(y, errors="coerce")
    plt.figure(figsize=(8, 4.5))
    plt.hist(y_num.dropna().values, bins=25, alpha=0.9, color="#2a6fdb")
    plt.title(title)
    plt.xlabel("target_yield_t_ha")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _save_year_spread(df: pd.DataFrame, *, year_col: str, out_path: Path) -> pd.DataFrame:
    tmp = df.copy()
    tmp[year_col] = pd.to_numeric(tmp[year_col], errors="coerce")
    tmp["target_yield_t_ha"] = pd.to_numeric(tmp["target_yield_t_ha"], errors="coerce")
    grp = (
        tmp.dropna(subset=[year_col, "target_yield_t_ha"])
        .groupby(year_col, as_index=False)
        .agg(mean_target=("target_yield_t_ha", "mean"), n=("target_yield_t_ha", "size"))
        .sort_values(year_col)
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grp.to_csv(out_path, index=False)
    return grp


def _simple_cv_gbdt(X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    Xn = X.copy()
    yn = pd.to_numeric(y, errors="coerce")
    keep = yn.notna()
    Xn = Xn.loc[keep].copy()
    yn = yn.loc[keep].copy()

    if len(Xn) < 10:
        return {"r2": float("nan"), "rmse": float("nan"), "mape": float("nan"), "n_used": float(len(Xn))}

    feature_cols = list(Xn.columns)
    id_like = [c for c in ["crop_id", "prev_crop_id"] if c in feature_cols]
    other_num = [c for c in feature_cols if c not in id_like]

    preprocess = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), other_num),
            ("id", SimpleImputer(strategy="constant", fill_value=-1), id_like),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    model = GradientBoostingRegressor(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        loss="squared_error",
    )

    pipe = Pipeline([("preprocess", preprocess), ("model", model)])

    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    oof = np.full(len(Xn), np.nan, dtype=float)
    for tr, te in kf.split(Xn):
        pipe.fit(Xn.iloc[tr], yn.iloc[tr])
        oof[te] = pipe.predict(Xn.iloc[te])

    y_true = yn.values.astype(float)
    y_pred = oof.astype(float)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": rmse,
        "mape": _mape(y_true, y_pred),
        "n_used": float(len(Xn)),
    }


def _print_and_collect(lines: list[str], msg: str) -> None:
    print(msg)
    lines.append(msg)


def run_one(name: str, df: pd.DataFrame, *, crops_df: pd.DataFrame) -> dict[str, object]:
    lines: list[str] = []
    _print_and_collect(lines, "=" * 80)
    _print_and_collect(lines, f"SANITY CHECK: {name}")
    _print_and_collect(lines, "=" * 80)

    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Structure + features
    _print_and_collect(lines, f"Shape: {df.shape[0]} rows × {df.shape[1]} cols")
    feat_cols = features_all_no_leak(df)
    _print_and_collect(lines, f"Model features (all_no_leak, numeric only): {len(feat_cols)}")
    (DEBUG_DIR / f"features_{name}.txt").write_text("\n".join(feat_cols) + "\n", encoding="utf-8")

    nan_counts = df[feat_cols].isna().sum().sort_values(ascending=False)
    top10 = nan_counts.head(10)
    _print_and_collect(lines, "Top-10 NaN features:")
    for k, v in top10.items():
        _print_and_collect(lines, f"  {k}: {int(v)}")
    nan_counts.to_csv(DEBUG_DIR / f"nan_counts_{name}.csv", header=["nan_count"])

    # 2) Target diagnostics
    y = df["target_yield_t_ha"] if "target_yield_t_ha" in df.columns else pd.Series(dtype=float)
    y_stats = _describe_series(y)
    _print_and_collect(
        lines,
        f"Target stats: min={y_stats['min']:.3f}, max={y_stats['max']:.3f}, mean={y_stats['mean']:.3f}, std={y_stats['std']:.3f}",
    )
    _plot_target_hist(y, title=f"Target histogram ({name})", out_path=DEBUG_DIR / f"target_hist_{name}.png")

    year_col = _safe_year_col(df)
    year_spread = _save_year_spread(df, year_col=year_col, out_path=DEBUG_DIR / f"target_by_year_{name}.csv")
    _print_and_collect(lines, f"Year column used: {year_col}")
    _print_and_collect(lines, f"Years: {int(year_spread[year_col].nunique()) if not year_spread.empty else 0}")

    # 3) Season-window aggregates check
    ndvi_cols = ["ndvi_mean_season", "ndvi_max_season", "ndvi_early", "ndvi_mid", "ndvi_late"]
    weather_cols = [
        "weather_temp_avg_season",
        "weather_gdd_season",
        "weather_precip_sum_season",
        "weather_precip_sum_early",
        "weather_hot_days",
    ]
    _print_and_collect(lines, "NDVI aggregate stats (min/mean/max):")
    for c in ndvi_cols:
        if c not in df.columns:
            _print_and_collect(lines, f"  {c}: MISSING")
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        _print_and_collect(lines, f"  {c}: min={np.nanmin(s.values):.4f} mean={np.nanmean(s.values):.4f} max={np.nanmax(s.values):.4f}")

    _print_and_collect(lines, "Weather aggregate stats (min/mean/max):")
    for c in weather_cols:
        if c not in df.columns:
            _print_and_collect(lines, f"  {c}: MISSING")
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().any():
            _print_and_collect(lines, f"  {c}: min={np.nanmin(s.values):.3f} mean={np.nanmean(s.values):.3f} max={np.nanmax(s.values):.3f}")
        else:
            _print_and_collect(lines, f"  {c}: all NaN")

    all_weather_nan = float(df[weather_cols].isna().all(axis=1).mean()) if all(c in df.columns for c in weather_cols) else float("nan")
    _print_and_collect(lines, f"Share of rows with ALL weather_* NaN: {all_weather_nan:.3f}")

    # If possible: show date columns (dataset may not have them)
    date_like = [c for c in df.columns if "date" in c.lower() or c.lower().endswith("_at")]
    if not date_like:
        _print_and_collect(lines, "No date-like columns in this dataset (cannot directly verify 5–9 vs 5–10 windows here).")

    # 4) Crop filter sanity (via crops.csv)
    if "crop_id" in df.columns and not crops_df.empty and "id" in crops_df.columns:
        tmp = df[["crop_id"]].copy()
        tmp["crop_id_num"] = pd.to_numeric(tmp["crop_id"], errors="coerce")
        map_df = crops_df.rename(columns={"id": "crop_id_num"}).copy()
        merged = tmp.merge(map_df, on="crop_id_num", how="left")
        uniq_crop_id = merged["crop_id_num"].dropna().unique()
        uniq_std = merged["standard_name"].dropna().astype(str).unique() if "standard_name" in merged.columns else np.array([])
        uniq_name = merged["name"].dropna().astype(str).unique() if "name" in merged.columns else np.array([])
        _print_and_collect(lines, f"Unique crop_id (count): {len(uniq_crop_id)}")
        _print_and_collect(lines, f"Unique standard_name (count): {len(uniq_std)}")
        if len(uniq_std) > 1:
            _print_and_collect(lines, f"WARNING: >1 standard_name in subset: {sorted(map(str, uniq_std))[:10]}")
        (DEBUG_DIR / f"crop_filter_uniques_{name}.txt").write_text(
            "unique_crop_id:\n" + "\n".join(map(str, sorted(uniq_crop_id))) + "\n\n"
            + "unique_standard_name:\n" + "\n".join(map(str, sorted(uniq_std))) + "\n\n"
            + "unique_name:\n" + "\n".join(map(str, sorted(uniq_name))) + "\n",
            encoding="utf-8",
        )

    # 5) Simple sanity model (random KFold)
    if "target_yield_t_ha" in df.columns:
        X = df[feat_cols].copy()
        y = df["target_yield_t_ha"].copy()
        cvm = _simple_cv_gbdt(X, y)
        _print_and_collect(lines, f"Simple KFold(3) GBDT: R²={cvm['r2']:.4f}, RMSE={cvm['rmse']:.4f}, MAPE={cvm['mape']:.2f}% (n={int(cvm['n_used'])})")
    else:
        cvm = {"r2": float("nan"), "rmse": float("nan"), "mape": float("nan"), "n_used": float("nan")}

    _save_text(DEBUG_DIR / f"console_{name}.txt", "\n".join(lines) + "\n")

    return {
        "name": name,
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "target_stats": y_stats,
        "all_weather_nan_share": all_weather_nan,
        "simple_kfold": cvm,
    }


def main() -> None:
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)
    crops_df = _load_crops_df()

    subsets = [
        ("all", df),
        ("wheat", filter_by_standard_name(df, crops_df=crops_df, standard_name=STANDARD_NAME_WHEAT_SPRING)),
        ("sunflower", filter_by_standard_name(df, crops_df=crops_df, standard_name=STANDARD_NAME_SUNFLOWER)),
    ]

    results: list[dict[str, object]] = []
    for name, subdf in subsets:
        results.append(run_one(name, subdf, crops_df=crops_df))

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    lines: list[str] = []
    for r in results:
        ts = r["target_stats"]
        sk = r["simple_kfold"]
        msg = (
            f"{r['name']}: N={r['n_rows']}, "
            f"target_range=[{ts['min']:.2f}, {ts['max']:.2f}], mean={ts['mean']:.2f}, std={ts['std']:.2f}; "
            f"all_weather_nan_share={float(r['all_weather_nan_share']):.3f}; "
            f"simpleKFold_R2={float(sk['r2']):.3f}, RMSE={float(sk['rmse']):.3f}, MAPE={float(sk['mape']):.2f}%"
        )
        print(msg)
        lines.append(msg)

    _save_text(DEBUG_DIR / "summary.txt", "\n".join(lines) + "\n")
    print(f"\nSaved reports to: {DEBUG_DIR.as_posix()}/")


if __name__ == "__main__":
    main()

