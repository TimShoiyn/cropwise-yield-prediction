"""v26 evaluation (Roadmap B3): models trained on REAL harvest truth, 475 fields.

Benchmark = Cropwise productivity estimate (cropwise_t_ha, already in dataset),
compared on the IDENTICAL rows where the ML model predicts.

Validation: walk-forward by year (train on past years, predict the next).
Models:
  - v26_full     : CatBoost with field_id (best ranking, but field_id must be seen).
  - v26_no_field : CatBoost without field_id (honest for brand-new fields).

LEAKAGE GUARD: harvest-time measurements (humidity/protein/oil/harvested_weight/
completed_area/completed_date) and the target itself are excluded from features.

Outputs:
  models_v2/asof_comparison/asof_results_v26_harvest.csv
  models_v2/asof_comparison/asof_predictions_v26_harvest.csv
  reports/V26_HARVEST_FINAL_RU.md
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

DATA_PROCESSED = Path("data_processed")
COMP_DIR = Path("models_v2/asof_comparison")
REPORTS = Path("reports")
COMP_DIR.mkdir(parents=True, exist_ok=True)

ASOF_TAGS = ["07_01", "08_01", "09_01"]
ASOF_LABELS = {"07_01": "1 Jul", "08_01": "1 Aug", "09_01": "1 Sep"}
SCENARIOS = [
    ("all_crops", None),
    ("wheat_combined", ["wheat_spring", "wheat_winter"]),
    ("sunflower", ["sunflower"]),
    ("barley", ["barley_spring"]),
]
MIN_TARGET_T_HA = 1.0
ASOF_MD = {"07_01": (7, 1), "08_01": (8, 1), "09_01": (9, 1)}
# As-of forecasting is only valid before harvest. By 1 Sep ~36% of fields are
# already harvested, so their satellite/weather series include post-harvest
# (bare-soil) signal. Keep only rows whose harvest happens strictly after as-of.
PREHARVEST_ONLY = os.getenv("V26_PREHARVEST_ONLY", "1") == "1"

# Never used as features: identifiers, target, benchmark, and harvest-time leakage.
DEAD_OR_ID = {
    "field_name", "shape_simplified_geojson", "sowing_date", "harvesting_date",
    "asof_tag", "target_yield_t_ha", "target_source", "yield_t_ha",
    "cropwise_t_ha", "estimate_value",
    "harvested_weight_t", "completed_area_ha", "completed_date", "operation_id",
    "humidity", "protein_content", "oil_content",
    "lat", "long",
}

CAT_FEATURES = [
    "field_id", "crop_id", "prev_crop_id", "standard_name",
    "prev_standard_name", "rotation_pair", "variety", "till_type",
]

PARAMS = dict(
    iterations=900, learning_rate=0.035, depth=4, l2_leaf_reg=30.0,
    loss_function="RMSE", random_seed=42, od_type="Iter", od_wait=80,
    allow_writing_files=False, verbose=False,
)


def metrics(y, pred) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    mask = np.isfinite(y) & np.isfinite(pred)
    y, pred = y[mask], pred[mask]
    if len(y) < 3:
        return {"r2": np.nan, "rmse": np.nan, "mae": np.nan, "mape": np.nan}
    return {
        "r2": float(r2_score(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "mae": float(mean_absolute_error(y, pred)),
        "mape": float(np.nanmean(np.abs((y - pred) / np.where(y > 0, y, np.nan))) * 100.0),
    }


def within_crop_year_r2(df: pd.DataFrame, pred_col: str) -> float:
    sub = df.dropna(subset=["target_yield_t_ha", pred_col, "standard_name", "year"]).copy()
    if len(sub) < 4:
        return np.nan
    y = sub["target_yield_t_ha"].astype(float)
    p = sub[pred_col].astype(float)
    groups = [sub["standard_name"], sub["year"]]
    y_dm = y - y.groupby(groups).transform("mean")
    p_dm = p - p.groupby(groups).transform("mean")
    ss_tot = float((y_dm**2).sum())
    if ss_tot <= 1e-9:
        return np.nan
    return 1.0 - float(((y_dm - p_dm) ** 2).sum()) / ss_tot


def load_dataset(tag: str, crop_filter: list[str] | None) -> pd.DataFrame:
    df = pd.read_csv(DATA_PROCESSED / f"ml_dataset_v26_harvest_asof_{tag}.csv")
    if crop_filter is not None:
        df = df[df["standard_name"].isin(crop_filter)].copy()
    df = df.dropna(subset=["target_yield_t_ha"])
    df = df[df["target_yield_t_ha"] >= MIN_TARGET_T_HA]
    if PREHARVEST_ONLY and "completed_date" in df.columns:
        month, day = ASOF_MD[tag]
        cd = pd.to_datetime(df["completed_date"], errors="coerce", utc=True).dt.tz_localize(None)
        asof = pd.to_datetime(dict(year=df["year"], month=month, day=day))
        df = df[cd.isna() | (cd.values > asof.values)]
    return df.reset_index(drop=True)


def prepare_X(df: pd.DataFrame, use_field_id: bool) -> tuple[pd.DataFrame, list[str], list[int]]:
    dead = set(DEAD_OR_ID)
    if not use_field_id:
        dead.add("field_id")
    feat_cols = [c for c in df.columns if c not in dead]
    X = df[feat_cols].copy()
    cat_cols = [c for c in CAT_FEATURES if c in X.columns and (use_field_id or c != "field_id")]
    for c in cat_cols:
        X[c] = X[c].fillna("missing").astype(str)
    for c in [c for c in feat_cols if c not in cat_cols]:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    cat_idx = [feat_cols.index(c) for c in cat_cols]
    return X, feat_cols, cat_idx


def walk_forward_predict(df: pd.DataFrame, use_field_id: bool) -> np.ndarray:
    X, _, cat_idx = prepare_X(df, use_field_id=use_field_id)
    y = df["target_yield_t_ha"].astype(float)
    years = df["year"].astype(int).values
    pred = np.full(len(df), np.nan)
    for test_year in sorted(df["year"].unique()):
        train_mask = years < int(test_year)
        test_mask = years == int(test_year)
        if test_mask.sum() == 0 or train_mask.sum() < 60 or len(np.unique(years[train_mask])) < 2:
            continue
        X_train, y_train = X.loc[train_mask], y.loc[train_mask]
        X_test = X.loc[test_mask]
        rng = np.random.default_rng(42)
        idx = np.arange(len(X_train))
        rng.shuffle(idx)
        val_n = max(15, int(len(X_train) * 0.15))
        val_idx, train_idx = idx[:val_n], idx[val_n:]
        model = CatBoostRegressor(**PARAMS)
        model.fit(
            Pool(X_train.iloc[train_idx], y_train.iloc[train_idx], cat_features=cat_idx),
            eval_set=Pool(X_train.iloc[val_idx], y_train.iloc[val_idx], cat_features=cat_idx),
        )
        pred[test_mask] = model.predict(X_test)
    return pred


def block(df: pd.DataFrame, scenario: str, tag: str, name: str, pred_col: str) -> dict[str, object]:
    cmp_df = df.dropna(subset=[pred_col]).copy()
    m = metrics(cmp_df["target_yield_t_ha"], cmp_df[pred_col])
    return {
        "scenario": scenario, "asof_tag": tag, "asof": ASOF_LABELS[tag], "model": name,
        "n": len(cmp_df), "r2": m["r2"], "wcy_r2": within_crop_year_r2(cmp_df, pred_col),
        "rmse": m["rmse"], "mae": m["mae"], "mape": m["mape"],
    }


def main() -> None:
    rows, pred_rows = [], []
    for scenario, crop_filter in SCENARIOS:
        print(f"\n{scenario}")
        for tag in ASOF_TAGS:
            df = load_dataset(tag, crop_filter)
            df["pred_v26_full"] = walk_forward_predict(df, use_field_id=True)
            df["pred_v26_no_field"] = walk_forward_predict(df, use_field_id=False)
            # Fair benchmark: Cropwise on the exact rows the ML model scored.
            fair = df[df["pred_v26_full"].notna()].copy()
            rows.append(block(df, scenario, tag, "v26_full", "pred_v26_full"))
            rows.append(block(df, scenario, tag, "v26_no_field", "pred_v26_no_field"))
            rows.append(block(fair, scenario, tag, "cropwise", "cropwise_t_ha"))
            tmp = df[["field_id", "year", "standard_name", "target_yield_t_ha",
                      "cropwise_t_ha", "pred_v26_full", "pred_v26_no_field"]].copy()
            tmp["scenario"] = scenario
            tmp["asof_tag"] = tag
            pred_rows.append(tmp)
            a, b, c = rows[-3], rows[-2], rows[-1]
            print(
                f"  {tag}: full MAPE={a['mape']:.1f}% R2={a['r2']:.3f} wcy={a['wcy_r2']:.3f} | "
                f"no_field MAPE={b['mape']:.1f}% R2={b['r2']:.3f} | "
                f"CW MAPE={c['mape']:.1f}% R2={c['r2']:.3f} wcy={c['wcy_r2']:.3f}"
            )

    res = pd.DataFrame(rows)
    preds = pd.concat(pred_rows, ignore_index=True)
    res.to_csv(COMP_DIR / "asof_results_v26_harvest.csv", index=False)
    preds.to_csv(COMP_DIR / "asof_predictions_v26_harvest.csv", index=False)

    show = res.copy()
    for col in ["r2", "wcy_r2", "rmse", "mae", "mape"]:
        show[col] = show[col].map(lambda x: round(float(x), 3))
    md = ["# v26 harvest-truth results (Roadmap B3)\n\n"]
    md.append("Target = real combine harvest (475 fields, 2021-2025), target >= 1 t/ha.\n")
    md.append("Benchmark = Cropwise productivity estimate on identical rows.\n")
    md.append("Validation = walk-forward by year. Leakage guard on harvest-time fields.\n\n")
    md.append(show[["scenario", "asof", "model", "n", "mape", "r2", "wcy_r2", "rmse", "mae"]].to_markdown(index=False))
    md.append("\n")
    (REPORTS / "V26_HARVEST_FINAL_RU.md").write_text("".join(md), encoding="utf-8")
    print("\nWrote results + report.")


if __name__ == "__main__":
    main()
