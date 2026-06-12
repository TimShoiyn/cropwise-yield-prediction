"""
Evaluate v23 full Cropwise dataset.

Models:
  - v23_full: CatBoost with field_id categorical feature.
  - v23_no_field: same, but without field_id.
  - cropwise: external benchmark, never used as feature.

Validation:
  Walk-forward by year. For each test year, train only on prior years.

Outputs:
  models_v2/asof_comparison/asof_results_v23_cropwise_full.csv
  models_v2/asof_comparison/asof_predictions_v23_cropwise_full.csv
  reports/V23_CROPWISE_FULL_FINAL_RU.md
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

from scripts.asof.train_compare_asof_v9 import attach_cropwise, cropwise_table  # noqa: E402

DATA_PROCESSED = Path("data_processed")
COMP_DIR = Path("models_v2/asof_comparison")
REPORTS = Path("reports")
COMP_DIR.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(exist_ok=True)

ASOF_TAGS = ["07_01", "08_01", "09_01"]
ASOF_LABELS = {"07_01": "1 Jul", "08_01": "1 Aug", "09_01": "1 Sep"}
SCENARIOS = [
    ("all_crops", None),
    ("wheat_combined", ["wheat_spring", "wheat_winter"]),
    ("sunflower", ["sunflower"]),
]

NON_FEATURES = {
    "history_item_id",
    "target_yield_t_ha",
    "target_source",
    "target_conflict",
    "cropwise_asof_t_ha",
    "estimate_history_dict",
    "asof_tag",
    "field_name",
    "shape_simplified_geojson",
    "sowing_date",
    "harvesting_date",
}

CAT_FEATURES = [
    "field_id",
    "crop_id",
    "prev_crop_id",
    "standard_name",
    "prev_standard_name",
    "rotation_pair",
    "variety",
    "till_type",
    "region_id",
    "district_id",
]

PARAMS = dict(
    iterations=900,
    learning_rate=0.035,
    depth=4,
    l2_leaf_reg=35.0,
    loss_function="RMSE",
    random_seed=42,
    od_type="Iter",
    od_wait=80,
    allow_writing_files=False,
    verbose=False,
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


def load_dataset(tag: str, crop_filter: list[str] | None, cw: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(DATA_PROCESSED / f"ml_dataset_v23_cropwise_full_asof_{tag}.csv")
    if crop_filter is not None:
        df = df[df["standard_name"].isin(crop_filter)].copy()
    df = attach_cropwise(df, tag, cw)
    return df.dropna(subset=["target_yield_t_ha"]).reset_index(drop=True)


def prepare_X(df: pd.DataFrame, use_field_id: bool) -> tuple[pd.DataFrame, list[str], list[int]]:
    non = set(NON_FEATURES)
    if not use_field_id:
        non.add("field_id")
    feat_cols = [c for c in df.columns if c not in non]
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
        if test_mask.sum() == 0 or train_mask.sum() < 80 or len(np.unique(years[train_mask])) < 2:
            continue

        X_train, y_train = X.loc[train_mask], y.loc[train_mask]
        X_test = X.loc[test_mask]
        rng = np.random.default_rng(42)
        idx = np.arange(len(X_train))
        rng.shuffle(idx)
        val_n = max(20, int(len(X_train) * 0.15))
        val_idx, train_idx = idx[:val_n], idx[val_n:]

        model = CatBoostRegressor(**PARAMS)
        model.fit(
            Pool(X_train.iloc[train_idx], y_train.iloc[train_idx], cat_features=cat_idx),
            eval_set=Pool(X_train.iloc[val_idx], y_train.iloc[val_idx], cat_features=cat_idx),
        )
        pred[test_mask] = model.predict(X_test)
    return pred


def evaluate_block(df: pd.DataFrame, scenario: str, tag: str, name: str, pred_col: str) -> dict[str, object]:
    cmp_df = df.dropna(subset=[pred_col]).copy()
    m = metrics(cmp_df["target_yield_t_ha"], cmp_df[pred_col])
    normal = cmp_df[cmp_df["target_yield_t_ha"] >= 1.0].copy()
    mn = metrics(normal["target_yield_t_ha"], normal[pred_col]) if len(normal) else {}
    return {
        "scenario": scenario,
        "asof_tag": tag,
        "asof": ASOF_LABELS[tag],
        "model": name,
        "n": len(cmp_df),
        "n_target_ge_1": len(normal),
        "r2": m["r2"],
        "wcy_r2": within_crop_year_r2(cmp_df, pred_col),
        "rmse": m["rmse"],
        "mae": m["mae"],
        "mape": m["mape"],
        "r2_target_ge_1": mn.get("r2", np.nan),
        "mape_target_ge_1": mn.get("mape", np.nan),
    }


def main() -> None:
    cw = cropwise_table()
    rows = []
    pred_rows = []

    for scenario, crop_filter in SCENARIOS:
        print(f"\n{scenario}")
        for tag in ASOF_TAGS:
            df = load_dataset(tag, crop_filter, cw)
            df["pred_v23_full"] = walk_forward_predict(df, use_field_id=True)
            df["pred_v23_no_field"] = walk_forward_predict(df, use_field_id=False)

            fair_df = df[df["pred_v23_full"].notna()].copy()
            for col, name in [
                ("pred_v23_full", "v23_full"),
                ("pred_v23_no_field", "v23_no_field"),
            ]:
                rows.append(evaluate_block(df, scenario, tag, name, col))
            rows.append(evaluate_block(fair_df, scenario, tag, "cropwise", "cropwise_asof_t_ha"))

            tmp = df[
                [
                    "field_id",
                    "year",
                    "standard_name",
                    "target_yield_t_ha",
                    "cropwise_asof_t_ha",
                    "pred_v23_full",
                    "pred_v23_no_field",
                ]
            ].copy()
            tmp["scenario"] = scenario
            tmp["asof_tag"] = tag
            pred_rows.append(tmp)

            a, b, c = rows[-3], rows[-2], rows[-1]
            print(
                f"  {tag}: full MAPE={a['mape']:.1f}% R2={a['r2']:.3f} wcy={a['wcy_r2']:.3f}; "
                f"no_field MAPE={b['mape']:.1f}% R2={b['r2']:.3f} wcy={b['wcy_r2']:.3f}; "
                f"CW MAPE={c['mape']:.1f}% R2={c['r2']:.3f} wcy={c['wcy_r2']:.3f}"
            )

    res = pd.DataFrame(rows)
    preds = pd.concat(pred_rows, ignore_index=True)
    res.to_csv(COMP_DIR / "asof_results_v23_cropwise_full.csv", index=False)
    preds.to_csv(COMP_DIR / "asof_predictions_v23_cropwise_full.csv", index=False)

    best = res.sort_values(["scenario", "asof_tag", "mape"]).groupby(["scenario", "asof_tag"]).head(1)
    md = ["# v23 Cropwise full results\n\n"]
    md.append("Dataset: 1,903 strict target rows, 483 fields, 2017-2025.\n\n")
    md.append("Models: `v23_full` includes `field_id`; `v23_no_field` removes it; `cropwise` is benchmark only.\n\n")
    md.append("## Best by MAPE\n\n")
    md.append(best[["scenario", "asof", "model", "n", "mape", "r2", "wcy_r2", "mape_target_ge_1"]].to_markdown(index=False))
    md.append("\n\n## Full table\n\n")
    md.append(res[["scenario", "asof", "model", "n", "mape", "r2", "wcy_r2", "rmse", "mae", "mape_target_ge_1"]].to_markdown(index=False))
    md.append("\n")
    (REPORTS / "V23_CROPWISE_FULL_FINAL_RU.md").write_text("".join(md), encoding="utf-8")
    print("\nWrote results/report.")


if __name__ == "__main__":
    main()
