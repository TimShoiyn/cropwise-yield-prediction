"""
v25 evaluation: does Sentinel-2 red-edge/EVI/GCVI beat the v24 baseline?

For a fair ablation, all models are trained and scored on the SAME rows: only
rows that have a usable Sentinel-2 scene (s2_ndre_mean not null) and
target >= 1 t/ha. This isolates the value of the S2 indices.

Models:
  - v24_base   : curated v24 features, no S2 (baseline).
  - v25_s2     : curated v24 features + S2 indices.
  - cropwise   : external benchmark on the same rows.

Both ML models use no field_id (honest for new fields). Walk-forward by year.

Outputs:
  models_v2/asof_comparison/asof_results_v25_s2.csv
  models_v2/asof_comparison/asof_predictions_v25_s2.csv
  reports/V25_S2_FINAL_RU.md
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

from scripts.asof.feature_sets_v24 import FEATURE_SETS, feature_set_with_s2  # noqa: E402
from scripts.asof.train_compare_asof_v9 import attach_cropwise, cropwise_table  # noqa: E402

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
]
MIN_TARGET_T_HA = 1.0

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
    df = pd.read_csv(DATA_PROCESSED / f"ml_dataset_v25_s2_asof_{tag}.csv")
    if crop_filter is not None:
        df = df[df["standard_name"].isin(crop_filter)].copy()
    df = attach_cropwise(df, tag, cw)
    df = df.dropna(subset=["target_yield_t_ha"])
    df = df[df["target_yield_t_ha"] >= MIN_TARGET_T_HA]
    # Fair ablation: keep only rows that actually have a Sentinel-2 scene.
    if "s2_ndre_mean" in df.columns:
        df = df[df["s2_ndre_mean"].notna()]
    return df.reset_index(drop=True)


def prepare_X(df: pd.DataFrame, numeric: list[str], categorical: list[str]) -> tuple[pd.DataFrame, list[int]]:
    feat_cols = [c for c in numeric + categorical if c in df.columns]
    X = df[feat_cols].copy()
    cat_cols = [c for c in categorical if c in X.columns]
    for c in cat_cols:
        X[c] = X[c].fillna("missing").astype(str)
    for c in [c for c in feat_cols if c not in cat_cols]:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    cat_idx = [feat_cols.index(c) for c in cat_cols]
    return X, cat_idx


def walk_forward_predict(df: pd.DataFrame, numeric: list[str], categorical: list[str]) -> np.ndarray:
    X, cat_idx = prepare_X(df, numeric, categorical)
    y = df["target_yield_t_ha"].astype(float)
    years = df["year"].astype(int).values
    pred = np.full(len(df), np.nan)
    for test_year in sorted(df["year"].unique()):
        train_mask = years < int(test_year)
        test_mask = years == int(test_year)
        if test_mask.sum() == 0 or train_mask.sum() < 50 or len(np.unique(years[train_mask])) < 2:
            continue
        X_train, y_train = X.loc[train_mask], y.loc[train_mask]
        X_test = X.loc[test_mask]
        rng = np.random.default_rng(42)
        idx = np.arange(len(X_train))
        rng.shuffle(idx)
        val_n = max(12, int(len(X_train) * 0.15))
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
    return {
        "scenario": scenario,
        "asof_tag": tag,
        "asof": ASOF_LABELS[tag],
        "model": name,
        "n": len(cmp_df),
        "r2": m["r2"],
        "wcy_r2": within_crop_year_r2(cmp_df, pred_col),
        "rmse": m["rmse"],
        "mae": m["mae"],
        "mape": m["mape"],
    }


def main() -> None:
    cw = cropwise_table()
    rows = []
    pred_rows = []
    for scenario, crop_filter in SCENARIOS:
        base_num, base_cat = FEATURE_SETS[scenario]
        s2_num, s2_cat = feature_set_with_s2(scenario)
        print(f"\n{scenario}")
        for tag in ASOF_TAGS:
            df = load_dataset(tag, crop_filter, cw)
            if len(df) < 60:
                print(f"  {tag}: too few S2 rows ({len(df)}), skipping")
                continue
            df["pred_v24_base"] = walk_forward_predict(df, base_num, base_cat)
            df["pred_v25_s2"] = walk_forward_predict(df, s2_num, s2_cat)
            fair = df[df["pred_v25_s2"].notna()].copy()
            rows.append(evaluate_block(df, scenario, tag, "v24_base", "pred_v24_base"))
            rows.append(evaluate_block(df, scenario, tag, "v25_s2", "pred_v25_s2"))
            rows.append(evaluate_block(fair, scenario, tag, "cropwise", "cropwise_asof_t_ha"))
            tmp = df[
                ["field_id", "year", "standard_name", "target_yield_t_ha",
                 "cropwise_asof_t_ha", "pred_v24_base", "pred_v25_s2"]
            ].copy()
            tmp["scenario"] = scenario
            tmp["asof_tag"] = tag
            pred_rows.append(tmp)
            a, b, c = rows[-3], rows[-2], rows[-1]
            print(
                f"  {tag} (n={a['n']}): base MAPE={a['mape']:.1f}% R2={a['r2']:.3f} wcy={a['wcy_r2']:.3f}; "
                f"S2 MAPE={b['mape']:.1f}% R2={b['r2']:.3f} wcy={b['wcy_r2']:.3f}; "
                f"CW MAPE={c['mape']:.1f}% R2={c['r2']:.3f} wcy={c['wcy_r2']:.3f}"
            )

    if not rows:
        print("No results (B1 likely not finished). Re-run after extraction.")
        return

    res = pd.DataFrame(rows)
    preds = pd.concat(pred_rows, ignore_index=True)
    res.to_csv(COMP_DIR / "asof_results_v25_s2.csv", index=False)
    preds.to_csv(COMP_DIR / "asof_predictions_v25_s2.csv", index=False)

    show = res.copy()
    for col in ["r2", "wcy_r2", "rmse", "mae", "mape"]:
        show[col] = show[col].map(lambda x: round(float(x), 3))
    md = ["# v25 Sentinel-2 ablation results\n\n"]
    md.append("Fair ablation: identical rows (target >= 1 and S2 scene present).\n\n")
    md.append("- `v24_base`: curated features without S2.\n")
    md.append("- `v25_s2`: same features + NDRE/EVI/GCVI.\n")
    md.append("- `cropwise`: benchmark on same rows.\n\n")
    md.append(show[["scenario", "asof", "model", "n", "mape", "r2", "wcy_r2", "rmse", "mae"]].to_markdown(index=False))
    md.append("\n")
    (REPORTS / "V25_S2_FINAL_RU.md").write_text("".join(md), encoding="utf-8")
    print("\nWrote v25 results/report.")


if __name__ == "__main__":
    main()
