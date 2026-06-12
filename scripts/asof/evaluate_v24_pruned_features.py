"""
A2 evaluation: v24 with curated feature contracts.

Compares:
  - v24_pruned_full: curated features + field_id categorical.
  - v24_pruned_no_field: curated features without field_id.
  - cropwise: external benchmark on identical rows.

Target policy from A1: train/evaluate main regression only on target >= 1 t/ha.
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

from scripts.asof.feature_sets_v24 import FEATURE_SETS  # noqa: E402
from scripts.asof.train_compare_asof_v9 import attach_cropwise, cropwise_table  # noqa: E402

DATA_PROCESSED = Path("data_processed")
COMP_DIR = Path("models_v2/asof_comparison")
REPORTS = Path("reports")

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
    df = pd.read_csv(DATA_PROCESSED / f"ml_dataset_v24_agro_asof_{tag}.csv")
    if crop_filter is not None:
        df = df[df["standard_name"].isin(crop_filter)].copy()
    df = attach_cropwise(df, tag, cw)
    df = df.dropna(subset=["target_yield_t_ha"])
    df = df[df["target_yield_t_ha"] >= MIN_TARGET_T_HA].reset_index(drop=True)
    return df


def prepare_X(df: pd.DataFrame, scenario: str, use_field_id: bool) -> tuple[pd.DataFrame, list[str], list[int]]:
    numeric, categorical = FEATURE_SETS[scenario]
    feat_cols = [c for c in numeric + categorical if c in df.columns]
    if use_field_id and "field_id" in df.columns:
        feat_cols = ["field_id"] + [c for c in feat_cols if c != "field_id"]
    if not use_field_id:
        feat_cols = [c for c in feat_cols if c != "field_id"]

    X = df[feat_cols].copy()
    cat_cols = [c for c in categorical if c in X.columns]
    if use_field_id and "field_id" in X.columns:
        cat_cols = ["field_id"] + [c for c in cat_cols if c != "field_id"]

    for c in cat_cols:
        X[c] = X[c].fillna("missing").astype(str)
    for c in [c for c in feat_cols if c not in cat_cols]:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    cat_idx = [feat_cols.index(c) for c in cat_cols]
    return X, feat_cols, cat_idx


def walk_forward_predict(df: pd.DataFrame, scenario: str, use_field_id: bool) -> np.ndarray:
    X, _, cat_idx = prepare_X(df, scenario, use_field_id=use_field_id)
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
        print(f"\n{scenario}")
        for tag in ASOF_TAGS:
            df = load_dataset(tag, crop_filter, cw)
            df["pred_v24_pruned_full"] = walk_forward_predict(df, scenario, use_field_id=True)
            df["pred_v24_pruned_no_field"] = walk_forward_predict(df, scenario, use_field_id=False)
            fair = df[df["pred_v24_pruned_full"].notna()].copy()
            rows.append(evaluate_block(df, scenario, tag, "v24_pruned_full", "pred_v24_pruned_full"))
            rows.append(evaluate_block(df, scenario, tag, "v24_pruned_no_field", "pred_v24_pruned_no_field"))
            rows.append(evaluate_block(fair, scenario, tag, "cropwise", "cropwise_asof_t_ha"))
            tmp = df[
                [
                    "field_id",
                    "year",
                    "standard_name",
                    "target_yield_t_ha",
                    "cropwise_asof_t_ha",
                    "pred_v24_pruned_full",
                    "pred_v24_pruned_no_field",
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
    res.to_csv(COMP_DIR / "asof_results_v24_pruned_features.csv", index=False)
    preds.to_csv(COMP_DIR / "asof_predictions_v24_pruned_features.csv", index=False)

    show = res.copy()
    for col in ["r2", "wcy_r2", "rmse", "mae", "mape"]:
        show[col] = show[col].map(lambda x: round(float(x), 3))
    md = ["# A2 v24 pruned feature evaluation\n\n"]
    md.append("Target policy: `target >= 1 t/ha`. Cropwise benchmark is evaluated on identical OOF rows.\n\n")
    md.append(show[["scenario", "asof", "model", "n", "mape", "r2", "wcy_r2", "rmse", "mae"]].to_markdown(index=False))
    md.append("\n")
    (REPORTS / "A2_V24_PRUNED_FEATURES_RU.md").write_text("".join(md), encoding="utf-8")
    print("\nWrote A2 results/report.")


if __name__ == "__main__":
    main()
