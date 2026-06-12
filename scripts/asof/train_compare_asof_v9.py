"""
v9 experiment: factual target + real sowing-date features + conservative tuning.

Compares:
  - v7_default: clean factual baseline without sowing features
  - v9_default: v9 with same CatBoost params as before
  - v9_shallow_l10: depth=3, stronger regularization
  - v9_shallow_l30: depth=2, strong regularization
  - v9_mae_l30: depth=3, MAE objective
  - v9_no_field_id: v9 without field_id categorical memorization

All comparisons are walk-forward by year and against Cropwise as-of forecast.
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT_DIR = Path(__file__).resolve().parents[2]
os.chdir(ROOT_DIR)

DATA_PROCESSED = Path("data_processed")
DATA_RAW = Path("data_raw")
MODELS_V2 = Path("models_v2")
REPORTS = Path("reports")
FIGS = REPORTS / "figures"
COMP_DIR = MODELS_V2 / "asof_comparison"
COMP_DIR.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

AS_OF_TAGS = ["07_01", "08_01", "09_01"]
AS_OF_LABELS = {"07_01": "1 Jul", "08_01": "1 Aug", "09_01": "1 Sep"}
WALK_FORWARD_TEST_YEARS = list(range(2017, 2026))
MIN_TRAIN_YEARS = 3
MIN_TRAIN_ROWS = 20

BASE_NON_FEATURE = {
    "field_id",
    "year",
    "target_yield_t_ha",
    "target_source",
    "unit_fix_applied",
    "standard_name",
    "sowing_date",
    "harvesting_date",
    "estimate_history_dict",
    # Cropwise estimate is the external benchmark; never used as model input.
    "cropwise_asof_t_ha",
    # Internal cross-validation / non-numeric / duplicate identifiers.
    "history_item_id",
    "field_name",
    "prod_crop_ru",
}
BASE_CAT_FEATURES = ["crop_id", "prev_crop_id", "field_id", "rotation_pair"]

PARAM_SETS: dict[str, dict] = {
    "default": dict(iterations=600, learning_rate=0.05, depth=5, l2_leaf_reg=3.0, loss_function="RMSE"),
    "shallow_l10": dict(iterations=800, learning_rate=0.04, depth=3, l2_leaf_reg=10.0, loss_function="RMSE"),
    "shallow_l30": dict(iterations=900, learning_rate=0.035, depth=2, l2_leaf_reg=30.0, loss_function="RMSE"),
    "mae_l30": dict(iterations=900, learning_rate=0.035, depth=3, l2_leaf_reg=30.0, loss_function="MAE"),
}

MAX_T_HA = {
    "sunflower": 5.0,
    "wheat_spring": 7.0,
    "wheat_winter": 8.0,
    "barley_spring": 7.0,
    "maize": 12.0,
    "oil_seed_raps_spring": 4.5,
    "oil_seed_raps_winter": 5.0,
    "soya": 4.0,
    "pea": 4.5,
    "lentil": 3.5,
    "buckwheat": 3.0,
    "safflower": 2.5,
}
DEFAULT_MAX_T_HA = 5.0
CROPS_DF = pd.read_csv(DATA_RAW / "crops.csv")[["id", "standard_name"]].rename(columns={"id": "crop_id"})


def parse_estimate_history(value) -> dict:
    if pd.isna(value):
        return {}
    if isinstance(value, dict):
        return value
    text = str(value)
    try:
        return ast.literal_eval(text)
    except Exception:
        try:
            return json.loads(text.replace("'", '"'))
        except Exception:
            return {}


def cropwise_asof_value(history: dict, as_of_date: pd.Timestamp) -> float | None:
    if not history:
        return None
    parsed = []
    for key, value in history.items():
        date = pd.to_datetime(key, errors="coerce")
        if pd.isna(date):
            continue
        try:
            parsed.append((date, float(value)))
        except Exception:
            continue
    eligible = [(date, value) for date, value in parsed if date <= as_of_date]
    if not eligible:
        return None
    eligible.sort(key=lambda item: item[0])
    return eligible[-1][1]


def normalize_cropwise(value: float | None, std_name: str | None) -> float | None:
    if value is None:
        return None
    cap = MAX_T_HA.get(std_name or "", DEFAULT_MAX_T_HA)
    if value > cap:
        return value / 10.0
    return float(value)


def cropwise_table() -> pd.DataFrame:
    peh = pd.read_csv(DATA_RAW / "productivity_estimate_histories.csv", low_memory=False)
    peh["estimate_history_dict"] = peh["estimate_history"].apply(parse_estimate_history)
    peh = peh[["field_id", "year", "estimate_history_dict"]].copy()
    peh["field_id"] = peh["field_id"].astype(int)
    peh["year"] = peh["year"].astype(int)
    return peh


def metrics(y, pred) -> dict[str, float]:
    mask = np.isfinite(y) & np.isfinite(pred)
    y = np.asarray(y)[mask]
    pred = np.asarray(pred)[mask]
    if len(y) < 2:
        return {"r2": float("nan"), "rmse": float("nan"), "mae": float("nan"), "mape": float("nan")}
    return {
        "r2": float(r2_score(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "mae": float(mean_absolute_error(y, pred)),
        "mape": float(np.mean(np.abs((y - pred) / np.where(y > 0, y, np.nan))) * 100),
    }


def prepare_xy(df: pd.DataFrame, use_field_id: bool = True):
    non_feature = set(BASE_NON_FEATURE)
    cat_features = list(BASE_CAT_FEATURES)
    if not use_field_id:
        cat_features = [c for c in cat_features if c != "field_id"]
    feat_cols = [c for c in df.columns if c not in non_feature]
    if use_field_id and "field_id" not in feat_cols:
        feat_cols.append("field_id")
    if not use_field_id and "field_id" in feat_cols:
        feat_cols.remove("field_id")

    X = df[feat_cols].copy()
    cat_in_x = [c for c in cat_features if c in X.columns]
    for col in cat_in_x:
        if col == "rotation_pair":
            X[col] = X[col].fillna("missing").astype(str)
        else:
            X[col] = X[col].apply(lambda v: "missing" if pd.isna(v) else str(int(float(v))))
    for col in [c for c in feat_cols if c not in cat_in_x]:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    return X, df["target_yield_t_ha"].astype(float), feat_cols, cat_in_x


def walk_forward_oof(df: pd.DataFrame, params: dict, use_field_id: bool = True) -> np.ndarray:
    X, y, feat_cols, cat_in_x = prepare_xy(df, use_field_id=use_field_id)
    cat_idx = [feat_cols.index(c) for c in cat_in_x]
    years = df["year"].values
    oof = np.full(len(df), np.nan)
    for test_year in WALK_FORWARD_TEST_YEARS:
        test_mask = years == test_year
        train_mask = years < test_year
        if test_mask.sum() == 0:
            continue
        if train_mask.sum() < MIN_TRAIN_ROWS or len(pd.unique(years[train_mask])) < MIN_TRAIN_YEARS:
            continue

        X_train, y_train = X.iloc[train_mask], y.iloc[train_mask]
        X_test = X.iloc[test_mask]
        rng = np.random.default_rng(42)
        idx = np.arange(len(X_train))
        rng.shuffle(idx)
        val_n = max(5, int(len(X_train) * 0.2))
        val_idx, train_idx = idx[:val_n], idx[val_n:]

        model_params = {
            **params,
            "random_seed": 42,
            "od_type": "Iter",
            "od_wait": 50,
            "allow_writing_files": False,
            "verbose": False,
        }
        model = CatBoostRegressor(**model_params)
        model.fit(
            Pool(X_train.iloc[train_idx], y_train.iloc[train_idx], cat_features=cat_idx),
            eval_set=Pool(X_train.iloc[val_idx], y_train.iloc[val_idx], cat_features=cat_idx),
        )
        oof[test_mask] = model.predict(X_test)
    return oof


def attach_cropwise(df: pd.DataFrame, tag: str, cw_table: pd.DataFrame) -> pd.DataFrame:
    out = df.merge(cw_table, on=["field_id", "year"], how="left")
    month, day = map(int, tag.split("_"))
    values = []
    for _, row in out.iterrows():
        history = row.get("estimate_history_dict")
        value = cropwise_asof_value(
            history if isinstance(history, dict) else {},
            pd.Timestamp(year=int(row["year"]), month=month, day=day),
        )
        values.append(normalize_cropwise(value, row.get("standard_name")))
    out["cropwise_asof_t_ha"] = values
    return out


def load_dataset(version: str, tag: str, crop_filter: list[str] | None) -> pd.DataFrame:
    path = DATA_PROCESSED / f"ml_dataset_clean_{version}_asof_{tag}.csv"
    df = pd.read_csv(path)
    if "standard_name" not in df.columns:
        df = df.merge(CROPS_DF, on="crop_id", how="left")
    if crop_filter is not None:
        df = df[df["standard_name"].isin(crop_filter)].copy()
    return df.dropna(subset=["target_yield_t_ha"]).reset_index(drop=True)


def run_one(scenario: str, crop_filter: list[str] | None, tag: str, cw_table_df: pd.DataFrame):
    rows = []

    # v7 default
    df7 = attach_cropwise(load_dataset("v7", tag, crop_filter), tag, cw_table_df)
    if len(df7) >= 30:
        pred7 = walk_forward_oof(df7, PARAM_SETS["default"], use_field_id=True)
        cmp7 = df7.assign(pred=pred7).dropna(subset=["pred"])
        cmp7_all = cmp7.dropna(subset=["cropwise_asof_t_ha"])
        m7 = metrics(cmp7["target_yield_t_ha"], cmp7["pred"])
        cw = metrics(cmp7_all["target_yield_t_ha"], cmp7_all["cropwise_asof_t_ha"]) if len(cmp7_all) else {}
        rows.append(
            {
                "scenario": scenario,
                "asof_tag": tag,
                "asof_label": AS_OF_LABELS[tag],
                "model": "v7_default",
                "n_ml": len(cmp7),
                "n_cw": len(cmp7_all),
                "mape": m7["mape"],
                "rmse": m7["rmse"],
                "r2": m7["r2"],
                "cw_mape": cw.get("mape", float("nan")),
                "cw_rmse": cw.get("rmse", float("nan")),
                "cw_r2": cw.get("r2", float("nan")),
            }
        )

    df9 = attach_cropwise(load_dataset("v9", tag, crop_filter), tag, cw_table_df)
    if len(df9) < 30:
        return pd.DataFrame(rows)

    for param_name, params in PARAM_SETS.items():
        pred = walk_forward_oof(df9, params, use_field_id=True)
        cmp_df = df9.assign(pred=pred).dropna(subset=["pred"])
        cmp_all = cmp_df.dropna(subset=["cropwise_asof_t_ha"])
        m = metrics(cmp_df["target_yield_t_ha"], cmp_df["pred"])
        cw = metrics(cmp_all["target_yield_t_ha"], cmp_all["cropwise_asof_t_ha"]) if len(cmp_all) else {}
        rows.append(
            {
                "scenario": scenario,
                "asof_tag": tag,
                "asof_label": AS_OF_LABELS[tag],
                "model": f"v9_{param_name}",
                "n_ml": len(cmp_df),
                "n_cw": len(cmp_all),
                "mape": m["mape"],
                "rmse": m["rmse"],
                "r2": m["r2"],
                "cw_mape": cw.get("mape", float("nan")),
                "cw_rmse": cw.get("rmse", float("nan")),
                "cw_r2": cw.get("r2", float("nan")),
            }
        )

    pred_nf = walk_forward_oof(df9, PARAM_SETS["shallow_l10"], use_field_id=False)
    cmp_nf = df9.assign(pred=pred_nf).dropna(subset=["pred"])
    cmp_nf_all = cmp_nf.dropna(subset=["cropwise_asof_t_ha"])
    m_nf = metrics(cmp_nf["target_yield_t_ha"], cmp_nf["pred"])
    cw_nf = metrics(cmp_nf_all["target_yield_t_ha"], cmp_nf_all["cropwise_asof_t_ha"]) if len(cmp_nf_all) else {}
    rows.append(
        {
            "scenario": scenario,
            "asof_tag": tag,
            "asof_label": AS_OF_LABELS[tag],
            "model": "v9_no_field_id",
            "n_ml": len(cmp_nf),
            "n_cw": len(cmp_nf_all),
            "mape": m_nf["mape"],
            "rmse": m_nf["rmse"],
            "r2": m_nf["r2"],
            "cw_mape": cw_nf.get("mape", float("nan")),
            "cw_rmse": cw_nf.get("rmse", float("nan")),
            "cw_r2": cw_nf.get("r2", float("nan")),
        }
    )
    return pd.DataFrame(rows)


def plot_best(summary: pd.DataFrame) -> Path:
    best = (
        summary.sort_values(["scenario", "asof_tag", "mape"])
        .groupby(["scenario", "asof_tag"], as_index=False)
        .first()
    )
    plt.figure(figsize=(9, 5.5))
    for scenario in best["scenario"].unique():
        s = best[best["scenario"] == scenario].sort_values("asof_tag")
        plt.plot(s["asof_label"], s["mape"], marker="o", lw=2.5, label=f"best ML — {scenario}")
        plt.plot(s["asof_label"], s["cw_mape"], marker="s", lw=2, ls="--", color="grey", label=f"Cropwise — {scenario}")
    plt.title("v9 best tuned ML vs Cropwise (factual target)")
    plt.xlabel("As-of forecast date")
    plt.ylabel("MAPE, %")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    path = FIGS / "asof_mape_v9_best.png"
    plt.savefig(path, dpi=140)
    plt.close()
    return path


def main() -> None:
    print("=" * 80)
    print("TRAIN v9 — FACTUAL TARGET + SOWING FEATURES + TUNING")
    print("=" * 80)
    cw = cropwise_table()
    scenarios = [
        ("sunflower", ["sunflower"]),
        ("wheat_combined", ["wheat_spring", "wheat_winter"]),
        ("all_crops", None),
    ]
    parts = []
    for scenario, crop_filter in scenarios:
        print(f"\n{scenario}")
        for tag in AS_OF_TAGS:
            result = run_one(scenario, crop_filter, tag, cw)
            if result.empty:
                print(f"  {tag}: no result")
                continue
            best = result.sort_values("mape").iloc[0]
            v7 = result[result["model"] == "v7_default"].iloc[0] if (result["model"] == "v7_default").any() else None
            print(
                f"  {tag}: best={best['model']} MAPE={best['mape']:.1f}% "
                f"R2={best['r2']:.3f}; CW={best['cw_mape']:.1f}%"
                + (f"; v7={v7['mape']:.1f}%" if v7 is not None else "")
            )
            parts.append(result)
    if not parts:
        print("No results")
        return

    summary = pd.concat(parts, ignore_index=True)
    summary.to_csv(COMP_DIR / "asof_results_v9.csv", index=False)
    best = (
        summary.sort_values(["scenario", "asof_tag", "mape"])
        .groupby(["scenario", "asof_tag"], as_index=False)
        .first()
    )
    plot_path = plot_best(summary)

    md = ["# v9 — Factual Target + Real Sowing-Date Features + Tuning\n\n"]
    md.append("_Generated by `scripts/asof/train_compare_asof_v9.py`._\n\n")
    md.append("## Best model per scenario/date\n\n")
    md.append(best.round(3).to_markdown(index=False))
    md.append("\n\n## Full grid\n\n")
    md.append(summary.round(3).to_markdown(index=False))
    md.append("\n\n")
    md.append(f"![v9 best](figures/{plot_path.name})\n")
    (REPORTS / "asof_v9_results.md").write_text("".join(md), encoding="utf-8")

    print("\n" + "=" * 80)
    print("FINAL BEST")
    print("=" * 80)
    print(best.round(3).to_string(index=False))
    print(f"\nSaved {COMP_DIR / 'asof_results_v9.csv'}")
    print(f"Saved {REPORTS / 'asof_v9_results.md'}")


if __name__ == "__main__":
    main()
