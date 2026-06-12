"""
v19 hybrid calibration.

Idea:
  1. v18 peer model learns the global agronomic function on 26k peer rows.
  2. v19 calibrates those peer predictions on our 30 local fields using only
     local past years.

Input:
  models_v2/asof_comparison/asof_predictions_v18_peer_transfer.csv

The input `pred` column is already leakage-safe:
  for each local test year Y, it was produced by a peer model trained on
  peer rows with year < Y.

Calibration validation:
  For each local test year Y, fit calibration only on local rows with year < Y.
  If not enough local rows exist, fall back to raw peer prediction.

Calibration candidates:
  - peer_raw: raw v18 peer transfer prediction
  - residual_global: peer_pred + mean(local residual)
  - residual_crop: peer_pred + mean(local residual by crop)
  - linear_global: Ridge target ~ peer_pred
  - linear_crop: Ridge target ~ peer_pred + crop one-hot
  - blend_crop_mean: alpha * peer_pred + (1-alpha) * local crop mean,
    alpha selected on local train by MAPE grid.

Outputs:
  models_v2/asof_comparison/asof_results_v19_hybrid_calibration.csv
  models_v2/asof_comparison/asof_predictions_v19_hybrid_calibration.csv
  reports/asof_v19_hybrid_calibration_results.md
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT_DIR = Path(__file__).resolve().parents[2]
os.chdir(ROOT_DIR)

MODELS_V2 = Path("models_v2")
REPORTS = Path("reports")
COMP_DIR = MODELS_V2 / "asof_comparison"
COMP_DIR.mkdir(parents=True, exist_ok=True)

INPUT = COMP_DIR / "asof_predictions_v18_peer_transfer.csv"
ASOF_LABELS = {"07_01": "1 Jul", "08_01": "1 Aug", "09_01": "1 Sep"}
MIN_LOCAL_TRAIN_ROWS = 20


def metrics(y, pred) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    mask = np.isfinite(y) & np.isfinite(pred)
    y = y[mask]
    pred = pred[mask]
    if len(y) < 2:
        return {"r2": np.nan, "rmse": np.nan, "mae": np.nan, "mape": np.nan}
    return {
        "r2": float(r2_score(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "mae": float(mean_absolute_error(y, pred)),
        "mape": float(np.mean(np.abs((y - pred) / np.where(y > 0, y, np.nan))) * 100.0),
    }


def _crop_means(train: pd.DataFrame) -> tuple[pd.Series, float]:
    global_mean = float(train["target_yield_t_ha"].mean())
    means = train.groupby("standard_name")["target_yield_t_ha"].mean()
    return means, global_mean


def pred_residual_global(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    residual = (train["target_yield_t_ha"] - train["pred"]).mean()
    return test["pred"].values + float(residual)


def pred_residual_crop(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    train_resid = train.assign(residual=train["target_yield_t_ha"] - train["pred"])
    global_resid = float(train_resid["residual"].mean())
    resid_by_crop = train_resid.groupby("standard_name")["residual"].mean()
    correction = test["standard_name"].map(resid_by_crop).fillna(global_resid).astype(float).values
    return test["pred"].values + correction


def pred_linear_global(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    if train["pred"].nunique(dropna=True) < 2:
        return test["pred"].values
    model = Pipeline([
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=1.0)),
    ])
    model.fit(train[["pred"]], train["target_yield_t_ha"])
    return model.predict(test[["pred"]])


def pred_linear_crop(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    model = Pipeline([
        (
            "prep",
            ColumnTransformer(
                transformers=[
                    ("num", StandardScaler(), ["pred"]),
                    ("cat", OneHotEncoder(handle_unknown="ignore"), ["standard_name"]),
                ]
            ),
        ),
        ("ridge", Ridge(alpha=3.0)),
    ])
    model.fit(train[["pred", "standard_name"]], train["target_yield_t_ha"])
    return model.predict(test[["pred", "standard_name"]])


def pred_blend_crop_mean(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    means, global_mean = _crop_means(train)
    train_crop_mean = train["standard_name"].map(means).fillna(global_mean).astype(float).values
    test_crop_mean = test["standard_name"].map(means).fillna(global_mean).astype(float).values

    # Select alpha by train MAPE. Use a small discrete grid to avoid overfitting.
    best_alpha = 1.0
    best_mape = np.inf
    y = train["target_yield_t_ha"].values
    for alpha in np.linspace(0.0, 1.0, 11):
        p = alpha * train["pred"].values + (1.0 - alpha) * train_crop_mean
        m = metrics(y, p)["mape"]
        if m < best_mape:
            best_mape = m
            best_alpha = float(alpha)
    return best_alpha * test["pred"].values + (1.0 - best_alpha) * test_crop_mean


CALIBRATORS = {
    "peer_raw": lambda train, test: test["pred"].values,
    "residual_global": pred_residual_global,
    "residual_crop": pred_residual_crop,
    "linear_global": pred_linear_global,
    "linear_crop": pred_linear_crop,
    "blend_crop_mean": pred_blend_crop_mean,
}


def calibrated_oof(df: pd.DataFrame, method: str) -> np.ndarray:
    out = np.full(len(df), np.nan)
    fn = CALIBRATORS[method]
    for test_year in sorted(df["year"].dropna().unique()):
        test_mask = df["year"].values == test_year
        train = df[df["year"] < test_year].dropna(subset=["target_yield_t_ha", "pred"]).copy()
        test = df[test_mask].dropna(subset=["pred"]).copy()
        if test.empty:
            continue
        if method == "peer_raw" or len(train) < MIN_LOCAL_TRAIN_ROWS or train["year"].nunique() < 2:
            p = test["pred"].values
        else:
            try:
                p = fn(train, test)
            except Exception:
                p = test["pred"].values
        out[test.index.values] = p
    return out


def evaluate_group(df: pd.DataFrame, scenario: str, tag: str) -> tuple[list[dict[str, object]], list[pd.DataFrame]]:
    rows: list[dict[str, object]] = []
    preds: list[pd.DataFrame] = []
    for method in CALIBRATORS:
        p = calibrated_oof(df, method)
        cmp_df = df.assign(v19_pred=p).dropna(subset=["v19_pred"])
        if cmp_df.empty:
            continue
        ml = metrics(cmp_df["target_yield_t_ha"], cmp_df["v19_pred"])
        cw_df = cmp_df.dropna(subset=["cropwise_asof_t_ha"])
        cw = metrics(cw_df["target_yield_t_ha"], cw_df["cropwise_asof_t_ha"]) if len(cw_df) else {}
        naive = metrics(cmp_df["target_yield_t_ha"], cmp_df["naive_crop_mean"])
        rows.append({
            "scenario": scenario,
            "asof_tag": tag,
            "asof_label": ASOF_LABELS[tag],
            "method": method,
            "n": len(cmp_df),
            "mape": ml["mape"],
            "mae": ml["mae"],
            "rmse": ml["rmse"],
            "r2": ml["r2"],
            "cw_mape": cw.get("mape", np.nan),
            "cw_mae": cw.get("mae", np.nan),
            "cw_rmse": cw.get("rmse", np.nan),
            "cw_r2": cw.get("r2", np.nan),
            "naive_mape": naive["mape"],
            "naive_mae": naive["mae"],
            "naive_rmse": naive["rmse"],
            "naive_r2": naive["r2"],
        })
        keep = cmp_df[[
            "scenario",
            "asof_tag",
            "field_id",
            "year",
            "standard_name",
            "target_yield_t_ha",
            "pred",
            "cropwise_asof_t_ha",
            "naive_crop_mean",
            "v19_pred",
        ]].copy()
        keep["method"] = method
        preds.append(keep)
    return rows, preds


def add_filtered_summary(summary: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    """Extra sensitivity views for low-yield denominator explosions."""
    rows = []
    filters = {
        "all": pred_df.index == pred_df.index,
        "target_ge_1": pred_df["target_yield_t_ha"] >= 1.0,
        "year_ge_2021": pred_df["year"] >= 2021,
        "target_ge_1_year_ge_2021": (pred_df["target_yield_t_ha"] >= 1.0) & (pred_df["year"] >= 2021),
    }
    for filter_name, mask in filters.items():
        cur = pred_df[mask].copy()
        for (scenario, tag, method), sub in cur.groupby(["scenario", "asof_tag", "method"]):
            if len(sub) < 2:
                continue
            ml = metrics(sub["target_yield_t_ha"], sub["v19_pred"])
            cw = metrics(sub["target_yield_t_ha"], sub["cropwise_asof_t_ha"])
            nv = metrics(sub["target_yield_t_ha"], sub["naive_crop_mean"])
            rows.append({
                "filter": filter_name,
                "scenario": scenario,
                "asof_tag": tag,
                "asof_label": ASOF_LABELS[tag],
                "method": method,
                "n": len(sub),
                "mape": ml["mape"],
                "mae": ml["mae"],
                "rmse": ml["rmse"],
                "r2": ml["r2"],
                "cw_mape": cw["mape"],
                "cw_mae": cw["mae"],
                "cw_rmse": cw["rmse"],
                "cw_r2": cw["r2"],
                "naive_mape": nv["mape"],
                "naive_mae": nv["mae"],
                "naive_rmse": nv["rmse"],
                "naive_r2": nv["r2"],
            })
    return pd.DataFrame(rows)


def main() -> None:
    print("=" * 80)
    print("v19 HYBRID CALIBRATION")
    print("=" * 80)

    raw = pd.read_csv(INPUT)
    all_rows: list[dict[str, object]] = []
    all_preds: list[pd.DataFrame] = []
    for (scenario, tag), group in raw.groupby(["scenario", "asof_tag"]):
        rows, preds = evaluate_group(group.reset_index(drop=True), scenario, tag)
        all_rows.extend(rows)
        all_preds.extend(preds)
        sub = pd.DataFrame(rows)
        best = sub.sort_values("mape").iloc[0]
        best_r2 = sub.sort_values("r2", ascending=False).iloc[0]
        print(
            f"{scenario} {tag}: best_mape={best['method']} {best['mape']:.1f}% "
            f"R2={best['r2']:.3f}; best_r2={best_r2['method']} "
            f"{best_r2['r2']:.3f} MAPE={best_r2['mape']:.1f}%; "
            f"CW={best['cw_mape']:.1f}% R2={best['cw_r2']:.3f}"
        )

    summary = pd.DataFrame(all_rows)
    pred_df = pd.concat(all_preds, ignore_index=True)
    filtered = add_filtered_summary(summary, pred_df)

    out_results = COMP_DIR / "asof_results_v19_hybrid_calibration.csv"
    out_preds = COMP_DIR / "asof_predictions_v19_hybrid_calibration.csv"
    out_filtered = COMP_DIR / "asof_results_v19_hybrid_calibration_filtered.csv"
    summary.to_csv(out_results, index=False)
    pred_df.to_csv(out_preds, index=False)
    filtered.to_csv(out_filtered, index=False)

    best = (
        summary.sort_values(["scenario", "asof_tag", "mape"])
        .groupby(["scenario", "asof_tag"], as_index=False)
        .first()
    )
    best_filtered = (
        filtered[filtered["filter"].isin(["target_ge_1", "target_ge_1_year_ge_2021"])]
        .sort_values(["filter", "scenario", "asof_tag", "mape"])
        .groupby(["filter", "scenario", "asof_tag"], as_index=False)
        .first()
    )

    md = ["# v19 hybrid calibration results\n\n"]
    md.append("Train calibration on local past years only; test on local target year.\n\n")
    md.append("## Best method per scenario/date (all rows)\n\n")
    md.append(best.round(4).to_markdown(index=False))
    md.append("\n\n## Best method under sensitivity filters\n\n")
    md.append(best_filtered.round(4).to_markdown(index=False))
    md.append("\n\n## Full grid\n\n")
    md.append(summary.round(4).to_markdown(index=False))
    md.append("\n")
    out_md = REPORTS / "asof_v19_hybrid_calibration_results.md"
    out_md.write_text("".join(md), encoding="utf-8")

    print(f"\nSaved {out_results}")
    print(f"Saved {out_preds}")
    print(f"Saved {out_filtered}")
    print(f"Saved {out_md}")


if __name__ == "__main__":
    main()
