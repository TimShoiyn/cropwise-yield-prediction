"""
v21: peer-transfer with leakage-safe NDVI domain alignment + honest metrics.

Problem found in Audit-2:
  - Our fields are systematically greener (NDVI mean 0.467 vs peers 0.407).
    A peer-trained model sees a shifted feature scale -> biased transfer.
  - Pooled R2 is dominated by crop/year level; the honest skill is
    within-crop-year R2, which is ~0 for everyone.

What v21 does:
  1. Domain alignment: for each test year, shift OUR level-NDVI features into
     peer space using ONLY past years (local_past mean vs peer_past mean).
     This is leakage-safe (no test-year info used).
  2. Optionally restrict peer training crops to crops we actually have.
  3. Reports pooled R2, within-crop-year R2 (honest ranking skill) and MAPE for
     v21, raw v18 transfer and Cropwise on identical rows.

Outputs:
  models_v2/asof_comparison/asof_results_v21_aligned.csv
  models_v2/asof_comparison/asof_predictions_v21_aligned.csv
  reports/AUDIT2_V21_RESULTS.md
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT_DIR = Path(__file__).resolve().parents[2]
os.chdir(ROOT_DIR)
sys.path.insert(0, str(ROOT_DIR))

from scripts.asof.evaluate_v18_peer_transfer import (  # noqa: E402
    ASOF_LABELS,
    ASOF_TAGS,
    CAT_COLS,
    FEATURE_COLS,
    PARAMS,
    POLICY,
    SCENARIOS,
    load_ours_as_v18,
    load_peers,
    model_params,
    prepare_X,
)
from scripts.asof.train_compare_asof_v9 import cropwise_table  # noqa: E402

COMP_DIR = Path("models_v2/asof_comparison")
REPORTS = Path("reports")
COMP_DIR.mkdir(parents=True, exist_ok=True)

# Level features that live on the raw-NDVI scale and suffer from the offset.
ALIGN_FEATURES = [
    "ndvi_mean_asof",
    "ndvi_max_asof",
    "ndvi_min_asof",
    "ndvi_p25_asof",
    "ndvi_p75_asof",
    "ndvi_last_value_asof",
    "ndvi_last30d_mean_asof",
]
MIN_LOCAL_PAST = 15  # need enough local past rows to estimate a stable shift


def metrics(y, pred) -> dict[str, float]:
    y = np.asarray(y, float)
    pred = np.asarray(pred, float)
    m = np.isfinite(y) & np.isfinite(pred)
    y, pred = y[m], pred[m]
    if len(y) < 2:
        return {"r2": np.nan, "rmse": np.nan, "mae": np.nan, "mape": np.nan}
    return {
        "r2": float(r2_score(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "mae": float(mean_absolute_error(y, pred)),
        "mape": float(np.mean(np.abs((y - pred) / np.where(y > 0, y, np.nan))) * 100.0),
    }


def within_crop_year_r2(df: pd.DataFrame, pred_col: str) -> float:
    sub = df.dropna(subset=["target_yield_t_ha", pred_col, "standard_name", "year"]).copy()
    if len(sub) < 4:
        return np.nan
    y = sub["target_yield_t_ha"].astype(float)
    p = sub[pred_col].astype(float)
    g = [sub["standard_name"], sub["year"]]
    y_dm = y - y.groupby(g).transform("mean")
    p_dm = p - p.groupby(g).transform("mean")
    ss_tot = float((y_dm ** 2).sum())
    if ss_tot <= 1e-9:
        return np.nan
    return 1.0 - float(((y_dm - p_dm) ** 2).sum()) / ss_tot


def predict_aligned(peers: pd.DataFrame, ours: pd.DataFrame, params_name: str, residual: bool):
    """Returns (pred, n_aligned_years). Shifts OUR features into peer space."""
    pred = np.full(len(ours), np.nan)
    cat_idx = [FEATURE_COLS.index(c) for c in CAT_COLS]
    aligned_years = 0

    for test_year in sorted(ours["year"].unique()):
        test_mask = ours["year"].values == test_year
        train = peers[peers["year"] < test_year].copy()
        if len(train) < 500:
            continue
        test = ours[test_mask].copy()
        local_past = ours[ours["year"] < test_year]

        # leakage-safe NDVI shift: move our test features into peer space
        if len(local_past) >= MIN_LOCAL_PAST:
            aligned_years += 1
            for f in ALIGN_FEATURES:
                lp = pd.to_numeric(local_past[f], errors="coerce").mean()
                pp = pd.to_numeric(train[f], errors="coerce").mean()
                if np.isfinite(lp) and np.isfinite(pp):
                    test[f] = pd.to_numeric(test[f], errors="coerce") + (pp - lp)

        X_train = prepare_X(train)
        X_test = prepare_X(test)
        y_train = train["target_yield_t_ha"].astype(float)
        global_mean = float(y_train.mean())
        y_fit = y_train.copy()
        crop_mean = None
        if residual:
            crop_mean = y_train.groupby(train["standard_name"]).mean()
            y_fit = y_train - train["standard_name"].map(crop_mean).fillna(global_mean).astype(float)

        rng = np.random.default_rng(42)
        idx = np.arange(len(X_train))
        rng.shuffle(idx)
        val_n = max(50, int(len(X_train) * 0.15))
        val_idx, train_idx = idx[:val_n], idx[val_n:]

        model = CatBoostRegressor(**model_params(PARAMS[params_name]))
        model.fit(
            Pool(X_train.iloc[train_idx], y_fit.iloc[train_idx], cat_features=cat_idx),
            eval_set=Pool(X_train.iloc[val_idx], y_fit.iloc[val_idx], cat_features=cat_idx),
        )
        p = model.predict(X_test)
        if residual and crop_mean is not None:
            p = p + test["standard_name"].map(crop_mean).fillna(global_mean).astype(float).values
        pred[test_mask] = p
    return pred, aligned_years


def main() -> None:
    print("=" * 80)
    print("v21 DOMAIN-ALIGNED PEER TRANSFER (restrict peers to our crops)")
    print("=" * 80)
    cw = cropwise_table()
    # raw v18 predictions for side-by-side comparison
    raw = pd.read_csv(COMP_DIR / "asof_predictions_v18_peer_transfer.csv")

    rows, pred_rows = [], []
    for scenario, crop_filter in SCENARIOS:
        print(f"\n{scenario}")
        for tag in ASOF_TAGS:
            ours = load_ours_as_v18(tag, crop_filter, cw)
            our_crops = list(ours["standard_name"].dropna().unique())
            # restrict peer training to crops we actually grow
            peers = load_peers(tag, our_crops if crop_filter is None else crop_filter)
            params_name, residual = POLICY[(scenario, tag)]
            pred, n_aln = predict_aligned(peers, ours, params_name, residual)

            cmp_df = ours.assign(pred=pred).dropna(subset=["pred"])
            cmp_cw = cmp_df.dropna(subset=["cropwise_asof_t_ha"])

            ml = metrics(cmp_df["target_yield_t_ha"], cmp_df["pred"])
            cw_m = metrics(cmp_cw["target_yield_t_ha"], cmp_cw["cropwise_asof_t_ha"]) if len(cmp_cw) else {}

            # raw v18 on the same scenario/tag for comparison
            raw_sub = raw[(raw["scenario"] == scenario) & (raw["asof_tag"] == tag)]
            raw_m = metrics(raw_sub["target_yield_t_ha"], raw_sub["pred"]) if len(raw_sub) else {}

            row = {
                "scenario": scenario, "asof": ASOF_LABELS[tag], "n": len(cmp_df),
                "aligned_years": n_aln,
                "v21_mape": ml["mape"], "v21_r2": ml["r2"],
                "v21_wcy_r2": within_crop_year_r2(cmp_df, "pred"),
                "v18raw_mape": raw_m.get("mape", np.nan), "v18raw_r2": raw_m.get("r2", np.nan),
                "v18raw_wcy_r2": within_crop_year_r2(raw_sub, "pred") if len(raw_sub) else np.nan,
                "cw_mape": cw_m.get("mape", np.nan), "cw_r2": cw_m.get("r2", np.nan),
                "cw_wcy_r2": within_crop_year_r2(cmp_cw, "cropwise_asof_t_ha") if len(cmp_cw) else np.nan,
            }
            rows.append(row)
            t = cmp_df[["field_id", "year", "standard_name", "target_yield_t_ha", "pred", "cropwise_asof_t_ha"]].copy()
            t["scenario"], t["asof_tag"] = scenario, tag
            pred_rows.append(t)
            print(
                f"  {tag}: v21 MAPE={row['v21_mape']:.1f}% R2={row['v21_r2']:.3f} wcyR2={row['v21_wcy_r2']:.3f} | "
                f"v18raw MAPE={row['v18raw_mape']:.1f}% R2={row['v18raw_r2']:.3f} | "
                f"CW MAPE={row['cw_mape']:.1f}% R2={row['cw_r2']:.3f} wcyR2={row['cw_wcy_r2']:.3f} (aln_yrs={n_aln})"
            )

    summary = pd.DataFrame(rows)
    summary.to_csv(COMP_DIR / "asof_results_v21_aligned.csv", index=False)
    pd.concat(pred_rows, ignore_index=True).to_csv(COMP_DIR / "asof_predictions_v21_aligned.csv", index=False)

    md = ["# v21 domain-aligned transfer + honest metrics\n\n"]
    md.append(
        "NDVI level features shifted into peer space using past-years stats only "
        "(leakage-safe). Peer training restricted to our crops. `wcy_r2` = "
        "within-crop-year R2 (honest field-ranking skill).\n\n"
    )
    md.append(summary.round(3).to_markdown(index=False) + "\n")
    (REPORTS / "AUDIT2_V21_RESULTS.md").write_text("".join(md), encoding="utf-8")
    print(f"\nSaved {COMP_DIR / 'asof_results_v21_aligned.csv'}")
    print(f"Saved {REPORTS / 'AUDIT2_V21_RESULTS.md'}")


if __name__ == "__main__":
    main()
