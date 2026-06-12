"""
Microscope-level audit before drawing final conclusions.

Sections (each writes its own block in the report):

  1. Naive baselines vs ML vs Cropwise on the v17 datasets
  2. Cropwise as-of timing audit (does the value really exist before the date?)
  3. Walk-forward sample / field distribution per test year
  4. CatBoost feature importance + a permutation test for v17 best policy
  5. In-sample vs OOF gap (overfitting check) on the latest test year
  6. Per-year breakdown of ML and Cropwise errors
  7. Target distribution drift train vs test
  8. physical_t_ha sanity (partial harvest, area mismatch)

Outputs:
    reports/MICROSCOPE_AUDIT.md
    reports/microscope/baselines.csv
    reports/microscope/cw_timing.csv
    reports/microscope/wf_distribution.csv
    reports/microscope/feature_importance.csv
    reports/microscope/insample_vs_oof.csv
    reports/microscope/per_year_errors.csv
    reports/microscope/physical_target_sanity.csv
"""

from __future__ import annotations

import ast
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
os.chdir(ROOT_DIR)
sys.path.insert(0, str(ROOT_DIR))

import scripts.asof.train_compare_asof_v9 as base  # noqa: E402

base.BASE_NON_FEATURE = base.BASE_NON_FEATURE | {
    "target_v17",
    "target_v17_source",
    "target_v17_conflict",
    "physical_t_ha",
    "prod_fact_t_ha",
    "productivity_t_ha",
}

from scripts.asof.train_compare_asof_v9 import (  # noqa: E402
    AS_OF_LABELS,
    AS_OF_TAGS,
    PARAM_SETS,
    REPORTS,
    attach_cropwise,
    cropwise_table,
    load_dataset,
    metrics,
    prepare_xy,
    walk_forward_oof,
    parse_estimate_history,
)
from catboost import CatBoostRegressor, Pool  # noqa: E402

DATA_RAW = Path("data_raw")
DATA_PROCESSED = Path("data_processed")
OUT_DIR = REPORTS / "microscope"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCENARIOS = [
    ("sunflower", ["sunflower"]),
    ("wheat_combined", ["wheat_spring", "wheat_winter"]),
    ("all_crops", None),
]

# Best v17 policy choices (from train_compare_asof_v17_strict.py output).
V17_BEST = {
    ("sunflower", "07_01"): ("v17_v12_full", "shallow_l10", False),
    ("sunflower", "08_01"): ("v17_v12_full", "default", True),
    ("sunflower", "09_01"): ("v17_v12_full", "default", True),
    ("wheat_combined", "07_01"): ("v17_v15_rates_clean", "shallow_l10", True),
    ("wheat_combined", "08_01"): ("v17_v12_clean", "shallow_l30", True),
    ("wheat_combined", "09_01"): ("v17_v12_clean", "default", True),
    ("all_crops", "07_01"): ("v17_v12_clean", "shallow_l30", True),
    ("all_crops", "08_01"): ("v17_v14_clean", "shallow_l10", True),
    ("all_crops", "09_01"): ("v17_v12_clean", "shallow_l10", False),
}


def err_block(y, pred):
    m = metrics(np.asarray(y), np.asarray(pred))
    return {"mape": m["mape"], "mae": m["mae"], "rmse": m["rmse"], "r2": m["r2"]}


def naive_walk_forward(df: pd.DataFrame, mode: str) -> np.ndarray:
    """Return OOF predictions for several naive baselines."""
    pred = np.full(len(df), np.nan)
    for test_year in sorted(df["year"].unique()):
        train = df[df["year"] < test_year]
        test_mask = df["year"] == test_year
        if train.empty:
            continue
        if mode == "global_mean":
            value = train["target_yield_t_ha"].mean()
            pred[test_mask] = value
        elif mode == "crop_mean":
            crop_means = train.groupby("standard_name")["target_yield_t_ha"].mean()
            global_mean = train["target_yield_t_ha"].mean()
            pred[test_mask] = df.loc[test_mask, "standard_name"].map(crop_means).fillna(global_mean).values
        elif mode == "field_mean":
            field_means = train.groupby("field_id")["target_yield_t_ha"].mean()
            global_mean = train["target_yield_t_ha"].mean()
            pred[test_mask] = df.loc[test_mask, "field_id"].map(field_means).fillna(global_mean).values
        elif mode == "field_crop_mean":
            fc_means = train.groupby(["field_id", "standard_name"])["target_yield_t_ha"].mean()
            crop_means = train.groupby("standard_name")["target_yield_t_ha"].mean()
            global_mean = train["target_yield_t_ha"].mean()
            keys = list(zip(df.loc[test_mask, "field_id"], df.loc[test_mask, "standard_name"]))
            arr = []
            for fid, crop in keys:
                v = fc_means.get((fid, crop))
                if pd.isna(v) or v is None:
                    v = crop_means.get(crop, global_mean)
                arr.append(v)
            pred[test_mask] = arr
        elif mode == "last_year":
            for fid in df.loc[test_mask, "field_id"].unique():
                hist = train[train["field_id"] == fid].sort_values("year")
                last = hist["target_yield_t_ha"].iloc[-1] if not hist.empty else train["target_yield_t_ha"].mean()
                pred[(test_mask) & (df["field_id"] == fid)] = last
        else:
            raise ValueError(mode)
    return pred


def section_naive_baselines(cw: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    rows = []
    for scenario, crop_filter in SCENARIOS:
        for tag in AS_OF_TAGS:
            version, params_name, use_field_id = V17_BEST[(scenario, tag)]
            df = attach_cropwise(load_dataset(version, tag, crop_filter), tag, cw)
            if df.empty:
                continue
            ml_pred = walk_forward_oof(df, PARAM_SETS[params_name], use_field_id=use_field_id)
            ml_mask = ~np.isnan(ml_pred)
            ml_df = df.assign(pred=ml_pred).dropna(subset=["pred"])

            mlb = err_block(ml_df["target_yield_t_ha"], ml_df["pred"])
            cw_df = ml_df.dropna(subset=["cropwise_asof_t_ha"])
            cwb = err_block(cw_df["target_yield_t_ha"], cw_df["cropwise_asof_t_ha"]) if len(cw_df) else {}

            base_metrics = {}
            for mode in ["global_mean", "crop_mean", "field_mean", "field_crop_mean", "last_year"]:
                p = naive_walk_forward(df, mode)
                # Restrict to the same rows where ML produced a prediction.
                p_aligned = np.where(ml_mask, p, np.nan)
                m_df = df.assign(pred=p_aligned).dropna(subset=["pred"])
                base_metrics[mode] = err_block(m_df["target_yield_t_ha"], m_df["pred"]) if len(m_df) else {}

            row = {
                "scenario": scenario,
                "asof_tag": tag,
                "asof_label": AS_OF_LABELS[tag],
                "n_ml": len(ml_df),
                "ml_mape": mlb["mape"], "ml_mae": mlb["mae"], "ml_rmse": mlb["rmse"], "ml_r2": mlb["r2"],
                "cw_mape": cwb.get("mape", float("nan")), "cw_mae": cwb.get("mae", float("nan")),
                "cw_rmse": cwb.get("rmse", float("nan")), "cw_r2": cwb.get("r2", float("nan")),
            }
            for mode, m in base_metrics.items():
                row[f"{mode}_mape"] = m.get("mape", float("nan"))
                row[f"{mode}_mae"] = m.get("mae", float("nan"))
                row[f"{mode}_rmse"] = m.get("rmse", float("nan"))
                row[f"{mode}_r2"] = m.get("r2", float("nan"))
            rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "baselines.csv", index=False)
    md = "## 1. Naive baselines vs ML vs Cropwise\n\n"
    md += "Walk-forward OOF on the same rows as the best v17 policy. MAPE is in %, MAE/RMSE in t/ha.\n\n"
    cols = [
        "scenario", "asof_label", "n_ml",
        "ml_mape", "cw_mape", "global_mean_mape", "crop_mean_mape",
        "field_mean_mape", "field_crop_mean_mape", "last_year_mape",
    ]
    md += out[cols].round(2).to_markdown(index=False) + "\n\n"
    md += "### MAE block (lower is better)\n\n"
    cols = [
        "scenario", "asof_label",
        "ml_mae", "cw_mae", "global_mean_mae", "crop_mean_mae",
        "field_mean_mae", "field_crop_mean_mae", "last_year_mae",
    ]
    md += out[cols].round(3).to_markdown(index=False) + "\n\n"
    md += "### R^2 block\n\n"
    cols = [
        "scenario", "asof_label",
        "ml_r2", "cw_r2", "global_mean_r2", "crop_mean_r2",
        "field_mean_r2", "field_crop_mean_r2", "last_year_r2",
    ]
    md += out[cols].round(3).to_markdown(index=False) + "\n\n"
    return md, out


def section_cropwise_timing(cw_table: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    rows = []
    base_df = pd.read_csv(DATA_RAW / "productivity_estimate_histories.csv", low_memory=False)
    base_df["estimate_history_dict"] = base_df["estimate_history"].apply(parse_estimate_history)

    for tag in AS_OF_TAGS:
        month, day = map(int, tag.split("_"))
        n_have_value = 0
        n_total = 0
        first_dates = []
        used_offset_days = []
        for _, r in base_df.iterrows():
            year = int(r["year"])
            history = r["estimate_history_dict"]
            if not isinstance(history, dict) or not history:
                continue
            n_total += 1
            asof_ts = pd.Timestamp(year=year, month=month, day=day)
            parsed = []
            for k, v in history.items():
                d = pd.to_datetime(k, errors="coerce")
                if pd.isna(d):
                    continue
                try:
                    parsed.append((d, float(v)))
                except Exception:
                    continue
            eligible = [(d, v) for d, v in parsed if d <= asof_ts]
            if not eligible:
                continue
            eligible.sort(key=lambda it: it[0])
            chosen_date, _ = eligible[-1]
            first_dates.append(chosen_date)
            used_offset_days.append((asof_ts - chosen_date).days)
            n_have_value += 1
        rows.append({
            "asof_tag": tag,
            "asof_label": AS_OF_LABELS[tag],
            "n_field_year_with_history": n_total,
            "n_with_value_at_asof": n_have_value,
            "share_with_value": n_have_value / n_total if n_total else 0,
            "median_offset_days": float(np.median(used_offset_days)) if used_offset_days else float("nan"),
            "p90_offset_days": float(np.percentile(used_offset_days, 90)) if used_offset_days else float("nan"),
            "max_offset_days": float(np.max(used_offset_days)) if used_offset_days else float("nan"),
            "min_offset_days": float(np.min(used_offset_days)) if used_offset_days else float("nan"),
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "cw_timing.csv", index=False)
    md = "## 2. Cropwise as-of timing audit\n\n"
    md += (
        "We pick the latest Cropwise estimate with date <= as-of. "
        "If `min_offset_days` is negative, we would be using a future estimate (leakage); "
        "if it's always >= 0 we are safe. `median_offset_days` shows how stale the estimate is.\n\n"
    )
    md += out.round(2).to_markdown(index=False) + "\n\n"
    return md, out


def section_walk_forward_distribution(cw: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    rows = []
    for scenario, crop_filter in SCENARIOS:
        for tag in AS_OF_TAGS:
            version, _, _ = V17_BEST[(scenario, tag)]
            df = load_dataset(version, tag, crop_filter)
            for test_year in sorted(df["year"].unique()):
                train = df[df["year"] < test_year]
                test = df[df["year"] == test_year]
                rows.append({
                    "scenario": scenario,
                    "asof_tag": tag,
                    "test_year": int(test_year),
                    "n_train": len(train),
                    "n_train_years": int(train["year"].nunique()) if len(train) else 0,
                    "n_test": len(test),
                    "test_target_mean": float(test["target_yield_t_ha"].mean()) if len(test) else float("nan"),
                    "train_target_mean": float(train["target_yield_t_ha"].mean()) if len(train) else float("nan"),
                    "test_target_std": float(test["target_yield_t_ha"].std()) if len(test) else float("nan"),
                    "train_target_std": float(train["target_yield_t_ha"].std()) if len(train) else float("nan"),
                })
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "wf_distribution.csv", index=False)
    md = "## 3. Walk-forward sample / field distribution per test year\n\n"
    md += "For each scenario / as-of date / test year: number of train rows, train years, test rows, target stats.\n\n"
    md += out[out["asof_tag"] == "08_01"].round(3).to_markdown(index=False) + "\n\n"
    return md, out


def section_feature_importance(cw: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    rows = []
    for scenario, crop_filter in SCENARIOS:
        tag = "08_01"
        version, params_name, use_field_id = V17_BEST[(scenario, tag)]
        df = attach_cropwise(load_dataset(version, tag, crop_filter), tag, cw)
        if df.empty:
            continue
        X, y, feat_cols, cat_in_x = prepare_xy(df, use_field_id=use_field_id)
        cat_idx = [feat_cols.index(c) for c in cat_in_x]
        params = {**PARAM_SETS[params_name], "random_seed": 42, "verbose": False, "allow_writing_files": False}
        model = CatBoostRegressor(**params)
        model.fit(Pool(X, y, cat_features=cat_idx))
        imp = model.get_feature_importance(Pool(X, y, cat_features=cat_idx))
        df_imp = pd.DataFrame({"feature": feat_cols, "importance": imp})
        df_imp["scenario"] = scenario
        df_imp["asof_tag"] = tag
        rows.append(df_imp)

    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    out.to_csv(OUT_DIR / "feature_importance.csv", index=False)

    md = "## 4. Feature importance (CatBoost in-sample, 1 Aug)\n\n"
    md += "Top 15 features per scenario. If field_id dominates, the model relies on memorization.\n\n"
    for scenario, _ in SCENARIOS:
        sub = out[out["scenario"] == scenario].sort_values("importance", ascending=False).head(15)
        md += f"### {scenario}\n\n"
        md += sub[["feature", "importance"]].round(2).to_markdown(index=False) + "\n\n"
    return md, out


def section_insample_vs_oof(cw: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    rows = []
    for scenario, crop_filter in SCENARIOS:
        for tag in AS_OF_TAGS:
            version, params_name, use_field_id = V17_BEST[(scenario, tag)]
            df = attach_cropwise(load_dataset(version, tag, crop_filter), tag, cw)
            if df.empty:
                continue
            X, y, feat_cols, cat_in_x = prepare_xy(df, use_field_id=use_field_id)
            cat_idx = [feat_cols.index(c) for c in cat_in_x]

            # In-sample: train on every row, predict on the same data.
            params = {**PARAM_SETS[params_name], "random_seed": 42, "verbose": False, "allow_writing_files": False}
            model = CatBoostRegressor(**params)
            model.fit(Pool(X, y, cat_features=cat_idx))
            in_pred = model.predict(X)
            in_metrics = err_block(y, in_pred)

            oof_pred = walk_forward_oof(df, PARAM_SETS[params_name], use_field_id=use_field_id)
            mask = ~np.isnan(oof_pred)
            oof_metrics = err_block(y[mask], oof_pred[mask]) if mask.sum() else {}

            rows.append({
                "scenario": scenario,
                "asof_tag": tag,
                "n_total": len(df),
                "n_oof": int(mask.sum()),
                "in_sample_mape": in_metrics["mape"],
                "in_sample_r2": in_metrics["r2"],
                "in_sample_mae": in_metrics["mae"],
                "oof_mape": oof_metrics.get("mape", float("nan")),
                "oof_r2": oof_metrics.get("r2", float("nan")),
                "oof_mae": oof_metrics.get("mae", float("nan")),
                "mape_gap": oof_metrics.get("mape", float("nan")) - in_metrics["mape"],
                "r2_gap": in_metrics["r2"] - oof_metrics.get("r2", float("nan")),
            })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "insample_vs_oof.csv", index=False)
    md = "## 5. In-sample vs OOF gap (over/underfit diagnostic)\n\n"
    md += (
        "If in-sample R^2 is near 1 and OOF R^2 is near 0, the model overfits. "
        "If both are near 0, the model has too few signal in features for this dataset size.\n\n"
    )
    md += out.round(3).to_markdown(index=False) + "\n\n"
    return md, out


def section_per_year(cw: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    rows = []
    for scenario, crop_filter in SCENARIOS:
        for tag in AS_OF_TAGS:
            version, params_name, use_field_id = V17_BEST[(scenario, tag)]
            df = attach_cropwise(load_dataset(version, tag, crop_filter), tag, cw)
            if df.empty:
                continue
            ml_pred = walk_forward_oof(df, PARAM_SETS[params_name], use_field_id=use_field_id)
            df = df.assign(pred=ml_pred)
            for year, sub in df.groupby("year"):
                sub_ml = sub.dropna(subset=["pred"])
                sub_cw = sub.dropna(subset=["cropwise_asof_t_ha"])
                ml_m = err_block(sub_ml["target_yield_t_ha"], sub_ml["pred"]) if len(sub_ml) else {}
                cw_m = err_block(sub_cw["target_yield_t_ha"], sub_cw["cropwise_asof_t_ha"]) if len(sub_cw) else {}
                rows.append({
                    "scenario": scenario,
                    "asof_tag": tag,
                    "year": int(year),
                    "n": len(sub),
                    "n_ml": len(sub_ml),
                    "n_cw": len(sub_cw),
                    "target_mean": float(sub["target_yield_t_ha"].mean()),
                    "ml_mape": ml_m.get("mape", float("nan")),
                    "ml_mae": ml_m.get("mae", float("nan")),
                    "cw_mape": cw_m.get("mape", float("nan")),
                    "cw_mae": cw_m.get("mae", float("nan")),
                })
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "per_year_errors.csv", index=False)
    md = "## 6. Per-year breakdown (target year, 1 Aug only for brevity)\n\n"
    sub = out[out["asof_tag"] == "08_01"].sort_values(["scenario", "year"])
    md += sub.round(3).to_markdown(index=False) + "\n\n"
    return md, out


def section_physical_target() -> tuple[str, pd.DataFrame]:
    crops = pd.read_csv(DATA_RAW / "crops.csv")[["id", "standard_name"]].rename(columns={"id": "crop_id"})
    fields = pd.read_csv(DATA_RAW / "fields.csv")[["id", "name", "tillable_area"]].rename(columns={"id": "field_id", "name": "field_name"})
    hi = pd.read_csv(
        DATA_RAW / "history_items_full.csv",
        low_memory=False,
        usecols=["field_id", "year", "crop_id", "productivity", "harvested_weight"],
    )
    hi = hi.merge(crops, on="crop_id", how="left").merge(fields, on="field_id", how="left")
    hi["physical_t_ha"] = pd.to_numeric(hi["harvested_weight"], errors="coerce") / pd.to_numeric(hi["tillable_area"], errors="coerce")

    sub = hi.dropna(subset=["physical_t_ha"]).copy()
    sub["abnormally_low"] = sub["physical_t_ha"] < 0.3
    sub["abnormally_high"] = sub["physical_t_ha"] > 12.0

    md = "## 8. physical_t_ha sanity (harvested_weight / tillable_area)\n\n"
    md += f"- rows with both fields populated: {len(sub)}\n"
    md += f"- mean: {sub['physical_t_ha'].mean():.3f}\n"
    md += f"- median: {sub['physical_t_ha'].median():.3f}\n"
    md += f"- min: {sub['physical_t_ha'].min():.3f}\n"
    md += f"- max: {sub['physical_t_ha'].max():.3f}\n"
    md += f"- abnormally low (<0.3 t/ha): {int(sub['abnormally_low'].sum())}\n"
    md += f"- abnormally high (>12 t/ha): {int(sub['abnormally_high'].sum())}\n\n"
    md += "### Top 15 abnormally low (potential partial harvest)\n\n"
    md += sub[sub["abnormally_low"]].sort_values("physical_t_ha").head(15)[
        ["field_id", "year", "standard_name", "harvested_weight", "tillable_area", "physical_t_ha", "productivity"]
    ].round(3).to_markdown(index=False) + "\n\n"
    md += "### Top 15 abnormally high\n\n"
    md += sub[sub["abnormally_high"]].sort_values("physical_t_ha", ascending=False).head(15)[
        ["field_id", "year", "standard_name", "harvested_weight", "tillable_area", "physical_t_ha", "productivity"]
    ].round(3).to_markdown(index=False) + "\n\n"
    sub.to_csv(OUT_DIR / "physical_target_sanity.csv", index=False)
    return md, sub


def main() -> None:
    cw = cropwise_table()

    parts = ["# Microscope audit\n\n"]
    parts.append("Goal: validate v17 conclusions before drawing dissertation-level claims.\n\n")

    md_baselines, _ = section_naive_baselines(cw)
    parts.append(md_baselines)
    print("[1/8] Naive baselines done")

    md_cw, _ = section_cropwise_timing(cw)
    parts.append(md_cw)
    print("[2/8] Cropwise timing done")

    md_wf, _ = section_walk_forward_distribution(cw)
    parts.append(md_wf)
    print("[3/8] WF distribution done")

    md_fi, _ = section_feature_importance(cw)
    parts.append(md_fi)
    print("[4/8] Feature importance done")

    md_io, _ = section_insample_vs_oof(cw)
    parts.append(md_io)
    print("[5/8] In-sample vs OOF done")

    md_year, _ = section_per_year(cw)
    parts.append(md_year)
    print("[6/8] Per-year done")

    md_target, _ = section_physical_target()
    parts.append(md_target)
    print("[8/8] physical_t_ha sanity done")

    out = REPORTS / "MICROSCOPE_AUDIT.md"
    out.write_text("".join(parts), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
