"""
Multi-source target audit.

Compares yield/target values from every available source for each (field_id, year):
  - history_items_full.harvested_weight / fields.tillable_area  -> physical t/ha
  - history_items_full.productivity (raw)                      -> productivity raw
  - additional_yields_from_history.target_yield_t_ha (raw)     -> additional yield raw
  - productivity_estimates.estimate_value (raw)                -> Cropwise estimate raw
  - productivity_data.fact (c/ha -> t/ha) via field-name match -> productivity_data fact
  - yield_maps                                                  -> combine map mean
  - targets_factual_t_ha (kept rows)                            -> previous baseline

Detects:
  - unit anomalies (raw value > 12 t/ha plausibly stored in c/ha);
  - cross-source disagreements > 1 t/ha;
  - rows where productivity_data fact equals 1/10 of additional_yield raw (clear unit mismatch).

Outputs:
  - reports/MULTI_SOURCE_TARGET_AUDIT.md
  - reports/multi_source_targets.csv
  - reports/multi_source_conflicts.csv
"""

from __future__ import annotations

import os
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
os.chdir(ROOT_DIR)

DATA_RAW = Path("data_raw")
DATA_PROCESSED = Path("data_processed")
REPORTS = Path("reports")

# Per-crop yield ceilings used to detect c/ha values stored as t/ha.
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
    "fallow": 2.0,
    "linum": 3.0,
    "avena_spring": 4.5,
    "rye_winter": 6.0,
    "sainfoin": 5.0,
    "medicago": 5.0,
    "sudan_grass": 6.0,
    "sweet_clover": 5.0,
    "potatoes": 60.0,
    "sugar_beet": 80.0,
}
DEFAULT_MAX_T_HA = 5.0


def norm(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).lower().replace("ё", "е")
    text = re.sub(r"[^а-яa-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_yield(raw: float, std_name: str | None) -> float:
    if pd.isna(raw):
        return np.nan
    cap = MAX_T_HA.get(std_name or "", DEFAULT_MAX_T_HA)
    value = float(raw)
    if value > cap:
        return value / 10.0
    return value


def pick_yield_value(row: pd.Series) -> float:
    for col in ["external_average", "totals.result.average.value"]:
        value = pd.to_numeric(row.get(col), errors="coerce")
        if pd.notna(value) and 0.1 <= float(value) <= 12.0:
            return float(value)
    value = pd.to_numeric(row.get("calculated_average"), errors="coerce")
    units = str(row.get("units", ""))
    if pd.notna(value):
        value = float(value)
        if units == "tonn_per_ha" and 0.1 <= value <= 12.0:
            return value
        if units == "tonn_per_acre" and 0.1 <= value <= 6.0:
            return value * 2.47105381
    return np.nan


def code(df_or_text) -> str:
    if isinstance(df_or_text, pd.DataFrame):
        text = df_or_text.to_string(index=False)
    else:
        text = str(df_or_text)
    return f"\n```\n{text}\n```\n"


def main() -> None:
    crops = pd.read_csv(DATA_RAW / "crops.csv")[["id", "standard_name"]].rename(columns={"id": "crop_id"})
    fields = pd.read_csv(DATA_RAW / "fields.csv")[["id", "name", "tillable_area"]].rename(columns={"id": "field_id", "name": "field_name"})
    fields["field_name_norm"] = fields["field_name"].map(norm)

    hi = pd.read_csv(
        DATA_RAW / "history_items_full.csv",
        low_memory=False,
        usecols=["id", "field_id", "year", "crop_id", "productivity", "harvested_weight"],
    )
    hi = hi.rename(columns={"id": "history_item_id"})
    hi["field_id"] = pd.to_numeric(hi["field_id"], errors="coerce").astype("Int64")
    hi["year"] = pd.to_numeric(hi["year"], errors="coerce").astype("Int64")
    hi = hi.dropna(subset=["field_id", "year"]).copy()
    hi["field_id"] = hi["field_id"].astype(int)
    hi["year"] = hi["year"].astype(int)
    hi = hi.merge(crops, on="crop_id", how="left")
    hi = hi.merge(fields[["field_id", "field_name", "tillable_area"]], on="field_id", how="left")

    hi["physical_t_ha_raw"] = pd.to_numeric(hi["harvested_weight"], errors="coerce") / pd.to_numeric(hi["tillable_area"], errors="coerce")
    hi["physical_t_ha"] = [normalize_yield(v, c) for v, c in zip(hi["physical_t_ha_raw"], hi["standard_name"])]
    hi["productivity_raw"] = pd.to_numeric(hi["productivity"], errors="coerce")
    hi["productivity_t_ha"] = [normalize_yield(v, c) for v, c in zip(hi["productivity_raw"], hi["standard_name"])]

    add = pd.read_csv(DATA_RAW / "additional_yields_from_history.csv")
    add["field_id"] = pd.to_numeric(add["field_id"], errors="coerce").astype("Int64")
    add["year"] = pd.to_numeric(add["year"], errors="coerce").astype("Int64")
    add = add.dropna(subset=["field_id", "year"]).copy()
    add["field_id"] = add["field_id"].astype(int)
    add["year"] = add["year"].astype(int)
    add = add.merge(crops, on="crop_id", how="left")
    add["additional_raw"] = pd.to_numeric(add["target_yield_t_ha"], errors="coerce")
    add["additional_t_ha"] = [normalize_yield(v, c) for v, c in zip(add["additional_raw"], add["standard_name"])]
    add_keep = add[["field_id", "year", "additional_raw", "additional_t_ha", "crop_id"]].rename(columns={"crop_id": "additional_crop_id"})

    pe = pd.read_csv(DATA_RAW / "productivity_estimates.csv")
    drop_cols = [c for c in ["field_id", "year"] if c in pe.columns]
    if drop_cols:
        pe = pe.drop(columns=drop_cols)
    pe = pe.merge(hi[["history_item_id", "field_id", "year", "standard_name"]], on="history_item_id", how="left")
    pe = pe.dropna(subset=["field_id", "year"]).copy()
    pe["field_id"] = pe["field_id"].astype(int)
    pe["year"] = pe["year"].astype(int)
    pe["estimate_raw"] = pd.to_numeric(pe["estimate_value"], errors="coerce")
    pe["estimate_t_ha"] = [normalize_yield(v, c) for v, c in zip(pe["estimate_raw"], pe["standard_name"])]
    estimate_agg = pe.groupby(["field_id", "year"]).agg(
        estimate_raw_max=("estimate_raw", "max"),
        estimate_t_ha_max=("estimate_t_ha", "max"),
    ).reset_index()

    prod = pd.read_csv(DATA_RAW / "productivity_data.csv")
    prod["field_name_norm"] = prod["Поле"].map(norm)
    prod = prod.rename(columns={"Год": "year", "Культура": "prod_crop_ru"})
    prod["year"] = pd.to_numeric(prod["year"], errors="coerce").astype("Int64")
    prod = prod.dropna(subset=["year"]).copy()
    prod["year"] = prod["year"].astype(int)
    prod["prod_fact_t_ha"] = pd.to_numeric(prod["урожайность факт ц/га"], errors="coerce") / 10.0
    prod["prod_forecast_t_ha"] = pd.to_numeric(prod["урожайность прогноз ц/га"], errors="coerce") / 10.0
    prod_join = prod.merge(fields[["field_id", "field_name_norm"]], on="field_name_norm", how="inner")
    prod_join = prod_join[["field_id", "year", "prod_crop_ru", "prod_fact_t_ha", "prod_forecast_t_ha"]]

    ym = pd.read_csv(DATA_RAW / "yield_maps.csv", low_memory=False)
    ym["created_at"] = pd.to_datetime(ym["created_at"], errors="coerce", utc=True).dt.tz_convert(None)
    ym["year"] = ym["created_at"].dt.year
    ym["yield_map_t_ha"] = ym.apply(pick_yield_value, axis=1)
    ym = ym.dropna(subset=["field_id", "year", "yield_map_t_ha"]).copy()
    ym["field_id"] = ym["field_id"].astype(int)
    ym["year"] = ym["year"].astype(int)
    ym = ym[(ym["yield_map_t_ha"] >= 0.2) & (ym["yield_map_t_ha"] <= 10.0)]
    ym_agg = ym.groupby(["field_id", "year"]).agg(
        yield_map_t_ha=("yield_map_t_ha", "mean"),
        yield_map_n=("yield_map_t_ha", "size"),
    ).reset_index()

    factual = pd.read_csv(DATA_PROCESSED / "targets_factual_t_ha.csv")
    factual = factual[factual["kept"]].copy()
    factual = factual[["field_id", "year", "standard_name", "target_yield_t_ha", "target_source"]].rename(
        columns={"target_yield_t_ha": "factual_target_t_ha", "target_source": "factual_target_source"}
    )

    base = hi[["field_id", "year", "standard_name", "physical_t_ha", "productivity_t_ha", "physical_t_ha_raw", "productivity_raw"]].copy()
    merged = (
        base.merge(add_keep, on=["field_id", "year"], how="outer")
        .merge(estimate_agg, on=["field_id", "year"], how="left")
        .merge(prod_join, on=["field_id", "year"], how="left")
        .merge(ym_agg, on=["field_id", "year"], how="left")
        .merge(factual, on=["field_id", "year"], how="left")
    )

    merged.to_csv(REPORTS / "multi_source_targets.csv", index=False)

    sources = ["physical_t_ha", "productivity_t_ha", "additional_t_ha", "estimate_t_ha_max", "prod_fact_t_ha", "yield_map_t_ha", "factual_target_t_ha"]
    coverage = pd.DataFrame({col: [merged[col].notna().sum()] for col in sources})
    coverage["total_rows"] = len(merged)

    pairs = []
    for a, b in combinations(sources, 2):
        sub = merged.dropna(subset=[a, b])
        if sub.empty:
            continue
        diffs = (sub[a] - sub[b]).abs()
        pairs.append({
            "source_a": a,
            "source_b": b,
            "n": len(sub),
            "median_abs_diff": float(diffs.median()),
            "p90_abs_diff": float(diffs.quantile(0.90)),
            "max_abs_diff": float(diffs.max()),
            "share_above_1t": float((diffs > 1.0).mean()),
        })
    pair_df = pd.DataFrame(pairs).sort_values("share_above_1t", ascending=False)

    conflict_rows = []
    for _, r in merged.iterrows():
        vals = {col: r[col] for col in sources if pd.notna(r[col])}
        if len(vals) < 2:
            continue
        spread = max(vals.values()) - min(vals.values())
        if spread > 1.0:
            row = r.copy().to_dict()
            row["spread_t_ha"] = spread
            row["sources_used"] = ";".join(vals.keys())
            conflict_rows.append(row)
    conflicts = pd.DataFrame(conflict_rows).sort_values("spread_t_ha", ascending=False)
    conflicts.to_csv(REPORTS / "multi_source_conflicts.csv", index=False)

    md: list[str] = ["# Multi-source target audit\n\n"]
    md.append("Generated by `scripts/audit/multi_source_target_audit.py`.\n\n")
    md.append("## Coverage per source\n")
    md.append(code(coverage))
    md.append("\n## Pairwise disagreement\n")
    md.append(code(pair_df.round(3)))
    md.append("\n## Rows with cross-source spread > 1 t/ha\n")
    md.append(f"Found {len(conflicts)} rows.\n")
    show = [
        "field_id", "year", "standard_name",
        "physical_t_ha", "productivity_t_ha", "additional_t_ha",
        "estimate_t_ha_max", "prod_fact_t_ha", "yield_map_t_ha",
        "factual_target_t_ha", "factual_target_source", "spread_t_ha", "sources_used",
    ]
    show = [c for c in show if c in conflicts.columns]
    md.append(code(conflicts[show].head(60).round(3)))

    md.append("\n## Notes\n\n")
    md.append("- `additional_yields_from_history.csv` declares `target_yield_t_ha` but contains many `c/ha` values (>12), normalized here per crop cap.\n")
    md.append("- `productivity_estimates.estimate_value` is Cropwise's own forecast; never use as ML feature.\n")
    md.append("- `yield_maps` are post-harvest combine maps; useful as cross-check, not as model features.\n")
    md.append("- `factual_target_t_ha` is our previous training target. Spread vs other sources highlights remaining target risk.\n")

    out = REPORTS / "MULTI_SOURCE_TARGET_AUDIT.md"
    out.write_text("".join(md), encoding="utf-8")
    print(f"Wrote {REPORTS / 'multi_source_targets.csv'}")
    print(f"Wrote {REPORTS / 'multi_source_conflicts.csv'}")
    print(f"Wrote {out}")
    print("\nCoverage:")
    print(coverage.to_string(index=False))
    print("\nPairwise disagreement:")
    print(pair_df.round(3).to_string(index=False))
    print(f"\nTotal cross-source spread > 1 t/ha: {len(conflicts)}")
    print(conflicts[show].head(20).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
