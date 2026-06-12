"""
Quick data overview to feed the final report.
Counts unique fields, crops, years per source. Saves a small JSON-like
summary that the report writer can embed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
os.chdir(ROOT_DIR)

DATA_RAW = Path("data_raw")
DATA_PROCESSED = Path("data_processed")
REPORTS = Path("reports")


def main() -> None:
    out: dict = {}

    fields = pd.read_csv(DATA_RAW / "fields.csv")
    out["fields_total"] = len(fields)
    out["fields_with_geometry"] = int(fields["geometry"].notna().sum()) if "geometry" in fields.columns else None
    out["fields_with_area"] = int(fields["tillable_area"].notna().sum())

    crops = pd.read_csv(DATA_RAW / "crops.csv")
    out["crops_total"] = len(crops)
    out["crops_with_standard_name"] = int(crops["standard_name"].notna().sum())

    hi = pd.read_csv(DATA_RAW / "history_items_full.csv", low_memory=False)
    out["history_items_total"] = len(hi)
    out["history_items_unique_fields"] = int(hi["field_id"].nunique())
    out["history_items_year_range"] = [int(hi["year"].min()), int(hi["year"].max())]
    out["history_items_with_harvested_weight"] = int(hi["harvested_weight"].notna().sum())
    out["history_items_with_productivity"] = int(hi["productivity"].notna().sum())

    ndvi = pd.read_csv(DATA_RAW / "ndvi_timeseries.csv")
    out["ndvi_rows"] = len(ndvi)
    out["ndvi_unique_fields"] = int(ndvi["field_id"].nunique()) if "field_id" in ndvi.columns else None
    if "date" in ndvi.columns:
        ndvi["date"] = pd.to_datetime(ndvi["date"], errors="coerce")
        out["ndvi_date_range"] = [
            str(ndvi["date"].min().date()) if ndvi["date"].notna().any() else None,
            str(ndvi["date"].max().date()) if ndvi["date"].notna().any() else None,
        ]

    om = pd.read_csv(DATA_RAW / "openmeteo_daily.csv")
    out["openmeteo_rows"] = len(om)
    out["openmeteo_unique_fields"] = int(om["field_id"].nunique()) if "field_id" in om.columns else None
    if "date" in om.columns:
        om["date"] = pd.to_datetime(om["date"], errors="coerce")
        out["openmeteo_date_range"] = [
            str(om["date"].min().date()) if om["date"].notna().any() else None,
            str(om["date"].max().date()) if om["date"].notna().any() else None,
        ]

    op = pd.read_csv(DATA_RAW / "operations.csv")
    out["operations_rows"] = len(op)
    out["operations_unique_fields"] = int(op["field_id"].nunique()) if "field_id" in op.columns else None

    sc = pd.read_csv(DATA_RAW / "field_scout_reports_aggregated.csv")
    out["scout_reports_rows"] = len(sc)
    out["scout_reports_unique_fields"] = int(sc["field_id"].nunique()) if "field_id" in sc.columns else None

    st = pd.read_csv(DATA_RAW / "soil_test_samples.csv")
    out["soil_samples_rows"] = len(st)
    out["soil_samples_unique_fields"] = int(st["field_id"].nunique()) if "field_id" in st.columns else None

    pd_data = pd.read_csv(DATA_RAW / "productivity_data.csv")
    out["productivity_data_rows"] = len(pd_data)
    out["productivity_data_year_range"] = [
        int(pd.to_numeric(pd_data["Год"], errors="coerce").min()),
        int(pd.to_numeric(pd_data["Год"], errors="coerce").max()),
    ]

    ym = pd.read_csv(DATA_RAW / "yield_maps.csv", low_memory=False)
    out["yield_maps_rows"] = len(ym)
    out["yield_maps_unique_fields"] = int(ym["field_id"].nunique()) if "field_id" in ym.columns else None

    # ML dataset stats
    for name in ["v17_v12_clean_asof_07_01", "v17_v12_clean_asof_08_01", "v17_v12_clean_asof_09_01"]:
        path = DATA_PROCESSED / f"ml_dataset_clean_{name}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        key = f"v17_clean_{name.split('_asof_')[1]}"
        out[key] = {
            "rows": len(df),
            "cols": len(df.columns),
            "unique_fields": int(df["field_id"].nunique()),
            "year_range": [int(df["year"].min()), int(df["year"].max())],
            "crops": df["standard_name"].value_counts().to_dict() if "standard_name" in df.columns else {},
            "target_mean": float(df["target_yield_t_ha"].mean()),
            "target_min": float(df["target_yield_t_ha"].min()),
            "target_max": float(df["target_yield_t_ha"].max()),
        }

    Path(REPORTS / "data_overview.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
