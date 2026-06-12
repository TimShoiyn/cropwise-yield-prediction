"""Build a fair as-of Cropwise benchmark from productivity_estimate_histories.

The raw `/api/v3/productivity_estimate_histories` pull is wide:
one row per (field_id, year), many columns like `estimate_history.2025-07-05`.

For each as-of date (07-01, 08-01, 09-01), this script takes the latest
non-null Cropwise estimate with `history_date <= asof_date` within the same
season year. API values are centner/ha (ц/га), so we convert to t/ha by /10.

Output:
  data_clean/cropwise_estimates_asof_clean.csv
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)

RAW = Path("data_raw")
OUT = Path("data_clean")
OUT.mkdir(exist_ok=True)

ASOF_DATES = {"07_01": (7, 1), "08_01": (8, 1), "09_01": (9, 1)}


def main() -> None:
    df = pd.read_csv(RAW / "cropwise_productivity_estimate_histories_full.csv", low_memory=False)
    df["field_id"] = pd.to_numeric(df["field_id"], errors="coerce").astype("Int64")
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["field_id", "year"]).copy()
    df["field_id"] = df["field_id"].astype(int)
    df["year"] = df["year"].astype(int)

    hist_cols = [c for c in df.columns if c.startswith("estimate_history.")]
    parsed_cols: list[tuple[str, pd.Timestamp]] = []
    for col in hist_cols:
        date = pd.to_datetime(col.replace("estimate_history.", ""), errors="coerce")
        if pd.notna(date):
            parsed_cols.append((col, pd.Timestamp(date)))

    rows = []
    for _, rec in df.iterrows():
        field_id = int(rec["field_id"])
        year = int(rec["year"])
        base = {"field_id": field_id, "year": year}
        for tag, (month, day) in ASOF_DATES.items():
            asof = pd.Timestamp(year=year, month=month, day=day)
            candidates = []
            for col, date in parsed_cols:
                if date.year != year or date > asof:
                    continue
                val = pd.to_numeric(rec.get(col), errors="coerce")
                if pd.notna(val) and np.isfinite(float(val)):
                    candidates.append((date, float(val)))
            if not candidates:
                cropwise_t_ha = np.nan
                history_date = pd.NaT
            else:
                history_date, raw_centner_ha = max(candidates, key=lambda x: x[0])
                cropwise_t_ha = raw_centner_ha / 10.0
            rows.append(
                {
                    **base,
                    "asof_tag": tag,
                    "cropwise_asof_t_ha": cropwise_t_ha,
                    "cropwise_history_date": history_date,
                }
            )

    out = pd.DataFrame(rows)
    out = out.dropna(subset=["cropwise_asof_t_ha"]).copy()
    out = out.sort_values(["field_id", "year", "asof_tag"]).reset_index(drop=True)
    path = OUT / "cropwise_estimates_asof_clean.csv"
    out.to_csv(path, index=False)

    print(f"asof cropwise benchmark: {len(out)} rows -> {path}")
    print(out.groupby("asof_tag")["cropwise_asof_t_ha"].agg(["count", "mean", "median"]).round(3).to_string())


if __name__ == "__main__":
    main()
