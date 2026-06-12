"""
Выгрузка «вторичных» агрономических данных, которые могут пригодиться позже:

- /api/v3a/growth_scales
- /api/v3/growth_stages
- /api/v3/growth_stage_groups
- /api/v3/growth_stages_predictions
- /api/v3/field_scout_reports_aggregated
- /api/v3b/plant_threats

Все сохранится в data_raw/*.csv, дальше можно уже спокойно играться с фичами.
"""

import os
from typing import Any, Dict, List

import pandas as pd

from fetch_cropwise_data import CropwiseClient, API_KEY, BASE_URL


DATA_DIR = "data_raw"
os.makedirs(DATA_DIR, exist_ok=True)

GROWTH_SCALES_CSV = os.path.join(DATA_DIR, "growth_scales.csv")
GROWTH_STAGES_CSV = os.path.join(DATA_DIR, "growth_stages.csv")
GROWTH_STAGE_GROUPS_CSV = os.path.join(DATA_DIR, "growth_stage_groups.csv")
GROWTH_STAGES_PRED_CSV = os.path.join(DATA_DIR, "growth_stages_predictions.csv")
FIELD_SCOUT_AGG_CSV = os.path.join(DATA_DIR, "field_scout_reports_aggregated.csv")
PLANT_THREATS_CSV = os.path.join(DATA_DIR, "plant_threats.csv")


def _ensure_api_key() -> None:
    if not API_KEY or API_KEY == "PASTE_YOUR_API_KEY_HERE":
        raise ValueError(
            "❌ CROPWISE_API_KEY не задан. "
            "В PowerShell:  $env:CROPWISE_API_KEY = '...токен...'"
        )


def _as_list(obj: Any) -> List[Any]:
    if obj is None:
        return []
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        if "data" in obj and isinstance(obj["data"], list):
            return obj["data"]
        if "items" in obj and isinstance(obj["items"], list):
            return obj["items"]
        return [obj]
    return [obj]


def _paged_fetch(client: CropwiseClient, path: str, limit: int = 1000) -> List[Dict[str, Any]]:
    all_items: List[Dict[str, Any]] = []
    from_id: int | None = None
    page = 1

    while True:
        params: Dict[str, Any] = {"limit": limit}
        if from_id is not None:
            params["from_id"] = from_id

        print(f"  -> {path} страница {page}, from_id={from_id} ...")
        resp = client.get(path, params=params)
        items = _as_list(resp)
        if not items:
            print("     ✅ Достигнут конец данных (пусто)")
            break

        all_items.extend(items)

        meta = resp.get("meta", {}) if isinstance(resp, dict) else {}
        last_id = None
        if isinstance(meta, dict):
            resp_meta = meta.get("response", {})
            if isinstance(resp_meta, dict):
                last_id = resp_meta.get("last_record_id")
        if not last_id:
            print("     ℹ️ meta.response.last_record_id не найден — останавливаемся")
            break

        from_id = int(last_id) + 1
        page += 1

    return all_items


def fetch_growth_scales_and_stages() -> None:
    print("=" * 80)
    print("ВЫГРУЗКА GROWTH SCALES / STAGES")
    print("=" * 80)

    _ensure_api_key()

    # growth_scales (v3a)
    client_v3a = CropwiseClient(API_KEY, base_url="https://operations.cropwise.com/api/v3a/")
    client_v3 = CropwiseClient(API_KEY, base_url=BASE_URL)

    # GrowthScales
    print("\nGrowthScales (v3a/growth_scales)")
    try:
        resp = client_v3a.get("/growth_scales")
        gs_items = _as_list(resp)
        gs_df = pd.DataFrame(gs_items)
        print(f"  ✅ GrowthScales: {len(gs_df)} записей, колонки: {list(gs_df.columns)}")
        gs_df.to_csv(GROWTH_SCALES_CSV, index=False)
        print(f"  💾 {GROWTH_SCALES_CSV}")
    except Exception as e:
        print(f"  ⚠️ Не удалось выкачать growth_scales: {e}")

    # GrowthStages
    print("\nGrowthStages (v3/growth_stages)")
    try:
        stages_items = _paged_fetch(client_v3, "/growth_stages")
        stages_df = pd.DataFrame(stages_items)
        print(f"  ✅ GrowthStages: {len(stages_df)} записей, колонки: {list(stages_df.columns)}")
        stages_df.to_csv(GROWTH_STAGES_CSV, index=False)
        print(f"  💾 {GROWTH_STAGES_CSV}")
    except Exception as e:
        print(f"  ⚠️ Не удалось выкачать growth_stages: {e}")

    # GrowthStageGroups
    print("\nGrowthStageGroups (v3/growth_stage_groups)")
    try:
        groups_items = _paged_fetch(client_v3, "/growth_stage_groups")
        groups_df = pd.DataFrame(groups_items)
        print(f"  ✅ GrowthStageGroups: {len(groups_df)} записей, колонки: {list(groups_df.columns)}")
        groups_df.to_csv(GROWTH_STAGE_GROUPS_CSV, index=False)
        print(f"  💾 {GROWTH_STAGE_GROUPS_CSV}")
    except Exception as e:
        print(f"  ⚠️ Не удалось выкачать growth_stage_groups: {e}")


def fetch_growth_stages_predictions() -> None:
    print("\n" + "=" * 80)
    print("ВЫГРУЗКА GROWTH_STAGES_PREDICTIONS")
    print("=" * 80)

    _ensure_api_key()
    client = CropwiseClient(API_KEY, base_url=BASE_URL)

    items = _paged_fetch(client, "/growth_stages_predictions")
    if not items:
        print("⚠️ growth_stages_predictions вернул пусто")
        return

    df = pd.DataFrame(items)
    print(f"  ✅ GrowthStagesPredictions: {len(df)} записей, колонки: {list(df.columns)}")
    # prediction_data и fact_data оставляем как JSON в колонках
    df.to_csv(GROWTH_STAGES_PRED_CSV, index=False)
    print(f"  💾 {GROWTH_STAGES_PRED_CSV}")


def fetch_field_scout_reports_aggregated() -> None:
    print("\n" + "=" * 80)
    print("ВЫГРУЗКА FIELD_SCOUT_REPORTS_AGGREGATED")
    print("=" * 80)

    _ensure_api_key()
    client = CropwiseClient(API_KEY, base_url=BASE_URL)

    items = _paged_fetch(client, "/field_scout_reports_aggregated")
    if not items:
        print("⚠️ field_scout_reports_aggregated вернул пусто")
        return

    df = pd.DataFrame(items)
    print(f"  ✅ FieldScoutReportsAggregated: {len(df)} записей, колонки: {list(df.columns)}")
    df.to_csv(FIELD_SCOUT_AGG_CSV, index=False)
    print(f"  💾 {FIELD_SCOUT_AGG_CSV}")


def fetch_plant_threats() -> None:
    print("\n" + "=" * 80)
    print("ВЫГРУЗКА PLANT_THREATS")
    print("=" * 80)

    _ensure_api_key()
    # v3b даёт расширенный threat_type и standard_name
    client_v3b = CropwiseClient(API_KEY, base_url="https://operations.cropwise.com/api/v3b/")

    try:
        items = _paged_fetch(client_v3b, "/plant_threats")
    except Exception:
        # fallback на v3, если v3b недоступен
        client_v3 = CropwiseClient(API_KEY, base_url=BASE_URL)
        items = _paged_fetch(client_v3, "/plant_threats")

    if not items:
        print("⚠️ plant_threats вернул пусто")
        return

    df = pd.DataFrame(items)
    print(f"  ✅ PlantThreats: {len(df)} записей, колонки: {list(df.columns)}")
    df.to_csv(PLANT_THREATS_CSV, index=False)
    print(f"  💾 {PLANT_THREATS_CSV}")


def main() -> None:
    fetch_growth_scales_and_stages()
    fetch_growth_stages_predictions()
    fetch_field_scout_reports_aggregated()
    fetch_plant_threats()
    print("\n" + "=" * 80)
    print("✅ ВЫГРУЗКА GROWTH/STAGE/SCOUTING/THREATS ЗАВЕРШЕНА")
    print("=" * 80)


if __name__ == "__main__":
    main()

