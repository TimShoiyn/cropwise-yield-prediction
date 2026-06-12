"""Curated v24 feature contracts.

These sets come from A2 audit logic:
  - keep agronomically meaningful features;
  - avoid dead columns and duplicated percentile soup;
  - keep separate contracts for all-crops, wheat and sunflower.
"""

COMMON_NUMERIC = [
    # Field scale / geography-free descriptors.
    "field_tillable_area",
    "field_calculated_area",
    # Sowing / phenology.
    "sowing_known_asof",
    "sowing_doy",
    "days_after_sowing_asof",
    # Rotation.
    "years_since_sunflower",
    "years_since_wheat",
    # NDVI state and curve shape.
    "ndvi_n_obs_asof",
    "ndvi_mean_asof",
    "ndvi_max_asof",
    "ndvi_p75_asof",
    "ndvi_slope_asof",
    "ndvi_integral_asof",
    "ndvi_last30d_mean_asof",
    "ndvi_above_05_days_asof",
    # Cropwise historical_values weather/agro proxies.
    "cw_temp_mean_asof",
    "cw_temp_max_asof",
    "cw_temp_last30d_mean_asof",
    "cw_sm_mean_asof",
    "cw_sm_min_asof",
    "cw_sm_last_value_asof",
    "cw_sm_last30d_mean_asof",
    "gdd_to_asof",
    "heat_days_ge30_to_asof",
    "cold_days_le0_to_asof",
    # Soil tests.
    "field_soil_pH",
    "field_soil_OM",
    "field_soil_N",
    "field_soil_P",
    "field_soil_K",
    "field_soil_Mg",
    "field_soil_S",
    "field_soil_Zn",
]

COMMON_CATEGORICAL = [
    "crop_id",
    "prev_crop_id",
    "standard_name",
    "prev_standard_name",
    "rotation_pair",
    "variety",
]

WHEAT_NUMERIC = [
    # Wheat signal audit: NDVI peak + heat/GDD/water stress dominate.
    "field_tillable_area",
    "sowing_known_asof",
    "sowing_doy",
    "days_after_sowing_asof",
    "years_since_wheat",
    "years_since_sunflower",
    "ndvi_n_obs_asof",
    "ndvi_mean_asof",
    "ndvi_max_asof",
    "ndvi_p75_asof",
    "ndvi_slope_asof",
    "ndvi_integral_asof",
    "ndvi_last30d_mean_asof",
    "ndvi_above_05_days_asof",
    "cw_temp_mean_asof",
    "cw_temp_max_asof",
    "cw_temp_last30d_mean_asof",
    "gdd_to_asof",
    "heat_days_ge30_to_asof",
    "cold_days_le0_to_asof",
    "cw_sm_mean_asof",
    "cw_sm_min_asof",
    "cw_sm_last30d_mean_asof",
    "field_soil_P",
    "field_soil_K",
    "field_soil_OM",
    "field_soil_pH",
]

WHEAT_CATEGORICAL = [
    "crop_id",
    "prev_crop_id",
    "standard_name",
    "prev_standard_name",
    "rotation_pair",
    "variety",
]

SUNFLOWER_NUMERIC = [
    # Sunflower signal audit: NDVI in July/August + P + root-zone moisture.
    "field_tillable_area",
    "sowing_known_asof",
    "sowing_doy",
    "days_after_sowing_asof",
    "years_since_sunflower",
    "years_since_wheat",
    "ndvi_n_obs_asof",
    "ndvi_mean_asof",
    "ndvi_max_asof",
    "ndvi_p75_asof",
    "ndvi_slope_asof",
    "ndvi_integral_asof",
    "ndvi_last30d_mean_asof",
    "ndvi_above_05_days_asof",
    "cw_sm_mean_asof",
    "cw_sm_min_asof",
    "cw_sm_last_value_asof",
    "cw_sm_last30d_mean_asof",
    "cw_temp_mean_asof",
    "cw_temp_max_asof",
    "gdd_to_asof",
    "heat_days_ge30_to_asof",
    "field_soil_P",
    "field_soil_K",
    "field_soil_OM",
    "field_soil_pH",
    "field_soil_Zn",
]

SUNFLOWER_CATEGORICAL = [
    "prev_crop_id",
    "prev_standard_name",
    "rotation_pair",
    "variety",
]


FEATURE_SETS = {
    "all_crops": (COMMON_NUMERIC, COMMON_CATEGORICAL),
    "wheat_combined": (WHEAT_NUMERIC, WHEAT_CATEGORICAL),
    "sunflower": (SUNFLOWER_NUMERIC, SUNFLOWER_CATEGORICAL),
}

# Sentinel-2 indices added in v25 (missing from Cropwise API).
S2_FEATURES = [
    "s2_ndre_mean",
    "s2_evi_mean",
    "s2_gcvi_mean",
    "s2_ndvi_mean",
    "s2_ndre_minus_ndvi",
    "s2_evi_minus_ndvi",
]


def feature_set_with_s2(scenario: str) -> tuple[list[str], list[str]]:
    """Return (numeric, categorical) feature lists augmented with S2 indices."""
    numeric, categorical = FEATURE_SETS[scenario]
    return numeric + S2_FEATURES, list(categorical)
