
"""
Project-specific constants extracted from the representative notebooks.

This package is intentionally notebook-faithful:
- phase1_01_svr_v2.ipynb
- phase1_02_lightgbm_v2.ipynb
- phase1_03_catboost_v2.ipynb
- phase2_01_stacking_v9_residual_meta.ipynb
- phase3_01_formula_recovery_linear.ipynb
- phase3_02_lightgbm_v10_feature_variant.ipynb
"""
from __future__ import annotations

RANDOM_STATE: int = 42
TARGET_COL: str = "calories_burned"

RAW_COLUMNS = [
    "ex_dura",
    "body_temp",
    "bpm",
    "height_feet",
    "height_inche",
    "weight_lb",
    "weight_status",
    "gender",
    "age",
]

AGE_GENDER_CORR_MAP = {
    ("20대", "M"): 1.0,
    ("30대", "M"): 2.0,
    ("20대", "F"): 3.0,
    ("30대", "F"): 4.0,
    ("40대", "M"): 4.0,
    ("50대", "F"): 5.0,
    ("40대", "F"): 6.0,
    ("50대", "M"): 6.0,
    ("60대", "F"): 7.0,
    ("70대", "F"): 7.5,
    ("70대", "M"): 8.0,
    ("60대", "M"): 9.0,
}

# Notebook-specific feature choices / saved dataset versions.
FEATURE_SETS = {
    "base": [
        "ex_dura", "bpm", "body_temp", "hr_ratio", "age", "gender",
        "height_cm", "weight_kg", "bmi", "log_exercise_stress_index",
        "log_bsa_intensity_time",
    ],
    "v2": [
        "ex_dura", "bpm", "body_temp", "hr_ratio", "age", "gender",
        "height_cm", "weight_kg", "weight_status", "bpm_per_kg",
        "temp_per_kg", "exercise_stress_index", "bsa_intensity_time",
    ],
    "v3": [
        "ex_dura", "bpm", "body_temp", "body_temp_c", "hr_ratio", "age",
        "gender", "height_cm", "weight_kg", "weight_status", "bpm_per_kg",
        "temp_per_kg", "exercise_stress_index", "bsa_intensity_time",
        "sqrt_age_gen_corr",
    ],
    "v4": [
        "ex_dura", "bpm", "body_temp", "body_temp_c", "hr_ratio", "age",
        "gender", "height_cm", "weight_kg", "weight_status", "bpm_per_kg",
        "temp_per_kg", "hr_load_per_kg", "hr_load_per_bsa",
        "exercise_stress_index", "bsa_intensity_time", "sqrt_age_gen_corr",
        "hr_load_short", "hr_load_mid", "hr_load_long",
    ],
    "v5": [
        "ex_dura", "bpm", "hr_ratio", "gender", "weight_status",
        "temp_per_kg", "hr_load_high", "high_load_male",
        "exercise_stress_index", "sqrt_age_gen_corr",
    ],
    "v6": [
        "ex_dura", "body_temp", "bpm", "height_feet", "height_inche", "weight_lb",
        "weight_status", "gender", "age", "age_section", "height_cm", "weight_kg",
        "body_temp_c", "bmi", "bmi_category", "bsa", "max_hr", "hr_ratio",
        "activity_proxy", "bsa_intensity_time", "bpm_per_kg", "hr_load_per_kg",
        "hr_load_per_bsa", "hr_load_short", "hr_load_mid", "hr_load_long",
        "high_load_male", "hr_load_high", "temp_diff", "temp_per_kg",
        "exercise_stress_index", "log_exercise_stress_index",
        "log_bsa_intensity_time", "temp_rise_rate", "heat_accumulation",
        "heat_load_per_mass", "hr_efficiency", "cardio_load", "hr_ratio_sq",
        "temp_diff_sq", "log_duration", "heat_per_surface", "metabolic_load",
        "log_metabolic_load", "hr_ratio_x_temp_diff", "hr_ratio_x_ex_dura",
        "hr_ratio_x_weight_kg", "hr_ratio_x_bsa", "temp_diff_x_ex_dura",
        "temp_diff_x_weight_kg", "temp_diff_x_bsa", "ex_dura_x_weight_kg",
        "ex_dura_x_bsa", "weight_kg_x_bsa", "age_gen_corr",
        "log_age_gen_corr", "sqrt_age_gen_corr",
    ],
    "v7": [
        "log_activity_proxy", "hr_load_per_kg", "bpm", "log_ex_dura_x_bsa",
        "log_metabolic_load", "sqrt_age_gen_corr", "exercise_stress_index",
        "bsa_intensity_time", "hr_load_per_bsa", "log_duration",
        "hr_load_long", "hr_load_high", "age", "log_ex_dura_x_weight_kg",
        "temp_diff_sq", "hr_ratio_x_weight_kg", "temp_rise_rate",
        "hr_load_mid", "height_cm", "hr_efficiency", "hr_load_short",
        "max_hr", "hr_ratio", "temp_diff", "bmi",
    ],
    "v8": [
        "log_activity_proxy", "hr_load_per_kg", "bpm", "log_ex_dura_x_bsa",
        "log_metabolic_load", "sqrt_age_gen_corr", "exercise_stress_index",
        "bsa_intensity_time", "hr_load_per_bsa", "ex_dura", "hr_load_long",
        "hr_load_high", "age", "gender", "log_ex_dura_x_weight_kg",
        "temp_diff_sq", "hr_ratio_x_weight_kg", "temp_rise_rate",
        "hr_load_mid", "height_cm", "hr_efficiency", "hr_load_short",
        "max_hr", "hr_ratio", "temp_diff", "bmi",
    ],
    "v9": [
        "ex_dura", "body_temp", "bpm", "height_feet", "height_inche",
        "weight_lb", "age", "bmi", "bsa_intensity_time",
        "exercise_stress_index_raw", "weight_status", "gender",
        "age_section", "bpm_section", "ex_section",
    ],
    "v10": [
        "ex_dura", "body_temp", "bpm", "height_feet", "height_inche",
        "weight_lb", "age", "bmi", "bsa_intensity_time",
        "exercise_stress_index_raw", "weight_status", "gender",
        "age_section", "bpm_section", "ex_section", "pre_cal_rounded",
    ],
    # Notebook aliases
    "phase1_01_svr_v2": [
        "ex_dura", "bpm", "body_temp", "hr_ratio", "age", "gender",
        "height_cm", "weight_kg", "bmi", "log_exercise_stress_index",
        "log_bsa_intensity_time",
    ],
    "phase1_02_lightgbm_v2": "base",   # actual notebook uses final_train_adj.csv
    "phase1_03_catboost_v2": [
        "ex_dura", "bpm", "body_temp", "hr_ratio", "age", "gender",
        "height_cm", "weight_kg", "bmi", "log_exercise_stress_index",
        "log_bsa_intensity_time",
    ],
    "phase2_01_stacking_v9": "v9",
    "phase3_01_formula_recovery_linear": ["pre_cal_rounded"],
    "phase3_02_lightgbm_v10_feature_variant": "v10",
}

def resolve_feature_set(name: str) -> list[str]:
    """Resolve aliases such as 'phase1_02_lightgbm_v2' -> actual column list."""
    value = FEATURE_SETS[name]
    if isinstance(value, str):
        return FEATURE_SETS[value]
    return value[:]  # copy
