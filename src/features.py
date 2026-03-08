
"""
Feature engineering rebuilt directly from the notebooks.

Main sources:
- 00_eda_feature_engineering_master.ipynb
- phase3_01_formula_recovery_linear.ipynb

The goal is not a generic tabular template.
The goal is to preserve the project's actual experimental feature logic.
"""
from __future__ import annotations

from itertools import combinations
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .config import AGE_GENDER_CORR_MAP, TARGET_COL, resolve_feature_set


def age_section(age: float) -> str:
    if age < 30:
        return "20대"
    if age < 40:
        return "30대"
    if age < 50:
        return "40대"
    if age < 60:
        return "50대"
    if age < 70:
        return "60대"
    return "70대"


def bmi_category(bmi: float) -> str:
    if bmi < 18.5:
        return "저체중"
    if bmi < 25:
        return "정상"
    if bmi < 30:
        return "과체중"
    return "비만"


def bpm_section(bpm: float) -> str:
    if bpm < 80:
        return "low"
    if bpm < 100:
        return "mid"
    return "high"


def ex_section(ex_dura: float) -> str:
    if ex_dura < 7:
        return "ex_low"
    if ex_dura < 15:
        return "ex_mid"
    return "ex_high"


def _ensure_columns(df: pd.DataFrame, cols: Iterable[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Notebook cells 6, 17, 18, 23, 46~51 기반.

    Produces the core feature block that later versions reuse:
    units, BMI/BSA, HR ratio, load features, temperature features,
    age-gender custom feature, and derived logs.
    """
    out = df.copy()

    _ensure_columns(out, ["ex_dura", "body_temp", "bpm", "height_feet", "height_inche", "weight_lb", "gender", "age"])

    out["age_section"] = out["age"].apply(age_section)

    out["height_cm"] = ((out["height_feet"] * 12) + out["height_inche"]) * 2.54
    out["weight_kg"] = out["weight_lb"] * 0.45359237
    out["body_temp_c"] = np.round((out["body_temp"] - 32) / 1.8, 1)

    out["bmi"] = np.round(out["weight_kg"] / ((out["height_cm"] / 100) ** 2), 2)
    out["bmi_category"] = out["bmi"].apply(bmi_category)
    out["bsa"] = np.sqrt((out["height_cm"] * out["weight_kg"]) / 3600)

    out["max_hr"] = 220 - out["age"]
    out["hr_ratio"] = out["bpm"] / out["max_hr"]

    out["activity_proxy"] = out["weight_kg"] * out["ex_dura"] * out["hr_ratio"]
    out["bsa_intensity_time"] = out["bsa"] * out["ex_dura"] * out["hr_ratio"]

    out["bpm_per_kg"] = out["bpm"] / (out["weight_kg"] + 1e-6)
    out["hr_load_per_kg"] = (out["bpm"] * out["ex_dura"]) / (out["weight_kg"] + 1e-6)
    out["hr_load_per_bsa"] = (out["bpm"] * out["ex_dura"]) / (out["bsa"] + 1e-6)

    # Conditional load features from EDA notebook cell 17~18.
    out["hr_load_short"] = out["bpm"] * out["ex_dura"] * (out["ex_dura"] < 7) * (out["gender"] == "M")
    out["hr_load_mid"] = out["bpm"] * out["ex_dura"] * ((out["ex_dura"] >= 7) & (out["ex_dura"] < 18)) * (out["gender"] == "M")
    out["hr_load_long"] = out["bpm"] * out["ex_dura"] * (out["ex_dura"] >= 18) * (out["gender"] == "M")

    out["high_load_male"] = (
        (out["gender"] == "M") &
        (out["ex_dura"] >= 15) &
        (out["bpm"] >= 110)
    ).astype(int)
    out["hr_load_high"] = out["bpm"] * out["ex_dura"] * out["high_load_male"]

    out["temp_diff"] = out["body_temp_c"] - 37
    out["temp_per_kg"] = out["temp_diff"] / (out["weight_kg"] + 1e-6)

    out["exercise_stress_index"] = out["ex_dura"] * out["hr_ratio"] * out["temp_diff"]
    out["exercise_stress_index"] = out["exercise_stress_index"].clip(lower=0)
    out["log_exercise_stress_index"] = np.log1p(out["exercise_stress_index"])
    out["log_bsa_intensity_time"] = np.log1p(out["bsa_intensity_time"])

    # Custom age-gender correction from EDA cell 46.
    out["w"] = list(map(lambda x: AGE_GENDER_CORR_MAP.get(x, np.nan), zip(out["age_section"], out["gender"])))
    out["age_gen_corr"] = out["w"] * (out["bpm"] / out["bmi"]) * out["ex_dura"]
    out["log_age_gen_corr"] = np.log1p(out["age_gen_corr"])
    out["sqrt_age_gen_corr"] = np.sqrt(np.clip(out["age_gen_corr"], a_min=0, a_max=None))

    return out.drop(columns=["w"])


def add_full_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Notebook cell 28 + 29 (add_features_full / add_selected_logs).

    Extends the core dataset into the v6-style master feature table.
    """
    out = df.copy()
    _ensure_columns(out, ["temp_diff", "ex_dura", "weight_kg", "hr_ratio", "bsa", "activity_proxy"])

    out["temp_rise_rate"] = out["temp_diff"] / (out["ex_dura"] + 1e-6)
    out["heat_accumulation"] = out["temp_diff"] * out["ex_dura"]
    out["heat_load_per_mass"] = out["temp_diff"] * out["ex_dura"] / (out["weight_kg"] + 1e-6)

    out["hr_efficiency"] = out["hr_ratio"] / (out["temp_diff"] + 1e-6)
    out["cardio_load"] = out["bpm_per_kg"] * out["ex_dura"]

    out["hr_ratio_sq"] = out["hr_ratio"] ** 2
    out["temp_diff_sq"] = out["temp_diff"] ** 2
    out["log_duration"] = np.log1p(out["ex_dura"])

    out["heat_per_surface"] = out["temp_diff"] / (out["bsa"] + 1e-6)

    out["metabolic_load"] = out["hr_ratio"] * out["ex_dura"] * out["temp_diff"] * out["weight_kg"]
    out["log_metabolic_load"] = np.log1p(np.clip(out["metabolic_load"], a_min=0, a_max=None))

    interaction_cols = ["hr_ratio", "temp_diff", "ex_dura", "weight_kg", "bsa"]
    for c1, c2 in combinations(interaction_cols, 2):
        out[f"{c1}_x_{c2}"] = out[c1] * out[c2]

    out["log_activity_proxy"] = np.log1p(np.clip(out["activity_proxy"], a_min=0, a_max=None))
    out["log_ex_dura_x_bsa"] = np.log1p(np.clip(out["ex_dura_x_bsa"], a_min=0, a_max=None))
    out["log_ex_dura_x_weight_kg"] = np.log1p(np.clip(out["ex_dura_x_weight_kg"], a_min=0, a_max=None))

    return out


def add_formula_recovery_features(df: pd.DataFrame, rounded_mode: str = "floor_half_up") -> pd.DataFrame:
    """
    Features used in the phase3 notebooks.

    Includes:
    - hr_wt_age_dura
    - exercise_stress_index_raw
    - ex_stress_index_new
    - bpm_section / ex_section
    - pre_cal / pre_cal_rounded
    """
    out = df.copy()
    _ensure_columns(out, ["gender", "ex_dura", "bpm", "weight_kg", "age", "bsa", "body_temp", "bmi", "age_section"])

    g = out["gender"].astype(str).str.upper().str.strip()

    male = (-55.0969 + 0.6309 * out["hr_ratio"] + 0.1988 * out["weight_kg"] + 0.2017 * out["age"]) * out["ex_dura"] / 4.184
    female = (-20.4022 + 0.4472 * out["hr_ratio"] - 0.1263 * out["weight_kg"] + 0.0740 * out["age"]) * out["ex_dura"] / 4.184
    out["hr_wt_age_dura"] = np.where(g.eq("M"), male, female).round()

    out["exercise_stress_index_raw"] = np.sqrt(np.clip(out["ex_dura"] * (out["bpm"] / 1.5) * (out["body_temp"] - 98.6), a_min=0, a_max=None))
    out["ex_stress_index_new"] = out["ex_dura"] * (out["bpm"] + out["bsa"] + out["age"]) * (out["body_temp"] - 98.6)

    out["bpm_section"] = out["bpm"].apply(bpm_section)
    out["ex_section"] = out["ex_dura"].apply(ex_section)

    male_cal = ((out["age"] * 0.2017) + (out["weight_kg"] * 0.09036) + (out["bpm"] * 0.6309) - 55.0969) * out["ex_dura"] / 4.184
    female_cal = ((out["age"] * 0.074) - (out["weight_kg"] * 0.05741) + (out["bpm"] * 0.4472) - 20.4022) * out["ex_dura"] / 4.184

    is_male = out["gender"].astype(str).str.upper().str.strip().eq("M")
    out["pre_cal"] = np.where(is_male, male_cal, female_cal)

    if rounded_mode == "round":
        out["pre_cal_rounded"] = np.round(out["pre_cal"])
    else:
        # phase3_01 notebook uses np.floor(x + 0.5)
        out["pre_cal_rounded"] = np.floor(out["pre_cal"] + 0.5)

    return out


def make_master_features(df: pd.DataFrame, rounded_mode: str = "floor_half_up") -> pd.DataFrame:
    """
    Reconstruct the wide feature table closest to final_train_adj_v6.csv
    and the formula-recovery additions used for v9/v10.
    """
    out = add_basic_features(df)
    out = add_full_interaction_features(out)
    out = add_formula_recovery_features(out, rounded_mode=rounded_mode)
    return out


# Backward-compatible alias for the earlier simplified module.
make_features = make_master_features


def select_feature_version(df: pd.DataFrame, version: str, include_target: bool = True) -> pd.DataFrame:
    """
    Select notebook/dataset-version columns from a wide dataframe.

    version examples:
    - base, v2, v3, ..., v10
    - phase1_01_svr_v2
    - phase2_01_stacking_v9
    """
    cols = resolve_feature_set(version)
    if include_target and TARGET_COL in df.columns and TARGET_COL not in cols:
        cols = cols + [TARGET_COL]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{version}: missing columns {missing}")
    return df[cols].copy()


def build_versioned_dataset(df: pd.DataFrame, version: str, include_target: bool = True, rounded_mode: str = "floor_half_up") -> pd.DataFrame:
    """
    Convenience wrapper:
    raw dataframe -> master engineered dataframe -> version selection.
    """
    engineered = make_master_features(df, rounded_mode=rounded_mode)
    return select_feature_version(engineered, version=version, include_target=include_target)
