
"""
Formula-recovery helpers from phase3_01_formula_recovery_linear.ipynb.
"""
from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LinearRegression

from .utils import rmse


def reverse_cal_function(gender_df: pd.DataFrame) -> dict:
    """
    Recover the hidden calorie equation coefficients for a single gender group.
    """
    X = pd.DataFrame({
        "Dur_Age": gender_df["ex_dura"] * gender_df["age"],
        "Dur_Weight": gender_df["ex_dura"] * gender_df["weight_kg"],
        "Dur_BPM": gender_df["ex_dura"] * gender_df["bpm"],
        "Duration_Only": gender_df["ex_dura"],
    })
    y = gender_df["calories_burned"]

    lr = LinearRegression(fit_intercept=False)
    lr.fit(X, y)
    coef_restored = lr.coef_ * 4.184

    return {
        "age_coef": float(coef_restored[0]),
        "weight_coef": float(coef_restored[1]),
        "bpm_coef": float(coef_restored[2]),
        "constant": float(coef_restored[3]),
    }


def fit_precal_linear(train_df: pd.DataFrame):
    X = train_df[["pre_cal_rounded"]].copy()
    y = train_df["calories_burned"].copy()
    model = LinearRegression()
    model.fit(X, y)
    return model


def evaluate_precal_linear(model, valid_df: pd.DataFrame) -> float:
    pred = model.predict(valid_df[["pre_cal_rounded"]])
    return rmse(valid_df["calories_burned"], pred)
