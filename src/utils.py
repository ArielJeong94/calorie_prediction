
"""
Utilities for evaluation and notebook-result summaries.
"""
from __future__ import annotations

import os
import random
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def summarize_regression(y_true, y_pred) -> dict:
    return {
        "rmse": rmse(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
    }


def compare_experiments(experiments: Dict[str, Dict[str, float]], sort_by: str = "rmse") -> pd.DataFrame:
    df = pd.DataFrame(experiments).T
    if sort_by in df.columns:
        df = df.sort_values(sort_by)
    return df


def extract_feature_importance(model, feature_names) -> pd.DataFrame:
    if hasattr(model, "feature_importances_"):
        vals = model.feature_importances_
    elif hasattr(model, "coef_"):
        vals = np.ravel(model.coef_)
    else:
        raise AttributeError("This model does not expose feature importances or coefficients.")
    return pd.DataFrame({"feature": feature_names, "importance": vals}).sort_values("importance", ascending=False)
