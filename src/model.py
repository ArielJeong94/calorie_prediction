
"""
Model factory and notebook-aligned experiment builders.
"""
from __future__ import annotations

from typing import Optional

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge

from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from .config import RANDOM_STATE
from .preprocessing import build_preprocessor


def build_svr_pipeline(X, C: float = 18000, gamma: float = 0.0055, epsilon: float = 0.019) -> Pipeline:
    preprocess = build_preprocessor(X, kind="svr")
    model = SVR(kernel="rbf", C=C, gamma=gamma, epsilon=epsilon)
    return Pipeline([
        ("prep", preprocess),
        ("model", model),
    ])


def build_lightgbm_pipeline(X, params: Optional[dict] = None) -> Pipeline:
    preprocess = build_preprocessor(X, kind="tree")
    default = {
        "random_state": RANDOM_STATE,
        "verbose": -1,
        "n_jobs": -1,
    }
    if params:
        default.update(params)
    model = LGBMRegressor(**default)
    return Pipeline([
        ("prep", preprocess),
        ("model", model),
    ])


def build_catboost_model(params: Optional[dict] = None) -> CatBoostRegressor:
    default = {
        "loss_function": "RMSE",
        "random_state": RANDOM_STATE,
        "verbose": 0,
    }
    if params:
        default.update(params)
    return CatBoostRegressor(**default)


def build_formula_recovery_model() -> LinearRegression:
    return LinearRegression()


def build_meta_ridge(alpha: float = 1.0) -> Ridge:
    return Ridge(alpha=alpha, random_state=RANDOM_STATE)


def fit_model(model, X_train, y_train, **fit_kwargs):
    m = clone(model)
    m.fit(X_train, y_train, **fit_kwargs)
    return m
