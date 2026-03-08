
"""
Residual/meta stacking rebuilt from phase2_01_stacking_v9_residual_meta.ipynb.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import Ridge

from .config import RANDOM_STATE
from .utils import rmse


def residual_stacking_cv_rmse(
    X: pd.DataFrame,
    y: pd.Series,
    base_model,
    residual_model,
    outer_splits: int = 5,
    inner_splits: int = 5,
    random_state: int = RANDOM_STATE,
):
    outer = KFold(n_splits=outer_splits, shuffle=True, random_state=random_state)
    preds_all = pd.Series(index=X.index, dtype=float)

    for fold, (tr_idx, va_idx) in enumerate(outer.split(X), 1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        inner = KFold(n_splits=inner_splits, shuffle=True, random_state=random_state)
        oof_base_tr = pd.Series(index=X_tr.index, dtype=float)

        for in_tr_idx, in_va_idx in inner.split(X_tr):
            X_in_tr, X_in_va = X_tr.iloc[in_tr_idx], X_tr.iloc[in_va_idx]
            y_in_tr = y_tr.iloc[in_tr_idx]

            m_base = clone(base_model)
            m_base.fit(X_in_tr, y_in_tr)
            oof_base_tr.loc[X_in_va.index] = m_base.predict(X_in_va)

        residual_tr = y_tr - oof_base_tr

        m_res = clone(residual_model)
        m_res.fit(X_tr, residual_tr)

        m_base_full = clone(base_model)
        m_base_full.fit(X_tr, y_tr)
        pred_base_va = m_base_full.predict(X_va)

        pred_res_va = m_res.predict(X_va)
        preds_all.loc[X_va.index] = pred_base_va + pred_res_va

    return rmse(y, preds_all), preds_all


def make_oof_predictions(X: pd.DataFrame, y: pd.Series, base_models: Dict[str, object], n_splits: int = 5, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof = pd.DataFrame(index=X.index)

    for name, model in base_models.items():
        pred = pd.Series(index=X.index, dtype=float)
        for tr_idx, va_idx in cv.split(X):
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr = y.iloc[tr_idx]
            m = clone(model)
            m.fit(X_tr, y_tr)
            pred.loc[X_va.index] = m.predict(X_va)
        oof[f"pred_{name}"] = pred

    return oof


def add_meta_features(base_pred_df: pd.DataFrame) -> pd.DataFrame:
    out = base_pred_df.copy()
    pred_cols = [c for c in out.columns if c.startswith("pred_")]

    out["meta_mean"] = out[pred_cols].mean(axis=1)
    out["meta_std"] = out[pred_cols].std(axis=1)
    out["meta_min"] = out[pred_cols].min(axis=1)
    out["meta_max"] = out[pred_cols].max(axis=1)
    out["meta_range"] = out["meta_max"] - out["meta_min"]

    if {"pred_svr", "pred_lgbm", "pred_cat"}.issubset(out.columns):
        out["svr_minus_lgbm"] = out["pred_svr"] - out["pred_lgbm"]
        out["svr_minus_cat"] = out["pred_svr"] - out["pred_cat"]
        out["lgbm_minus_cat"] = out["pred_lgbm"] - out["pred_cat"]

    return out


def make_oof_density_feature(
    X: pd.DataFrame,
    models_any_preprocess: Dict[str, object],
    n_splits: int = 5,
    random_state: int = RANDOM_STATE,
    k: int = 15,
) -> pd.Series:
    """
    Density = average KNN distance in the transformed feature space.
    The notebook uses the SVR preprocessor for this.
    """
    prep = models_any_preprocess["svr"].named_steps["prep"]
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    dens = pd.Series(index=X.index, dtype=float)

    for tr_idx, va_idx in cv.split(X):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        Xtr_t = prep.fit_transform(X_tr)
        Xva_t = prep.transform(X_va)

        nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
        nn.fit(Xtr_t)
        dists, _ = nn.kneighbors(Xva_t, return_distance=True)
        dens.loc[X_va.index] = dists.mean(axis=1)

    return dens


def stacking_cv_rmse(
    X: pd.DataFrame,
    y: pd.Series,
    base_models: Dict[str, object],
    meta_model=None,
    n_splits: int = 5,
    random_state: int = RANDOM_STATE,
):
    if meta_model is None:
        meta_model = Ridge(alpha=1.0, random_state=random_state)

    oof_base = make_oof_predictions(X, y, base_models, n_splits=n_splits, random_state=random_state)
    meta_X = add_meta_features(oof_base)

    density = make_oof_density_feature(X, base_models, n_splits=n_splits, random_state=random_state, k=15)
    meta_X["meta_density"] = density
    meta_y = y.copy()

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    final_pred = pd.Series(index=X.index, dtype=float)

    for tr_idx, va_idx in cv.split(meta_X):
        X_tr, X_va = meta_X.iloc[tr_idx], meta_X.iloc[va_idx]
        y_tr = meta_y.iloc[tr_idx]

        m = clone(meta_model)
        m.fit(X_tr, y_tr)
        final_pred.loc[meta_X.iloc[va_idx].index] = m.predict(X_va)

    return rmse(y, final_pred), final_pred, meta_X
