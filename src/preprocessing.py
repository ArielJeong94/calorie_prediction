
"""
Project-specific preprocessing and split helpers.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import RANDOM_STATE, TARGET_COL
from .feature_sets import get_feature_set

SplitMode = Literal["phase1", "stacking", "plain"]
PreprocessorKind = Literal["svr", "tree", "lightgbm_native", "catboost"]


def load_data(path: str | Path, drop_duplicates: bool = True) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    if drop_duplicates:
        df = df.drop_duplicates(keep="first")
    return df


def get_xy(df: pd.DataFrame, target_col: str = TARGET_COL):
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()
    return X, y


def make_outlier_flag(df: pd.DataFrame, mode: SplitMode = "phase1") -> pd.Series:
    """
    Notebook-faithful flags:
    - phase1 notebooks: body_temp >= 41 and bpm < 100
    - stacking notebook: body_temp >= 40
    - plain: all zeros
    """
    if mode == "phase1":
        return ((df["body_temp"] >= 41) & (df["bpm"] < 100.0)).astype(int)
    if mode == "stacking":
        return (df["body_temp"] >= 40).astype(int)
    return pd.Series(0, index=df.index, dtype=int)


def split_data(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    mode: SplitMode = "plain",
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
):
    X, y = get_xy(df, target_col=target_col)
    flag = make_outlier_flag(df, mode=mode)
    if flag.nunique() > 1:
        return train_test_split(
            X, y, flag,
            test_size=test_size,
            random_state=random_state,
            stratify=flag,
        )
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def infer_column_types(X: pd.DataFrame):
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]
    return num_cols, cat_cols


def build_preprocessor(X: pd.DataFrame, kind: PreprocessorKind = "svr") -> ColumnTransformer | None:
    """
    Mirrors the notebooks' preprocessing choices.

    - svr:
        median imputer + standard scaler for numeric
        most_frequent imputer + OHE for categoricals
    - tree / catboost:
        median imputer for numeric, most_frequent + OHE for categoricals
    - lightgbm_native:
        returns None because the notebook converts categorical columns to pandas 'category'
    """
    num_cols, cat_cols = infer_column_types(X)

    if kind == "lightgbm_native":
        return None

    if kind == "svr":
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
    else:
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
        ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
    )


def prepare_lightgbm_native_categorical(X: pd.DataFrame, categorical_columns: Optional[list[str]] = None) -> pd.DataFrame:
    """
    phase3_02 notebook-style conversion for LightGBM native categorical support.
    """
    out = X.copy()
    if categorical_columns is None:
        categorical_columns = [c for c in out.columns if out[c].dtype == "object"]
    for c in categorical_columns:
        out[c] = out[c].astype("category")
    return out


def subset_for_notebook(df: pd.DataFrame, notebook_name: str, target_col: str = TARGET_COL) -> pd.DataFrame:
    cols = get_feature_set(notebook_name, include_target=(target_col in df.columns))
    return df[cols].copy()
