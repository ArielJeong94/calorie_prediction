"""Plot utilities for the calorie prediction hackathon project.

This module focuses on two needs from the original notebooks:
1. EDA visualizations used in the feature engineering notebook
2. Model evaluation visualizations used in the training notebooks
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def setup_plot_style(font_family: Optional[str] = None, korean: bool = True) -> None:
    """Apply a clean default matplotlib style.

    Parameters
    ----------
    font_family : str, optional
        Preferred font family. When None, matplotlib default is used.
    korean : bool, default True
        If True, tries to avoid minus sign breaking and works better for Korean text.
    """
    if font_family:
        plt.rcParams['font.family'] = font_family
    if korean:
        plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = 120
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.25
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 9


def _ensure_series(data: pd.DataFrame | pd.Series, column: Optional[str] = None) -> pd.Series:
    if isinstance(data, pd.Series):
        return data.dropna()
    if column is None:
        raise ValueError('column must be provided when data is a DataFrame')
    return data[column].dropna()


def hist_one_feature(
    data: pd.DataFrame | pd.Series,
    column: Optional[str] = None,
    bins: int = 30,
    title: Optional[str] = None,
    figsize: tuple[int, int] = (6, 4),
) -> plt.Axes:
    """Draw a single-feature histogram."""
    s = _ensure_series(data, column)
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(s, bins=bins)
    ax.set_title(title or f'Histogram: {s.name}')
    ax.set_xlabel(s.name or column or 'value')
    ax.set_ylabel('count')
    fig.tight_layout()
    return ax


def boxplt_with_mean(
    data: pd.DataFrame,
    column: str,
    by: Optional[str] = None,
    title: Optional[str] = None,
    figsize: tuple[int, int] = (6, 4),
) -> plt.Axes:
    """Draw a box plot and overlay the mean."""
    fig, ax = plt.subplots(figsize=figsize)
    if by is None:
        vals = data[column].dropna()
        ax.boxplot(vals, vert=True)
        ax.scatter([1], [vals.mean()], marker='D', label='mean')
        ax.set_xticks([1])
        ax.set_xticklabels([column])
    else:
        groups = [grp[column].dropna().values for _, grp in data.groupby(by)]
        labels = [str(k) for k, _ in data.groupby(by)]
        ax.boxplot(groups, labels=labels)
        means = [np.mean(g) if len(g) else np.nan for g in groups]
        ax.scatter(range(1, len(means) + 1), means, marker='D', label='mean')
        ax.set_xlabel(by)
    ax.set_ylabel(column)
    ax.set_title(title or f'Box Plot: {column}')
    ax.legend()
    fig.tight_layout()
    return ax


def violin_with_median(
    data: pd.DataFrame,
    column: str,
    by: Optional[str] = None,
    title: Optional[str] = None,
    figsize: tuple[int, int] = (6, 4),
) -> plt.Axes:
    """Draw a violin plot and overlay the median."""
    fig, ax = plt.subplots(figsize=figsize)
    if by is None:
        vals = data[column].dropna().values
        parts = ax.violinplot(vals, showmeans=False, showmedians=False, showextrema=True)
        ax.scatter([1], [np.median(vals)], marker='o', label='median')
        ax.set_xticks([1])
        ax.set_xticklabels([column])
    else:
        groups = [grp[column].dropna().values for _, grp in data.groupby(by)]
        labels = [str(k) for k, _ in data.groupby(by)]
        ax.violinplot(groups, showmeans=False, showmedians=False, showextrema=True)
        medians = [np.median(g) if len(g) else np.nan for g in groups]
        ax.scatter(range(1, len(medians) + 1), medians, marker='o', label='median')
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels)
        ax.set_xlabel(by)
    ax.set_ylabel(column)
    ax.set_title(title or f'Violin Plot: {column}')
    ax.legend()
    fig.tight_layout()
    return ax


def all_feature_plot(
    data: pd.DataFrame,
    columns: Sequence[str],
    bins: int = 30,
    ncols: int = 3,
    figsize_per_ax: tuple[float, float] = (4.5, 3.5),
) -> np.ndarray:
    """Draw histograms for multiple numeric features."""
    columns = list(columns)
    n = len(columns)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(figsize_per_ax[0] * ncols, figsize_per_ax[1] * nrows),
        squeeze=False,
    )
    flat_axes = axes.ravel()
    for ax, col in zip(flat_axes, columns):
        vals = data[col].dropna()
        ax.hist(vals, bins=bins)
        ax.set_title(col)
    for ax in flat_axes[len(columns):]:
        ax.axis('off')
    fig.tight_layout()
    return axes


def plot_correlation_heatmap(
    data: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
    figsize: tuple[int, int] = (10, 8),
    annot: bool = False,
) -> plt.Axes:
    """Plot a correlation heatmap using matplotlib only."""
    corr_df = data[list(columns)].corr() if columns is not None else data.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(corr_df.values, aspect='auto')
    ax.set_xticks(range(len(corr_df.columns)))
    ax.set_xticklabels(corr_df.columns, rotation=90)
    ax.set_yticks(range(len(corr_df.index)))
    ax.set_yticklabels(corr_df.index)
    if annot:
        for i in range(corr_df.shape[0]):
            for j in range(corr_df.shape[1]):
                ax.text(j, i, f'{corr_df.iloc[i, j]:.2f}', ha='center', va='center', fontsize=7)
    fig.colorbar(im, ax=ax)
    ax.set_title('Correlation Heatmap')
    fig.tight_layout()
    return ax


def plot_3d_scatter_by_gender(
    data: pd.DataFrame,
    x: str,
    y: str,
    z: str,
    gender_col: str = 'gender',
    figsize: tuple[int, int] = (7, 5),
):
    """3D scatter plot split by gender category."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    for label, grp in data.groupby(gender_col):
        ax.scatter(grp[x], grp[y], grp[z], label=str(label), alpha=0.7)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_zlabel(z)
    ax.set_title(f'3D Scatter: {x} / {y} / {z}')
    ax.legend()
    fig.tight_layout()
    return ax


def _simple_lowess_like(x: np.ndarray, y: np.ndarray, frac: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    """A lightweight smooth curve approximation using sorted rolling mean.

    Used instead of statsmodels LOWESS to keep dependencies minimal.
    """
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    window = max(5, int(len(x_sorted) * frac))
    y_smooth = pd.Series(y_sorted).rolling(window=window, center=True, min_periods=1).mean().to_numpy()
    return x_sorted, y_smooth


def plot_scatter_linear_lowess_grid(
    data: pd.DataFrame,
    features: Sequence[str],
    target: str,
    ncols: int = 2,
    figsize_per_ax: tuple[float, float] = (5.0, 4.0),
) -> np.ndarray:
    """Scatter plots with linear fit and a LOWESS-like smooth line."""
    features = list(features)
    n = len(features)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(figsize_per_ax[0] * ncols, figsize_per_ax[1] * nrows),
        squeeze=False,
    )
    flat_axes = axes.ravel()

    for ax, feat in zip(flat_axes, features):
        sub = data[[feat, target]].dropna()
        x = sub[feat].to_numpy()
        y = sub[target].to_numpy()
        ax.scatter(x, y, alpha=0.5)
        if len(x) >= 2:
            coef = np.polyfit(x, y, deg=1)
            x_line = np.linspace(x.min(), x.max(), 200)
            y_line = coef[0] * x_line + coef[1]
            ax.plot(x_line, y_line, linewidth=2, label='linear fit')
            xs, ys = _simple_lowess_like(x, y)
            ax.plot(xs, ys, linewidth=2, linestyle='--', label='smooth')
        ax.set_title(f'{feat} vs {target}')
        ax.set_xlabel(feat)
        ax.set_ylabel(target)
        ax.legend()

    for ax in flat_axes[len(features):]:
        ax.axis('off')
    fig.tight_layout()
    return axes


def plot_group_median_line(
    data: pd.DataFrame,
    x: str,
    y: str,
    sort_index: bool = True,
    figsize: tuple[int, int] = (6, 4),
) -> plt.Axes:
    """Plot median of y grouped by x."""
    grp = data.groupby(x)[y].median()
    if sort_index:
        try:
            grp = grp.sort_index()
        except Exception:
            pass
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(grp.index.astype(str), grp.values, marker='o')
    ax.set_title(f'Median {y} by {x}')
    ax.set_xlabel(x)
    ax.set_ylabel(f'median({y})')
    fig.tight_layout()
    return ax


def plot_actual_vs_predicted(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    title: str = 'Actual vs Predicted',
    figsize: tuple[int, int] = (6, 5),
) -> plt.Axes:
    """Scatter plot for actual vs predicted values."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(y_true, y_pred, alpha=0.6)
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], linestyle='--', linewidth=2)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title(title)
    fig.tight_layout()
    return ax


def plot_residual_scatter(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    title: str = 'Residual Scatter',
    figsize: tuple[int, int] = (6, 4),
) -> plt.Axes:
    """Residuals vs predicted plot."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(y_pred, residuals, alpha=0.6)
    ax.axhline(0, linestyle='--', linewidth=2)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Residual')
    ax.set_title(title)
    fig.tight_layout()
    return ax


def plot_feature_importance(
    importance_df: pd.DataFrame,
    feature_col: str = 'feature',
    importance_col: str = 'importance',
    top_n: int = 20,
    title: str = 'Feature Importance',
    figsize: tuple[int, int] = (8, 6),
) -> plt.Axes:
    """Horizontal bar chart for feature importance."""
    imp = importance_df[[feature_col, importance_col]].copy()
    imp = imp.sort_values(importance_col, ascending=False).head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(imp[feature_col].astype(str), imp[importance_col])
    ax.set_title(title)
    ax.set_xlabel(importance_col)
    ax.set_ylabel(feature_col)
    fig.tight_layout()
    return ax


def plot_cv_rmse_comparison(
    cv_scores: dict[str, Sequence[float]],
    title: str = 'CV RMSE Comparison',
    figsize: tuple[int, int] = (8, 5),
) -> plt.Axes:
    """Compare per-fold RMSE across experiments."""
    fig, ax = plt.subplots(figsize=figsize)
    for name, scores in cv_scores.items():
        scores = list(scores)
        ax.plot(range(1, len(scores) + 1), scores, marker='o', label=name)
    ax.set_xlabel('Fold')
    ax.set_ylabel('RMSE')
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return ax


def plot_named_series_comparison(
    values: dict[str, float],
    title: str = 'Named Series Comparison',
    ylabel: str = 'value',
    figsize: tuple[int, int] = (8, 4),
) -> plt.Axes:
    """Simple bar comparison for named scalar values."""
    fig, ax = plt.subplots(figsize=figsize)
    names = list(values.keys())
    vals = list(values.values())
    ax.bar(names, vals)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis='x', rotation=45)
    fig.tight_layout()
    return ax


def plot_experiment_rmse_bar(
    experiments: pd.DataFrame,
    name_col: str = 'experiment',
    score_col: str = 'rmse',
    ascending: bool = True,
    title: str = 'Experiment RMSE',
    figsize: tuple[int, int] = (9, 5),
) -> plt.Axes:
    """Bar chart for experiment RMSE table."""
    df = experiments[[name_col, score_col]].copy().sort_values(score_col, ascending=ascending)
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(df[name_col].astype(str), df[score_col])
    ax.set_ylabel(score_col)
    ax.set_title(title)
    ax.tick_params(axis='x', rotation=45)
    fig.tight_layout()
    return ax


def save_current_figure(path: str | Path, dpi: int = 150) -> Path:
    """Save the current matplotlib figure and return the saved path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.gcf().savefig(path, dpi=dpi, bbox_inches='tight')
    return path
