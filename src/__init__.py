"""Visualization entry points for the calorie prediction project.

This lightweight package file is focused on the plotting helpers that were
re-sent after the earlier execution environment expired.
"""

from .plots import (
    setup_plot_style,
    hist_one_feature,
    boxplt_with_mean,
    violin_with_median,
    all_feature_plot,
    plot_correlation_heatmap,
    plot_3d_scatter_by_gender,
    plot_scatter_linear_lowess_grid,
    plot_group_median_line,
    plot_actual_vs_predicted,
    plot_residual_scatter,
    plot_feature_importance,
    plot_cv_rmse_comparison,
    plot_named_series_comparison,
    plot_experiment_rmse_bar,
    save_current_figure,
)

__all__ = [
    'setup_plot_style',
    'hist_one_feature',
    'boxplt_with_mean',
    'violin_with_median',
    'all_feature_plot',
    'plot_correlation_heatmap',
    'plot_3d_scatter_by_gender',
    'plot_scatter_linear_lowess_grid',
    'plot_group_median_line',
    'plot_actual_vs_predicted',
    'plot_residual_scatter',
    'plot_feature_importance',
    'plot_cv_rmse_comparison',
    'plot_named_series_comparison',
    'plot_experiment_rmse_bar',
    'save_current_figure',
]
