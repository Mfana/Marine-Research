"""SHAP explanations for fitted tree and linear models.

SHAP values describe how a *fitted model* uses its inputs. They are not causal
statements about ocean physics; lagged features are strongly autocorrelated, so
attributions can be shared arbitrarily among correlated lags.
"""

from __future__ import annotations

import pandas as pd


def shap_values(model, X: pd.DataFrame, background: pd.DataFrame | None = None):
    import shap
    name = type(model).__name__
    if name in ("RandomForestRegressor", "XGBRegressor", "GradientBoostingRegressor"):
        explainer = shap.TreeExplainer(model)
    elif name in ("LinearRegression", "Ridge", "Lasso"):
        explainer = shap.LinearExplainer(model, background if background is not None else X)
    else:
        explainer = shap.Explainer(model.predict, background if background is not None else X)
    return explainer(X)


def mean_abs_importance(explanation, feature_names) -> pd.Series:
    import numpy as np
    vals = np.abs(explanation.values).mean(axis=0)
    return pd.Series(vals, index=feature_names, name="mean_abs_shap").sort_values(ascending=False)


def summary_plot(explanation, X: pd.DataFrame, show: bool = False):
    import matplotlib.pyplot as plt
    import shap
    shap.summary_plot(explanation, X, show=show)
    return plt.gcf()
