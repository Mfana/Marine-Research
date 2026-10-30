"""Forecasting scaffolding with chronological splits and leakage-safe features.

Do not train on real data until remote access and preprocessing are validated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


@dataclass
class Split:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def make_features(s: pd.Series, lags=(1, 2, 3, 12)) -> pd.DataFrame:
    """Lagged features + calendar encoding. Only past values are used for each row."""
    df = pd.DataFrame({"y": s})
    for k in lags:
        df[f"lag_{k}"] = s.shift(k)
    month = s.index.month
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    return df.dropna()


def chronological_split(df: pd.DataFrame, train_end: str, validation_end: str,
                        test_end: str | None = None) -> Split:
    """Split by date; no shuffling, so later data never informs earlier predictions."""
    train = df.loc[:train_end]
    val = df.loc[pd.Timestamp(train_end) + pd.Timedelta(days=1):validation_end]
    test = df.loc[pd.Timestamp(validation_end) + pd.Timedelta(days=1):test_end]
    return Split(train, val, test)


# ---- baselines -------------------------------------------------------------

def persistence_forecast(df: pd.DataFrame) -> pd.Series:
    """Next month = this month."""
    return df["lag_1"].rename("persistence")


def seasonal_persistence_forecast(df: pd.DataFrame) -> pd.Series:
    """Next month = same month last year."""
    return df["lag_12"].rename("seasonal_persistence")


def climatology_forecast(train: pd.DataFrame, target: pd.DataFrame) -> pd.Series:
    """Monthly mean from the training period only (no leakage)."""
    clim = train["y"].groupby(train.index.month).mean()
    return pd.Series(clim.reindex(target.index.month).values, index=target.index, name="climatology")


# ---- models ----------------------------------------------------------------

def build_models(seed: int = 42) -> dict:
    from xgboost import XGBRegressor
    return {
        "linear": LinearRegression(),
        "random_forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=3,
                                               random_state=seed, n_jobs=-1),
        "xgboost": XGBRegressor(n_estimators=400, max_depth=3, learning_rate=0.05,
                                subsample=0.8, colsample_bytree=0.8, random_state=seed),
    }


def feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c != "y"]


def fit_predict(model, split: Split, on: str = "validation") -> pd.Series:
    X = feature_cols(split.train)
    model.fit(split.train[X], split.train["y"])
    target = getattr(split, on)
    return pd.Series(model.predict(target[X]), index=target.index)


# ---- metrics ---------------------------------------------------------------

def metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    e = (y_pred - y_true).dropna()
    return {"MAE": float(e.abs().mean()), "RMSE": float(np.sqrt((e ** 2).mean())),
            "bias": float(e.mean()), "n": int(len(e))}


def skill_score(model_rmse: float, baseline_rmse: float) -> float:
    """1 - RMSE_model / RMSE_baseline (>0 means better than the baseline)."""
    return float(1 - model_rmse / baseline_rmse) if baseline_rmse else float("nan")


def evaluate(split: Split, seed: int = 42, on: str = "validation") -> pd.DataFrame:
    target = getattr(split, on)
    preds = {
        "persistence": persistence_forecast(target),
        "seasonal_persistence": seasonal_persistence_forecast(target),
        "climatology": climatology_forecast(split.train, target),
    }
    for name, m in build_models(seed).items():
        preds[name] = fit_predict(m, split, on)
    rows = {k: metrics(target["y"], v) for k, v in preds.items()}
    out = pd.DataFrame(rows).T
    out["skill_vs_climatology"] = [skill_score(r, out.loc["climatology", "RMSE"]) for r in out["RMSE"]]
    out["skill_vs_persistence"] = [skill_score(r, out.loc["persistence", "RMSE"]) for r in out["RMSE"]]
    return out
