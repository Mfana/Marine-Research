"""Multi-horizon SST anomaly forecasting: leakage-safe datasets, baselines, tree models,
an LSTM, metrics and the Diebold–Mariano test (paper 1).

All models predict the *anomaly* at t+h relative to a monthly climatology fitted on the
training period only; forecasts are converted back to °C by adding that climatology.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


# ---- data -------------------------------------------------------------------

def train_climatology(df: pd.DataFrame, train_end: str) -> pd.DataFrame:
    """Monthly climatology (12 × zones) from the training period only."""
    tr = df.loc[:train_end]
    return tr.groupby(tr.index.month).mean()


def to_anomalies(df: pd.DataFrame, clim: pd.DataFrame) -> pd.DataFrame:
    return df - clim.reindex(df.index.month).set_axis(df.index)


@dataclass
class Dataset:
    X: pd.DataFrame            # tabular features at forecast origin t
    seq: np.ndarray            # (n, n_lags, n_channels) sequences for the LSTM
    y: pd.Series               # target anomaly at t+h
    origin: pd.DatetimeIndex   # forecast origin t
    target_time: pd.DatetimeIndex
    persistence: pd.Series     # anomaly at t (persistence forecast of the anomaly)


def make_dataset(anom: pd.DataFrame, target: str, h: int, n_lags: int = 12,
                 cross_zone: bool = True) -> Dataset:
    """Features at origin t use only anomalies at t, t-1, ..., t-n_lags+1 (no leakage).
    Calendar terms describe the *target* month, which is known in advance."""
    cols = list(anom.columns) if cross_zone else [target]
    a = anom[cols]
    feats = {}
    for k in range(n_lags):
        feats[f"{target}_lag{k}"] = anom[target].shift(k)
    if cross_zone:
        for c in cols:
            if c != target:
                feats[f"{c}_lag0"] = a[c]
    X = pd.DataFrame(feats, index=anom.index)
    tgt_time = anom.index + pd.DateOffset(months=h)
    month = tgt_time.month
    X["target_month_sin"] = np.sin(2 * np.pi * month / 12)
    X["target_month_cos"] = np.cos(2 * np.pi * month / 12)
    y = anom[target].shift(-h).rename("y")

    seq_cols = [target] + [c for c in cols if c != target]
    arr = a[seq_cols].to_numpy()
    seqs = np.full((len(anom), n_lags, len(seq_cols) + 2), np.nan)
    for i in range(n_lags - 1, len(anom)):
        seqs[i, :, :len(seq_cols)] = arr[i - n_lags + 1:i + 1]
    seqs[:, :, -2] = X["target_month_sin"].to_numpy()[:, None]
    seqs[:, :, -1] = X["target_month_cos"].to_numpy()[:, None]

    ok = X.notna().all(axis=1) & y.notna() & ~np.isnan(seqs).any(axis=(1, 2))
    idx = anom.index[ok]
    return Dataset(X=X.loc[idx], seq=seqs[ok.to_numpy()], y=y.loc[idx], origin=idx,
                   target_time=pd.DatetimeIndex(idx + pd.DateOffset(months=h)),
                   persistence=anom[target].loc[idx])


def split_mask(ds: Dataset, start: str | None, end: str) -> np.ndarray:
    """Select samples whose *target* time falls in [start, end] (so no target leaks
    across split boundaries)."""
    t = ds.target_time
    m = t <= pd.Timestamp(end)
    if start:
        m &= t >= pd.Timestamp(start)
    return np.asarray(m)


def subset(ds: Dataset, mask: np.ndarray) -> Dataset:
    return Dataset(X=ds.X[mask], seq=ds.seq[mask], y=ds.y[mask], origin=ds.origin[mask],
                   target_time=ds.target_time[mask], persistence=ds.persistence[mask])


# ---- models -----------------------------------------------------------------

def fit_tree(kind: str, params: dict, X: pd.DataFrame, y: pd.Series, seed: int):
    if kind == "ridge":
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        m = make_pipeline(StandardScaler(), Ridge(alpha=params.get("alpha", 1.0)))
    elif kind == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        m = RandomForestRegressor(n_estimators=500, random_state=seed, n_jobs=-1, **params)
    elif kind == "xgboost":
        from xgboost import XGBRegressor
        # Single-threaded: on macOS, torch and xgboost load separate OpenMP runtimes and
        # deadlock when both run multi-threaded in one process (see scripts/paper1_forecasting.py).
        m = XGBRegressor(n_estimators=params.pop("n_estimators", 400), random_state=seed,
                         subsample=0.8, colsample_bytree=0.8, n_jobs=1, **params)
    else:
        raise ValueError(kind)
    return m.fit(X, y)


class LSTMForecaster:
    """Small LSTM on (n_lags × channels) anomaly sequences; early stopping on a
    validation set; inputs standardised with training statistics."""

    def __init__(self, hidden: int = 32, layers: int = 1, dropout: float = 0.1, lr: float = 3e-3,
                 weight_decay: float = 1e-4, max_epochs: int = 400, patience: int = 40, seed: int = 0):
        self.__dict__.update(hidden=hidden, layers=layers, dropout=dropout, lr=lr, weight_decay=weight_decay,
                             max_epochs=max_epochs, patience=patience, seed=seed)

    def _net(self, n_in):
        import torch.nn as nn

        class Net(nn.Module):
            def __init__(s):
                super().__init__()
                s.lstm = nn.LSTM(n_in, self.hidden, self.layers, batch_first=True,
                                 dropout=self.dropout if self.layers > 1 else 0.0)
                s.drop = nn.Dropout(self.dropout)
                s.head = nn.Linear(self.hidden, 1)

            def forward(s, x):
                out, _ = s.lstm(x)
                return s.head(s.drop(out[:, -1])).squeeze(-1)
        return Net()

    def fit(self, Xs, y, Xv=None, yv=None):
        import torch
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.mu = Xs.reshape(-1, Xs.shape[-1]).mean(0)
        self.sd = Xs.reshape(-1, Xs.shape[-1]).std(0) + 1e-8
        self.ymu, self.ysd = float(np.mean(y)), float(np.std(y)) + 1e-8
        f = lambda a: torch.tensor((a - self.mu) / self.sd, dtype=torch.float32)
        g = lambda a: torch.tensor((np.asarray(a) - self.ymu) / self.ysd, dtype=torch.float32)
        xt, yt = f(Xs), g(y)
        has_val = Xv is not None and len(Xv) > 0
        if has_val:
            xv, yvt = f(Xv), g(yv)
        self.net = self._net(Xs.shape[-1])
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        best, best_state, wait = np.inf, None, 0
        gen = torch.Generator().manual_seed(self.seed)
        for epoch in range(self.max_epochs):
            self.net.train()
            perm = torch.randperm(len(xt), generator=gen)
            for i in range(0, len(xt), 32):
                b = perm[i:i + 32]
                opt.zero_grad()
                loss = torch.mean((self.net(xt[b]) - yt[b]) ** 2)
                loss.backward()
                opt.step()
            if has_val:
                self.net.eval()
                with torch.no_grad():
                    v = float(torch.mean((self.net(xv) - yvt) ** 2))
                if v < best - 1e-5:
                    best, wait = v, 0
                    best_state = {k: t.clone() for k, t in self.net.state_dict().items()}
                    self.best_epoch = epoch + 1
                else:
                    wait += 1
                    if wait >= self.patience:
                        break
        if has_val and best_state is not None:
            self.net.load_state_dict(best_state)
        else:
            self.best_epoch = self.max_epochs
        return self

    def predict(self, Xs):
        import torch
        self.net.eval()
        with torch.no_grad():
            p = self.net(torch.tensor((Xs - self.mu) / self.sd, dtype=torch.float32)).numpy()
        return p * self.ysd + self.ymu


# ---- evaluation -------------------------------------------------------------

def scores(y: np.ndarray, p: np.ndarray, clim_mse: float | None = None,
           pers_mse: float | None = None) -> dict:
    """y and p are anomalies; RMSE/MAE/bias are identical in °C after adding climatology."""
    e = np.asarray(p) - np.asarray(y)
    mse = float(np.mean(e ** 2))
    out = {"RMSE": np.sqrt(mse), "MAE": float(np.mean(np.abs(e))), "bias": float(np.mean(e)),
           "ACC": float(np.corrcoef(y, p)[0, 1]) if np.std(p) > 0 else np.nan, "n": len(e)}
    if clim_mse is not None:
        out["MSSS_clim"] = 1 - mse / clim_mse
    if pers_mse is not None:
        out["MSSS_pers"] = 1 - mse / pers_mse
    return out


def diebold_mariano(y, p1, p2, h: int = 1) -> tuple[float, float]:
    """DM test with the Harvey–Leybourne–Newbold small-sample correction, squared-error
    loss. Negative statistic: forecast 1 more accurate. Returns (statistic, two-sided p)."""
    y, p1, p2 = map(np.asarray, (y, p1, p2))
    d = (p1 - y) ** 2 - (p2 - y) ** 2
    n = len(d)
    dbar = d.mean()
    gamma = [np.mean((d[k:] - dbar) * (d[:n - k] - dbar)) for k in range(h)]
    var = (gamma[0] + 2 * sum(gamma[1:])) / n
    if var <= 0:
        return np.nan, np.nan
    dm = dbar / np.sqrt(var)
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat = dm * hln
    return float(stat), float(2 * stats.t.sf(abs(stat), df=n - 1))
