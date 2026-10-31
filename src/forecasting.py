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
    seq: np.ndarray            # (n, n_lags, 3) own-zone sequence + target-month sin/cos (LSTM)
    static: np.ndarray         # (n, k) other zones' anomalies at t (LSTM; k = 0 if univariate)
    y: pd.Series               # target anomaly at t+h
    origin: pd.DatetimeIndex   # forecast origin t
    target_time: pd.DatetimeIndex
    persistence: pd.Series     # anomaly at t (persistence forecast of the anomaly)


def make_dataset(anom: pd.DataFrame, target: str, h: int, n_lags: int = 12,
                 cross_zone: bool = True) -> Dataset:
    """Features at origin t use only anomalies at t, t-1, ..., t-n_lags+1 (no leakage).
    Calendar terms describe the *target* month, which is known in advance.

    The tabular features and the LSTM inputs carry the same information: the target zone's
    last n_lags anomalies, the target month and (cross_zone) the other zones' anomalies at t.
    The LSTM gets the first two as a sequence and the other zones as static inputs."""
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

    arr = anom[target].to_numpy()
    seqs = np.full((len(anom), n_lags, 3), np.nan)
    for i in range(n_lags - 1, len(anom)):
        seqs[i, :, 0] = arr[i - n_lags + 1:i + 1]
    seqs[:, :, 1] = X["target_month_sin"].to_numpy()[:, None]
    seqs[:, :, 2] = X["target_month_cos"].to_numpy()[:, None]
    static = a[[c for c in cols if c != target]].to_numpy()

    ok = X.notna().all(axis=1) & y.notna() & ~np.isnan(seqs).any(axis=(1, 2))
    idx = anom.index[ok]
    okn = ok.to_numpy()
    return Dataset(X=X.loc[idx], seq=seqs[okn], static=static[okn], y=y.loc[idx], origin=idx,
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


def fit_mask(ds: Dataset, eval_mask: np.ndarray) -> np.ndarray:
    """Training samples for a model that is evaluated on ``eval_mask``: only those whose
    target was observed by the earliest evaluation origin. A month's value is taken to be
    available once that month ends, so a forecast issued at origin t may use labels up to
    t. Splitting by target time alone is not enough at leads h > 1: a model fitted on
    targets up to the month before the evaluation period would have seen labels that were
    still in the future at the first h - 1 evaluation origins."""
    cutoff = ds.origin[eval_mask].min()
    return np.asarray(ds.target_time <= cutoff)


def evaluation_masks(ds: Dataset, val_start: str, val_end: str,
                     test_start: str, test_end: str) -> tuple[np.ndarray, ...]:
    """Keep *selection*, including early stopping, available at the first test origin.

    Target-date disjointness and an origin-safe final fit alone are insufficient:
    validation outcomes used to select hyperparameters must also be observed then.
    """
    te = split_mask(ds, test_start, test_end)
    if not te.any():
        raise ValueError("empty evaluation period")
    cutoff = ds.origin[te].min()
    va = split_mask(ds, val_start, val_end) & (ds.target_time <= cutoff)
    if not va.any():
        raise ValueError("empty validation period before first evaluation origin")
    tr, refit = fit_mask(ds, va), fit_mask(ds, te)
    assert ds.target_time[va].max() <= cutoff
    assert ds.target_time[tr].max() <= ds.origin[va].min()
    return tr, va, refit, te


def damped_persistence(train: Dataset, evaluation: Dataset) -> tuple[np.ndarray, float]:
    """Direct, zero-intercept least-squares damping fitted separately at each lead.

    Restrict the multiplier to [0, 1], so the baseline interpolates between anomaly
    climatology and undamped persistence rather than adding an extrapolation model.
    """
    x, y = train.persistence.to_numpy(), train.y.to_numpy()
    denom = float(x @ x)
    beta = float(np.clip((x @ y) / denom, 0, 1)) if denom > 0 else 0.0
    return beta * evaluation.persistence.to_numpy(), beta


def subset(ds: Dataset, mask: np.ndarray) -> Dataset:
    return Dataset(X=ds.X[mask], seq=ds.seq[mask], static=ds.static[mask], y=ds.y[mask],
                   origin=ds.origin[mask],
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
    """Small LSTM on (n_lags × channels) anomaly sequences, with optional static inputs
    joined to the final hidden state before the output layer; early stopping on a
    validation set; inputs standardised with training statistics."""

    def __init__(self, hidden: int = 32, layers: int = 1, dropout: float = 0.1, lr: float = 3e-3,
                 weight_decay: float = 1e-4, max_epochs: int = 400, patience: int = 40, seed: int = 0):
        self.__dict__.update(hidden=hidden, layers=layers, dropout=dropout, lr=lr, weight_decay=weight_decay,
                             max_epochs=max_epochs, patience=patience, seed=seed)

    def _net(self, n_in, n_static):
        import torch
        import torch.nn as nn

        class Net(nn.Module):
            def __init__(s):
                super().__init__()
                s.lstm = nn.LSTM(n_in, self.hidden, self.layers, batch_first=True,
                                 dropout=self.dropout if self.layers > 1 else 0.0)
                s.drop = nn.Dropout(self.dropout)
                s.head = nn.Linear(self.hidden + n_static, 1)

            def forward(s, x, z):
                out, _ = s.lstm(x)
                return s.head(torch.cat([s.drop(out[:, -1]), z], dim=1)).squeeze(-1)
        return Net()

    def _static(self, Xs, S):
        return np.zeros((len(Xs), 0)) if S is None else np.asarray(S, dtype=float)

    def _tensors(self, Xs, S):
        import torch
        S = self._static(Xs, S)
        x = torch.tensor((Xs - self.mu) / self.sd, dtype=torch.float32)
        z = torch.tensor((S - self.smu) / self.ssd, dtype=torch.float32)
        return x, z

    def fit(self, Xs, y, Xv=None, yv=None, S=None, Sv=None):
        import torch
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        S = self._static(Xs, S)
        self.mu = Xs.reshape(-1, Xs.shape[-1]).mean(0)
        self.sd = Xs.reshape(-1, Xs.shape[-1]).std(0) + 1e-8
        self.smu, self.ssd = S.mean(0), S.std(0) + 1e-8
        self.ymu, self.ysd = float(np.mean(y)), float(np.std(y)) + 1e-8
        g = lambda a: torch.tensor((np.asarray(a) - self.ymu) / self.ysd, dtype=torch.float32)
        (xt, zt), yt = self._tensors(Xs, S), g(y)
        has_val = Xv is not None and len(Xv) > 0
        if has_val:
            (xv, zv), yvt = self._tensors(Xv, Sv), g(yv)
        self.net = self._net(Xs.shape[-1], S.shape[1])
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        best, best_state, wait = np.inf, None, 0
        self.history_ = []
        gen = torch.Generator().manual_seed(self.seed)
        for epoch in range(self.max_epochs):
            self.net.train()
            perm = torch.randperm(len(xt), generator=gen)
            for i in range(0, len(xt), 32):
                b = perm[i:i + 32]
                opt.zero_grad()
                loss = torch.mean((self.net(xt[b], zt[b]) - yt[b]) ** 2)
                loss.backward()
                opt.step()
            if has_val:
                self.net.eval()
                with torch.no_grad():
                    v = float(torch.mean((self.net(xv, zv) - yvt) ** 2))
                with torch.no_grad():
                    train_loss = float(torch.mean((self.net(xt, zt) - yt) ** 2))
                self.history_.append({"epoch": epoch + 1, "validation_mse_standardised": v,
                                      "training_mse_standardised": train_loss})
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

    def predict(self, Xs, S=None):
        import torch
        self.net.eval()
        with torch.no_grad():
            p = self.net(*self._tensors(Xs, S)).numpy()
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


def diebold_mariano(y, p1, p2, h: int = 1, bandwidth: int | None = None) -> tuple[float, float]:
    """DM test with the Harvey–Leybourne–Newbold small-sample correction, squared-error
    loss. Negative statistic: forecast 1 more accurate. Returns (statistic, two-sided p)."""
    y, p1, p2 = map(np.asarray, (y, p1, p2))
    d = (p1 - y) ** 2 - (p2 - y) ** 2
    n = len(d)
    dbar = d.mean()
    # Bartlett/Newey-West weights and denominator n give a nonnegative HAC
    # estimator. Allow dependence beyond mechanical h-step forecast overlap.
    automatic = int(np.floor(4 * (n / 100) ** (2 / 9)))
    lag = min(n - 1, max(h - 1, automatic) if bandwidth is None else bandwidth)
    if lag < 0:
        raise ValueError("bandwidth must be nonnegative")
    centered = d - dbar
    gamma = [float(centered[k:] @ centered[:n-k]) / n for k in range(lag + 1)]
    var = (gamma[0] + 2 * sum((1 - k / (lag + 1)) * gamma[k]
                             for k in range(1, lag + 1))) / n
    if var <= 0:
        return np.nan, np.nan
    dm = dbar / np.sqrt(var)
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat = dm * hln
    return float(stat), float(2 * stats.t.sf(abs(stat), df=n - 1))
