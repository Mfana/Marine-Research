"""Marine heatwave (MHW) detection following Hobday et al. (2016), with the
categories of Hobday et al. (2018). Works on a daily pd.Series (one location or a
zone mean) and, vectorised, on (time, ...) numpy arrays for gridded data.

Climatology and threshold: for each day of year (366-day calendar; non-leap years
skip Feb 29), the mean and percentile of all baseline days within ±`window_half_width`
days, then smoothed with a circular `smooth_width`-day moving average.
Events: threshold exceeded for ≥ `min_duration` days; events separated by
≤ `max_gap` days are joined.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def doy366(index: pd.DatetimeIndex) -> np.ndarray:
    """Day of year on a 366-day calendar: Feb 29 = 60, and in non-leap years days
    after February are shifted by one so a calendar date always maps to one value."""
    d = index.dayofyear.to_numpy().copy()
    shift = (~index.is_leap_year) & (index.month > 2)
    d[shift] += 1
    return d


def _circular_smooth(a: np.ndarray, width: int) -> np.ndarray:
    """Moving average along axis 0 of a 366-row array, wrapping around the year."""
    k = width // 2
    padded = np.concatenate([a[-k:], a, a[:k]], axis=0)
    c = np.cumsum(np.insert(padded, 0, 0, axis=0), axis=0)
    return (c[width:] - c[:-width]) / width


def climatology(values: np.ndarray, index: pd.DatetimeIndex, baseline: tuple[str, str],
                pctile: float = 90, window_half_width: int = 5, smooth_width: int = 31):
    """Seasonal mean and threshold, each shaped (366, ...). `values` is (time, ...)."""
    values = np.asarray(values, dtype="float64")
    in_base = (index >= pd.Timestamp(baseline[0])) & (index <= pd.Timestamp(baseline[1]))
    d = doy366(index)[in_base]
    v = values[in_base]
    shape = (366,) + values.shape[1:]
    seas, thresh = np.full(shape, np.nan), np.full(shape, np.nan)
    for day in range(1, 367):
        if day == 60:
            continue
        # Actual consecutive days around each calendar date, clipped to the baseline.
        # This matches marineHeatWaves with a baseline-only alternateClimatology and
        # cannot draw evaluation-period observations across a training boundary.
        centres = np.flatnonzero(d == day)
        positions = (centres[:, None] + np.arange(-window_half_width, window_half_width + 1)).ravel()
        positions = positions[(positions >= 0) & (positions < len(v))]
        sel = v[positions]
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                seas[day - 1] = np.nanmean(sel, axis=0)
                thresh[day - 1] = np.nanpercentile(sel, pctile, axis=0)
    seas[59] = (seas[58] + seas[60]) / 2
    thresh[59] = (thresh[58] + thresh[60]) / 2
    return _circular_smooth(seas, smooth_width), _circular_smooth(thresh, smooth_width)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """(start, end_inclusive) index pairs of consecutive True runs."""
    m = np.concatenate([[False], mask, [False]])
    diff = np.diff(m.astype(int))
    starts, ends = np.where(diff == 1)[0], np.where(diff == -1)[0] - 1
    return list(zip(starts, ends))


def event_mask(exceed: np.ndarray, min_duration: int = 5, max_gap: int = 2) -> np.ndarray:
    """Boolean MHW-day mask from a boolean exceedance series (1-D)."""
    runs = [(s, e) for s, e in _runs(exceed) if e - s + 1 >= min_duration]
    merged = []
    for s, e in runs:
        if merged and s - merged[-1][1] - 1 <= max_gap:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    out = np.zeros_like(exceed, dtype=bool)
    for s, e in merged:
        out[s:e + 1] = True
    return out


def detect(sst: pd.Series, baseline: tuple[str, str] = ("1993-01-01", "2022-12-31"),
           pctile: float = 90, min_duration: int = 5, max_gap: int = 2,
           clim: tuple[np.ndarray, np.ndarray] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Detect MHWs in a daily series. Returns (daily, events):
    daily  — sst, seas, thresh, anomaly, in_mhw, event_id, category
    events — one row per event with Hobday et al. (2016, 2018) metrics.
    Pass `clim=(seas366, thresh366)` to reuse a climatology fitted elsewhere (e.g. training years)."""
    sst = sst.asfreq("D")
    idx = sst.index
    seas366, thresh366 = clim if clim is not None else climatology(sst.to_numpy(), idx, baseline, pctile)
    d = doy366(idx) - 1
    seas, thresh = seas366[d], thresh366[d]
    x = sst.to_numpy()
    exceed = np.nan_to_num(x > thresh, nan=False).astype(bool)
    in_mhw = event_mask(exceed, min_duration, max_gap)
    daily = pd.DataFrame({"sst": x, "seas": seas, "thresh": thresh, "anomaly": x - seas,
                          "in_mhw": in_mhw}, index=idx)
    daily["event_id"] = 0
    rows = []
    for i, (s, e) in enumerate(_runs(in_mhw), start=1):
        daily.iloc[s:e + 1, daily.columns.get_loc("event_id")] = i
        seg = daily.iloc[s:e + 1]
        anom = seg["anomaly"].to_numpy()
        diff = (seg["thresh"] - seg["seas"]).to_numpy()
        peak = int(np.nanargmax(anom))
        ratio = np.nanmax(np.divide(anom, diff, out=np.full_like(anom, np.nan), where=diff > 0))
        category = int(min(max(np.floor(ratio), 1), 4)) if np.isfinite(ratio) else 1
        onset_rate = ((anom[peak] - 0.5 * (daily["anomaly"].iloc[s - 1] + anom[0])) / (peak + 0.5)
                      if s > 0 else (anom[peak] - anom[0]) / max(peak, 1))
        rows.append({"event_id": i, "start": seg.index[0], "end": seg.index[-1], "peak": seg.index[peak],
                     "duration_days": e - s + 1, "intensity_max": float(anom[peak]),
                     "intensity_mean": float(np.nanmean(anom)), "intensity_cumulative": float(np.nansum(anom)),
                     "category": category, "onset_rate": float(onset_rate)})
    events = pd.DataFrame(rows)
    cat = pd.Series(0, index=idx)
    for r in rows:
        cat.loc[r["start"]:r["end"]] = r["category"]
    daily["event_category"] = cat.to_numpy()
    ratio = np.divide(x - seas, thresh - seas, out=np.zeros_like(x), where=thresh > seas)
    daily["category"] = np.where(in_mhw, np.clip(np.floor(ratio), 0, 4), 0).astype(int)
    return daily, events


def annual_metrics(daily: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Per calendar year: event count (by start year), MHW days, mean and max intensity."""
    years = range(daily.index.year.min(), daily.index.year.max() + 1)
    out = pd.DataFrame(index=pd.Index(years, name="year"))
    out["mhw_days"] = daily["in_mhw"].groupby(daily.index.year).sum()
    if len(events):
        ev = events.assign(year=pd.DatetimeIndex(events["start"]).year)
        g = ev.groupby("year")
        out["events"] = g.size()
        out["mean_duration"] = g["duration_days"].mean()
        out["mean_intensity_max"] = g["intensity_max"].mean()
        out["cumulative_intensity"] = g["intensity_cumulative"].sum()
    out = out.fillna({"events": 0, "mhw_days": 0, "cumulative_intensity": 0})
    return out


def detect_grid(values: np.ndarray, index: pd.DatetimeIndex, baseline: tuple[str, str],
                pctile: float = 90, min_duration: int = 5, max_gap: int = 2) -> np.ndarray:
    """Boolean MHW-day array (time, ...) for gridded data; NaN cells stay False."""
    seas, thresh = climatology(values, index, baseline, pctile)
    th = thresh[doy366(index) - 1]
    exceed = np.nan_to_num(values > th, nan=False).astype(bool)
    flat = exceed.reshape(len(index), -1)
    out = np.zeros_like(flat)
    for j in range(flat.shape[1]):
        if flat[:, j].any():
            out[:, j] = event_mask(flat[:, j], min_duration, max_gap)
    return out.reshape(exceed.shape)
