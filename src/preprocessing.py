"""Selection, quality control, masking and area-weighted aggregation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr


def surface(da: xr.DataArray) -> xr.DataArray:
    """Select the shallowest model level (nearest surface)."""
    if "depth" in da.dims:
        return da.isel(depth=int(np.argmin(da["depth"].values)))
    return da


def ocean_mask(da: xr.DataArray) -> xr.DataArray:
    """True where the cell is ocean. Land cells are NaN at every time step in the
    reanalysis; for the full grid prefer the static dataset's `mask` variable."""
    return da.notnull().any("time") if "time" in da.dims else da.notnull()


def area_weights(da: xr.DataArray) -> xr.DataArray:
    """cos(latitude) weights for a regular lat/lon grid (proportional to cell area)."""
    return np.cos(np.deg2rad(da["latitude"])).rename("area_weight")


def area_weighted_mean(da: xr.DataArray) -> xr.DataArray:
    """Area-weighted spatial mean over ocean cells (NaN/land ignored)."""
    spatial = [d for d in ("latitude", "longitude") if d in da.dims]
    if not spatial:
        return da
    return da.weighted(area_weights(da).fillna(0)).mean(spatial, skipna=True)


def zone_masks(static: xr.Dataset, sub_regions: dict, isobath: float = 200.0) -> dict:
    """Boolean masks '<region>_shelf' (sea, depth <= isobath) and '<region>_offshore'
    for each sub-region, on the static dataset's grid."""
    sea = static["mask"] == 1
    shelf = sea & (static["deptho"] <= isobath)
    out = {}
    for key, r in sub_regions.items():
        inside = ((static.longitude >= r["min_lon"] - 1e-3) & (static.longitude <= r["max_lon"] + 1e-3)
                  & (static.latitude >= r["min_lat"] - 1e-3) & (static.latitude <= r["max_lat"] + 1e-3))
        out[f"{key}_shelf"] = shelf & inside
        out[f"{key}_offshore"] = sea & ~shelf & inside
    return out


def distance_to_land_km(sea: xr.DataArray) -> xr.DataArray:
    """Approximate distance (km) from each sea cell to the nearest land cell in the grid."""
    from scipy.spatial import cKDTree
    lat2, lon2 = xr.broadcast(sea.latitude, sea.longitude)
    lat2, lon2 = lat2.transpose(*sea.dims).values, lon2.transpose(*sea.dims).values
    y = lat2 * 110.57
    x = lon2 * 111.32 * np.cos(np.deg2rad(lat2))
    land = ~sea.values
    tree = cKDTree(np.column_stack([x[land], y[land]]))
    d, _ = tree.query(np.column_stack([x.ravel(), y.ravel()]))
    out = xr.DataArray(d.reshape(sea.shape), coords=sea.coords, dims=sea.dims, name="dist_km")
    return out.where(sea)


def trend_map(monthly: xr.DataArray, months: tuple[int, ...] | None = None) -> xr.DataArray:
    """Per-cell OLS trend (°C/decade) of anomalies. With `months`, trend of the yearly
    mean over those months (e.g. (12, 1, 2) for DJF, with December assigned to the next year)."""
    anom = monthly.groupby("time.month") - monthly.groupby("time.month").mean("time")
    if months:
        sel = anom.where(anom["time.month"].isin(months), drop=True)
        year = sel["time.year"] + (sel["time.month"] == 12) * (12 in months and 1 in months)
        anom = sel.groupby(year.rename("year")).mean("time")
        n = sel.groupby(year.rename("year")).count("time")
        anom = anom.where(n == len(months))
        t = anom["year"].astype(float)
        dim = "year"
    else:
        t = (anom["time"] - anom["time"][0]) / np.timedelta64(1, "D") / 365.25
        dim = "time"
    t = t.where(anom.notnull())
    tm, am = t.mean(dim), anom.mean(dim)
    slope = ((t - tm) * (anom - am)).sum(dim) / ((t - tm) ** 2).sum(dim)
    return (slope * 10).rename("trend_degC_per_decade")


def to_series(da: xr.DataArray) -> pd.Series:
    """1-D time series (spatially averaged if needed) as a pandas Series."""
    da = area_weighted_mean(surface(da))
    extra = [d for d in da.dims if d != "time"]
    if extra:
        da = da.squeeze(extra, drop=True)
    s = da.to_series()
    s.name = da.name
    return s


def missing_report(da: xr.DataArray) -> dict:
    """Missing-data summary separating permanent land from gaps in ocean cells."""
    total = int(da.size)
    n_nan = int(da.isnull().sum())
    out = {"total_values": total, "nan_values": n_nan,
           "nan_fraction": round(n_nan / total, 4) if total else None}
    if "time" in da.dims and da.sizes.get("time", 0) > 0:
        mask = ocean_mask(da)
        ocean_vals = da.where(mask)
        n_ocean = int(mask.sum()) * da.sizes["time"]
        gaps = int(ocean_vals.isnull().sum()) - (total - n_ocean)
        out.update({"ocean_cells": int(mask.sum()), "land_cells": int((~mask).sum()),
                    "ocean_gap_values": max(gaps, 0)})
        ts = pd.DatetimeIndex(da["time"].values)
        if len(ts) > 1:
            expected = pd.date_range(ts.min(), ts.max(), freq="MS")
            out["missing_time_steps"] = int(len(expected.difference(ts)))
    return out


def plausibility_check(s: pd.Series, lo: float = -2.5, hi: float = 35.0) -> dict:
    """Physical range check for sea water temperature (degC)."""
    bad = s[(s < lo) | (s > hi)]
    return {"min": float(s.min()), "max": float(s.max()), "out_of_range": int(len(bad))}


def monthly_climatology(s: pd.Series) -> pd.Series:
    return s.groupby(s.index.month).mean().rename_axis("month")


def anomalies(s: pd.Series) -> pd.Series:
    clim = monthly_climatology(s)
    return (s - clim.reindex(s.index.month).values).rename(f"{s.name}_anom")


def hac_trend(anom: pd.Series, maxlags: int = 12) -> dict:
    """Linear trend (°C/decade) of anomalies with HAC (Newey–West) errors,
    which account for autocorrelation in monthly anomalies."""
    import statsmodels.api as sm
    a = anom.dropna()
    x = ((a.index - a.index[0]).days / 365.25).to_numpy()
    fit = sm.OLS(a.to_numpy(), sm.add_constant(x)).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return {"trend_degC_per_decade": float(fit.params[1] * 10),
            "ci95": float(1.96 * fit.bse[1] * 10),
            "p_value": float(fit.pvalues[1]),
            "lag1_autocorr": float(a.autocorr(1))}


SEASONS = {"DJF": 12, "MAM": 3, "JJA": 6, "SON": 9}


def seasonal_trends(anom: pd.Series) -> pd.DataFrame:
    """Trend per season from yearly seasonal-mean anomalies (complete seasons only).
    DJF is labelled by the December that starts it."""
    q = anom.resample("QS-DEC")
    means, counts = q.mean(), q.count()
    means = means[counts == 3]
    rows = {name: hac_trend(means[means.index.month == m], maxlags=2) for name, m in SEASONS.items()}
    return pd.DataFrame(rows).T


def fdr_adjust(p_values, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini–Hochberg false-discovery-rate control across a family of tests.
    Returns (reject, adjusted p-values)."""
    from statsmodels.stats.multitest import multipletests
    p = np.asarray(p_values, dtype=float)
    reject, p_adj, _, _ = multipletests(p, alpha=alpha, method="fdr_bh")
    return reject, p_adj


def describe_series(s: pd.Series) -> pd.DataFrame:
    d = s.describe().to_frame("value")
    d.loc["n_missing"] = int(s.isna().sum())
    d.loc["start"] = str(s.index.min().date())
    d.loc["end"] = str(s.index.max().date())
    return d
