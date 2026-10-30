"""Zone series and trend maps for the reanalysis vs ESA CCI / C3S satellite comparison.

Reads the cached monthly grids (scripts/cache_monthly_grids.py) — no network access.
Both sources get identical zones (20°E split, shelf isobath, nearshore band) over the
validation period. Saves:
  data/processed/sst_satellite_zones_monthly.csv
  data/processed/thetao_reanalysis_zones_1993_2024.csv
  data/processed/trend_maps.nc                       annual + DJF trend maps, both sources
  outputs/tables/satellite_validation_qc.json

`build()` takes nearshore_km / isobath overrides for sensitivity tests.

Usage:  .venv/bin/python scripts/build_satellite_validation.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402

from src import data_access, preprocessing, utils  # noqa: E402

PROCESSED = utils.PROJECT_ROOT / "data" / "processed"
TABLES = utils.PROJECT_ROOT / "outputs" / "tables"


def zone_masks_on(grid_sea: xr.DataArray, deptho: xr.DataArray, s: dict,
                  nearshore_km: float | None = None, isobath: float | None = None) -> dict:
    """Shelf/offshore/nearshore masks per sub-region on any lat/lon grid. Sea cells the
    model treats as land (deptho NaN) are nearshore water, so they count as shelf."""
    iso = s["zones"]["shelf_isobath_m"] if isobath is None else isobath
    near_km = s["zones"]["nearshore_km"] if nearshore_km is None else nearshore_km
    d = deptho.interp(latitude=grid_sea.latitude, longitude=grid_sea.longitude, method="nearest")
    shelf = grid_sea & ((d <= iso) | d.isnull())
    dist = preprocessing.distance_to_land_km(grid_sea)
    masks = {}
    for key, r in s["sub_regions"].items():
        inside = ((grid_sea.longitude >= r["min_lon"] - 1e-3) & (grid_sea.longitude <= r["max_lon"] + 1e-3)
                  & (grid_sea.latitude >= r["min_lat"] - 1e-3) & (grid_sea.latitude <= r["max_lat"] + 1e-3))
        masks[f"{key}_shelf"] = shelf & inside
        masks[f"{key}_offshore"] = grid_sea & ~shelf & inside
        masks[f"{key}_nearshore"] = (dist <= near_km) & inside
    return masks


def zone_series(field: xr.DataArray, masks: dict) -> pd.DataFrame:
    return pd.DataFrame({k: preprocessing.area_weighted_mean(field.where(m)).to_series()
                         for k, m in masks.items()})


def load_inputs(s: dict) -> tuple[xr.DataArray, xr.DataArray, xr.Dataset]:
    v = s["validation"]
    sat = data_access.load_monthly_grid("satellite", s).sel(time=slice(v["start"], v["end"]))
    rean = data_access.load_monthly_grid("reanalysis", s).sel(time=slice(v["start"], v["end"]))
    static = data_access.open_static(s)
    return sat, rean, static


def build(s: dict, nearshore_km: float | None = None, isobath: float | None = None,
          inputs: tuple | None = None) -> dict:
    """Zone series for both sources. Pass `inputs` from load_inputs() to reuse loaded grids."""
    sat, rean, static = inputs or load_inputs(s)
    sat_sea, rean_sea = sat.notnull().any("time"), rean.notnull().any("time")
    sat_masks = zone_masks_on(sat_sea, static["deptho"], s, nearshore_km, isobath)
    rean_masks = zone_masks_on(rean_sea, static["deptho"], s, nearshore_km, isobath)
    return {"sat": zone_series(sat, sat_masks), "rean": zone_series(rean, rean_masks),
            "sat_masks": sat_masks, "rean_masks": rean_masks, "sat_sea": sat_sea, "rean_sea": rean_sea}


def main() -> None:
    s = utils.load_settings()
    log = utils.setup_logging(s, "marine.validation")
    v = s["validation"]
    t0 = time.time()
    sat, rean, static = inputs = load_inputs(s)
    out = build(s, inputs=inputs)

    out["sat"].to_csv(PROCESSED / "sst_satellite_zones_monthly.csv")
    out["rean"].to_csv(PROCESSED / "thetao_reanalysis_zones_1993_2024.csv")

    maps = xr.Dataset({
        "sat_annual": preprocessing.trend_map(sat),
        "sat_djf": preprocessing.trend_map(sat, (12, 1, 2)),
        "rean_annual": preprocessing.trend_map(rean).rename(latitude="lat_r", longitude="lon_r"),
        "rean_djf": preprocessing.trend_map(rean, (12, 1, 2)).rename(latitude="lat_r", longitude="lon_r"),
        "sat_dist_km": preprocessing.distance_to_land_km(out["sat_sea"]),
    })
    maps.attrs.update(units="degC per decade", period=f"{v['start']}..{v['end']}")
    maps.to_netcdf(PROCESSED / "trend_maps.nc")

    qc = {"seconds_total": round(time.time() - t0), "period": [v["start"], v["end"]],
          "inputs": {k: s["analysis"]["monthly_grids"][k] for k in ("satellite", "reanalysis")}}
    for key, field in (("satellite", sat), ("reanalysis", rean)):
        short = key[:3] if key == "satellite" else "rean"
        z = out[short]
        qc[key] = {"dims": dict(field.sizes), "sea_cells": int(out[f"{short}_sea"].sum()),
                   "zone_cells": {k: int(m.sum()) for k, m in out[f"{short}_masks"].items()},
                   "missing_months": {k: int(z[k].isna().sum()) for k in z}}
    qc["satellite_provenance"] = dict(sat.attrs)
    with open(TABLES / "satellite_validation_qc.json", "w") as f:
        json.dump(qc, f, indent=2, default=str)
    log.info("Validation outputs rebuilt from cached grids in %ss", qc["seconds_total"])
    print(json.dumps({k: qc[k] for k in ("seconds_total", "period", "inputs")}, indent=2))


if __name__ == "__main__":
    main()
