"""Shelf vs offshore surface thetao series within each sub-region.

Streams the study area once (~67 MB, nothing raw written to disk) and the static
bathymetry (<1 MB), then builds area-weighted monthly series for every zone:
  <region>_shelf      sea cells shallower than the shelf isobath (default 200 m)
  <region>_offshore   deeper sea cells
Saves:
  data/processed/thetao_zones_monthly.csv
  data/processed/zone_masks.nc                    masks + bathymetry for mapping
  outputs/tables/zone_qc.json

Usage:  .venv/bin/python scripts/build_zone_series.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402

from src import data_access, preprocessing, utils  # noqa: E402

PROCESSED = utils.PROJECT_ROOT / "data" / "processed"
TABLES = utils.PROJECT_ROOT / "outputs" / "tables"


def main() -> None:
    s = utils.load_settings()
    log = utils.setup_logging(s, "marine.zones")
    isobath = s["zones"]["shelf_isobath_m"]

    static = data_access.open_static(s)
    masks = preprocessing.zone_masks(static, s["sub_regions"], isobath)

    ds = data_access.open_remote(s)
    t0 = time.time()
    # Approved by the user on 2026-10-30 (shelf/offshore split; same volume as the study area).
    th = preprocessing.surface(data_access.safe_load(ds["thetao"], s, approved=True))
    load_s = round(time.time() - t0, 1)
    if not (np.allclose(th.latitude, static.latitude) and np.allclose(th.longitude, static.longitude)):
        raise ValueError("thetao and static grids differ")
    th = th.assign_coords(latitude=static.latitude, longitude=static.longitude)

    land_agree = int(((static["mask"] == 1) != th.notnull().any("time")).sum())
    report = {"load_seconds": load_s, "shelf_isobath_m": isobath,
              "static_mask_vs_thetao_disagreements": land_agree, "zones": {}}
    series = {}
    for name, m in masks.items():
        z = th.where(m)
        ts = preprocessing.area_weighted_mean(z).to_series().rename(name)
        series[name] = ts
        depth = static["deptho"].where(m)
        report["zones"][name] = {
            "cells": int(m.sum()),
            "area_km2": round(float(static["cell_area"].where(m).sum()) / 1e6),
            "depth_m_median": round(float(depth.median()), 1),
            "depth_m_range": [round(float(depth.min()), 1), round(float(depth.max()), 1)],
            "missing_months": int(ts.isna().sum()),
            "plausibility": preprocessing.plausibility_check(ts),
        }
        log.info("%s: %s", name, report["zones"][name])

    pd.DataFrame(series).to_csv(PROCESSED / "thetao_zones_monthly.csv")
    xr.Dataset({k: v.astype("int8") for k, v in masks.items()} | {"deptho": static["deptho"]},
               attrs=static.attrs).to_netcdf(PROCESSED / "zone_masks.nc")
    report["provenance"] = {k: v for k, v in ds.attrs.items()
                            if k.startswith(("source_", "access", "toolbox", "request"))}
    with open(TABLES / "zone_qc.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(json.dumps(report["zones"], indent=2), "\nmask disagreements:", land_agree, "load s:", load_s)


if __name__ == "__main__":
    main()
