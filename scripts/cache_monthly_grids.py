"""Stream once and cache monthly-mean surface temperature grids for the study area.

Approved by the user on 2026-10-30 (~1–2 GB satellite + ~67 MB reanalysis transfer).
Daily satellite SST is reduced to monthly means band by band in memory; only the
monthly grids are written (float32, zlib-compressed):
  data/cache/sst_satellite_monthly_1993_2024.nc      ESA CCI / C3S, 0.05°
  data/cache/thetao_reanalysis_monthly_1993_2025.nc  reanalysis surface thetao, 1/12°

Afterwards, data_access.load_monthly_grid() serves both without any network access.

Usage:  .venv/bin/python scripts/cache_monthly_grids.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import copernicusmarine as cm  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402

from src import data_access, preprocessing, utils  # noqa: E402


def satellite_monthly(s: dict, log) -> xr.DataArray:
    v, a = s["validation"], s["study_area"]
    ds = cm.open_dataset(dataset_id=v["dataset_id"], dataset_version=v["dataset_version"],
                         variables=[v["variable"]], service=v["service"],
                         minimum_longitude=a["min_lon"], maximum_longitude=a["max_lon"],
                         minimum_latitude=a["min_lat"], maximum_latitude=a["max_lat"],
                         start_datetime=v["start"], end_datetime=v["end"])
    sst = ds[v["variable"]]
    log.info("Satellite selection: %s, lazy %.0f MB (float64)", dict(sst.sizes), sst.nbytes / 1e6)
    edges = np.concatenate([[0], np.cumsum(sst.chunks[1])])
    bands = []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        t0 = time.time()
        band = sst.isel(latitude=slice(lo, hi)).resample(time="MS").mean().astype("float32").load()
        bands.append(band - np.float32(273.15))
        log.info("Band %d/%d (lat rows %d–%d) in %.0fs", i + 1, len(edges) - 1, lo, hi, time.time() - t0)
    out = xr.concat(bands, "latitude").rename("sst")
    out.attrs.update(units="degC", long_name="monthly mean sea surface temperature (~20 cm)",
                     source_product_id=v["product_id"], source_dataset_id=v["dataset_id"],
                     source_dataset_version=v["dataset_version"],
                     source_title=v["product_title"], access_date=pd.Timestamp.today().date().isoformat(),
                     processing="daily analysed_sst -> calendar-month mean; K -> degC")
    return out


def write(da: xr.DataArray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    enc = {da.name: {"zlib": True, "complevel": 4, "dtype": "float32"}}
    da.to_dataset().to_netcdf(path, encoding=enc)


def main() -> None:
    s = utils.load_settings()
    log = utils.setup_logging(s, "marine.cache")
    grids = s["analysis"]["monthly_grids"]
    t0 = time.time()

    sat = satellite_monthly(s, log)
    write(sat, utils.PROJECT_ROOT / grids["satellite"])
    log.info("Satellite grid saved: %s", dict(sat.sizes))

    ds = data_access.open_remote(s)
    # Approved by the user on 2026-10-30.
    rean = preprocessing.surface(data_access.safe_load(ds["thetao"], s, approved=True)).astype("float32")
    rean = rean.drop_vars("depth", errors="ignore").rename("thetao")
    rean.attrs.update({k: v for k, v in ds.attrs.items() if k.startswith(("source_", "access", "toolbox", "request"))})
    rean.attrs.update(units="degC", depth_m=0.494025)
    write(rean, utils.PROJECT_ROOT / grids["reanalysis"])
    log.info("Reanalysis grid saved: %s", dict(rean.sizes))

    for key in ("satellite", "reanalysis"):
        p = utils.PROJECT_ROOT / grids[key]
        print(f"{key}: {p.relative_to(utils.PROJECT_ROOT)}  {p.stat().st_size / 1e6:.1f} MB on disk")
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
