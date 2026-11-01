"""Stream daily ESA CCI / C3S satellite SST once and cache compact daily products.

Approved by the user on 2026-10-30 (one daily stream, ~1–2 GB transfer). Daily data are
processed in blocks aligned to the store's native chunks (2112 days × 16 rows) so each
chunk is fetched once, and reduced in memory to:
  data/cache/sst_satellite_daily_zones_1993_2024.csv   area-weighted daily means, six zones
  data/cache/sst_satellite_daily_025deg_1993_2024.nc   daily SST block-averaged to 0.25°
No 0.05° daily data are written.

Usage:  .venv/bin/python scripts/cache_daily_satellite.py
"""

import sys
import time
import hashlib
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import copernicusmarine as cm  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402

from src import data_access, utils  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_satellite_validation import zone_masks_on  # noqa: E402

CACHE = utils.PROJECT_ROOT / "data" / "cache"
PRODUCT_START = pd.Timestamp("1982-01-01")   # first day of the C3S record
TIME_CHUNK, LAT_BAND, COARSEN = 2112, 80, 5   # native chunk length; 80 rows = 5×16; 0.05° → 0.25°


def main() -> None:
    s = utils.load_settings()
    log = utils.setup_logging(s, "marine.daily")
    v, a = s["validation"], s["study_area"]
    t_all = time.time()

    ds = cm.open_dataset(dataset_id=v["dataset_id"], dataset_version=v["dataset_version"],
                         variables=[v["variable"]], service=v["service"],
                         minimum_longitude=a["min_lon"], maximum_longitude=a["max_lon"],
                         minimum_latitude=a["min_lat"], maximum_latitude=a["max_lat"],
                         start_datetime=v["start"], end_datetime=v["end"])
    sst = ds[v["variable"]]
    times = pd.DatetimeIndex(sst.time.values)
    log.info("Daily selection %s (lazy %.0f MB float64)", dict(sst.sizes), sst.nbytes / 1e6)

    # Zone masks on the satellite grid (sea mask from the cached monthly grid).
    monthly = data_access.load_monthly_grid("satellite", s)
    if not (np.allclose(monthly.latitude, sst.latitude) and np.allclose(monthly.longitude, sst.longitude)):
        raise ValueError("daily and monthly satellite grids differ")
    static = data_access.open_static(s)
    masks = zone_masks_on(monthly.notnull().any("time"), static["deptho"], s)
    w = np.cos(np.deg2rad(sst.latitude.values))[:, None] * np.ones(sst.sizes["longitude"])
    zone_w = {k: (m.values * w).astype("float64") for k, m in masks.items()}
    num = {k: np.zeros(len(times)) for k in masks}
    den = {k: np.zeros(len(times)) for k in masks}

    n_lat, n_lon = sst.sizes["latitude"], sst.sizes["longitude"]
    coarse = np.full((len(times), n_lat // COARSEN, n_lon // COARSEN), np.nan, dtype="float32")

    # Time blocks aligned to the native 2112-day chunks of the store.
    day0 = (times[0] - PRODUCT_START).days
    edges = [0] + [k * TIME_CHUNK - day0 for k in range(day0 // TIME_CHUNK + 1, 100)
                   if 0 < k * TIME_CHUNK - day0 < len(times)] + [len(times)]
    blocks = [(t0, t1, r0, min(r0 + LAT_BAND, n_lat)) for t0, t1 in zip(edges[:-1], edges[1:])
              for r0 in range(0, n_lat, LAT_BAND)]
    # Checkpoint after every block so a network failure resumes instead of re-fetching.
    fingerprint = {"dataset_id": v["dataset_id"], "dataset_version": str(v["dataset_version"]),
                   "variable": v["variable"], "start": v["start"], "end": v["end"],
                   "study_area": a, "zones": s["zones"], "sub_regions": s["sub_regions"],
                   "coarsen": COARSEN,
                   "mask_hash": hashlib.sha256(b"".join(m.values.tobytes() for m in masks.values())).hexdigest()}
    key = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()[:16]
    ckpt_dir = CACHE / "_daily_checkpoints" / key
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "provenance.json").write_text(json.dumps({**fingerprint, "returned_dataset_attrs": dict(ds.attrs)},
                                                      indent=2, default=str))
    for i, (t0, t1, r0, r1) in enumerate(blocks):
        ckpt = ckpt_dir / f"block_{t0}_{r0}.npz"
        nb = (r1 - r0) // COARSEN
        if ckpt.exists():
            z = np.load(ckpt)
            for k in masks:
                num[k][t0:t1] += z[f"num_{k}"]
                den[k][t0:t1] += z[f"den_{k}"]
            coarse[t0:t1, r0 // COARSEN:r0 // COARSEN + nb] = z["coarse"]
            log.info("Block %d/%d restored from checkpoint", i + 1, len(blocks))
            continue
        tb = time.time()
        for attempt in range(1, 11):
            try:
                x = sst.isel(time=slice(t0, t1), latitude=slice(r0, r1)).values.astype("float32") - np.float32(273.15)
                break
            except Exception as e:          # transient S3 / TLS timeouts
                if attempt == 10:
                    raise
                log.warning("Block %d attempt %d failed (%s); retrying in %ds", i + 1, attempt,
                            type(e).__name__, min(30 * attempt, 180))
                time.sleep(min(30 * attempt, 180))
        valid = ~np.isnan(x)
        xz = np.where(valid, x, np.float32(0.0))
        part = {}
        for k in masks:
            wk = zone_w[k][r0:r1]
            part[f"num_{k}"] = np.einsum("tij,ij->t", xz, wk)
            part[f"den_{k}"] = np.einsum("tij,ij->t", valid.astype("float32"), wk)
            num[k][t0:t1] += part[f"num_{k}"]
            den[k][t0:t1] += part[f"den_{k}"]
        blk = x[:, :nb * COARSEN].reshape(t1 - t0, nb, COARSEN, n_lon // COARSEN, COARSEN)
        with np.errstate(invalid="ignore"), __import__("warnings").catch_warnings():
            __import__("warnings").simplefilter("ignore", RuntimeWarning)
            part["coarse"] = np.nanmean(blk, axis=(2, 4)).astype("float32")
        coarse[t0:t1, r0 // COARSEN:r0 // COARSEN + nb] = part["coarse"]
        np.savez(ckpt, **part)
        del x, valid, xz, blk
        log.info("Block %d/%d days %s..%s rows %d–%d in %.0fs", i + 1, len(blocks),
                 times[t0].date(), times[t1 - 1].date(), r0, r1, time.time() - tb)

    zones = pd.DataFrame({k: num[k] / np.where(den[k] > 0, den[k], np.nan) for k in masks}, index=times)
    zones.index.name = "time"
    zones.to_csv(CACHE / "sst_satellite_daily_zones_1993_2024.csv", float_format="%.4f")

    lat_c = sst.latitude.values[:(n_lat // COARSEN) * COARSEN].reshape(-1, COARSEN).mean(1)
    lon_c = sst.longitude.values[:(n_lon // COARSEN) * COARSEN].reshape(-1, COARSEN).mean(1)
    da = xr.DataArray(coarse, dims=("time", "latitude", "longitude"),
                      coords={"time": times, "latitude": lat_c, "longitude": lon_c}, name="sst",
                      attrs={"units": "degC", "processing": "daily analysed_sst, K->degC, 5x5 block mean (0.05->0.25 deg), "
                             "cells with any sea pixel kept", "source_product_id": v["product_id"],
                             "source_dataset_id": v["dataset_id"],
                             "source_dataset_version": str(v["dataset_version"]),
                             "extraction_fingerprint": key,
                             "access_date": pd.Timestamp.today().date().isoformat()})
    da.to_dataset().to_netcdf(CACHE / "sst_satellite_daily_025deg_1993_2024.nc",
                              encoding={"sst": {"zlib": True, "complevel": 4}})
    (CACHE / "sst_satellite_daily_provenance.json").write_text(
        json.dumps({**fingerprint, "returned_dataset_attrs": dict(ds.attrs), "key": key}, indent=2, default=str))
    print(zones.describe().round(2))
    print("missing days per zone:", zones.isna().sum().to_dict())
    for f in ("sst_satellite_daily_zones_1993_2024.csv", "sst_satellite_daily_025deg_1993_2024.nc"):
        print(f, f"{(CACHE / f).stat().st_size / 1e6:.1f} MB")
    print(f"total {time.time() - t_all:.0f}s")


if __name__ == "__main__":
    main()
