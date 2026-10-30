"""Lazy remote access to Copernicus Marine datasets.

Everything here returns *lazy* xarray objects backed by remote ARCO Zarr chunks.
Data is transferred only when values are computed (.load(), .compute(), .values,
plotting, statistics). Always narrow variable, time, space and depth first.
"""

from __future__ import annotations

from datetime import date

import copernicusmarine as cm
import xarray as xr

from .utils import (estimate_uncompressed_mb, grid_cells, load_settings,
                    months_between, setup_logging)

log = setup_logging(name="marine.data_access")


class TransferTooLarge(RuntimeError):
    """Raised when a selection exceeds the configured no-approval transfer limit."""


def dataset_id(settings: dict, resolution: str | None = None) -> str:
    ds_cfg = settings["dataset"]
    return ds_cfg["datasets"][resolution or ds_cfg["active_resolution"]]


def describe_dataset(settings: dict | None = None) -> dict:
    """Catalogue metadata (no authentication or data transfer required)."""
    settings = settings or load_settings()
    did = dataset_id(settings)
    r = cm.describe(dataset_id=did)
    product = r.products[0]
    ds = product.datasets[0]
    version = ds.versions[0]
    services = [str(s.service_name.value if hasattr(s.service_name, "value") else s.service_name)
                for s in version.parts[0].services]
    return {"product_id": product.product_id, "title": product.title,
            "doi": product.digital_object_identifier, "dataset_id": ds.dataset_id,
            "version": version.label, "services": services}


def open_remote(settings: dict | None = None, *, variables=None, min_lon=None, max_lon=None,
                min_lat=None, max_lat=None, start=None, end=None, min_depth=None,
                max_depth=None, resolution: str | None = None) -> xr.Dataset:
    """Open a lazily-loaded, server-side-subset view of the dataset.

    Defaults come from settings.yaml (study area, primary variable, surface level).
    Credentials are read from the Copernicus Marine credentials file created by
    `copernicusmarine login`; they are never passed or stored by this project.
    """
    settings = settings or load_settings()
    area, depth, ds_cfg = settings["study_area"], settings["depth"], settings["dataset"]
    kwargs = dict(
        dataset_id=dataset_id(settings, resolution),
        dataset_version=ds_cfg.get("dataset_version"),
        variables=variables or settings["variables"]["primary"],
        minimum_longitude=area["min_lon"] if min_lon is None else min_lon,
        maximum_longitude=area["max_lon"] if max_lon is None else max_lon,
        minimum_latitude=area["min_lat"] if min_lat is None else min_lat,
        maximum_latitude=area["max_lat"] if max_lat is None else max_lat,
        start_datetime=start or area["start"],
        end_datetime=end or area["end"],
        minimum_depth=depth["min_depth"] if min_depth is None else min_depth,
        maximum_depth=depth["max_depth"] if max_depth is None else max_depth,
        coordinates_selection_method=depth.get("method", "nearest"),
        service=ds_cfg.get("service"),
    )
    log.info("Opening lazily: %s", {k: v for k, v in kwargs.items() if k != "dataset_id"})
    ds = cm.open_dataset(**kwargs)
    # "nearest" can snap an end date (e.g. 31 Mar) onto the next monthly stamp (1 Apr);
    # trim lazily so the period is exactly what was requested.
    ds = ds.sel(time=slice(kwargs["start_datetime"], kwargs["end_datetime"]))
    ds.attrs.update(provenance(settings, kwargs))
    return ds


def open_point(lon: float, lat: float, start: str, end: str, settings: dict | None = None,
               variables=None) -> xr.Dataset:
    """Open a single grid point (nearest cell) for a short period."""
    return open_remote(settings, variables=variables, min_lon=lon, max_lon=lon,
                       min_lat=lat, max_lat=lat, start=start, end=end)


def open_static(settings: dict | None = None, *, min_lon=None, max_lon=None, min_lat=None,
                max_lat=None) -> xr.Dataset:
    """Bathymetry, surface land/sea mask and cell sizes for the study area (2-D, small).

    Returns deptho (m), mask (1 = sea, surface level), e1t/e2t (m) and cell_area (m²).
    """
    settings = settings or load_settings()
    area = settings["study_area"]
    bounds = dict(minimum_longitude=area["min_lon"] if min_lon is None else min_lon,
                  maximum_longitude=area["max_lon"] if max_lon is None else max_lon,
                  minimum_latitude=area["min_lat"] if min_lat is None else min_lat,
                  maximum_latitude=area["max_lat"] if max_lat is None else max_lat,
                  coordinates_selection_method="nearest")
    did = dataset_id(settings, "static")
    log.info("Opening static fields: %s", bounds)
    bathy = cm.open_dataset(dataset_id=did, dataset_part="bathy", variables=["deptho", "mask"],
                            minimum_depth=0.494025, maximum_depth=0.494025, **bounds)
    coords = cm.open_dataset(dataset_id=did, dataset_part="coords", variables=["e1t", "e2t"], **bounds)
    mask = bathy["mask"]
    if "depth" in mask.dims:
        mask = mask.isel(depth=0, drop=True)
    ds = xr.Dataset({"deptho": bathy["deptho"], "mask": mask,
                     "e1t": coords["e1t"], "e2t": coords["e2t"]}).load()
    ds["cell_area"] = (ds["e1t"] * ds["e2t"]).assign_attrs(units="m2", long_name="grid cell area")
    ds.attrs.update(provenance(settings))
    ds.attrs["source_dataset_id"] = did
    return ds


def load_zone_series(source: str | None = None, settings: dict | None = None):
    """Monthly zone series (°C) from a processed CSV. `source` is 'satellite' or
    'reanalysis'; default is the configured coastal source."""
    import pandas as pd
    from .utils import PROJECT_ROOT
    settings = settings or load_settings()
    source = source or settings["analysis"]["coastal_source"]
    path = PROJECT_ROOT / settings["analysis"]["zone_series"][source]
    df = pd.read_csv(path, index_col=0, parse_dates=True).asfreq("MS")
    df.attrs["source"] = source
    return df


def load_monthly_grid(source: str, settings: dict | None = None) -> xr.DataArray:
    """Cached monthly-mean surface temperature grid (°C) for 'satellite' or 'reanalysis'.
    Created once by scripts/cache_monthly_grids.py; reading it transfers nothing."""
    from .utils import PROJECT_ROOT
    settings = settings or load_settings()
    path = PROJECT_ROOT / settings["analysis"]["monthly_grids"][source]
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run scripts/cache_monthly_grids.py (streams ~1–2 GB).")
    ds = xr.open_dataset(path)
    da = ds[list(ds.data_vars)[0]].load()
    da.attrs.update(ds.attrs)
    return da


def provenance(settings: dict, request: dict | None = None) -> dict:
    ds_cfg = settings["dataset"]
    prov = {
        "source_provider": ds_cfg["provider"],
        "source_product_id": ds_cfg["product_id"],
        "source_doi": ds_cfg["doi"],
        "source_dataset_version": str(ds_cfg.get("dataset_version")),
        "source_licence": ds_cfg["licence"],
        "source_citation": ds_cfg["citation"].strip(),
        "source_data_type": ds_cfg["data_type"],
        "access_date": date.today().isoformat(),
        "toolbox_version": cm.__version__,
    }
    if request:
        prov["source_dataset_id"] = request["dataset_id"]
        prov["request_bounds"] = (f"lon[{request['minimum_longitude']},{request['maximum_longitude']}] "
                                  f"lat[{request['minimum_latitude']},{request['maximum_latitude']}] "
                                  f"time[{request['start_datetime']},{request['end_datetime']}] "
                                  f"depth[{request['minimum_depth']},{request['maximum_depth']}]")
    return prov


def estimate_selection(ds: xr.Dataset) -> dict:
    """Estimate in-memory size of a lazy selection without transferring data."""
    mb = sum(v.size * v.dtype.itemsize for v in ds.data_vars.values()) / 1e6
    return {"dims": dict(ds.sizes), "uncompressed_mb": round(mb, 2)}


def estimate_study_area(settings: dict | None = None, n_vars: int = 1) -> dict:
    """Offline size estimate of the full configured study area (no network)."""
    settings = settings or load_settings()
    a = settings["study_area"]
    n_lon, n_lat = grid_cells(a["min_lon"], a["max_lon"]), grid_cells(a["min_lat"], a["max_lat"])
    n_time = months_between(a["start"], a["end"])
    daily_factor = 30.4
    mb = estimate_uncompressed_mb(n_time, n_lat, n_lon, n_vars=n_vars, bytes_per_value=8)
    return {"n_lon": n_lon, "n_lat": n_lat, "n_months": n_time,
            "monthly_mb_per_level": round(mb, 1),
            "daily_mb_per_level": round(mb * daily_factor, 1)}


def safe_load(obj, settings: dict | None = None, approved: bool = False):
    """Load a lazy selection only if it is under the configured limit (or approved)."""
    settings = settings or load_settings()
    limit = settings["safeguards"]["max_transfer_mb_without_approval"]
    mb = (obj.nbytes if hasattr(obj, "nbytes") else 0) / 1e6
    if mb > limit and not approved:
        raise TransferTooLarge(f"Selection is ~{mb:.1f} MB (> {limit} MB). "
                               "Narrow it, or pass approved=True after review.")
    log.info("Loading ~%.2f MB", mb)
    return obj.load()
