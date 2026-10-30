"""Stage 1 + 2 remote validation.

1. Tiny point test: one ocean point, three months, thetao, nearest-surface level.
2. Lazy open of the full study area (metadata only, nothing loaded) + size estimate.
3. Tiny worked example: monthly series for a small coastal box, with plot and QC.

Usage:  .venv/bin/python scripts/connectivity_test.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src import data_access, preprocessing, visualisation, utils  # noqa: E402

OUT_TABLES = utils.PROJECT_ROOT / "outputs" / "tables"
OUT_SAMPLES = utils.PROJECT_ROOT / "data" / "samples"


def main(sample_start: str = "2020-01-01", sample_end: str = "2022-12-31") -> dict:
    s = utils.load_settings()
    utils.set_seed(s["project"]["random_seed"])
    log = utils.setup_logging(s, "marine.connectivity")
    report = {"catalogue": data_access.describe_dataset(s)}

    # 1. Tiny point test
    c = s["connectivity_test"]
    t0 = time.time()
    ds_pt = data_access.open_point(c["lon"], c["lat"], c["start"], c["end"], s)
    pt = ds_pt["thetao"].load()
    report["point_test"] = {
        "location": c["name"],
        "grid_lon": float(pt.longitude.values.ravel()[0]),
        "grid_lat": float(pt.latitude.values.ravel()[0]),
        "depth_m": float(pt.depth.values.ravel()[0]),
        "units": pt.attrs.get("units"),
        "values": {str(pd.Timestamp(t).date()): round(float(v), 3)
                   for t, v in zip(pt.time.values, pt.values.ravel())},
        "seconds": round(time.time() - t0, 1),
    }
    log.info("Point test OK: %s", report["point_test"]["values"])

    # 2. Lazy study-area selection — no data loaded
    t0 = time.time()
    ds_area = data_access.open_remote(s)
    th = ds_area["thetao"]
    report["study_area_lazy"] = {
        **data_access.estimate_selection(ds_area),
        "lon_range": [float(th.longitude.min()), float(th.longitude.max())],
        "lat_range": [float(th.latitude.min()), float(th.latitude.max())],
        "time_range": [str(pd.Timestamp(th.time.values[0]).date()),
                       str(pd.Timestamp(th.time.values[-1]).date())],
        "depth_levels_m": [round(float(d), 3) for d in th.depth.values],
        "is_lazy_dask": th.chunks is not None,
        "chunks": {d: list(set(ch)) for d, ch in zip(th.dims, th.chunks)} if th.chunks else None,
        "seconds_to_open": round(time.time() - t0, 1),
    }
    report["study_area_offline_estimate"] = data_access.estimate_study_area(s)

    # 3. Tiny worked example: small coastal box, 3 years
    b = s["sample_region"]
    t0 = time.time()
    ds_box = data_access.open_remote(s, min_lon=b["min_lon"], max_lon=b["max_lon"],
                                     min_lat=b["min_lat"], max_lat=b["max_lat"],
                                     start=sample_start, end=sample_end)
    box = data_access.safe_load(ds_box["thetao"], s)
    qc = preprocessing.missing_report(preprocessing.surface(box))
    series = preprocessing.to_series(box)
    qc["plausibility"] = preprocessing.plausibility_check(series)
    report["sample_example"] = {"region": b["name"], "period": [sample_start, sample_end],
                                "dims": dict(box.sizes), "qc": qc,
                                "seconds": round(time.time() - t0, 1)}

    stats = preprocessing.describe_series(series)
    stats.to_csv(OUT_TABLES / "sample_thetao_stats.csv")
    series.to_frame("thetao_degC").to_csv(OUT_SAMPLES / "sample_thetao_monthly.csv")
    fig, _ = visualisation.plot_timeseries(
        series, f"Surface potential temperature, {b['name']} (area-weighted, monthly)")
    report["figures"] = [str(p) for p in visualisation.save(fig, "sample_thetao_timeseries")]
    report["provenance"] = {k: v for k, v in ds_box.attrs.items() if k.startswith(("source_", "access", "toolbox", "request"))}

    with open(OUT_TABLES / "connectivity_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(json.dumps(report, indent=2, default=str))
    print(stats)
    return report


if __name__ == "__main__":
    main()
