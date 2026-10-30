"""Build the full 1993–2025 area-weighted surface thetao series for the study area.

Streams ~67 MB (float64, one surface level, monthly) into memory; nothing raw is
cached to disk. Saves only derived products:
  data/processed/thetao_study_area_monthly.csv    area-weighted monthly series
  data/processed/thetao_monthly_climatology.nc    per-cell 12-month climatology (~2 MB)
  outputs/tables/study_area_qc.json               QC + provenance

Usage:  .venv/bin/python scripts/build_study_area_series.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import data_access, preprocessing, utils, visualisation  # noqa: E402

PROCESSED = utils.PROJECT_ROOT / "data" / "processed"
TABLES = utils.PROJECT_ROOT / "outputs" / "tables"


def main() -> None:
    s = utils.load_settings()
    log = utils.setup_logging(s, "marine.build")
    ds = data_access.open_remote(s)
    est = data_access.estimate_selection(ds)
    log.info("Selection: %s", est)

    t0 = time.time()
    # Approved by the user on 2026-10-30 (~67 MB, above the 50 MB guard).
    th = preprocessing.surface(data_access.safe_load(ds["thetao"], s, approved=True))
    load_s = round(time.time() - t0, 1)
    log.info("Loaded in %ss", load_s)

    qc = preprocessing.missing_report(th)
    series = preprocessing.to_series(th).rename("thetao_degC")
    qc["plausibility"] = preprocessing.plausibility_check(series)
    qc["cell_plausibility"] = {"min": float(th.min()), "max": float(th.max())}

    PROCESSED.mkdir(parents=True, exist_ok=True)
    series.to_frame().to_csv(PROCESSED / "thetao_study_area_monthly.csv")
    clim = th.groupby("time.month").mean("time").astype("float32")
    clim.attrs.update(ds.attrs)
    clim.to_dataset(name="thetao_climatology").to_netcdf(PROCESSED / "thetao_monthly_climatology.nc")
    preprocessing.describe_series(series).to_csv(TABLES / "study_area_thetao_stats.csv")

    fig, _ = visualisation.plot_timeseries(
        series, "Surface potential temperature, 16–34°E, 36–28°S (area-weighted, monthly)")
    visualisation.save(fig, "study_area_thetao_timeseries")
    fig, _ = visualisation.plot_map(th.mean("time"), "Mean surface potential temperature, 1993–2025")
    visualisation.save(fig, "study_area_thetao_mean_map")

    report = {"selection": est, "load_seconds": load_s, "qc": qc,
              "provenance": {k: v for k, v in ds.attrs.items()
                             if k.startswith(("source_", "access", "toolbox", "request"))}}
    with open(TABLES / "study_area_qc.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(json.dumps(report, indent=2, default=str))
    print(preprocessing.describe_series(series))


if __name__ == "__main__":
    main()
