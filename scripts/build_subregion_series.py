"""Build area-weighted monthly surface thetao series for each sub-region in settings.yaml.

Streams each sub-region lazily (Benguela ~15 MB, Agulhas ~52 MB; together the same
~67 MB as the full study area). Nothing raw is written to disk. Saves:
  data/processed/thetao_<region>_monthly.csv
  data/processed/thetao_subregions_monthly.csv   all regions side by side
  outputs/tables/subregion_qc.json

Usage:  .venv/bin/python scripts/build_subregion_series.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src import data_access, preprocessing, utils  # noqa: E402

PROCESSED = utils.PROJECT_ROOT / "data" / "processed"
TABLES = utils.PROJECT_ROOT / "outputs" / "tables"


def main() -> None:
    s = utils.load_settings()
    log = utils.setup_logging(s, "marine.subregions")
    report, series = {}, {}
    for key, r in s["sub_regions"].items():
        ds = data_access.open_remote(s, min_lon=r["min_lon"], max_lon=r["max_lon"],
                                     min_lat=r["min_lat"], max_lat=r["max_lat"])
        est = data_access.estimate_selection(ds)
        t0 = time.time()
        # Approved by the user on 2026-10-30 (sub-region split; same total volume as the study area).
        th = preprocessing.surface(data_access.safe_load(ds["thetao"], s, approved=True))
        qc = preprocessing.missing_report(th)
        ts = preprocessing.to_series(th).rename(key)
        qc["plausibility"] = preprocessing.plausibility_check(ts)
        ts.to_frame("thetao_degC").to_csv(PROCESSED / f"thetao_{key}_monthly.csv")
        series[key] = ts
        report[key] = {"name": r["name"], "selection": est, "load_seconds": round(time.time() - t0, 1),
                       "qc": qc, "request_bounds": ds.attrs.get("request_bounds")}
        log.info("%s: %s", key, report[key])

    pd.DataFrame(series).to_csv(PROCESSED / "thetao_subregions_monthly.csv")
    report["provenance"] = {k: v for k, v in ds.attrs.items()
                            if k.startswith(("source_", "access", "toolbox"))}
    with open(TABLES / "subregion_qc.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
