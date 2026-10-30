# marine-ai-research

A reproducible local research environment for AI/ML analysis of the **Copernicus Marine Global Ocean
Physics Reanalysis** (`GLOBAL_MULTIYEAR_PHY_001_030`, DOI [10.48670/moi-00021](https://doi.org/10.48670/moi-00021)),
using **remote, lazy access**. The dataset is never downloaded in full.

| | |
|---|---|
| Dataset (monthly) | `cmems_mod_glo_phy_my_0.083deg_P1M-m`, version `202311` (daily `…_P1D-m` ready in config) |
| Access | Copernicus Marine Toolbox → ARCO Zarr (`arco-geo-series`), opened lazily with xarray + Dask |
| Study area | 16°E–34°E, 36°S–28°S, Jan 1993 – Dec 2025 |
| Variable | `thetao` (sea water potential temperature, °C), nearest-surface level (0.494 m) |
| Data type | Model reanalysis (NEMO + assimilation): **not** independent observations |

## Setup

```bash
cd marine-ai-research
/opt/homebrew/bin/python3.13 -m venv .venv          # already done
.venv/bin/pip install -r requirements.txt           # top-level pins; full lock in requirements-lock.txt
.venv/bin/copernicusmarine login                    # once; credentials stay outside the project
```

XGBoost on Apple Silicon needs `brew install libomp` (installed).
In VS Code, open this folder; `.vscode/settings.json` selects `.venv` and the Jupyter kernel
**Python (marine-ai-research)**.

## Layout

```
config/settings.yaml   study area, dataset IDs, depth, safeguards, model splits (edit here, not in code)
src/data_access.py     lazy remote open, provenance, size estimates, safe_load() transfer guard
src/preprocessing.py   surface selection, land mask, cos-lat area-weighted means, QC, anomalies
src/visualisation.py   publication figures (PNG 300 dpi + PDF) with data attribution
src/modelling.py       lag features, chronological splits, persistence/seasonal/climatology baselines,
                       linear, Random Forest, XGBoost; MAE, RMSE, bias, skill scores
src/explainability.py  SHAP for tree/linear models (non-causal)
notebooks/01–09        access → EDA → time series → ML → XAI → regions → shelf/offshore →
                       satellite validation → coastal analysis (satellite)
scripts/               connectivity_test.py, build_study_area_series.py, build_subregion_series.py,
                       build_zone_series.py, cache_monthly_grids.py, build_satellite_validation.py,
                       make_notebooks.py
tests/                 offline unit tests; `-m remote` for the live test
data/, outputs/        local caches and results (git-ignored)
```

## Usage

```bash
.venv/bin/python -m pytest                          # offline tests
.venv/bin/python -m pytest -m remote                # live 3-month point test (needs login)
.venv/bin/python scripts/connectivity_test.py       # writes outputs/tables/connectivity_report.json
```

```python
from src import data_access, preprocessing
ds = data_access.open_remote()                      # lazy: metadata only
data_access.estimate_selection(ds)                  # size before computing
small = ds["thetao"].sel(longitude=slice(18, 19), latitude=slice(-35, -34))
series = preprocessing.to_series(data_access.safe_load(small))
```

## Cached monthly grids (no re-streaming)

`scripts/cache_monthly_grids.py` streamed the data once and saved two derived monthly-mean grids to
`data/cache/` (git-ignored; no daily data):

| File | Source | Size |
|---|---|---|
| `sst_satellite_monthly_1993_2024.nc` | ESA CCI / C3S satellite SST, 0.05° | 21 MB |
| `thetao_reanalysis_monthly_1993_2025.nc` | Reanalysis surface thetao, 1/12° | 10 MB |

`data_access.load_monthly_grid("satellite" | "reanalysis")` reads them with no network access, and
`scripts/build_satellite_validation.py` rebuilds all zone series and trend maps from them in seconds. For
sensitivity tests, call `build(settings, nearshore_km=..., isobath=...)` from that script. Re-run
`cache_monthly_grids.py` only to extend the period (it streams ~1–2 GB).

## Which SST source to use

The SST source is set per kind of analysis in `settings.yaml` (`analysis.coastal_source`,
`analysis.offshore_source`) and read through `data_access.load_zone_series()`. Coastal trend and upwelling
work uses the ESA CCI / C3S satellite record; the reanalysis is used for offshore, subsurface and
dynamical variables. Notebook 08 compares the two sources; notebooks 03–07 use the reanalysis and
notebook 09 the satellite record.

## Safeguards

* Remote ≠ free: computing on a lazy remote array transfers the Zarr chunks it touches, even
  though code runs locally in VS Code. Always subset variable/time/space/depth first.
* `safe_load()` refuses anything above `safeguards.max_transfer_mb_without_approval` (50 MB).
* No `get()`/`subset()` bulk downloads. Credentials are never stored in the project or committed.
* Fixed random seed, pinned dependencies, chronological splits, and provenance attributes
  (product, version, DOI, licence, access date, bounds) attached to every opened dataset.

## Citation & licence

Data: E.U. Copernicus Marine Service Information (CMEMS), Global Ocean Physics Reanalysis,
DOI 10.48670/moi-00021. Free under the Copernicus Marine Service licence; attribution required.
