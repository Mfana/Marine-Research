"""Generate the starter notebooks (run once; edit the notebooks afterwards)."""

from pathlib import Path

import nbformat as nbf

NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"

SETUP = """import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent if pathlib.Path.cwd().name == 'notebooks' else pathlib.Path.cwd()))
from src import utils, data_access, preprocessing, visualisation
settings = utils.load_settings()
utils.set_seed(settings['project']['random_seed'])
%matplotlib inline"""

NOTEBOOKS = {
    "01_remote_data_access": [
        ("md", "# 01 · Remote data access\nLazy access to the Copernicus Marine Global Ocean Physics Reanalysis "
               "(GLOBAL_MULTIYEAR_PHY_001_030) via ARCO Zarr. **Nothing here loads the full dataset.** "
               "Requires `copernicusmarine login` once (credentials stored outside the project)."),
        ("code", SETUP),
        ("md", "## Catalogue check (no authentication, no data)"),
        ("code", "data_access.describe_dataset(settings)"),
        ("md", "## Tiny connectivity test — one point, three months, one variable, one depth"),
        ("code", "c = settings['connectivity_test']\n"
                 "pt = data_access.open_point(c['lon'], c['lat'], c['start'], c['end'], settings)\n"
                 "pt['thetao'].load()"),
        ("md", "## Lazy selection of the study area\nThis only reads metadata. Check the size before computing anything."),
        ("code", "ds = data_access.open_remote(settings)\nds"),
        ("code", "data_access.estimate_selection(ds), data_access.estimate_study_area(settings)"),
        ("md", "> Any `.load()`, `.compute()`, `.values`, plot or statistic on `ds` transfers data. "
               "Use `data_access.safe_load(obj)`, which refuses selections above the configured limit."),
    ],
    "02_exploratory_analysis": [
        ("md", "# 02 · Exploratory analysis\nMetadata, coverage, missingness and a small coastal time series. "
               "Lightweight by design."),
        ("code", SETUP),
        ("code", "ds = data_access.open_remote(settings)   # lazy\n"
                 "print(ds.attrs.get('source_citation'))\nds['thetao'].attrs"),
        ("md", "## Coverage (from coordinates only — no data transfer)"),
        ("code", "import pandas as pd\nt = pd.DatetimeIndex(ds.time.values)\n"
                 "print(t.min().date(), '→', t.max().date(), len(t), 'months')\n"
                 "print('lon', float(ds.longitude.min()), float(ds.longitude.max()), ds.sizes['longitude'])\n"
                 "print('lat', float(ds.latitude.min()), float(ds.latitude.max()), ds.sizes['latitude'])\n"
                 "print('depth', ds.depth.values)"),
        ("md", "## Small coastal box time series"),
        ("code", "b = settings['sample_region']\n"
                 "box = data_access.open_remote(settings, min_lon=b['min_lon'], max_lon=b['max_lon'],\n"
                 "                              min_lat=b['min_lat'], max_lat=b['max_lat'])['thetao']\n"
                 "data_access.estimate_selection(box.to_dataset())"),
        ("code", "box = data_access.safe_load(box, settings)\n"
                 "preprocessing.missing_report(preprocessing.surface(box))"),
        ("code", "s = preprocessing.to_series(box)\npreprocessing.describe_series(s)"),
        ("code", "fig, ax = visualisation.plot_timeseries(s, f\"Surface temperature, {b['name']}\")\n"
                 "visualisation.save(fig, '02_box_timeseries')"),
        ("code", "clim = preprocessing.monthly_climatology(s)\n"
                 "fig, ax = visualisation.plot_seasonality(clim, f\"Monthly climatology, {b['name']}\")\n"
                 "visualisation.save(fig, '02_box_seasonality')"),
        ("code", "# One snapshot map of the study area (single month, ~0.1 MB)\n"
                 "snap = data_access.open_remote(settings, start='2020-01-01', end='2020-01-31')['thetao']\n"
                 "snap = data_access.safe_load(preprocessing.surface(snap).isel(time=0), settings)\n"
                 "fig, ax = visualisation.plot_map(snap, 'Surface potential temperature, Jan 2020')\n"
                 "visualisation.save(fig, '02_map_2020_01')"),
    ],
    "03_time_series_analysis": [
        ("md", "# 03 · Time-series analysis\nAnomalies, trend and seasonal decomposition of the area-weighted series. "
               "Reanalysis values are model output constrained by observations — not independent observations."),
        ("code", SETUP),
        ("md", "Input: `data/processed/thetao_study_area_monthly.csv` from `scripts/build_study_area_series.py` "
               "(area-weighted over 16–34°E, 36–28°S). The box mixes the cold Benguela upwelling (west) and the "
               "warm Agulhas Current (east), so regional means blend two regimes; consider sub-regions."),
        ("code", "import pandas as pd\n"
                 "s = pd.read_csv('../data/processed/thetao_study_area_monthly.csv', index_col=0, parse_dates=True).iloc[:, 0]\n"
                 "s = s.asfreq('MS')\npreprocessing.describe_series(s)"),
        ("code", "clim = preprocessing.monthly_climatology(s)\n"
                 "fig, ax = visualisation.plot_seasonality(clim, 'Monthly climatology 1993–2025, study area')\n"
                 "visualisation.save(fig, '03_seasonality')"),
        ("code", "anom = preprocessing.anomalies(s)\n"
                 "fig, ax = visualisation.plot_timeseries(anom, 'Monthly anomaly vs 1993–2025 climatology', 'Anomaly (°C)')\n"
                 "ax.axhline(0, color='0.3', lw=0.8)\nvisualisation.save(fig, '03_anomalies')"),
        ("code", "from statsmodels.tsa.seasonal import STL\nres = STL(s, period=12, robust=True).fit()\n"
                 "fig = res.plot(); fig.set_size_inches(8, 6); visualisation.save(fig, '03_stl_decomposition')"),
        ("md", "## Linear trend of anomalies\nOLS with HAC (Newey–West) standard errors, because monthly anomalies "
               "are autocorrelated and naive p-values are too optimistic."),
        ("code", "import numpy as np\nimport statsmodels.api as sm\n"
                 "x = ((s.index - s.index[0]).days / 365.25).to_numpy()\n"
                 "ols = sm.OLS(anom.to_numpy(), sm.add_constant(x)).fit(cov_type='HAC', cov_kwds={'maxlags': 12})\n"
                 "slope, se = ols.params[1] * 10, ols.bse[1] * 10\n"
                 "print(f'Trend: {slope:.3f} ± {1.96*se:.3f} °C/decade (95% CI, HAC), p = {ols.pvalues[1]:.3g}')\n"
                 "print(f'Lag-1 autocorrelation of anomalies: {anom.autocorr(1):.2f}')"),
        ("code", "pd.DataFrame({'trend_degC_per_decade': [slope], 'ci95': [1.96*se], 'p_value': [ols.pvalues[1]],\n"
                 "              'lag1_autocorr': [anom.autocorr(1)]}).to_csv('../outputs/tables/03_trend.csv', index=False)"),
    ],
    "04_machine_learning": [
        ("md", "# 04 · Machine learning (scaffold)\nChronological splits, leakage-safe lag features, baselines vs ML. "
               "Remote access and preprocessing were validated on 2026-10-30; the series has ~400 rows, so "
               "training takes seconds. **Model selection uses the validation period only; touch the test "
               "period once, at the end.**"),
        ("code", SETUP + "\nfrom src import modelling"),
        ("code", "RUN_TRAINING = True\nEVALUATE_TEST = False   # set True only for the final, one-off evaluation"),
        ("code", "import pandas as pd\n"
                 "s = pd.read_csv('../data/processed/thetao_study_area_monthly.csv', index_col=0, parse_dates=True).iloc[:, 0]\n"
                 "df = modelling.make_features(s, settings['modelling']['lags'])\n"
                 "m = settings['modelling']\nsplit = modelling.chronological_split(df, m['train_end'], m['validation_end'], m['test_end'])\n"
                 "{k: (len(v), str(v.index.min().date()), str(v.index.max().date()))\n"
                 " for k, v in vars(split).items()}"),
        ("code", "if RUN_TRAINING:\n"
                 "    results = modelling.evaluate(split, settings['project']['random_seed'], on='validation')\n"
                 "    results.round(3).to_csv('../outputs/tables/04_validation_metrics.csv')\n"
                 "    display(results.round(3))"),
        ("md", "Skill > 0 means better than the baseline. Seasonal-cycle–dominated targets make climatology a strong "
               "baseline; beating it, not persistence, is the meaningful test."),
        ("code", "if RUN_TRAINING and EVALUATE_TEST:\n"
                 "    display(modelling.evaluate(split, settings['project']['random_seed'], on='test').round(3))"),
    ],
    "05_explainable_ai": [
        ("md", "# 05 · Explainable AI (scaffold)\nSHAP attributions for fitted models. "
               "**SHAP explains the model, not the ocean** — do not interpret attributions causally; "
               "correlated lags share credit arbitrarily."),
        ("code", SETUP + "\nfrom src import modelling, explainability"),
        ("code", "RUN_TRAINING = True"),
        ("code", "import pandas as pd\n"
                 "s = pd.read_csv('../data/processed/thetao_study_area_monthly.csv', index_col=0, parse_dates=True).iloc[:, 0]\n"
                 "m = settings['modelling']\n"
                 "split = modelling.chronological_split(modelling.make_features(s, m['lags']),\n"
                 "                                      m['train_end'], m['validation_end'], m['test_end'])\n"
                 "X = modelling.feature_cols(split.train)"),
        ("code", "if RUN_TRAINING:\n"
                 "    imp = {}\n"
                 "    for name in ('xgboost', 'random_forest'):\n"
                 "        model = modelling.build_models(settings['project']['random_seed'])[name]\n"
                 "        model.fit(split.train[X], split.train['y'])\n"
                 "        exp = explainability.shap_values(model, split.validation[X])\n"
                 "        imp[name] = explainability.mean_abs_importance(exp, X)\n"
                 "    imp = pd.DataFrame(imp).round(3)\n"
                 "    imp.to_csv('../outputs/tables/05_shap_importance.csv')\n"
                 "    display(imp)"),
        ("code", "if RUN_TRAINING:\n"
                 "    import matplotlib.pyplot as plt\n"
                 "    fig = explainability.summary_plot(exp, split.validation[X])\n"
                 "    visualisation.save(plt.gcf(), '05_shap_summary_random_forest')"),
    ],
    "06_regional_comparison": [
        ("md", "# 06 · Regional comparison: Benguela vs Agulhas\nThe study area is split at Cape Agulhas "
               "(20°E): **Benguela** (16–20°E, Atlantic, upwelling-dominated) and **Agulhas** (20–34°E, Indian "
               "Ocean, warm western boundary current). Input: `scripts/build_subregion_series.py`. Each series "
               "is a cos-latitude area-weighted mean of surface (0.494 m) ocean cells, so it includes offshore "
               "water as well as the coastal upwelling band."),
        ("code", SETUP + "\nfrom src import modelling\nimport pandas as pd, numpy as np, matplotlib.pyplot as plt\n"
                 "visualisation.apply_style()\n"
                 "df = pd.read_csv('../data/processed/thetao_subregions_monthly.csv', index_col=0, parse_dates=True).asfreq('MS')\n"
                 "names = {k: v['name'] for k, v in settings['sub_regions'].items()}\n"
                 "COLORS = {'benguela': '#2b6cb0', 'agulhas': '#c53030'}"),
        ("md", "## Where the regions are"),
        ("code", "import xarray as xr\n"
                 "clim = xr.open_dataset('../data/processed/thetao_monthly_climatology.nc')['thetao_climatology'].mean('month')\n"
                 "fig, ax = visualisation.plot_map(clim, 'Mean surface temperature 1993–2025 and sub-regions')\n"
                 "ax.axvline(20.0, color='k', ls='--', lw=1)\n"
                 "ax.text(18.0, -28.4, 'Benguela', ha='center', fontsize=9)\nax.text(27.0, -28.4, 'Agulhas', ha='center', fontsize=9)\n"
                 "visualisation.save(fig, '06_region_map')"),
        ("md", "## Descriptive statistics"),
        ("code", "stats = pd.concat({k: preprocessing.describe_series(df[k])['value'] for k in df}, axis=1)\n"
                 "stats.to_csv('../outputs/tables/06_region_stats.csv')\nstats"),
        ("code", "fig, ax = plt.subplots(figsize=(8, 3))\n"
                 "for k in df:\n    ax.plot(df.index, df[k], lw=1, color=COLORS[k], label=names[k])\n"
                 "ax.set(title='Area-weighted surface temperature by region', ylabel='Temperature (°C)')\n"
                 "ax.set_ylim(top=df.max().max() + 1.8)\nax.legend(loc='upper left', ncol=2, fontsize=8); visualisation.source_note(ax); fig.tight_layout()\n"
                 "visualisation.save(fig, '06_region_timeseries')"),
        ("md", "## Seasonal cycle"),
        ("code", "clims = pd.DataFrame({k: preprocessing.monthly_climatology(df[k]) for k in df})\n"
                 "fig, ax = plt.subplots(figsize=(5.5, 3))\n"
                 "for k in clims:\n    ax.plot(clims.index, clims[k], marker='o', color=COLORS[k], label=names[k])\n"
                 "ax.set(title='Monthly climatology 1993–2025', xlabel='Month', ylabel='Temperature (°C)', xticks=range(1, 13))\n"
                 "ax.legend(fontsize=8); visualisation.source_note(ax); fig.tight_layout()\n"
                 "visualisation.save(fig, '06_region_seasonality')\n"
                 "amp = (clims.max() - clims.min()).rename('seasonal_amplitude_degC')\n"
                 "peak = clims.idxmax().rename('warmest_month'); low = clims.idxmin().rename('coolest_month')\n"
                 "pd.concat([amp, peak, low], axis=1).round(2)"),
        ("md", "## Anomalies and trends\nHAC (Newey–West) standard errors account for autocorrelation."),
        ("code", "anoms = pd.DataFrame({k: preprocessing.anomalies(df[k]) for k in df})\n"
                 "trends = pd.DataFrame({k: preprocessing.hac_trend(anoms[k]) for k in anoms}).T\n"
                 "trends.to_csv('../outputs/tables/06_region_trends.csv')\ntrends.round(4)"),
        ("code", "fig, axes = plt.subplots(2, 1, figsize=(8, 4.5), sharex=True)\n"
                 "for ax, k in zip(axes, anoms):\n"
                 "    a = anoms[k]; ax.fill_between(a.index, 0, a, where=a > 0, color='#c53030', lw=0)\n"
                 "    ax.fill_between(a.index, 0, a, where=a < 0, color='#2b6cb0', lw=0)\n"
                 "    ax.plot(a.rolling(12, center=True).mean(), color='k', lw=1)\n"
                 "    t = trends.loc[k]\n"
                 "    ax.set_title(f\"{names[k]}: {t.trend_degC_per_decade:+.2f} ± {t.ci95:.2f} °C/decade\", fontsize=9)\n"
                 "    ax.set_ylabel('Anomaly (°C)')\n"
                 "visualisation.source_note(axes[-1]); fig.tight_layout()\nvisualisation.save(fig, '06_region_anomalies')"),
        ("code", "r = anoms.corr().iloc[0, 1]\n"
                 "r12 = anoms.rolling(12).mean().corr().iloc[0, 1]\n"
                 "print(f'Correlation of monthly anomalies: {r:.2f}; of 12-month running means: {r12:.2f}')"),
        ("md", "## One-month-ahead forecast skill by region (validation 2016–2019; test period untouched)"),
        ("code", "m = settings['modelling']; rows = {}\n"
                 "for k in df:\n"
                 "    sp = modelling.chronological_split(modelling.make_features(df[k], m['lags']),\n"
                 "                                       m['train_end'], m['validation_end'], m['test_end'])\n"
                 "    rows[k] = modelling.evaluate(sp, settings['project']['random_seed'], on='validation')\n"
                 "skill = pd.concat(rows, names=['region', 'model'])\n"
                 "skill.round(3).to_csv('../outputs/tables/06_region_validation_metrics.csv')\n"
                 "skill[['MAE', 'RMSE', 'bias', 'skill_vs_climatology']].round(3)"),
    ],
    "07_shelf_vs_offshore": [
        ("md", "# 07 · Shelf vs offshore within each region\nEach sub-region is split at the 200 m isobath "
               "(static bathymetry `deptho`). **Benguela shelf** holds the coastal upwelling; **Agulhas shelf** is "
               "mostly the Agulhas Bank; **Agulhas offshore** contains the Agulhas Current core just seaward of the "
               "shelf break. Input: `scripts/build_zone_series.py`. Surface (0.494 m) reanalysis temperature — "
               "model output constrained by observations, not independent observations."),
        ("code", SETUP + "\nfrom src import modelling\nimport pandas as pd, numpy as np, xarray as xr, matplotlib.pyplot as plt\n"
                 "visualisation.apply_style()\n"
                 "df = pd.read_csv('../data/processed/thetao_zones_monthly.csv', index_col=0, parse_dates=True).asfreq('MS')\n"
                 "zm = xr.open_dataset('../data/processed/zone_masks.nc')\n"
                 "LABEL = {'benguela_shelf': 'Benguela shelf', 'benguela_offshore': 'Benguela offshore',\n"
                 "         'agulhas_shelf': 'Agulhas shelf', 'agulhas_offshore': 'Agulhas offshore'}\n"
                 "COLORS = {'benguela_shelf': '#1e3a8a', 'benguela_offshore': '#60a5fa',\n"
                 "          'agulhas_shelf': '#f59e0b', 'agulhas_offshore': '#b91c1c'}"),
        ("md", "## Zone map"),
        ("code", "from matplotlib.colors import ListedColormap\n"
                 "zid = xr.zeros_like(zm['deptho']) * np.nan\n"
                 "for i, k in enumerate(LABEL):\n    zid = zid.where(zm[k] != 1, i)\n"
                 "fig, ax = plt.subplots(figsize=(6.5, 3.5))\n"
                 "zid.plot(ax=ax, cmap=ListedColormap([COLORS[k] for k in LABEL]), add_colorbar=False, vmin=-0.5, vmax=3.5)\n"
                 "zm['deptho'].plot.contour(ax=ax, levels=[200], colors='k', linewidths=0.6)\n"
                 "ax.axvline(20, color='k', ls='--', lw=0.8)\n"
                 "from matplotlib.patches import Patch\n"
                 "ax.legend(handles=[Patch(color=COLORS[k], label=LABEL[k]) for k in LABEL], fontsize=7, loc='center left', bbox_to_anchor=(1.01, 0.5))\n"
                 "ax.set(title='Zones: 200 m isobath (black) and 20°E split', xlabel='Longitude (°E)', ylabel='Latitude (°N)')\n"
                 "ax.grid(False); fig.tight_layout(); visualisation.save(fig, '07_zone_map')"),
        ("md", "## Climatology and seasonal cycle"),
        ("code", "stats = pd.concat({k: preprocessing.describe_series(df[k])['value'] for k in df}, axis=1)\n"
                 "stats.to_csv('../outputs/tables/07_zone_stats.csv')\nstats"),
        ("code", "clims = pd.DataFrame({k: preprocessing.monthly_climatology(df[k]) for k in df})\n"
                 "fig, ax = plt.subplots(figsize=(6, 3.2))\n"
                 "for k in clims:\n    ax.plot(clims.index, clims[k], marker='o', ms=3, color=COLORS[k], label=LABEL[k])\n"
                 "ax.set(title='Monthly climatology 1993–2025', xlabel='Month', ylabel='Temperature (°C)', xticks=range(1, 13))\n"
                 "ax.legend(fontsize=7, ncol=2); visualisation.source_note(ax); fig.tight_layout()\n"
                 "visualisation.save(fig, '07_zone_seasonality')"),
        ("md", "## Trends (annual, all months)\nHAC (Newey–West) 95 % confidence intervals."),
        ("code", "anoms = pd.DataFrame({k: preprocessing.anomalies(df[k]) for k in df})\n"
                 "trends = pd.DataFrame({k: preprocessing.hac_trend(anoms[k]) for k in anoms}).T\n"
                 "trends.to_csv('../outputs/tables/07_zone_trends.csv')\ntrends.round(4)"),
        ("code", "fig, axes = plt.subplots(4, 1, figsize=(8, 7.5), sharex=True)\n"
                 "for ax, k in zip(axes, anoms):\n"
                 "    a = anoms[k]; ax.fill_between(a.index, 0, a, where=a > 0, color='#c53030', lw=0)\n"
                 "    ax.fill_between(a.index, 0, a, where=a < 0, color='#2b6cb0', lw=0)\n"
                 "    ax.plot(a.rolling(12, center=True).mean(), color='k', lw=1)\n"
                 "    t = trends.loc[k]\n"
                 "    ax.set_title(f\"{LABEL[k]}: {t.trend_degC_per_decade:+.2f} ± {t.ci95:.2f} °C/decade\", fontsize=9)\n"
                 "    ax.set_ylabel('°C')\n"
                 "visualisation.source_note(axes[-1]); fig.tight_layout()\nvisualisation.save(fig, '07_zone_anomalies')"),
        ("md", "## Seasonal trends\nThe southern Benguela upwelling peaks in austral spring–summer (SON–DJF). "
               "Trends from yearly seasonal means; DJF is labelled by its December."),
        ("code", "st = pd.concat({k: preprocessing.seasonal_trends(anoms[k]) for k in anoms}, names=['zone', 'season'])\n"
                 "st.to_csv('../outputs/tables/07_zone_seasonal_trends.csv')\n"
                 "tbl = st['trend_degC_per_decade'].unstack().round(3)[list(preprocessing.SEASONS)]\n"
                 "sig = (st['p_value'] < 0.05).unstack()[list(preprocessing.SEASONS)]\n"
                 "tbl.astype(str) + sig.replace({True: ' *', False: ''})"),
        ("code", "fig, ax = plt.subplots(figsize=(6.5, 3.2))\n"
                 "x = np.arange(4); w = 0.2\n"
                 "for i, k in enumerate(anoms):\n"
                 "    d = st.loc[k].loc[list(preprocessing.SEASONS)]\n"
                 "    ax.bar(x + (i - 1.5) * w, d['trend_degC_per_decade'], w, yerr=d['ci95'], color=COLORS[k],\n"
                 "           label=LABEL[k], capsize=2, error_kw={'lw': 0.7})\n"
                 "ax.axhline(0, color='0.3', lw=0.8)\n"
                 "ax.set(xticks=x, xticklabels=list(preprocessing.SEASONS), ylabel='°C per decade',\n"
                 "       title='Seasonal SST trends 1993–2025 (95 % CI)')\n"
                 "ax.legend(fontsize=7, ncol=2); visualisation.source_note(ax); fig.tight_layout()\n"
                 "visualisation.save(fig, '07_zone_seasonal_trends')"),
        ("md", "## Cross-shelf temperature difference (upwelling proxy)\n"
               "ΔT = offshore − shelf. Stronger upwelling cools the shelf relative to offshore, raising ΔT. "
               "This is a thermal proxy only; wind stress and vertical velocity would be needed to attribute changes."),
        ("code", "dT = pd.DataFrame({r: df[f'{r}_offshore'] - df[f'{r}_shelf'] for r in ('benguela', 'agulhas')})\n"
                 "dT_trend = pd.DataFrame({r: preprocessing.hac_trend(preprocessing.anomalies(dT[r])) for r in dT}).T\n"
                 "dT_seas = pd.concat({r: preprocessing.seasonal_trends(preprocessing.anomalies(dT[r])) for r in dT})\n"
                 "dT_trend.to_csv('../outputs/tables/07_crossshelf_dT_trends.csv')\n"
                 "dT_seas.to_csv('../outputs/tables/07_crossshelf_dT_seasonal_trends.csv')\n"
                 "display(dT.groupby(dT.index.month).mean().round(2).T)\n"
                 "display(dT_trend.round(4)); dT_seas.round(4)"),
        ("code", "fig, ax = plt.subplots(figsize=(8, 3))\n"
                 "for r, c in (('benguela', '#1e3a8a'), ('agulhas', '#b91c1c')):\n"
                 "    ax.plot(dT[r], color=c, lw=0.6, alpha=0.4)\n"
                 "    ax.plot(dT[r].rolling(12, center=True).mean(), color=c, lw=1.5, label=f'{r.title()} (12-month mean)')\n"
                 "ax.set(title='Cross-shelf ΔT (offshore − shelf)', ylabel='ΔT (°C)')\n"
                 "ax.legend(fontsize=8); visualisation.source_note(ax); fig.tight_layout()\n"
                 "visualisation.save(fig, '07_crossshelf_dT')"),
        ("md", "## One-month-ahead skill by zone (validation 2016–2019; test untouched)"),
        ("code", "m = settings['modelling']; rows = {}\n"
                 "for k in df:\n"
                 "    sp = modelling.chronological_split(modelling.make_features(df[k], m['lags']),\n"
                 "                                       m['train_end'], m['validation_end'], m['test_end'])\n"
                 "    rows[k] = modelling.evaluate(sp, settings['project']['random_seed'], on='validation')\n"
                 "skill = pd.concat(rows, names=['zone', 'model'])\n"
                 "skill.round(3).to_csv('../outputs/tables/07_zone_validation_metrics.csv')\n"
                 "skill['RMSE'].unstack().round(3)"),
    ],
    "08_satellite_validation": [
        ("md", "# 08 · Validation against satellite SST\nReanalysis (GLOBAL_MULTIYEAR_PHY_001_030, 1/12°, 0.494 m) vs the "
               "**ESA CCI / C3S** satellite climate data record (SST_GLO_SST_L4_REP_OBSERVATIONS_010_024, 0.05°, "
               "daily-mean SST at ~20 cm), **1993–2024**. Both use identical zones: 20°E split, 200 m isobath, and a "
               "**50 km nearshore band**. Input: `scripts/build_satellite_validation.py`.\n\n"
               "The L4 analysis is gap-filled by optimal interpolation and assimilates some of the same observations "
               "as the reanalysis, so agreement is not fully independent, but its trends do not depend on the ocean model."),
        ("code", SETUP + "\nimport pandas as pd, numpy as np, xarray as xr, matplotlib.pyplot as plt\n"
                 "visualisation.apply_style()\n"
                 "rd = lambda p: pd.read_csv(p, index_col=0, parse_dates=True).asfreq('MS')\n"
                 "sat = rd('../data/processed/sst_satellite_zones_monthly.csv')\n"
                 "rea = rd('../data/processed/thetao_reanalysis_zones_1993_2024.csv')[sat.columns]\n"
                 "maps = xr.open_dataset('../data/processed/trend_maps.nc')\n"
                 "ZONES = list(sat.columns)\n"
                 "LABEL = {z: z.replace('_', ' ').title() for z in ZONES}\nsat.index.min(), sat.index.max(), ZONES"),
        ("md", "## Agreement of monthly values"),
        ("code", "rows = {}\n"
                 "for z in ZONES:\n"
                 "    a_s, a_r = preprocessing.anomalies(sat[z]), preprocessing.anomalies(rea[z])\n"
                 "    rows[z] = {'mean_sat': sat[z].mean(), 'mean_rean': rea[z].mean(),\n"
                 "               'bias_rean_minus_sat': (rea[z] - sat[z]).mean(),\n"
                 "               'rmsd': float(np.sqrt(((rea[z] - sat[z]) ** 2).mean())),\n"
                 "               'r_monthly': rea[z].corr(sat[z]), 'r_anomaly': a_r.corr(a_s)}\n"
                 "agree = pd.DataFrame(rows).T\nagree.round(3).to_csv('../outputs/tables/08_agreement.csv')\nagree.round(3)"),
        ("md", "## Trends: annual and by season (HAC 95 % CI)"),
        ("code", "tr = []\n"
                 "for z in ZONES:\n"
                 "    for src, df in (('satellite', sat), ('reanalysis', rea)):\n"
                 "        a = preprocessing.anomalies(df[z])\n"
                 "        tr.append({'zone': z, 'source': src, 'season': 'ANN', **preprocessing.hac_trend(a)})\n"
                 "        for season, r in preprocessing.seasonal_trends(a).iterrows():\n"
                 "            tr.append({'zone': z, 'source': src, 'season': season, **r.to_dict()})\n"
                 "tr = pd.DataFrame(tr).set_index(['zone', 'season', 'source'])\n"
                 "tr.round(4).to_csv('../outputs/tables/08_trends_satellite_vs_reanalysis.csv')\n"
                 "fmt = lambda r: f\"{r.trend_degC_per_decade:+.2f} ± {r.ci95:.2f}{' *' if r.p_value < 0.05 else ''}\"\n"
                 "tr.apply(fmt, axis=1).unstack('source').unstack('season')"
                 ".reindex(columns=['ANN', 'DJF', 'MAM', 'JJA', 'SON'], level=1)"),
        ("code", "fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)\n"
                 "for ax, season in zip(axes, ('ANN', 'DJF')):\n"
                 "    x = np.arange(len(ZONES)); w = 0.38\n"
                 "    for i, (src, c) in enumerate((('reanalysis', '#7c3aed'), ('satellite', '#0f766e'))):\n"
                 "        d = tr.xs((season, src), level=('season', 'source')).loc[ZONES]\n"
                 "        ax.bar(x + (i - 0.5) * w, d['trend_degC_per_decade'], w, yerr=d['ci95'], color=c, label=src,\n"
                 "               capsize=2, error_kw={'lw': 0.7})\n"
                 "    ax.axhline(0, color='0.3', lw=0.8)\n"
                 "    ax.set_xticks(x, [LABEL[z].replace(' ', '\\n', 1) for z in ZONES], fontsize=7)\n"
                 "    ax.set_title({'ANN': 'Annual', 'DJF': 'Summer (DJF)'}[season] + ' trend 1993–2024')\n"
                 "axes[0].set_ylabel('°C per decade'); axes[0].legend(fontsize=8)\n"
                 "visualisation.source_note(axes[1], 'Data: Copernicus Marine (doi:10.48670/moi-00021; SST_GLO_SST_L4_REP_OBSERVATIONS_010_024)')\n"
                 "fig.tight_layout(); visualisation.save(fig, '08_trend_comparison')"),
        ("md", "## Is the trend difference significant?\nTrend of the monthly difference series (reanalysis − satellite), "
               "HAC 95 % CI. A significant positive value means the reanalysis drifts warm relative to the satellite record."),
        ("code", "rows = {}\n"
                 "for z in ZONES:\n"
                 "    a = preprocessing.anomalies(rea[z] - sat[z])\n"
                 "    r, dj = preprocessing.hac_trend(a), preprocessing.seasonal_trends(a).loc['DJF']\n"
                 "    rows[z] = {'ann_diff': r['trend_degC_per_decade'], 'ann_ci': r['ci95'], 'ann_p': r['p_value'],\n"
                 "               'djf_diff': dj['trend_degC_per_decade'], 'djf_ci': dj['ci95'], 'djf_p': dj['p_value']}\n"
                 "diff = pd.DataFrame(rows).T\ndiff.round(3).to_csv('../outputs/tables/08_trend_difference_rean_minus_sat.csv')\ndiff.round(3)"),
        ("md", "## Trend maps\nPer-cell OLS trend of monthly anomalies (annual) and of yearly DJF means. Dashed: 50 km "
               "from the coast (satellite land mask). Point estimates only — individual cells are noisy."),
        ("code", "fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey=True)\n"
                 "kw = dict(cmap='RdBu_r', vmin=-0.6, vmax=0.6, add_colorbar=False)\n"
                 "for row, season in enumerate(('annual', 'djf')):\n"
                 "    for col, (src, var) in enumerate((('Reanalysis', f'rean_{season}'), ('Satellite (C3S)', f'sat_{season}'))):\n"
                 "        ax = axes[row, col]; da = maps[var]\n"
                 "        if src == 'Reanalysis': da = da.rename(lat_r='latitude', lon_r='longitude')\n"
                 "        im = da.plot(ax=ax, **kw)\n"
                 "        maps['sat_dist_km'].plot.contour(ax=ax, levels=[50], colors='k', linewidths=0.5, linestyles='--')\n"
                 "        ax.axvline(20, color='k', lw=0.6)\n"
                 "        ax.set(title=f\"{src}: {'annual' if season == 'annual' else 'summer (DJF)'}\", xlabel='', ylabel='')\n"
                 "        ax.grid(False)\n"
                 "fig.colorbar(im, ax=axes, label='°C per decade', shrink=0.8)\n"
                 "visualisation.save(fig, '08_trend_maps')"),
        ("md", "## Benguela nearshore summer (DJF) temperature"),
        ("code", "def djf(s):\n"
                 "    q = s.resample('QS-DEC'); m = q.mean()[q.count() == 3]\n"
                 "    m = m[m.index.month == 12]; m.index = m.index.year + 1; return m\n"
                 "fig, ax = plt.subplots(figsize=(7, 3))\n"
                 "for src, df, c in (('Reanalysis', rea, '#7c3aed'), ('Satellite', sat, '#0f766e')):\n"
                 "    y = djf(df['benguela_nearshore']); ax.plot(y.index, y, marker='o', ms=3, color=c, label=src)\n"
                 "    b = np.polyfit(y.index, y.values, 1); ax.plot(y.index, np.polyval(b, y.index), color=c, ls='--', lw=1)\n"
                 "ax.set(title='Benguela nearshore (≤50 km) summer mean SST', ylabel='°C', xlabel='Year (Jan–Feb)')\n"
                 "ax.legend(fontsize=8); visualisation.source_note(ax, 'Data: Copernicus Marine'); fig.tight_layout()\n"
                 "visualisation.save(fig, '08_benguela_nearshore_djf')"),
        ("md", "## Nearshore − offshore ΔT (upwelling proxy), both sources\n"
               "ΔT = nearshore − offshore, usually negative. A **negative trend** means the nearshore cooled relative "
               "to offshore (consistent with stronger upwelling); a positive trend means a weaker thermal contrast."),
        ("code", "rows = []\n"
                 "for r in ('benguela', 'agulhas'):\n"
                 "    for src, df in (('satellite', sat), ('reanalysis', rea)):\n"
                 "        a = preprocessing.anomalies(df[f'{r}_nearshore'] - df[f'{r}_offshore'])\n"
                 "        rows.append({'region': r, 'source': src, 'season': 'ANN', **preprocessing.hac_trend(a)})\n"
                 "        for season, x in preprocessing.seasonal_trends(a).iterrows():\n"
                 "            rows.append({'region': r, 'source': src, 'season': season, **x.to_dict()})\n"
                 "dT = pd.DataFrame(rows).set_index(['region', 'season', 'source'])\n"
                 "dT.round(4).to_csv('../outputs/tables/08_nearshore_offshore_dT_trends.csv')\n"
                 "dT.apply(fmt, axis=1).unstack('source')"),
    ],
    "09_coastal_satellite_analysis": [
        ("md", "# 09 · Coastal SST analysis from the satellite record\nSource: **ESA CCI / C3S** satellite SST "
               "(`settings.yaml → analysis.coastal_source`), zone series from `scripts/build_satellite_validation.py`, "
               "1993–2024. Coastal trends and upwelling are analysed here from observations, following the "
               "validation in notebook 08. Zones: 20°E split; shelf ≤ 200 m; "
               "nearshore ≤ 50 km from the coast."),
        ("code", SETUP + "\nfrom src import modelling, explainability\n"
                 "import pandas as pd, numpy as np, matplotlib.pyplot as plt\nvisualisation.apply_style()\n"
                 "df = data_access.load_zone_series(settings=settings)\nprint('source:', df.attrs['source'], df.index.min().date(), '→', df.index.max().date())\n"
                 "ZONES = ['benguela_nearshore', 'benguela_shelf', 'benguela_offshore', 'agulhas_nearshore', 'agulhas_shelf', 'agulhas_offshore']\n"
                 "df = df[ZONES]\nLABEL = {z: z.replace('_', ' ').title() for z in ZONES}\n"
                 "COLORS = {'benguela_nearshore': '#0b2559', 'benguela_shelf': '#1e3a8a', 'benguela_offshore': '#60a5fa',\n"
                 "          'agulhas_nearshore': '#7c2d12', 'agulhas_shelf': '#f59e0b', 'agulhas_offshore': '#b91c1c'}\n"
                 "CREDIT = 'Data: ESA CCI/C3S SST via Copernicus Marine (SST_GLO_SST_L4_REP_OBSERVATIONS_010_024)'"),
        ("md", "## Mean state and seasonal cycle"),
        ("code", "stats = pd.concat({z: preprocessing.describe_series(df[z])['value'] for z in ZONES}, axis=1)\n"
                 "stats.to_csv('../outputs/tables/09_sat_zone_stats.csv')\nstats"),
        ("code", "clims = pd.DataFrame({z: preprocessing.monthly_climatology(df[z]) for z in ZONES})\n"
                 "fig, ax = plt.subplots(figsize=(6.5, 3.4))\n"
                 "for z in ZONES:\n    ax.plot(clims.index, clims[z], marker='o', ms=3, color=COLORS[z], label=LABEL[z])\n"
                 "ax.set(title='Satellite SST monthly climatology 1993–2024', xlabel='Month', ylabel='SST (°C)', xticks=range(1, 13))\n"
                 "ax.legend(fontsize=7, ncol=2); visualisation.source_note(ax, CREDIT); fig.tight_layout()\n"
                 "visualisation.save(fig, '09_sat_seasonality')"),
        ("md", "## Upwelling index: nearshore − offshore ΔT\nMore negative = colder nearshore relative to offshore "
               "(stronger upwelling signature). Thermal proxy only; wind data would be needed for attribution."),
        ("code", "dT = pd.DataFrame({r: df[f'{r}_nearshore'] - df[f'{r}_offshore'] for r in ('benguela', 'agulhas')})\n"
                 "dT.to_csv('../data/processed/sst_satellite_upwelling_index.csv')\n"
                 "dT.groupby(dT.index.month).mean().round(2).T"),
        ("code", "fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), gridspec_kw={'width_ratios': [1, 2.2]})\n"
                 "for r, c in (('benguela', '#1e3a8a'), ('agulhas', '#b91c1c')):\n"
                 "    m = dT[r].groupby(dT.index.month).mean(); axes[0].plot(m.index, m, marker='o', ms=3, color=c, label=r.title())\n"
                 "    axes[1].plot(dT[r], color=c, lw=0.5, alpha=0.35)\n"
                 "    axes[1].plot(dT[r].rolling(12, center=True).mean(), color=c, lw=1.6, label=f'{r.title()} (12-month mean)')\n"
                 "axes[0].set(title='Seasonal cycle of ΔT', xlabel='Month', ylabel='ΔT (°C)', xticks=range(1, 13))\n"
                 "axes[1].set(title='ΔT (nearshore − offshore), satellite', ylabel='ΔT (°C)')\n"
                 "axes[0].axhline(0, color='0.3', lw=0.8); axes[1].axhline(0, color='0.3', lw=0.8)\n"
                 "axes[0].legend(fontsize=8); axes[1].legend(fontsize=8)\n"
                 "visualisation.source_note(axes[1], CREDIT); fig.tight_layout(); visualisation.save(fig, '09_upwelling_index')"),
        ("md", "## Trends with false-discovery-rate control\nAnnual and seasonal trends for the six zones and both "
               "ΔT indices (40 tests). HAC 95 % CIs; significance after **Benjamini–Hochberg FDR** at 5 % across all 40."),
        ("code", "series = {**{z: df[z] for z in ZONES}, 'benguela_dT': dT['benguela'], 'agulhas_dT': dT['agulhas']}\n"
                 "rows = []\n"
                 "for name, s in series.items():\n"
                 "    a = preprocessing.anomalies(s)\n"
                 "    rows.append({'series': name, 'season': 'ANN', **preprocessing.hac_trend(a)})\n"
                 "    for season, r in preprocessing.seasonal_trends(a).iterrows():\n"
                 "        rows.append({'series': name, 'season': season, **r.to_dict()})\n"
                 "tr = pd.DataFrame(rows)\n"
                 "tr['significant_fdr'], tr['p_fdr'] = preprocessing.fdr_adjust(tr['p_value'])\n"
                 "tr = tr.set_index(['series', 'season'])\n"
                 "tr.round(4).to_csv('../outputs/tables/09_sat_trends_fdr.csv')\n"
                 "fmt = lambda r: f\"{r.trend_degC_per_decade:+.2f} ± {r.ci95:.2f}{' **' if r.significant_fdr else (' *' if r.p_value < 0.05 else '')}\"\n"
                 "print('** significant after FDR; * nominal p < 0.05 only')\n"
                 "tr.apply(fmt, axis=1).unstack('season')[['ANN', 'DJF', 'MAM', 'JJA', 'SON']].loc[list(series)]"),
        ("code", "names = list(series); seasons = ['ANN', 'DJF', 'MAM', 'JJA', 'SON']\n"
                 "M = tr['trend_degC_per_decade'].unstack('season').loc[names, seasons]\n"
                 "S = tr['significant_fdr'].unstack('season').loc[names, seasons]\n"
                 "N = (tr['p_value'] < 0.05).unstack('season').loc[names, seasons]\n"
                 "fig, ax = plt.subplots(figsize=(6, 4.6))\n"
                 "im = ax.imshow(M.values, cmap='RdBu_r', vmin=-0.3, vmax=0.3, aspect='auto')\n"
                 "for i in range(len(names)):\n"
                 "    for j in range(len(seasons)):\n"
                 "        mark = '**' if S.iloc[i, j] else ('*' if N.iloc[i, j] else '')\n"
                 "        ax.text(j, i, f'{M.iloc[i, j]:+.2f}{mark}', ha='center', va='center', fontsize=7)\n"
                 "ax.set_xticks(range(len(seasons)), seasons)\n"
                 "ax.set_yticks(range(len(names)), [n.replace('_dT', ' ΔT').replace('_', ' ').title().replace('Δt', 'ΔT') for n in names], fontsize=8)\n"
                 "ax.axhline(5.5, color='k', lw=0.8); ax.grid(False)\n"
                 "fig.colorbar(im, ax=ax, label='°C per decade', shrink=0.8)\n"
                 "ax.set_title('Satellite SST trends 1993–2024\\n** FDR-significant, * nominal p < 0.05', fontsize=10)\n"
                 "visualisation.source_note(ax, 'Data: ESA CCI/C3S SST via Copernicus Marine'); fig.tight_layout()\n"
                 "visualisation.save(fig, '09_sat_trend_matrix')"),
        ("md", "## One-month-ahead forecasting on satellite zone series\nChronological split from `settings.yaml` "
               "(train ≤ 2015, validation 2016–2019, test 2020–2024 **untouched**). Lags 1, 2, 3, 12 + month encoding."),
        ("code", "m = settings['modelling']; rows = {}; splits = {}\n"
                 "for z in ZONES:\n"
                 "    sp = modelling.chronological_split(modelling.make_features(df[z], m['lags']),\n"
                 "                                       m['train_end'], m['validation_end'], m['test_end'])\n"
                 "    splits[z] = sp\n"
                 "    rows[z] = modelling.evaluate(sp, settings['project']['random_seed'], on='validation')\n"
                 "skill = pd.concat(rows, names=['zone', 'model'])\n"
                 "skill.round(3).to_csv('../outputs/tables/09_sat_validation_metrics.csv')\n"
                 "print('Test period:', splits[ZONES[0]].test.index.min().date(), '→', splits[ZONES[0]].test.index.max().date(), '(not evaluated)')\n"
                 "skill['skill_vs_climatology'].unstack().round(3)[['seasonal_persistence', 'linear', 'random_forest', 'xgboost']]"),
        ("md", "## SHAP: Benguela nearshore\nExplains how the best tree model uses its inputs, not what drives upwelling."),
        ("code", "z = 'benguela_nearshore'; sp = splits[z]; X = modelling.feature_cols(sp.train)\n"
                 "best = skill.loc[z].loc[['random_forest', 'xgboost'], 'RMSE'].idxmin()\n"
                 "model = modelling.build_models(settings['project']['random_seed'])[best]\n"
                 "model.fit(sp.train[X], sp.train['y'])\n"
                 "exp = explainability.shap_values(model, sp.validation[X])\n"
                 "imp = explainability.mean_abs_importance(exp, X)\n"
                 "imp.round(3).to_csv('../outputs/tables/09_shap_benguela_nearshore.csv')\nprint('model:', best); imp.round(3)"),
        ("code", "fig = explainability.summary_plot(exp, sp.validation[X])\n"
                 "visualisation.save(plt.gcf(), '09_shap_benguela_nearshore')"),
    ],
}


def build():
    NB_DIR.mkdir(exist_ok=True)
    for name, cells in NOTEBOOKS.items():
        nb = nbf.v4.new_notebook()
        nb.metadata["kernelspec"] = {"name": "marine-ai-research", "display_name": "Python (marine-ai-research)",
                                     "language": "python"}
        nb.cells = [nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src)
                    for kind, src in cells]
        path = NB_DIR / f"{name}.ipynb"
        if path.exists():
            print("skip (exists)", path.name)
            continue
        nbf.write(nb, path)
        print("wrote", path.name)


if __name__ == "__main__":
    build()
