"""Offline tests (no network, synthetic data)."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src import modelling, preprocessing, utils
from src.data_access import estimate_study_area


@pytest.fixture
def grid():
    time = pd.date_range("1993-01-01", periods=36, freq="MS")
    lat = np.array([-34.0, -33.0])
    lon = np.array([18.0, 19.0])
    data = 15 + 3 * np.sin(2 * np.pi * time.month.to_numpy() / 12)[:, None, None] * np.ones((1, 2, 2))
    data = data.astype("float32")
    data[:, 0, 1] = np.nan  # a land cell
    return xr.DataArray(data, dims=("time", "latitude", "longitude"),
                        coords={"time": time, "latitude": lat, "longitude": lon}, name="thetao")


def test_settings_load():
    s = utils.load_settings()
    assert s["dataset"]["datasets"]["monthly"] == "cmems_mod_glo_phy_my_0.083deg_P1M-m"
    a = s["study_area"]
    assert a["min_lon"] < a["max_lon"] and a["min_lat"] < a["max_lat"]


def test_area_weighted_mean_ignores_land(grid):
    m = preprocessing.area_weighted_mean(grid)
    assert m.dims == ("time",)
    assert not np.isnan(m).any()
    np.testing.assert_allclose(m.values, grid.isel(latitude=1, longitude=0).values, rtol=1e-5)


def test_cos_lat_weights_favour_equator(grid):
    w = preprocessing.area_weights(grid)
    assert w.sel(latitude=-33.0) > w.sel(latitude=-34.0)


def test_missing_report_separates_land(grid):
    r = preprocessing.missing_report(grid)
    assert r["land_cells"] == 1 and r["ocean_cells"] == 3
    assert r["ocean_gap_values"] == 0 and r["missing_time_steps"] == 0


def test_features_have_no_future_leakage():
    s = pd.Series(np.arange(30, dtype=float), index=pd.date_range("2000-01-01", periods=30, freq="MS"))
    df = modelling.make_features(s, lags=(1, 12))
    assert (df["lag_1"] == df["y"] - 1).all()
    assert (df["lag_12"] == df["y"] - 12).all()


def test_chronological_split_is_ordered():
    s = pd.Series(np.random.rand(120), index=pd.date_range("2000-01-01", periods=120, freq="MS"))
    sp = modelling.chronological_split(modelling.make_features(s), "2005-12-31", "2007-12-31")
    assert sp.train.index.max() < sp.validation.index.min()
    assert sp.validation.index.max() < sp.test.index.min()


def test_metrics():
    y = pd.Series([1.0, 2.0, 3.0])
    m = modelling.metrics(y, y + 1)
    assert m["MAE"] == 1 and m["RMSE"] == 1 and m["bias"] == 1


def test_study_area_estimate():
    e = estimate_study_area()
    assert e["n_months"] == 396
    assert 60 < e["monthly_mb_per_level"] < 70


def test_trend_map_recovers_known_trend():
    time = pd.date_range("1993-01-01", "2022-12-01", freq="MS")
    years = ((time - time[0]).days / 365.25).to_numpy()
    season = 3 * np.sin(2 * np.pi * time.month.to_numpy() / 12)
    data = (15 + season + 0.02 * years)[:, None, None] * np.ones((1, 2, 3))
    da = xr.DataArray(data, dims=("time", "latitude", "longitude"),
                      coords={"time": time, "latitude": [-34.0, -33.0], "longitude": [17.0, 18.0, 19.0]})
    np.testing.assert_allclose(preprocessing.trend_map(da).values, 0.2, atol=1e-3)
    np.testing.assert_allclose(preprocessing.trend_map(da, (12, 1, 2)).values, 0.2, atol=1e-2)


def test_distance_to_land():
    lon = np.arange(18.0, 19.01, 0.1)
    sea = xr.DataArray(np.ones((3, lon.size), bool), dims=("latitude", "longitude"),
                       coords={"latitude": [-34.1, -34.0, -33.9], "longitude": lon})
    sea[:, 0] = False  # land column at 18.0°E
    d = preprocessing.distance_to_land_km(sea)
    assert np.isnan(d[:, 0]).all()
    # 0.5° of longitude at 34°S ≈ 46 km
    np.testing.assert_allclose(float(d.sel(latitude=-34.0, longitude=18.5, method="nearest")), 46.1, atol=1.0)


def test_seasonal_trends_complete_seasons_only():
    time = pd.date_range("1993-01-01", "2022-12-01", freq="MS")
    a = pd.Series(0.002 * np.arange(len(time)), index=time)
    st = preprocessing.seasonal_trends(a)
    assert set(st.index) == {"DJF", "MAM", "JJA", "SON"}
    np.testing.assert_allclose(st["trend_degC_per_decade"].values, 0.24, atol=0.01)


def test_fdr_adjust_is_more_conservative():
    p = [0.001, 0.01, 0.04, 0.049, 0.3]
    reject, p_adj = preprocessing.fdr_adjust(p)
    assert (p_adj >= np.array(p)).all()
    assert reject[0] and not reject[-1]


def test_load_zone_series_defaults_to_coastal_source():
    from src import data_access
    s = utils.load_settings()
    df = data_access.load_zone_series(settings=s)
    assert df.attrs["source"] == s["analysis"]["coastal_source"]
    assert {"benguela_nearshore", "agulhas_nearshore"} <= set(df.columns)
    assert df.index.freqstr == "MS"


def test_static_cache_matches_domain_and_release(tmp_path, monkeypatch):
    from src import data_access
    monkeypatch.setattr(data_access, "PROJECT_ROOT", tmp_path)
    folder=tmp_path/"data/cache"
    folder.mkdir(parents=True)
    cached=xr.Dataset({"deptho":(("latitude","longitude"),np.ones((2,2)))},
                      coords={"latitude":[-36.,-28.],"longitude":[16.,34.]},
                      attrs={"source_dataset_version":"202311"})
    cached.to_netcdf(folder/"study_area_static.nc")
    settings={"study_area":{"min_lon":16.,"max_lon":34.,"min_lat":-36.,"max_lat":-28.},
              "dataset":{"dataset_version":"202311","datasets":{"static":"test"}}}
    def no_remote(**kwargs):
        raise RuntimeError("remote fallback")
    monkeypatch.setattr(data_access.cm,"open_dataset",no_remote)
    xr.testing.assert_equal(data_access.open_static(settings),cached)
    settings["dataset"]["dataset_version"]="different"
    with pytest.raises(RuntimeError,match="remote fallback"):
        data_access.open_static(settings)


def _toy_anoms(n=240, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("1993-01-01", periods=n, freq="MS")
    a = np.zeros(n)
    for t in range(1, n):
        a[t] = 0.7 * a[t - 1] + rng.normal(0, 0.3)
    return pd.DataFrame({"z1": a, "z2": a * 0.5 + rng.normal(0, 0.1, n)}, index=idx)


def test_forecast_features_use_only_past():
    from src import forecasting as fc
    an = _toy_anoms()
    ds = fc.make_dataset(an, "z1", h=3, n_lags=4)
    t = ds.origin[10]
    assert ds.X.loc[t, "z1_lag0"] == an.loc[t, "z1"]
    assert ds.X.loc[t, "z1_lag3"] == an["z1"].shift(3).loc[t]
    assert ds.y.loc[t] == an["z1"].shift(-3).loc[t]
    assert (ds.target_time == ds.origin + pd.DateOffset(months=3)).all()
    np.testing.assert_allclose(ds.seq[10, -1, 0], an.loc[t, "z1"])   # last step = origin
    np.testing.assert_allclose(ds.seq[10, 0, 0], an["z1"].shift(3).loc[t])


def test_lstm_inputs_match_tabular_information():
    """The LSTM sees the own-zone lags, target month and other zones at t, and nothing more."""
    from src import forecasting as fc
    an = _toy_anoms()
    ds = fc.make_dataset(an, "z1", h=3, n_lags=4, cross_zone=True)
    t = ds.origin[10]
    assert ds.seq.shape[1:] == (4, 3)
    np.testing.assert_allclose(ds.seq[10, :, 0], [ds.X.loc[t, f"z1_lag{k}"] for k in (3, 2, 1, 0)])
    np.testing.assert_allclose(ds.seq[10, 0, 1:], ds.X.loc[t, ["target_month_sin", "target_month_cos"]])
    assert ds.static.shape[1] == 1 and ds.static[10, 0] == ds.X.loc[t, "z2_lag0"]
    assert fc.make_dataset(an, "z1", h=3, n_lags=4, cross_zone=False).static.shape[1] == 0


def test_lstm_static_inputs_run():
    from src import forecasting as fc
    ds = fc.make_dataset(_toy_anoms(), "z1", h=1, n_lags=4)
    m = fc.LSTMForecaster(hidden=4, max_epochs=3).fit(ds.seq, ds.y.to_numpy(), S=ds.static)
    assert m.predict(ds.seq[:5], ds.static[:5]).shape == (5,)


def test_split_by_target_time_prevents_leakage():
    from src import forecasting as fc
    ds = fc.make_dataset(_toy_anoms(), "z1", h=6)
    tr = fc.subset(ds, fc.split_mask(ds, None, "2005-12-31"))
    te = fc.subset(ds, fc.split_mask(ds, "2006-01-01", "2012-12-31"))
    assert tr.target_time.max() < te.target_time.min()


def test_fit_cutoff_respects_forecast_origin():
    """At lead h, a model evaluated from target month T may only be fitted on labels
    observed by the first origin T - h, not on the h - 1 months before T."""
    from src import forecasting as fc
    for h in (1, 3, 6):
        ds = fc.make_dataset(_toy_anoms(), "z1", h=h)
        te = fc.split_mask(ds, "2006-01-01", "2012-12-31")
        fit = fc.subset(ds, fc.fit_mask(ds, te))
        first_origin = ds.origin[te].min()
        assert first_origin == pd.Timestamp("2006-01-01") - pd.DateOffset(months=h)
        assert fit.target_time.max() == first_origin


def test_model_selection_respects_first_test_origin():
    from src import forecasting as fc
    for h in (1, 3, 6):
        ds = fc.make_dataset(_toy_anoms(n=384), "z1", h=h)
        tr, va, refit, te = fc.evaluation_masks(ds, "2016-01-01", "2019-12-31",
                                              "2020-01-01", "2024-12-31")
        cutoff = ds.origin[te].min()
        assert ds.target_time[va].max() <= cutoff
        assert ds.target_time[refit].max() <= cutoff
        assert ds.target_time[tr].max() <= ds.origin[va].min()
        assert va.sum() == 49 - h
        assert te.sum() == 60


def test_damped_persistence_recovers_known_multiplier():
    from src import forecasting as fc
    ds = fc.make_dataset(_toy_anoms(), "z1", h=1)
    ds.y = 0.4 * ds.persistence
    prediction, beta = fc.damped_persistence(ds, ds)
    np.testing.assert_allclose(beta, 0.4)
    np.testing.assert_allclose(prediction, ds.y)


def test_dm_hac_accounts_for_serially_correlated_loss():
    from src import forecasting as fc
    rng = np.random.default_rng(21)
    loss = np.zeros(500)
    for i in range(1, len(loss)):
        loss[i] = 0.9 * loss[i-1] + rng.normal()
    loss = loss - loss.min() + 0.1
    y = np.zeros(len(loss))
    independent, _ = fc.diebold_mariano(y, np.sqrt(loss), y, bandwidth=0)
    dependent, _ = fc.diebold_mariano(y, np.sqrt(loss), y, bandwidth=12)
    assert 0 < dependent < independent


def test_diebold_mariano_direction():
    from src import forecasting as fc
    rng = np.random.default_rng(1)
    y = rng.normal(size=200)
    good, bad = y + rng.normal(0, 0.1, 200), y + rng.normal(0, 1.0, 200)
    stat, p = fc.diebold_mariano(y, good, bad)
    assert stat < 0 and p < 0.01


def test_climatology_from_training_only():
    from src import forecasting as fc
    an = _toy_anoms()
    an.loc["2010":] += 5.0
    clim = fc.train_climatology(an, "2005-12-31")
    assert clim.abs().max().max() < 2.0


def _daily_series(years=12, seed=0, noise=0.2):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("1993-01-01", periods=365 * years + years // 4, freq="D")
    seas = 18 + 3 * np.sin(2 * np.pi * idx.dayofyear.to_numpy() / 365.25)
    return pd.Series(seas + rng.normal(0, noise, len(idx)), index=idx)


def test_doy366_maps_dates_consistently():
    from src import mhw
    idx = pd.DatetimeIndex(["2001-02-28", "2001-03-01", "2004-02-29", "2004-03-01", "2004-12-31", "2001-12-31"])
    np.testing.assert_array_equal(mhw.doy366(idx), [59, 61, 60, 61, 366, 366])


def test_event_mask_duration_and_gap_rules():
    from src import mhw
    ex = np.zeros(40, bool)
    ex[2:6] = True                      # 4 days: too short
    ex[10:15] = True; ex[17:22] = True  # 5 + gap 2 + 5 -> one joined event
    ex[30:35] = True                    # gap 8 from previous -> separate event
    m = mhw.event_mask(ex)
    assert not m[2:6].any()
    assert m[10:22].all()
    assert m[30:35].all() and not m[22:30].any()


def test_detect_finds_injected_heatwave_with_metrics():
    from src import mhw
    s = _daily_series()
    s.loc["2000-02-01":"2000-02-20"] += 2.0
    daily, events = mhw.detect(s, baseline=("1993-01-01", "2004-12-31"))
    ev = events[(events.start <= "2000-02-10") & (events.end >= "2000-02-10")]
    assert len(ev) == 1
    e = ev.iloc[0]
    assert 15 <= e.duration_days <= 22 and e.intensity_max > 1.5
    assert e.category >= 3   # +2 °C vs a ~0.26 °C threshold excess
    clim_frac = daily.loc["1993":"1999", "in_mhw"].mean()
    assert clim_frac < 0.05  # pure noise rarely makes 5-day runs above the 90th percentile


def test_detect_grid_matches_series():
    from src import mhw
    s = _daily_series(years=8, seed=3, noise=0.5)
    s.loc["1998-07-01":"1998-07-15"] += 2.0
    daily, _ = mhw.detect(s, baseline=("1993-01-01", "2000-12-31"))
    grid = mhw.detect_grid(s.to_numpy()[:, None, None], s.index, ("1993-01-01", "2000-12-31"))
    np.testing.assert_array_equal(grid[:, 0, 0], daily["in_mhw"].to_numpy())


def test_eof_recovers_single_mode():
    from src import spatiotemporal as st
    rng = np.random.default_rng(0)
    t, ny, nx = 200, 6, 8
    lat = np.linspace(-36, -30, ny)
    pattern = np.outer(np.ones(ny), np.linspace(-1, 1, nx))
    pc = rng.normal(size=t)
    anom = pc[:, None, None] * pattern + rng.normal(0, 0.05, (t, ny, nx))
    anom[:, 0, 0] = np.nan
    r = st.eof(anom, lat, n_modes=2)
    assert r["frac"][0] > 0.95
    assert abs(np.corrcoef(r["pcs"][:, 0], pc)[0, 1]) > 0.99


def test_decorrelation_length_of_smooth_field():
    from src import spatiotemporal as st
    rng = np.random.default_rng(1)
    lat, lon = np.linspace(-35, -30, 21), np.linspace(20, 30, 41)
    t = 300
    # Field = shared signal (perfect correlation) -> e-folding length beyond the domain.
    sig = rng.normal(size=t)[:, None, None] * np.ones((1, 21, 41))
    out = st.decorrelation_length(sig, lat, lon, [(10, 20)], max_km=300)
    assert np.isnan(out[(10, 20)]["efold_km"])
    # Independent noise -> decorrelates within the first bin.
    noise = rng.normal(size=(t, 21, 41))
    out = st.decorrelation_length(noise, lat, lon, [(10, 20)], max_km=300)
    assert out[(10, 20)]["efold_km"] <= 50
