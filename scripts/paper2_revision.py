"""Additional paper-2 analyses requested by the scientific review.

Local cached data only. Run after paper2_mhw.py. All design choices and the reused
test period remain exploratory. Artifacts are written under papers/paper2/tables.
"""
import argparse
import hashlib
import importlib.util
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paper2_mhw as p
from build_satellite_validation import zone_masks_on
from src import mhw, preprocessing, utils

ROOT, TAB = utils.PROJECT_ROOT, p.TAB
PROC = ROOT / "data/processed"
SEED = 42


def logistic_candidates():
    return [make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000))
            for c in (0.1, 1, 10)]


def fit_selected(X, y, train, validation):
    models = logistic_candidates()
    for m in models:
        m.fit(X.loc[train], y.loc[train])
    return max(models, key=lambda m: average_precision_score(y.loc[validation], m.predict_proba(X.loc[validation])[:, 1]))


def forecast_baselines(labels, anom, excess):
    pr = pd.read_csv(PROC / "paper2_ew_test_probs.csv", parse_dates=["date"])
    pr = pr[pr.date <= pd.Timestamp(p.TEST_END)-p.EMBARGO].copy()
    va_saved = pd.read_csv(PROC / "paper2_ew_validation_probs.csv", parse_dates=["date"])
    selected = pd.read_csv(TAB / "selected_models.csv")
    raw_rows = []
    for (zone,k),g in pr.groupby(["zone","lead_days"]):
        for model in ("base_rate","persistence","logistic","random_forest","rf_unweighted","xgboost"):
            threshold = (0.5 if model=="persistence" else float(g.base_rate.iloc[0]) if model=="base_rate"
                         else float(selected[(selected.zone==zone)&(selected.lead_days==k)&(selected.model==model)].threshold.iloc[0]))
            row=dict(zone=zone,lead_days=k,model=model,**p.cls_scores(g.y.to_numpy(),g[model].to_numpy(),threshold))
            row["brier"]=brier_score_loss(g.y,g[model]);row["BSS_vs_base_rate"]=1-row["brier"]/brier_score_loss(g.y,g.base_rate)
            raw_rows.append(row)
    scores=pd.DataFrame(raw_rows)
    scores.to_csv(TAB/"early_warning_test.csv",index=False)
    pr.to_csv(PROC/"paper2_ew_test_probs.csv",index=False,float_format="%.9f")
    selection, extra, frames, calibration_rows, reliability = [], [], [], [], []
    for zone in p.ZONES:
        full = p.ew_features(anom, labels, excess, zone)
        for k in p.LEADS:
            X = full.copy()
            y = labels[zone].shift(-k)
            valid = X.notna().all(axis=1) & y.notna()
            X, y = X.loc[valid], y.loc[valid].astype(int)
            target = X.index + pd.Timedelta(days=k)
            tr = p.split(target, None, pd.Timestamp(p.TRAIN_END) - p.EMBARGO)
            va = p.split(target, p.VAL_START, pd.Timestamp(p.VAL_END) - p.EMBARGO)
            te = p.split(target, p.TEST_START, pd.Timestamp(p.TEST_END)-p.EMBARGO)
            q = pr[(pr.zone == zone) & (pr.lead_days == k)].set_index("date").loc[target[te]].reset_index()
            assert np.array_equal(q.y, y.loc[te])
            val_q = va_saved[(va_saved.zone == zone) & (va_saved.lead_days == k)].set_index("date").loc[target[va]]
            new = {"threshold_distance": X.loc[te, "tgt_excess_to_thresh"].to_numpy()}
            validation_extra = {}
            thresholds = {"threshold_distance": p.best_f1_threshold(y.loc[va], X.loc[va, "tgt_excess_to_thresh"])}
            for name, cols in {
                "univariate_logistic": ["tgt_excess_to_thresh"],
                "state_logistic": ["tgt_excess_to_thresh", "tgt_exceed_run", "tgt_trend7",
                                   f"{zone}_mean7", "doy_sin", "doy_cos"],
            }.items():
                model = fit_selected(X[cols], y, tr, va)
                vp = model.predict_proba(X.loc[va, cols])[:, 1]
                validation_extra[name] = vp
                new[name] = model.predict_proba(X.loc[te, cols])[:, 1]
                thresholds[name] = p.best_f1_threshold(y.loc[va], vp)
                selection.append(dict(zone=zone, lead_days=k, model=name,
                    C=model[-1].C, threshold=thresholds[name], validation_AP=average_precision_score(y.loc[va], vp)))
            # Sigmoid calibration uses the same (not refitted) model's held-out
            # validation predictions. It never fits on test probabilities/labels.
            for raw in ("logistic", "random_forest", "rf_unweighted", "xgboost", "univariate_logistic", "state_logistic"):
                name = raw + "_sigmoid"
                vp = validation_extra[raw] if raw in validation_extra else val_q[raw].to_numpy()
                test_probability = new[raw] if raw in validation_extra else q[raw].to_numpy()
                cal = LogisticRegression(C=1e6, max_iter=2000).fit(logit(np.clip(vp, 1e-6, 1-1e-6))[:, None], val_q.y)
                cv = cal.predict_proba(logit(np.clip(vp, 1e-6, 1-1e-6))[:, None])[:, 1]
                new[name] = cal.predict_proba(logit(np.clip(test_probability, 1e-6, 1-1e-6))[:, None])[:, 1]
                thresholds[name] = p.best_f1_threshold(val_q.y, cv)
            for name, value in new.items():
                q[name] = value
                row = dict(zone=zone, lead_days=k, model=name, **p.cls_scores(q.y.to_numpy(), value, thresholds[name]))
                row["brier"] = np.nan if name == "threshold_distance" else brier_score_loss(q.y, value)
                row["BSS_vs_base_rate"] = 1 - row["brier"] / brier_score_loss(q.y, q.base_rate)
                extra.append(row)
            for name in [c for c in q.columns if c not in ("zone", "lead_days", "date", "y", "threshold_distance")]:
                values = q[name].to_numpy()
                if name in ("base_rate", "persistence"):
                    continue
                lp = logit(np.clip(values, 1e-6, 1 - 1e-6))
                fit = sm.GLM(q.y.to_numpy(), sm.add_constant(lp), family=sm.families.Binomial()).fit()
                offset = sm.GLM(q.y.to_numpy(), np.ones((len(q), 1)), offset=lp,
                                family=sm.families.Binomial()).fit()
                bins = np.minimum((values * 10).astype(int), 9)
                rel, res = 0., 0.
                for b in np.unique(bins):
                    take = bins == b
                    obs = q.y.to_numpy()[take].mean()
                    pred = values[take].mean()
                    rel += take.mean() * (pred - obs) ** 2
                    res += take.mean() * (obs - q.y.mean()) ** 2
                    # Wilson binomial intervals are descriptive only: the bootstrap
                    # calibration intervals below account for serial dependence.
                    reliability.append(dict(zone=zone, lead_days=k, model=name, bin=int(b),
                        count=int(take.sum()), mean_prediction=pred, observed_fraction=obs))
                calibration_rows.append(dict(zone=zone, lead_days=k, model=name,
                    observed=q.y.mean(), mean_prediction=values.mean(),
                    intercept=fit.params[0], slope=fit.params[1], calibration_in_large=offset.params[0],
                    reliability_binned=rel, resolution_binned=res,
                    uncertainty=q.y.mean() * (1-q.y.mean())))
            q["issue_date"] = X.index[te]
            q["issue_in_mhw_retrospective"] = labels[zone].reindex(X.index[te]).to_numpy()
            q["issue_exceed_run"] = X.loc[te, "tgt_exceed_run"].to_numpy()
            frames.append(q)
    pr = pd.concat(frames, ignore_index=True)
    pr.to_csv(PROC / "paper2_ew_test_probs_extended.csv", index=False, float_format="%.17g")
    pd.concat([scores, pd.DataFrame(extra)], ignore_index=True).to_csv(TAB / "forecast_scores_extended.csv", index=False)
    pd.DataFrame(selection).to_csv(TAB / "baseline_selection.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(TAB / "calibration.csv", index=False)
    pd.DataFrame(reliability).to_csv(TAB / "reliability_bins.csv", index=False)
    return pr


def paired_intervals(pr, draws=2000):
    """Fast AP for repeated circular samples, including exact tie handling.

    Multiplicity weights on each original date are equivalent to explicit repeated
    rows; all zones and models use the same sampled test dates.
    """
    rng = np.random.default_rng(SEED)
    comparisons = ["threshold_distance", "univariate_logistic", "state_logistic", "xgboost",
                   "random_forest", "rf_unweighted", "persistence"]
    rows, calibration_intervals = [], []
    for k in p.LEADS:
        d = pr[pr.lead_days == k]
        dates = np.sort(d.date.unique())
        gs = [d[d.zone == z].set_index("date").loc[dates] for z in p.ZONES]
        n = len(dates)
        Y = np.array([g.y.to_numpy() for g in gs])
        models = ["logistic"] + comparisons
        probabilities = np.array([[g[m].to_numpy() for m in models] for g in gs])
        order = np.argsort(-probabilities, axis=2, kind="stable")
        ties = np.concatenate([np.diff(np.take_along_axis(probabilities, order, axis=2), axis=2) != 0,
                               np.ones((*order.shape[:2], 1), bool)], axis=2)

        def stats(weights):
            positive = Y @ weights
            if np.any(positive == 0):
                return None
            sw = np.take_along_axis(np.broadcast_to(weights, order.shape), order, axis=2)
            sy = np.take_along_axis(np.broadcast_to(Y[:, None, :], order.shape), order, axis=2)
            tp = np.cumsum(sw * sy, axis=2)
            total = np.cumsum(sw, axis=2)
            precision = np.divide(tp, total, out=np.zeros_like(tp, dtype=float), where=total > 0)
            ap = np.empty(order.shape[:2])
            for zi in range(len(gs)):
                for mi in range(len(models)):
                    ends = ties[zi, mi]
                    change = np.diff(np.r_[0, tp[zi, mi, ends]])
                    ap[zi, mi] = np.sum(change * precision[zi, mi, ends]) / positive[zi]
            ref = np.array([np.sum(weights * (Y[i] - gs[i].base_rate.to_numpy()) ** 2) / weights.sum() for i in range(len(gs))])
            bs = np.sum(weights * (Y[:, None, :] - probabilities) ** 2, axis=2) / weights.sum()
            bss = 1 - bs / ref[:, None]
            bss[:, 1] = np.nan  # threshold distance is a ranking score, not a probability
            return ap, bss

        point = stats(np.ones(n))
        expected = np.array([[average_precision_score(g.y, g[m]) for m in models] for g in gs])
        np.testing.assert_allclose(point[0], expected, atol=1e-12)
        for block in p.BOOT_BLOCKS:
            inds = p.block_indices(n, block, draws, rng)
            samples = []
            cal_bias, cal_bins = [], []
            for iteration, ix in enumerate(inds):
                weights = np.bincount(ix, minlength=n)
                out = stats(weights)
                if iteration == 0 and out is not None:
                    explicit = np.array([[average_precision_score(g.y.to_numpy()[ix],g[m].to_numpy()[ix])
                                          for m in models] for g in gs])
                    np.testing.assert_allclose(out[0],explicit,atol=1e-12)
                if out is not None:
                    samples.append(out)
                # Pooled reliability is a descriptive summary across dependent zones.
                bias = np.array([(g.logistic.to_numpy() - g.y.to_numpy()) @ weights / n for g in gs])
                cal_bias.append(bias)
                binvals = []
                for g in gs:
                    values = g.logistic.to_numpy(); y = g.y.to_numpy()
                    bins = np.minimum((values * 10).astype(int), 9)
                    binvals.append([(np.sum(weights[bins == b] * y[bins == b]) / np.sum(weights[bins == b])
                                     if np.sum(weights[bins == b]) else np.nan) for b in range(10)])
                cal_bins.append(binvals)
            aps = np.array([v[0] for v in samples]); bsss = np.array([v[1] for v in samples])
            for mi, model in enumerate(comparisons, 1):
                for metric, pts, boot in [("AUPRC", point[0], aps), ("BSS", point[1], bsss)]:
                    if model == "threshold_distance" and metric == "BSS":
                        continue
                    diff = pts[:, 0] - pts[:, mi]
                    bd = boot[:, :, 0] - boot[:, :, mi]
                    for zi, zone in enumerate(p.ZONES + ["mean"]):
                        estimate = diff[zi] if zi < 6 else diff.mean()
                        sample = bd[:, zi] if zi < 6 else bd.mean(axis=1)
                        rows.append(dict(lead_days=k, block_days=block, comparison="logistic-"+model,
                            metric=metric, zone=zone, diff=estimate, lo=np.quantile(sample, .025),
                            hi=np.quantile(sample, .975), n_draws=len(samples)))
            cb, rb = np.asarray(cal_bias), np.asarray(cal_bins)
            for zi, zone in enumerate(p.ZONES):
                for b in range(10):
                    sample = rb[:, zi, b]; sample = sample[np.isfinite(sample)]
                    if len(sample):
                        calibration_intervals.append(dict(zone=zone, lead_days=k, model="logistic", block_days=block,
                            bin=b, lo=np.quantile(sample,.025), hi=np.quantile(sample,.975),
                            bias_lo=np.quantile(cb[:,zi],.025), bias_hi=np.quantile(cb[:,zi],.975)))
            print("paired bootstrap", k, block, len(samples), flush=True)
    pd.DataFrame(rows).to_csv(TAB / "early_warning_uncertainty.csv", index=False)
    pd.DataFrame(calibration_intervals).to_csv(TAB / "calibration_uncertainty.csv", index=False)


def onset_target(label, horizon):
    """Whether an event starts in (t, t+horizon]; target needs horizon+6 future days."""
    starts = label.astype(bool) & ~label.shift(1, fill_value=0).astype(bool)
    future = pd.concat([starts.shift(-j) for j in range(1, horizon + 1)], axis=1)
    return future.max(axis=1).where(future.notna().all(axis=1))


def causal_inactive(exceed_run):
    """Below threshold, outside an event confirmed from past observations only."""
    known, active, gap = [], False, 0
    for run in exceed_run:
        gap = gap + 1 if run == 0 else 0
        if run >= 5:
            active = True
        if gap > 2:
            active = False
        known.append(active)
    return (exceed_run == 0) & ~np.asarray(known)


def onset_analysis(labels, anom, excess):
    rows, prediction_frames = [], []
    for zone in p.ZONES:
        X = p.ew_features(anom, labels, excess, zone)
        # Strictly inactive state: today's SST is not above the threshold. Unlike a
        # retrospective MHW label, this can be determined at the issue date.
        inactive = causal_inactive(X.tgt_exceed_run)
        for k in p.LEADS:
            y = onset_target(labels[zone], k)
            ok = X.notna().all(axis=1) & y.notna() & inactive
            xf, yf = X.loc[ok], y.loc[ok].astype(int)
            end = xf.index + pd.Timedelta(days=k)
            tr = p.split(end, None, pd.Timestamp(p.TRAIN_END)-p.EMBARGO)
            va = p.split(end, p.VAL_START, pd.Timestamp(p.VAL_END)-p.EMBARGO)
            # Full issue-to-horizon window must lie inside test; final 7 days are
            # censored since event duration/gap qualification needs future evidence.
            te = (xf.index >= p.TEST_START) & (end <= pd.Timestamp(p.TEST_END)-p.EMBARGO)
            for name, cols in {"onset_multivariate": list(xf.columns),
                               "onset_univariate": ["tgt_excess_to_thresh"]}.items():
                model = fit_selected(xf[cols], yf, tr, va)
                vp = model.predict_proba(xf.loc[va, cols])[:, 1]
                prob = model.predict_proba(xf.loc[te, cols])[:, 1]
                threshold = p.best_f1_threshold(yf.loc[va], vp)
                dates = xf.index[te]
                alerts = pd.Series(prob >= threshold, index=dates).reindex(
                    pd.date_range(p.TEST_START, pd.Timestamp(p.TEST_END)-p.EMBARGO), fill_value=False)
                episodes = mhw._runs(alerts.to_numpy())
                starts = labels[zone].astype(bool) & ~labels[zone].shift(1, fill_value=0).astype(bool)
                event_dates = starts.index[starts & (starts.index >= p.TEST_START) &
                                          (starts.index <= pd.Timestamp(p.TEST_END)-p.EMBARGO)]
                eligible, detected, lead_times = 0, 0, []
                for event in event_dates:
                    windows = dates[(dates >= event-pd.Timedelta(days=k)) & (dates < event)]
                    if len(windows) == 0:
                        continue
                    eligible += 1
                    hits = windows[alerts.reindex(windows).to_numpy()]
                    if len(hits):
                        detected += 1
                        lead_times.append((event-hits[0]).days)
                false = sum(not any((event_dates > alerts.index[a]) &
                                   (event_dates <= alerts.index[b]+pd.Timedelta(days=k))) for a,b in episodes)
                row = dict(zone=zone, lead_days=k, model=name,
                    **p.cls_scores(yf.loc[te].to_numpy(), prob, threshold), threshold=threshold,
                    all_test_events=len(event_dates), eligible_events=eligible, detected_events=detected,
                    event_recall=detected/eligible if eligible else np.nan,
                    false_alerts_per_year=false / (len(alerts)/365.25), alert_episodes=len(episodes),
                    mean_alert_duration=np.mean([b-a+1 for a,b in episodes]) if episodes else 0,
                    mean_warning_lead_days=np.mean(lead_times) if lead_times else np.nan)
                rows.append(row)
                prediction_frames.append(pd.DataFrame(dict(zone=zone, lead_days=k, model=name,
                    issue_date=dates, y=yf.loc[te].to_numpy(), probability=prob, threshold=threshold)))
    pd.DataFrame(rows).to_csv(TAB / "onset_scores.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(PROC / "paper2_onset_probs.csv", index=False)


def geometry_and_trends(daily):
    monthly = xr.open_dataset(p.CACHE / "sst_satellite_monthly_1993_2024.nc").sst
    static = xr.open_dataset(p.CACHE / "study_area_static.nc")
    masks = zone_masks_on(monthly.notnull().any("time"), static.deptho, utils.load_settings())
    weights = np.cos(np.deg2rad(monthly.latitude)).broadcast_like(monthly.isel(time=0))
    cell_area = (6371*np.pi/180*.05)**2 * weights
    overlap = []
    for region in ("benguela", "agulhas"):
        near, shelf, off = [masks[f"{region}_{x}"] for x in ("nearshore", "shelf", "offshore")]
        for name, mask in (("nearshore",near),("shelf",shelf),("offshore",off),
                           ("nearshore_shelf",near&shelf),("nearshore_offshore",near&off)):
            overlap.append(dict(region=region, mask=name, cells=int(mask.sum()),
                                area_km2=float(cell_area.where(mask).sum())))
    pd.DataFrame(overlap).to_csv(TAB / "zone_overlap.csv", index=False)

    sst = xr.open_dataset(p.CACHE / "sst_satellite_daily_025deg_1993_2024.nc").sst.load()
    sea = sst.notnull().any("time")
    # Use distance on the original sea mask, interpolated to coarse cell centres;
    # measuring distance after coarsening shifts the coast.
    distance = preprocessing.distance_to_land_km(monthly.notnull().any("time")).interp(
        latitude=sst.latitude, longitude=sst.longitude, method="nearest")
    depth = static.deptho.interp(latitude=sst.latitude, longitude=sst.longitude, method="nearest")
    w = np.cos(np.deg2rad(sst.latitude)).broadcast_like(sea)
    rows = []
    for width in (25, 50, 75):
        for region, region_mask in (("benguela",sst.longitude<20), ("agulhas",sst.longitude>20)):
            near = (distance<=width)&sea&region_mask
            shelf = (distance>width)&((depth<=200)|depth.isnull())&sea&region_mask
            offshore = (distance>width)&(depth>200)&sea&region_mask
            assert not bool(((near&shelf)|(near&offshore)|(shelf&offshore)).any())
            for name, mask in (("nearshore",near),("shelf",shelf),("offshore",offshore)):
                series = sst.where(mask).weighted(w).mean(("latitude","longitude")).to_series()
                for baseline in [("1993-01-01","2022-12-31"),("1993-01-01","2015-12-31")]:
                    d,e = mhw.detect(series, baseline=baseline)
                    annual = mhw.annual_metrics(d,e)
                    trend = preprocessing.hac_trend(pd.Series(annual.mhw_days.to_numpy(),
                        index=pd.to_datetime(annual.index.astype(str))), maxlags=2)
                    rows.append(dict(region=region, zone=name, width_km=width, baseline_end=baseline[1],
                        cells=int(mask.sum()), events=len(e), mean_duration=e.duration_days.mean(),
                        mean_peak=e.intensity_max.mean(), days_per_year=d.in_mhw.sum()/32,
                        trend=trend["trend_degC_per_decade"], ci95=trend["ci95"]))
    pd.DataFrame(rows).to_csv(TAB / "disjoint_zone_sensitivity.csv", index=False)

    grid = xr.open_dataset(PROC / "paper2_mhw_grid.nc")
    coastal = grid.annual_mhw_days.where(distance<=50).weighted(w).mean(("latitude","longitude")).to_numpy()
    offshore = grid.annual_mhw_days.where(distance>100).weighted(w).mean(("latitude","longitude")).to_numpy()
    years = grid.year.to_numpy()
    trend_rows = []
    for name, values in (("coastal",coastal),("offshore",offshore),("offshore-minus-coastal",offshore-coastal)):
        for omit in (None,2020):
            take = np.ones(len(years),bool) if omit is None else years != omit
            X = sm.add_constant((years[take]-years[take].mean())/10)
            fit = sm.OLS(values[take],X).fit(cov_type="HAC",cov_kwds={"maxlags":2})
            lo,hi = fit.conf_int()[1]
            trend_rows.append(dict(series=name, omitted_year=omit, trend=fit.params[1],lo=lo,hi=hi,p=fit.pvalues[1]))
    pd.DataFrame(trend_rows).to_csv(TAB / "grid_band_trend_contrast.csv", index=False)
    pd.DataFrame(dict(year=years,coastal=coastal,offshore=offshore)).to_csv(TAB / "grid_band_annual.csv",index=False)

    # Whole-year circular shifts retain calendar season and each series' persistence.
    co_rows=[]
    for region in ("benguela","agulhas"):
        n,_=mhw.detect(daily[region+"_nearshore"]);o,_=mhw.detect(daily[region+"_offshore"])
        index=n.index; obs=float((n.in_mhw&o.in_mhw).sum()/n.in_mhw.sum())
        surrogates=[]
        for shift in range(1,32):
            dates=[pd.Timestamp(year=1993+(d.year-1993+shift)%32,month=d.month,
                     day=min(d.day,28) if d.month==2 and d.day==29 else d.day) for d in index]
            shuffled=o.in_mhw.reindex(dates).to_numpy()
            surrogates.append((n.in_mhw.to_numpy()&shuffled).sum()/n.in_mhw.sum())
        for lag in (-14,-7,0,7,14):
            shifted=o.in_mhw.shift(lag,fill_value=False)
            co_rows.append(dict(region=region,lag_days=lag,share=float((n.in_mhw&shifted).sum()/n.in_mhw.sum()),
                null_mean=np.mean(surrogates),null_lo=np.quantile(surrogates,.025),null_hi=np.quantile(surrogates,.975),
                shift_p=(1+sum(np.asarray(surrogates)>=obs))/32))
    pd.DataFrame(co_rows).to_csv(TAB / "cooccurrence_null_lags.csv",index=False)


def historical(labels, anom, excess):
    # Recompute climatologies at each historical origin, avoiding use of later
    # years in earlier features/labels. Choices are still retrospective/exploratory.
    daily=pd.read_csv(p.CACHE/"sst_satellite_daily_zones_1993_2024.csv",index_col=0,parse_dates=True)
    rows=[]
    for train_end,validation_end,test_end in ((2007,2011,2015),(2011,2015,2019)):
        a,e,l={},{},{}
        for z in p.ZONES:
            seas,threshold=mhw.climatology(daily[z].to_numpy(),daily.index,("1993-01-01",f"{train_end}-12-31"))
            dd,_=mhw.detect(daily[z],clim=(seas,threshold));a[z]=dd.anomaly;e[z]=dd.thresh-dd.seas;l[z]=dd.in_mhw.astype(int)
        a,e,l=pd.DataFrame(a),pd.DataFrame(e),pd.DataFrame(l)
        for zone in p.ZONES:
            X=p.ew_features(a,l,e,zone)
            for k in p.LEADS:
                y=l[zone].shift(-k);ok=X.notna().all(axis=1)&y.notna();xf,yf=X.loc[ok],y.loc[ok].astype(int)
                target=xf.index+pd.Timedelta(days=k)
                tr=p.split(target,None,pd.Timestamp(f"{train_end}-12-31")-p.EMBARGO)
                va=p.split(target,f"{train_end+1}-01-01",pd.Timestamp(f"{validation_end}-12-31")-p.EMBARGO)
                te=p.split(target,f"{validation_end+1}-01-01",f"{test_end}-12-31")
                for name,cols in {"logistic":list(xf.columns),"univariate_logistic":["tgt_excess_to_thresh"]}.items():
                    model=fit_selected(xf[cols],yf,tr,va);prob=model.predict_proba(xf.loc[te,cols])[:,1]
                    rows.append(dict(train_end=train_end,test_start=validation_end+1,test_end=test_end,zone=zone,
                        lead_days=k,model=name,AP=average_precision_score(yf.loc[te],prob),base_rate=yf.loc[te].mean()))
    pd.DataFrame(rows).to_csv(TAB/"historical_scores.csv",index=False)


def reference_check(path):
    import types
    ref=types.ModuleType("reference_mhw")
    source=Path(path).read_text()
    # Upstream assigns one-element arrays to scalars, rejected by NumPy 2.
    # .item() changes only that compatibility detail, not the algorithm.
    source=source.replace("doy[tt] = doy_leapYear[(month_leapYear == month[tt]) * (day_leapYear == day[tt])]",
                          "doy[tt] = doy_leapYear[(month_leapYear == month[tt]) * (day_leapYear == day[tt])].item()")
    source=source.replace("doyClim[i] = doy_leapYear[(month_leapYear == monthClim[i]) * (day_leapYear == dayClim[i])]",
                          "doyClim[i] = doy_leapYear[(month_leapYear == monthClim[i]) * (day_leapYear == dayClim[i])].item()")
    exec(compile(source,str(path),"exec"),ref.__dict__)
    # Compatibility alias only: upstream uses the removed NumPy NaN spelling.
    if not hasattr(np,"NaN"): np.NaN=np.nan
    daily=pd.read_csv(p.CACHE/"sst_satellite_daily_zones_1993_2024.csv",index_col=0,parse_dates=True)
    t=np.array([v.date().toordinal() for v in daily.index]);rows=[]
    for z in p.ZONES:
        for end in ("2015-12-31","2022-12-31"):
            base=(daily.index>=pd.Timestamp("1993-01-01"))&(daily.index<=pd.Timestamp(end))
            re,cl=ref.detect(t,daily[z].to_numpy().copy(),climatologyPeriod=[1993,int(end[:4])],alternateClimatology=[t[base],daily[z].to_numpy()[base].copy()])
            d,ev=mhw.detect(daily[z],baseline=("1993-01-01",end));mask=np.zeros(len(d),bool)
            for start,finish in zip(re["index_start"],re["index_end"]):mask[start:finish+1]=True
            err=max(np.max(np.abs(d.seas-cl["seas"])),np.max(np.abs(d.thresh-cl["thresh"])))
            assert np.array_equal(mask,d.in_mhw)
            for a,b in (("duration_days","duration"),("intensity_max","intensity_max"),
                        ("intensity_cumulative","intensity_cumulative"),("onset_rate","rate_onset")):
                np.testing.assert_allclose(ev[a],re[b],atol=1e-9,rtol=1e-9)
            cat=np.array([{"Moderate":1,"Strong":2,"Severe":3,"Extreme":4}.get(v,v) for v in re["category"]])
            np.testing.assert_array_equal(ev.category,cat)
            rows.append(dict(zone=z,baseline_end=end,events=len(ev),max_climatology_error=err,day_mask_matches=True,metrics_match=True))
    pd.DataFrame(rows).to_csv(TAB/"reference_validation.csv",index=False)
    (TAB/"reference_provenance.json").write_text(json.dumps(dict(url="https://raw.githubusercontent.com/ecjoliver/marineHeatWaves/master/marineHeatWaves.py",
        sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),comparison="baseline-only alternateClimatology"),indent=2))


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--reference");parser.add_argument("--only",choices=["geometry","forecast","historical","reference"]);args=parser.parse_args()
    warnings.filterwarnings("ignore",category=FutureWarning)
    daily=pd.read_csv(p.CACHE/"sst_satellite_daily_zones_1993_2024.csv",index_col=0,parse_dates=True)
    labels,anom,excess=p.train_labels_features(daily)
    if args.only in (None,"reference") and args.reference:reference_check(args.reference)
    if args.only in (None,"geometry"):geometry_and_trends(daily)
    if args.only in (None,"historical"):historical(labels,anom,excess)
    if args.only in (None,"forecast"):
        pr=forecast_baselines(labels,anom,excess);onset_analysis(labels,anom,excess);paired_intervals(pr)
        path=TAB/"run_meta.json"
        meta=json.loads(path.read_text())
        meta.update(revision_date="2026-11-01",supervised_final_fit="training_only_no_post_validation_refit",
                    test_target_end="2024-12-24",test_end_censor_days=7,
                    exploratory=True,bootstrap={"draws":2000,"block_days":list(p.BOOT_BLOCKS)},
                    onset_eligibility="below threshold and outside causally confirmed event",
                    legacy_daily_version="requested 202506; returned version unrecorded",
                    historical_origins=[2007,2011])
        path.write_text(json.dumps(meta,indent=2))


if __name__=="__main__":main()
