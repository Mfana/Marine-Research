"""Paper 2 — marine heatwaves along the South African coast: detection, statistics and ML.

Data: daily ESA CCI / C3S SST (scripts/cache_daily_satellite.py), 1993–2024: six zone series
and a 0.25° grid.

Part A (descriptive; Hobday et al. 2016 with a 1993–2022 baseline): events and metrics per
zone, annual MHW days/events with HAC trends + FDR, nearshore–offshore co-occurrence, and
gridded mean/trend of annual MHW days.

Part B (machine learning; climatology and labels from the TRAINING years 1993–2015 only, so
no test information leaks into features or labels):
  B1 unsupervised detection — Isolation Forest, an MLP autoencoder and a z-score rule scored
     against Hobday MHW days (AUROC, AUPRC, F1 at a validation-chosen threshold);
  B2 early warning — is day t+k an MHW day? (k = 1, 3, 7, 14) with logistic regression, Random
     Forest and XGBoost vs persistence and base-rate baselines; SHAP for XGBoost.
Splits: train 1993–2015, validation 2016–2019, test 2020–2024 (evaluated once).

Outputs: papers/paper2/{tables,figures}/, data/processed/paper2_*.

Usage:  .venv/bin/python scripts/paper2_mhw.py
"""

import itertools
import json
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402

from src import mhw, preprocessing, utils  # noqa: E402

ROOT = utils.PROJECT_ROOT
OUT = ROOT / "papers" / "paper2"
TAB, FIG = OUT / "tables", OUT / "figures"
CACHE = ROOT / "data" / "cache"
BASELINE = ("1993-01-01", "2022-12-31")
TRAIN_END, VAL_START, VAL_END, TEST_START, TEST_END = (
    "2015-12-31", "2016-01-01", "2019-12-31", "2020-01-01", "2024-12-31")
ZONES = ["benguela_nearshore", "benguela_shelf", "benguela_offshore",
         "agulhas_nearshore", "agulhas_shelf", "agulhas_offshore"]
LEADS = (1, 3, 7, 14)


# ---- Part A -------------------------------------------------------------------

def part_a(daily_z: pd.DataFrame, log) -> dict:
    events_all, annual_all, daily_all = [], {}, {}
    for z in ZONES:
        d, ev = mhw.detect(daily_z[z], baseline=BASELINE)
        daily_all[z], annual_all[z] = d, mhw.annual_metrics(d, ev)
        events_all.append(ev.assign(zone=z))
    events = pd.concat(events_all, ignore_index=True)
    events.to_csv(TAB / "events.csv", index=False)

    summ = events.groupby("zone").agg(
        n_events=("event_id", "size"), duration_mean=("duration_days", "mean"),
        duration_max=("duration_days", "max"), intensity_max_mean=("intensity_max", "mean"),
        intensity_max_max=("intensity_max", "max"),
        cum_intensity_mean=("intensity_cumulative", "mean")).reindex(ZONES)
    n_years = daily_z.index.year.nunique()
    summ["events_per_year"] = summ["n_events"] / n_years
    summ["mhw_days_per_year"] = [daily_all[z]["in_mhw"].sum() / n_years for z in ZONES]
    cats = events.pivot_table(index="zone", columns="category", values="event_id", aggfunc="size",
                              fill_value=0).reindex(ZONES)
    cats.columns = [f"cat_{c}" for c in cats.columns]
    summ = summ.join(cats)
    summ.round(3).to_csv(TAB / "zone_summary.csv")

    # Trends of annual metrics (complete years), HAC errors, FDR across zones × metrics.
    rows = []
    for z in ZONES:
        a = annual_all[z]
        for metric in ("mhw_days", "events", "cumulative_intensity"):
            y = a[metric].astype(float)
            t = preprocessing.hac_trend(pd.Series(y.to_numpy(), index=pd.to_datetime(y.index.astype(str))), maxlags=2)
            rows.append({"zone": z, "metric": metric, "trend_per_decade": t["trend_degC_per_decade"],
                         "ci95": t["ci95"], "p_value": t["p_value"],
                         "mean_1993_2008": y.loc[:2008].mean(), "mean_2009_2024": y.loc[2009:].mean()})
    tr = pd.DataFrame(rows)
    tr["significant_fdr"], tr["p_fdr"] = preprocessing.fdr_adjust(tr["p_value"])
    tr.round(4).to_csv(TAB / "annual_trends.csv", index=False)
    pd.concat({z: annual_all[z] for z in ZONES}, names=["zone"]).to_csv(TAB / "annual_metrics.csv")

    # Nearshore–offshore co-occurrence (share of nearshore MHW days that are also offshore MHW days).
    co = []
    for r in ("benguela", "agulhas"):
        n, o = daily_all[f"{r}_nearshore"]["in_mhw"], daily_all[f"{r}_offshore"]["in_mhw"]
        both = (n & o).sum()
        co.append({"region": r, "nearshore_days": int(n.sum()), "offshore_days": int(o.sum()),
                   "both": int(both), "share_of_nearshore": both / max(n.sum(), 1),
                   "share_of_offshore": both / max(o.sum(), 1),
                   "jaccard": both / max((n | o).sum(), 1)})
    pd.DataFrame(co).round(3).to_csv(TAB / "cooccurrence.csv", index=False)

    # Top events.
    top = events.sort_values("intensity_cumulative", ascending=False).groupby("zone").head(3)
    top.round(3).to_csv(TAB / "top_events.csv", index=False)
    flags = pd.DataFrame({z: daily_all[z]["in_mhw"].astype(int) for z in ZONES})
    flags.join(pd.DataFrame({f"{z}_anomaly": daily_all[z]["anomaly"] for z in ZONES})).to_csv(
        ROOT / "data" / "processed" / "paper2_daily_mhw.csv", float_format="%.4f")
    log.info("Part A done: %d events", len(events))
    return {"daily": daily_all, "annual": annual_all, "events": events}


def part_a_grid(log) -> None:
    ds = xr.open_dataset(CACHE / "sst_satellite_daily_025deg_1993_2024.nc")
    sst = ds["sst"].load()
    idx = pd.DatetimeIndex(sst.time.values)
    vals = sst.to_numpy()
    sea = ~np.isnan(vals).all(0)
    t0 = time.time()
    in_mhw = mhw.detect_grid(vals, idx, BASELINE)
    in_mhw[:, ~sea] = False
    years = idx.year
    yrs = np.unique(years)
    annual = np.stack([in_mhw[years == y].sum(0) for y in yrs]).astype(float)
    annual[:, ~sea] = np.nan
    t = yrs - yrs.mean()
    slope = ((t[:, None, None] * (annual - annual.mean(0))).sum(0) / (t ** 2).sum()) * 10
    out = xr.Dataset({"mean_annual_mhw_days": (("latitude", "longitude"), annual.mean(0)),
                      "trend_mhw_days_per_decade": (("latitude", "longitude"), slope),
                      "annual_mhw_days": (("year", "latitude", "longitude"), annual)},
                     coords={"latitude": sst.latitude, "longitude": sst.longitude, "year": yrs})
    out.to_netcdf(ROOT / "data" / "processed" / "paper2_mhw_grid.nc")
    log.info("Gridded MHW detection on %d sea cells in %.0fs", int(sea.sum()), time.time() - t0)


# ---- Part B -------------------------------------------------------------------

def train_labels_features(daily_z: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Anomalies and MHW labels using a climatology fitted on the training years only."""
    lab, anom, excess = {}, {}, {}
    for z in ZONES:
        s = daily_z[z].asfreq("D")
        seas, thresh = mhw.climatology(s.to_numpy(), s.index, ("1993-01-01", TRAIN_END))
        d, _ = mhw.detect(s, clim=(seas, thresh))
        lab[z], anom[z] = d["in_mhw"].astype(int), d["anomaly"]
        excess[z] = d["thresh"] - d["seas"]
    return pd.DataFrame(lab), pd.DataFrame(anom), pd.DataFrame(excess)


def unsup_features(a: pd.Series) -> pd.DataFrame:
    f = pd.DataFrame({"anom": a})
    for w in (3, 7, 15, 30):
        f[f"mean{w}"] = a.rolling(w, min_periods=w).mean()
    f["std30"] = a.rolling(30, min_periods=30).std()
    f["max7"] = a.rolling(7, min_periods=7).max()
    doy = a.index.dayofyear.to_numpy()
    f["doy_sin"], f["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    return f


def split(idx, start, end):
    return (idx >= pd.Timestamp(start) if start else np.ones(len(idx), bool)) & (idx <= pd.Timestamp(end))


def best_f1_threshold(y, score):
    from sklearn.metrics import precision_recall_curve
    p, r, th = precision_recall_curve(y, score)
    f1 = 2 * p * r / np.maximum(p + r, 1e-12)
    i = int(np.nanargmax(f1[:-1])) if len(th) else 0
    return float(th[i]) if len(th) else 0.5


def cls_scores(y, score, thr) -> dict:
    from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
    pred = score >= thr
    base = y.mean()
    return {"AUROC": roc_auc_score(y, score) if 0 < base < 1 else np.nan,
            "AUPRC": average_precision_score(y, score) if base > 0 else np.nan,
            "F1": f1_score(y, pred, zero_division=0), "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0), "base_rate": base, "n": len(y)}


def part_b1(labels, anom, excess, seed, log) -> None:
    from sklearn.ensemble import IsolationForest
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    rows = []
    for z in ZONES:
        f = unsup_features(anom[z]).dropna()
        y = labels[z].reindex(f.index)
        idx = f.index
        tr, va, te = split(idx, None, TRAIN_END), split(idx, VAL_START, VAL_END), split(idx, TEST_START, TEST_END)
        sc = StandardScaler().fit(f[tr])
        X = sc.transform(f)
        scores = {}
        iso = IsolationForest(n_estimators=300, contamination="auto", random_state=seed).fit(X[tr])
        scores["isolation_forest"] = -iso.score_samples(X)
        # Warm-only variant: anomalies flagged only when the 7-day mean anomaly is positive.
        scores["isolation_forest_warm"] = np.where(f["mean7"] > 0, scores["isolation_forest"], scores["isolation_forest"].min())
        ae = MLPRegressor(hidden_layer_sizes=(16, 3, 16), max_iter=500, random_state=seed,
                          early_stopping=True).fit(X[tr], X[tr])
        rec = ((ae.predict(X) - X) ** 2).mean(1)
        scores["autoencoder"] = rec
        scores["autoencoder_warm"] = np.where(f["mean7"] > 0, rec, rec.min())
        # Rule baseline: 5-day running minimum of the standardised anomaly (no threshold knowledge).
        sd = anom[z][split(anom.index, None, TRAIN_END)].groupby(anom[z].index.dayofyear[split(anom.index, None, TRAIN_END)]).std()
        zsc = (anom[z] / sd.reindex(anom[z].index.dayofyear).to_numpy()).rolling(5, min_periods=5).min()
        scores["zscore_run5"] = zsc.reindex(idx).to_numpy()
        for name, sco in scores.items():
            sco = np.nan_to_num(sco, nan=np.nanmin(sco))
            thr = best_f1_threshold(y[va], sco[va])
            rows.append({"zone": z, "detector": name, **cls_scores(y[te].to_numpy(), sco[te], thr)})
        log.info("B1 %s done", z)
    pd.DataFrame(rows).round(4).to_csv(TAB / "unsupervised_detection_test.csv", index=False)


def ew_features(anom: pd.DataFrame, labels: pd.DataFrame, excess: pd.DataFrame, target: str) -> pd.DataFrame:
    f = {}
    for z in ZONES:
        a = anom[z]
        f[f"{z}_anom"] = a
        for w in (7, 30):
            f[f"{z}_mean{w}"] = a.rolling(w, min_periods=w).mean()
    a = anom[target]
    f["tgt_trend7"] = a - a.shift(7)
    f["tgt_excess_to_thresh"] = a - excess[target]
    # Causal only: whether day t is an MHW day depends on days t+1..t+4 (5-day minimum), so the
    # label itself is not used as a feature. Use the exceedance run length known at t instead.
    exceed = (a > excess[target]).astype(int)
    run = exceed.groupby((exceed != exceed.shift()).cumsum()).cumcount() + 1
    f["tgt_exceed_run"] = run.where(exceed == 1, 0)
    f["dT_nearshore_offshore"] = anom[f"{target.split('_')[0]}_nearshore"] - anom[f"{target.split('_')[0]}_offshore"]
    doy = a.index.dayofyear.to_numpy()
    f["doy_sin"], f["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    return pd.DataFrame(f, index=a.index)


def part_b2(labels, anom, excess, seed, log) -> None:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier
    import shap
    rows, shap_rows = [], []
    for z, k in itertools.product(ZONES, LEADS):
        X = ew_features(anom, labels, excess, z)
        y = labels[z].shift(-k).rename("y")
        ok = X.notna().all(1) & y.notna()
        X, y = X[ok], y[ok].astype(int)
        tt = X.index + pd.Timedelta(days=k)        # split on the target day
        tr, va, te = split(tt, None, TRAIN_END), split(tt, VAL_START, VAL_END), split(tt, TEST_START, TEST_END)
        trva = split(tt, None, VAL_END)
        base_rate = y[tr].mean()
        # Persistence: an event is known to be under way at t (≥5 consecutive exceedance days so far).
        probs = {"base_rate": np.full(te.sum(), base_rate),
                 "persistence": (X.loc[te, "tgt_exceed_run"] >= 5).to_numpy().astype(float)}
        cands = {
            "logistic": [make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000)) for c in (0.1, 1.0, 10.0)],
            "random_forest": [RandomForestClassifier(n_estimators=400, max_depth=d, min_samples_leaf=l, n_jobs=-1,
                                                     random_state=seed, class_weight="balanced_subsample")
                              for d in (6, 12, None) for l in (5, 20)],
            "xgboost": [XGBClassifier(n_estimators=n, max_depth=d, learning_rate=0.05, subsample=0.8,
                                      colsample_bytree=0.8, n_jobs=1, random_state=seed, eval_metric="logloss")
                        for n in (200, 600) for d in (3, 5)],
        }
        from sklearn.base import clone
        from sklearn.metrics import average_precision_score
        thr = {}
        for name, models in cands.items():
            best, best_m = -1, None
            for m in models:
                m.fit(X[tr], y[tr])
                ap = average_precision_score(y[va], m.predict_proba(X[va])[:, 1])
                if ap > best:
                    best, best_m = ap, m
            thr[name] = best_f1_threshold(y[va], best_m.predict_proba(X[va])[:, 1])
            final = clone(best_m).fit(X[trva], y[trva])
            probs[name] = final.predict_proba(X[te])[:, 1]
            if name == "xgboost" and k in (1, 7):
                sv = shap.TreeExplainer(final)(X[te])
                for feat, v in zip(X.columns, np.abs(sv.values).mean(0)):
                    shap_rows.append({"zone": z, "lead": k, "feature": feat, "mean_abs_shap": v})
        thr["persistence"], thr["base_rate"] = 0.5, base_rate
        y_te = y[te].to_numpy()
        bs_ref = brier_score_loss(y_te, probs["base_rate"])
        for name, p in probs.items():
            r = {"zone": z, "lead_days": k, "model": name, **cls_scores(y_te, p, thr[name])}
            bs = brier_score_loss(y_te, np.clip(p, 0, 1))
            r["brier"], r["BSS_vs_base_rate"] = bs, 1 - bs / bs_ref
            rows.append(r)
        log.info("B2 %s lead %d done", z, k)
    pd.DataFrame(rows).round(4).to_csv(TAB / "early_warning_test.csv", index=False)
    pd.DataFrame(shap_rows).round(5).to_csv(TAB / "early_warning_shap.csv", index=False)


def main() -> None:
    s = utils.load_settings()
    seed = s["project"]["random_seed"]
    utils.set_seed(seed)
    log = utils.setup_logging(s, "marine.paper2")
    TAB.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    t0 = time.time()
    daily_z = pd.read_csv(CACHE / "sst_satellite_daily_zones_1993_2024.csv", index_col=0, parse_dates=True)[ZONES]
    part_a(daily_z, log)
    part_a_grid(log)
    labels, anom, excess = train_labels_features(daily_z)
    part_b1(labels, anom, excess, seed, log)
    part_b2(labels, anom, excess, seed, log)
    meta = {"seconds": round(time.time() - t0), "baseline_descriptive": BASELINE,
            "ml_climatology": f"1993-01-01..{TRAIN_END}", "leads_days": LEADS,
            "splits": {"train": f"..{TRAIN_END}", "validation": f"{VAL_START}..{VAL_END}",
                       "test": f"{TEST_START}..{TEST_END}"}}
    (TAB / "run_meta.json").write_text(json.dumps(meta, indent=2))
    log.info("Paper 2 finished in %ss", meta["seconds"])


if __name__ == "__main__":
    main()
