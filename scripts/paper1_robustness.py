"""Reproducible supplementary checks for paper 1.

--prepare builds historical ridge benchmarks and disjoint-zone sensitivity from
cached data. Default analyses the corrected main-run predictions. No network calls.
"""
import argparse
import json
import hashlib
import importlib.metadata
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

from src import data_access, forecasting as fc, preprocessing, utils
from paper1_forecasting import tune_and_test

ROOT = utils.PROJECT_ROOT
TAB = ROOT / "papers/paper1/tables"
ML = ("ridge", "random_forest", "xgboost", "lstm")
SEED = 42


def block_indices(n, length, draws=2000):
    """Circular moving-block indices, shared across paired models and all zones."""
    rng = np.random.default_rng(SEED + length)
    starts = rng.integers(0, n, (draws, int(np.ceil(n / length))))
    return ((starts[:, :, None] + np.arange(length)) % n).reshape(draws, -1)[:, :n]


def historical():
    """Nested historical blocks for ridge; not a stability claim about all models."""
    raw = data_access.load_zone_series("satellite")
    seed = utils.load_settings()["project"]["random_seed"]
    rows, params = [], []
    for first, last, val_start, clim_end in [(2010, 2014, 2006, 2004), (2015, 2019, 2011, 2009)]:
        an = fc.to_anomalies(raw, fc.train_climatology(raw, f"{clim_end}-12-31"))
        for z in raw.columns:
            for h in (1, 3, 6):
                ds = fc.make_dataset(an, z, h, cross_zone=True)
                _, _, refit, te_mask = fc.evaluation_masks(
                    ds, f"{val_start}-01-01", f"{first-1}-12-31", f"{first}-01-01", f"{last}-12-31")
                te = fc.subset(ds, te_mask)
                p, par, info, _ = tune_and_test(ds, "ridge", seed, None,
                    f"{val_start}-01-01", f"{first-1}-12-31", f"{first}-01-01", f"{last}-12-31")
                dp, beta = fc.damped_persistence(fc.subset(ds, refit), te)
                y = te.y.to_numpy()
                for model, prediction in [("ridge", p), ("damped_persistence", dp),
                                           ("persistence", te.persistence.to_numpy())]:
                    rows.append(dict(period=f"{first}--{last}", zone=z, h=h, model=model,
                                     **fc.scores(y, prediction, np.mean(y*y))))
                params.append(dict(period=f"{first}--{last}", zone=z, h=h,
                                   params=json.dumps(par), damping=beta, **info))
    pd.DataFrame(rows).to_csv(TAB / "historical_scores.csv", index=False)
    pd.DataFrame(params).to_csv(TAB / "historical_settings.csv", index=False)


def disjoint():
    import xarray as xr
    from build_satellite_validation import zone_masks_on
    settings = utils.load_settings()
    sat = data_access.load_monthly_grid("satellite", settings)
    static = xr.load_dataset(ROOT / "data/cache/study_area_static.nc")
    masks = zone_masks_on(sat.notnull().any("time"), static.deptho, settings)
    zones = list(masks)
    overlap = []
    for a in zones:
        for b in zones:
            overlap.append(dict(target=a, predictor=b,
                fraction_target_area_shared=float(preprocessing.area_weights(sat).where(masks[a] & masks[b]).sum()
                    / preprocessing.area_weights(sat).broadcast_like(masks[a]).where(masks[a]).sum())))
    pd.DataFrame(overlap).to_csv(TAB / "zone_overlap.csv", index=False)
    disjoint_masks = dict(masks)
    for region in ("benguela", "agulhas"):
        near = masks[f"{region}_nearshore"]
        for kind in ("shelf", "offshore"):
            disjoint_masks[f"{region}_{kind}"] = masks[f"{region}_{kind}"] & ~near
    for i, a in enumerate(zones):
        assert int(disjoint_masks[a].sum()) > 0
        for b in zones[i+1:]:
            assert not bool((disjoint_masks[a] & disjoint_masks[b]).any()), (a, b)
    raw = pd.DataFrame({z: preprocessing.area_weighted_mean(sat.where(m)).to_series()
                        for z, m in disjoint_masks.items()})
    raw.to_csv(ROOT / "data/processed/paper1_disjoint_zones_monthly.csv")
    an = fc.to_anomalies(raw, fc.train_climatology(raw, "2014-12-31"))
    rows = []
    seed = settings["project"]["random_seed"]
    for z in zones:
        for fs in ("univariate", "multizone"):
            ds = fc.make_dataset(an, z, 1, cross_zone=fs == "multizone")
            _, _, _, te_mask = fc.evaluation_masks(ds, "2016-01-01", "2019-12-31", "2020-01-01", "2024-12-31")
            te = fc.subset(ds, te_mask)
            pred, _, _, model = tune_and_test(ds, "ridge", seed, None)
            y = te.y.to_numpy()
            rows.append(dict(zone=z, features=fs, cells=int(disjoint_masks[z].sum()),
                             **fc.scores(y, pred, np.mean(y*y))))
    pd.DataFrame(rows).to_csv(TAB / "disjoint_scores.csv", index=False)
    original = data_access.load_zone_series("satellite")
    fc.to_anomalies(original, fc.train_climatology(original, "2014-12-31")).loc["2020":"2024"].corr().to_csv(
        TAB / "test_zone_correlations.csv")


def uncertainty(preds):
    rows, aggregate, ablations = [], [], []
    for h in (1, 3, 6):
        for fs in ("univariate", "multizone"):
            dd = preds[(preds.h == h) & (preds.features == fs)]
            error = dd.assign(sq=(dd.pred_anom - dd.y_anom)**2).pivot(
                index="target_time", columns=["model", "zone"], values="sq").sort_index()
            for length in (6, 12):
                ix = block_indices(len(error), length)
                for m in ML:
                    for ref in ("climatology", "persistence", "damped_persistence"):
                        losses, skills = [], []
                        for z in error[m].columns:
                            a, b = error[m][z].to_numpy(), error[ref][z].to_numpy()
                            boot = (a-b)[ix].mean(1)
                            skill = 1-a[ix].mean(1)/b[ix].mean(1)
                            lo, hi = np.quantile(boot, [.025, .975])
                            rows.append(dict(h=h, features=fs, model=m, reference=ref, zone=z,
                                block_months=length, delta_mse=np.mean(a-b), lower=lo, upper=hi))
                            losses.append(boot); skills.append(skill)
                        for metric, vals in [("delta_mse", losses), ("mean_MSSS", skills)]:
                            boot = np.mean(vals, axis=0)
                            lo, hi = np.quantile(boot, [.025, .975])
                            aggregate.append(dict(h=h, features=fs, model=m, reference=ref,
                                block_months=length, metric=metric, lower=lo, upper=hi))
        # Paired ablation on the same target months, with shared resampled blocks.
        dd = preds[preds.h == h].assign(sq=lambda a:(a.pred_anom-a.y_anom)**2)
        e = dd.pivot(index="target_time", columns=["model", "features", "zone"], values="sq").sort_index()
        for length in (6, 12):
            ix=block_indices(len(e), length)
            for m in ML:
                boots=[]; points=[]
                for z in e[m]["univariate"].columns:
                    u=e[m]["univariate"][z].to_numpy(); v=e[m]["multizone"][z].to_numpy()
                    c=e["climatology"]["univariate"][z].to_numpy()
                    boot=(u-v)[ix].mean(1)/c[ix].mean(1)
                    lo,hi=np.quantile(boot,[.025,.975]); point=np.mean(u-v)/np.mean(c)
                    ablations.append(dict(h=h,model=m,zone=z,block_months=length,delta_MSSS=point,lower=lo,upper=hi))
                    boots.append(boot);points.append(point)
                lo,hi=np.quantile(np.mean(boots,axis=0),[.025,.975])
                ablations.append(dict(h=h,model=m,zone="mean",block_months=length,
                    delta_MSSS=np.mean(points),lower=lo,upper=hi))
    pd.DataFrame(rows).to_csv(TAB / "paired_uncertainty.csv", index=False)
    pd.DataFrame(aggregate).to_csv(TAB / "aggregate_uncertainty.csv", index=False)
    pd.DataFrame(ablations).to_csv(TAB / "ablation_uncertainty.csv", index=False)


def hac_sensitivity(preds):
    comparisons=pd.read_csv(TAB / "diebold_mariano.csv")
    out=[]
    for bandwidth in (0, 6, 12):
        rows=[]
        for _,r in comparisons.iterrows():
            d=preds[(preds.zone==r.zone)&(preds.h==r.h)&(preds.features==r.features)]
            a=d[d.model==r.model].sort_values("target_time")
            b=d[d.model==r.reference].sort_values("target_time")
            lag=max(int(r.h)-1,bandwidth)
            stat,p=fc.diebold_mariano(a.y_anom,a.pred_anom,b.pred_anom,int(r.h),bandwidth=lag)
            rows.append(dict(zone=r.zone,h=r.h,features=r.features,model=r.model,reference=r.reference,
                             bandwidth=lag,bandwidth_rule=bandwidth,dm_stat=stat,p_value=p))
        df=pd.DataFrame(rows)
        df["significant_fdr"],df["p_fdr"]=preprocessing.fdr_adjust(df.p_value.fillna(1))
        out.append(df)
    pd.concat(out).to_csv(TAB / "hac_sensitivity.csv",index=False)


def seasonality(preds):
    d=preds.copy()
    d["season"]=d.target_time.dt.month.map({12:"DJF",1:"DJF",2:"DJF",3:"MAM",4:"MAM",5:"MAM",
                                           6:"JJA",7:"JJA",8:"JJA",9:"SON",10:"SON",11:"SON"})
    rows=[]
    for key,g in d.groupby(["zone","h","features","model","season"]):
        y=g.y_anom.to_numpy();p=g.pred_anom.to_numpy()
        rows.append(dict(zip(["zone","h","features","model","season"],key)),)
        rows[-1].update(fc.scores(y,p,np.mean(y*y)))
    pd.DataFrame(rows).to_csv(TAB / "seasonal_scores.csv",index=False)


def coefficients():
    raw=data_access.load_zone_series("satellite")
    an=fc.to_anomalies(raw,fc.train_climatology(raw,"2014-12-31"))
    settings=pd.read_csv(TAB / "tuned_hyperparameters.csv")
    rows=[]
    for _,r in settings[settings.model=="ridge"].iterrows():
        ds=fc.make_dataset(an,r.zone,int(r.h),cross_zone=r.features=="multizone")
        tr,va,refit,te=fc.evaluation_masks(ds,"2016-01-01","2019-12-31","2020-01-01","2024-12-31")
        params=json.loads(r.params)
        for label,mask in [("initial",tr),("refit",refit)]:
            fit=fc.subset(ds,mask);m=fc.fit_tree("ridge",{"alpha":params["alpha"]},fit.X,fit.y,SEED)
            for f,b in zip(fit.X.columns,m[-1].coef_ / m[0].scale_):
                rows.append(dict(zone=r.zone,h=r.h,features=r.features,fit=label,feature=f,coefficient=b))
    pd.DataFrame(rows).to_csv(TAB / "ridge_coefficients.csv",index=False)


def learning_curves():
    raw=data_access.load_zone_series("satellite")
    an=fc.to_anomalies(raw,fc.train_climatology(raw,"2014-12-31"))
    settings=pd.read_csv(TAB / "tuned_hyperparameters.csv")
    rows=[]
    for z in ("benguela_nearshore", "agulhas_offshore"):
        r=settings[(settings.model=="lstm")&(settings.zone==z)&(settings.h==1)&(settings.features=="multizone")].iloc[0]
        par=json.loads(r.params)
        ds=fc.make_dataset(an,z,1,cross_zone=True)
        tr,va,_,_=fc.evaluation_masks(ds,"2016-01-01","2019-12-31","2020-01-01","2024-12-31")
        train,val=fc.subset(ds,tr),fc.subset(ds,va)
        seed=utils.load_settings()["project"]["random_seed"]
        for s in range(3):
            m=fc.LSTMForecaster(seed=seed+s,**{k:par[k] for k in ("hidden","dropout","lr")})
            m.fit(train.seq,train.y.to_numpy(),val.seq,val.y.to_numpy(),S=train.static,Sv=val.static)
            rows.extend(dict(zone=z,seed=seed+s,**entry) for entry in m.history_)
    pd.DataFrame(rows).to_csv(TAB/"lstm_learning_curves.csv",index=False)


def provenance():
    paths=[ROOT/"data/cache/sst_satellite_monthly_1993_2024.nc",
           ROOT/"data/processed/sst_satellite_zones_monthly.csv",
           ROOT/"data/cache/study_area_static.nc",ROOT/"config/settings.yaml",
           ROOT/"scripts/paper1_forecasting.py",ROOT/"src/forecasting.py",
           ROOT/"scripts/paper1_robustness.py",ROOT/"papers/paper1/make_results_text.py",
           ROOT/"scripts/paper1_tables.py",ROOT/"scripts/paper1_figures.py",
           ROOT/"scripts/paper_figures_common.py",ROOT/"src/data_access.py",
           ROOT/"src/preprocessing.py",ROOT/"requirements-lock.txt",
           ROOT/"data/processed/paper1_test_predictions.csv"]
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    qc=json.loads((ROOT/"outputs/tables/satellite_validation_qc.json").read_text())
    info=dict(sha256=hashes, packages={p:importlib.metadata.version(p) for p in
        ("numpy","pandas","scipy","torch","xgboost","scikit-learn","shap","xarray","copernicusmarine")},
        satellite_configured_version=utils.load_settings()["validation"]["dataset_version"],
        satellite_cached_version=qc.get("satellite_provenance",{}).get("source_dataset_version","not recorded in original cache"),
        inference="exploratory; conditional on fitted models; bootstrap does not repeat fitting or selection",
        bootstrap=dict(draws=2000,block_months=[6,12],seed=SEED,shared_time_indices_across_zones=True))
    (TAB/"reproduction_manifest.json").write_text(json.dumps(info,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--prepare",action="store_true")
    args=parser.parse_args()
    if args.prepare:
        historical();disjoint()
    else:
        predictions=pd.read_csv(ROOT/"data/processed/paper1_test_predictions.csv",parse_dates=["target_time","origin"])
        uncertainty(predictions);hac_sensitivity(predictions);seasonality(predictions);coefficients();learning_curves();provenance()
    print("Supplementary checks complete",flush=True)
