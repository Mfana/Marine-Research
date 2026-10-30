"""Paper 1 — explainable multi-horizon SST anomaly forecasting along the South African coast.

Data: ESA CCI / C3S satellite SST zone series (monthly, 1993–2024), six zones.
Design: anomalies vs a training-period climatology; horizons 1, 3, 6 months; feature sets
'univariate' (own 12 lags) and 'multizone' (+ current anomalies of the other zones);
models climatology, anomaly persistence, ridge, Random Forest, XGBoost, LSTM.
Splits by target time: train 1994–2015, validation 2016–2019 (tuning), test 2020–2024
(evaluated once, after refitting on train+validation with the tuned settings).

Outputs: papers/paper1/{tables,figures}/ and data/processed/paper1_test_predictions.csv

Usage:  .venv/bin/python scripts/paper1_forecasting.py
"""

import itertools
import json
import os
import sys

# torch and xgboost each bundle an OpenMP runtime; on macOS the two deadlock when both
# run multi-threaded in one process. One OpenMP thread avoids this (data are small).
os.environ.setdefault("OMP_NUM_THREADS", "1")
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import data_access, forecasting as fc, utils  # noqa: E402

ROOT = utils.PROJECT_ROOT
OUT = ROOT / "papers" / "paper1"
TAB, FIG = OUT / "tables", OUT / "figures"

TRAIN_END, VAL_START, VAL_END, TEST_START, TEST_END = (
    "2015-12-31", "2016-01-01", "2019-12-31", "2020-01-01", "2024-12-31")
HORIZONS = (1, 3, 6)
FEATURE_SETS = ("univariate", "multizone")
N_SEEDS = 5
GRIDS = {
    "ridge": [{"alpha": a} for a in (0.1, 1.0, 10.0, 100.0)],
    "random_forest": [dict(max_depth=d, min_samples_leaf=l, max_features=f)
                      for d, l, f in itertools.product((3, 6, None), (1, 5, 10), (0.33, 1.0))],
    "xgboost": [dict(max_depth=d, learning_rate=lr, n_estimators=n, min_child_weight=w)
                for d, lr, n, w in itertools.product((2, 3, 4), (0.03, 0.1), (200, 600), (1, 5))],
    # Comparable tuning effort to the tree models (12 configurations).
    "lstm": [dict(hidden=h, dropout=d, lr=lr)
             for h, d, lr in itertools.product((8, 16, 32), (0.1, 0.3), (1e-3, 3e-3))],
}


def mse(y, p):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def tune_and_test(ds, kind, seed, log):
    tr = fc.subset(ds, fc.split_mask(ds, None, TRAIN_END))
    va = fc.subset(ds, fc.split_mask(ds, VAL_START, VAL_END))
    trva = fc.subset(ds, fc.split_mask(ds, None, VAL_END))
    te = fc.subset(ds, fc.split_mask(ds, TEST_START, TEST_END))
    best, best_params, best_extra = np.inf, None, {}
    for params in GRIDS[kind]:
        if kind == "lstm":
            m = fc.LSTMForecaster(seed=seed, **params).fit(tr.seq, tr.y.to_numpy(), va.seq, va.y.to_numpy())
            v = mse(va.y, m.predict(va.seq))
            extra = {"epochs": m.best_epoch}
        else:
            m = fc.fit_tree(kind, dict(params), tr.X, tr.y, seed)
            v = mse(va.y, m.predict(va.X))
            extra = {}
        if v < best:
            best, best_params, best_extra = v, params, extra
    # Refit on train+validation with tuned settings; evaluate the test set once.
    if kind == "lstm":
        preds, vals = [], []
        for s in range(N_SEEDS):
            m = fc.LSTMForecaster(seed=seed + s, max_epochs=max(best_extra["epochs"], 20), **best_params)
            m.fit(trva.seq, trva.y.to_numpy())
            preds.append(m.predict(te.seq))
        preds = np.array(preds)
        seed_rmse = [float(np.sqrt(mse(te.y, p))) for p in preds]
        return preds.mean(0), best_params | best_extra, {"val_mse": best, "seed_rmse": seed_rmse}, None
    m = fc.fit_tree(kind, dict(best_params), trva.X, trva.y, seed)
    return m.predict(te.X), best_params, {"val_mse": best}, m


def main() -> None:
    import torch
    torch.set_num_threads(1)
    s = utils.load_settings()
    seed = s["project"]["random_seed"]
    utils.set_seed(seed)
    log = utils.setup_logging(s, "marine.paper1")
    TAB.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    df = data_access.load_zone_series("satellite", s)
    zones = ["benguela_nearshore", "benguela_shelf", "benguela_offshore",
             "agulhas_nearshore", "agulhas_shelf", "agulhas_offshore"]
    df = df[zones].loc[:TEST_END]
    clim = fc.train_climatology(df, TRAIN_END)
    anom = fc.to_anomalies(df, clim)

    rows, preds_all, dm_rows, tuned, tree_models = [], [], [], [], {}
    t_all = time.time()
    for z, h, fs in itertools.product(zones, HORIZONS, FEATURE_SETS):
        t0 = time.time()
        ds = fc.make_dataset(anom, z, h, n_lags=12, cross_zone=(fs == "multizone"))
        te = fc.subset(ds, fc.split_mask(ds, TEST_START, TEST_END))
        y = te.y.to_numpy()
        base = {"climatology": np.zeros_like(y), "persistence": te.persistence.to_numpy()}
        clim_mse, pers_mse = mse(y, base["climatology"]), mse(y, base["persistence"])
        preds = dict(base)
        for kind in ("ridge", "random_forest", "xgboost", "lstm"):
            p, params, info, model = tune_and_test(ds, kind, seed, log)
            preds[kind] = p
            tuned.append({"zone": z, "h": h, "features": fs, "model": kind, "params": json.dumps(params, default=str),
                          **{k: v for k, v in info.items() if k != "seed_rmse"},
                          "lstm_seed_rmse_mean": np.mean(info["seed_rmse"]) if "seed_rmse" in info else np.nan,
                          "lstm_seed_rmse_sd": np.std(info["seed_rmse"]) if "seed_rmse" in info else np.nan})
            if kind == "xgboost":
                tree_models[(z, h, fs)] = (model, te.X)
        for name, p in preds.items():
            rows.append({"zone": z, "h": h, "features": fs, "model": name,
                         **fc.scores(y, p, clim_mse, pers_mse)})
            preds_all.append(pd.DataFrame({"zone": z, "h": h, "features": fs, "model": name,
                                           "origin": te.origin, "target_time": te.target_time,
                                           "y_anom": y, "pred_anom": p,
                                           "clim": clim.loc[te.target_time.month, z].to_numpy()}))
        for name in ("ridge", "random_forest", "xgboost", "lstm"):
            for ref in ("persistence", "climatology"):
                st, pv = fc.diebold_mariano(y, preds[name], preds[ref], h)
                dm_rows.append({"zone": z, "h": h, "features": fs, "model": name, "reference": ref,
                                "dm_stat": st, "p_value": pv})
        st, pv = fc.diebold_mariano(y, preds["xgboost"], preds["lstm"], h)
        dm_rows.append({"zone": z, "h": h, "features": fs, "model": "xgboost", "reference": "lstm",
                        "dm_stat": st, "p_value": pv})
        log.info("%s h=%d %s done in %.0fs", z, h, fs, time.time() - t0)

    res = pd.DataFrame(rows)
    # Baselines do not depend on the feature set; keep one copy.
    res = res[~((res.features == "univariate") & res.model.isin(["climatology", "persistence"]))]
    res.loc[res.model.isin(["climatology", "persistence"]), "features"] = "-"
    res.round(4).to_csv(TAB / "test_scores.csv", index=False)
    pd.DataFrame(tuned).to_csv(TAB / "tuned_hyperparameters.csv", index=False)

    from src.preprocessing import fdr_adjust
    dm = pd.DataFrame(dm_rows)
    dm["significant_fdr"], dm["p_fdr"] = fdr_adjust(dm["p_value"].fillna(1.0))
    dm.round(4).to_csv(TAB / "diebold_mariano.csv", index=False)
    pd.concat(preds_all).to_csv(ROOT / "data" / "processed" / "paper1_test_predictions.csv", index=False)

    # SHAP for the tuned multizone XGBoost models on the test period.
    import shap
    shap_rows = []
    shap_store = {}
    for (z, h, fs), (model, X) in tree_models.items():
        if fs != "multizone":
            continue
        sv = shap.TreeExplainer(model)(X)
        imp = np.abs(sv.values).mean(0)
        for f, v in zip(X.columns, imp):
            shap_rows.append({"zone": z, "h": h, "feature": f, "mean_abs_shap": v})
        shap_store[(z, h)] = (sv, X)
    pd.DataFrame(shap_rows).round(5).to_csv(TAB / "shap_importance.csv", index=False)
    import pickle
    with open(ROOT / "data" / "processed" / "paper1_shap.pkl", "wb") as f:
        pickle.dump({k: (v[0].values, v[0].base_values, v[1]) for k, v in shap_store.items()}, f)

    meta = {"seconds": round(time.time() - t_all), "zones": zones, "horizons": HORIZONS,
            "splits": {"train": f"1994..{TRAIN_END}", "validation": f"{VAL_START}..{VAL_END}",
                       "test": f"{TEST_START}..{TEST_END}"},
            "n_seeds_lstm": N_SEEDS, "grids": {k: len(v) for k, v in GRIDS.items()},
            "torch": torch.__version__}
    (TAB / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    log.info("Paper 1 experiments finished in %ss", meta["seconds"])


if __name__ == "__main__":
    main()
