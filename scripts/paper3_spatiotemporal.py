"""Paper 3 — spatiotemporal modelling of SST variability in the Benguela and Agulhas systems.

Part A (spatial analysis; monthly ESA CCI / C3S SST, 0.05° cache coarsened to 0.1°, 1993–2024):
  EOFs of monthly anomalies (whole domain, Benguela < 20°E, Agulhas > 20°E) with North et al.
  (1982) errors using an autocorrelation-adjusted sample size; spatial e-folding decorrelation
  lengths at nearshore, shelf and offshore reference sites.
Part B (deep learning; daily 0.25° cache aggregated to weekly means): forecast weekly anomaly
  fields 1, 2 and 4 weeks ahead from the previous 8 weeks with a ConvLSTM and a U-Net-style
  CNN (3 seeds each), against anomaly persistence, damped persistence and per-pixel linear
  regression. Weekly climatology from training years only. Splits by the latest target week:
  train ≤ 2015, validation 2016–2019, test 2020–2024 (evaluated once).

Outputs: papers/paper3/{tables,figures}/, data/processed/paper3_*.

Usage:  .venv/bin/python scripts/paper3_spatiotemporal.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402

from src import data_access, preprocessing, spatiotemporal as stm, utils  # noqa: E402

ROOT = utils.PROJECT_ROOT
OUT = ROOT / "papers" / "paper3"
TAB, FIG = OUT / "tables", OUT / "figures"
CACHE = ROOT / "data" / "cache"
TRAIN_END, VAL_START, VAL_END, TEST_START, TEST_END = (
    "2015-12-31", "2016-01-01", "2019-12-31", "2020-01-01", "2024-12-31")
T_IN, LEADS, SEEDS = 8, (1, 2, 4), (0, 1, 2)
SITES = {  # name: (lon, lat) — nearest sea cell is used
    "benguela_nearshore_stHelenaBay": (17.9, -32.6),
    "benguela_offshore": (16.5, -33.0),
    "agulhas_bank_shelf": (22.0, -35.0),
    "agulhas_nearshore_portAlfred": (27.0, -33.75),
    "agulhas_current_core": (30.5, -31.5),
    "agulhas_offshore_south": (28.0, -35.5),
}


# ---- Part A -------------------------------------------------------------------

def part_a(s, log):
    m = data_access.load_monthly_grid("satellite", s)
    m = m.coarsen(latitude=2, longitude=2, boundary="trim").mean()          # 0.05° -> 0.1°
    clim = m.groupby("time.month").mean("time")
    anom = (m.groupby("time.month") - clim).drop_vars("month")
    lat, lon = anom.latitude.values, anom.longitude.values
    A = anom.values.astype("float64")
    times = pd.DatetimeIndex(anom.time.values)
    rows, pcs_out, pats = [], {}, {}
    for dom, sel in (("domain", lon >= -999), ("benguela", lon <= 20.0), ("agulhas", lon > 20.0)):
        sub = A[:, :, sel]
        r = stm.eof(sub, lat, n_modes=4)
        n_eff = stm.effective_sample_size(r["pcs"][:, 0])
        north = r["frac"] * np.sqrt(2.0 / n_eff)
        for k in range(4):
            rows.append({"domain": dom, "mode": k + 1, "variance_frac": r["frac"][k], "north_err": north[k],
                         "separated_from_next": bool(k < 3 and r["frac"][k] - r["frac"][k + 1] > north[k]),
                         "n_eff": n_eff})
            pcs_out[f"{dom}_pc{k + 1}"] = r["pcs"][:, k]
        full = np.full((4, len(lat), len(lon)), np.nan)
        full[:, :, sel] = r["patterns"]
        pats[dom] = full
    eof_tab = pd.DataFrame(rows)
    eof_tab.round(4).to_csv(TAB / "eof_variance.csv", index=False)
    pcs = pd.DataFrame(pcs_out, index=times)
    pcs.to_csv(ROOT / "data" / "processed" / "paper3_eof_pcs.csv")
    # PC trends and cross-system correlation.
    pc_tr = []
    for c in pcs:
        t = preprocessing.hac_trend(pcs[c])
        pc_tr.append({"pc": c, "trend_sd_per_decade": t["trend_degC_per_decade"], "ci95": t["ci95"],
                      "p_value": t["p_value"]})
    pd.DataFrame(pc_tr).round(4).to_csv(TAB / "eof_pc_trends.csv", index=False)
    corr = pcs[[f"benguela_pc{k}" for k in (1, 2)] + [f"agulhas_pc{k}" for k in (1, 2)]].corr()
    corr.round(3).to_csv(TAB / "eof_pc_correlations.csv")
    xr.Dataset({f"{d}_patterns": (("mode", "latitude", "longitude"), p) for d, p in pats.items()},
               coords={"mode": [1, 2, 3, 4], "latitude": lat, "longitude": lon}).to_netcdf(
        ROOT / "data" / "processed" / "paper3_eof_patterns.nc")

    # Decorrelation lengths.
    sea = ~np.isnan(A).any(0)
    pts, names = [], []
    LAT, LON = np.meshgrid(lat, lon, indexing="ij")
    for name, (x, y) in SITES.items():
        d = np.where(sea, np.hypot((LON - x) * np.cos(np.deg2rad(y)), LAT - y), np.inf)
        iy, ix = np.unravel_index(np.argmin(d), d.shape)
        pts.append((int(iy), int(ix)))
        names.append(name)
    dec = stm.decorrelation_length(A, lat, lon, pts, max_km=800, bin_km=20)
    drows, curves = [], {}
    for name, p in zip(names, pts):
        drows.append({"site": name, "lon": float(lon[p[1]]), "lat": float(lat[p[0]]),
                      "efold_km": dec[p]["efold_km"]})
        curves[name] = pd.Series(dec[p]["mean_r"], index=dec[p]["distance_km"])
    pd.DataFrame(drows).round(2).to_csv(TAB / "decorrelation_lengths.csv", index=False)
    pd.DataFrame(curves).to_csv(ROOT / "data" / "processed" / "paper3_decorrelation_curves.csv")
    log.info("Part A done")


# ---- Part B -------------------------------------------------------------------

def weekly_fields():
    ds = xr.open_dataset(CACHE / "sst_satellite_daily_025deg_1993_2024.nc")
    sst = ds["sst"].load()
    n = (sst.sizes["time"] // 7) * 7
    v = sst.values[:n].reshape(n // 7, 7, *sst.shape[1:])
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            wk = np.nanmean(v, 1)
    starts = pd.DatetimeIndex(sst.time.values[:n:7])
    return wk, starts, sst.latitude.values, sst.longitude.values


def coarse_regions(lat, lon, s):
    """Region masks on the 0.25° grid: coastal = any of the cell's 0.05° pixels within the
    nearshore distance of the coast (from the satellite land mask)."""
    maps = xr.open_dataset(ROOT / "data" / "processed" / "trend_maps.nc")
    dist = maps["sat_dist_km"].values
    near = np.nan_to_num(dist <= s["zones"]["nearshore_km"], nan=False)
    ny, nx = len(lat), len(lon)
    coastal = near[:ny * 5, :nx * 5].reshape(ny, 5, nx, 5).any(axis=(1, 3))
    west = (lon <= 20.0)[None, :] * np.ones((ny, 1), bool)
    return {"benguela_coastal": coastal & west, "benguela_offshore": ~coastal & west,
            "agulhas_coastal": coastal & ~west, "agulhas_offshore": ~coastal & ~west}


def part_b(s, log):
    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    wk, starts, lat, lon = weekly_fields()
    sea = ~np.isnan(wk).any(0)
    woy = np.minimum((starts.dayofyear.to_numpy() - 1) // 7, 51)
    train_w = starts <= pd.Timestamp(TRAIN_END)
    clim = np.stack([np.nanmean(wk[train_w & (woy == k)], 0) for k in range(52)])
    clim = (np.roll(clim, 1, 0) + clim + np.roll(clim, -1, 0)) / 3          # 3-week circular smoothing
    anom = wk - clim[woy]
    anom[:, ~sea] = 0.0
    anom = np.nan_to_num(anom)                                                  # rare gaps -> 0 anomaly

    N, H, W = anom.shape
    L = max(LEADS)
    origins = np.arange(T_IN - 1, N - L)
    # Two channels per input week: anomaly and sea mask. Seasonal terms are omitted because
    # the inputs are already anomalies; keeping the array small matters on an 8 GB machine.
    sea_ch = sea.astype("float32")
    X = np.zeros((len(origins), T_IN, 2, H, W), dtype="float32")
    Y = np.zeros((len(origins), len(LEADS), H, W), dtype="float32")
    for i, t in enumerate(origins):
        X[i, :, 0] = anom[t - T_IN + 1:t + 1]
        X[i, :, 1] = sea_ch
        Y[i] = anom[[t + l for l in LEADS]]
    last_target = starts[origins + L] + pd.Timedelta(days=6)
    tr = last_target <= pd.Timestamp(TRAIN_END)
    va = (starts[origins + 1] >= pd.Timestamp(VAL_START)) & (last_target <= pd.Timestamp(VAL_END))
    te = (starts[origins + 1] >= pd.Timestamp(TEST_START)) & (last_target <= pd.Timestamp(TEST_END) + pd.Timedelta(days=6))
    trva = tr | va
    log.info("Weekly samples: train %d, val %d, test %d; grid %dx%d, sea %d", tr.sum(), va.sum(), te.sum(), H, W, sea.sum())

    preds = {"climatology": np.zeros_like(Y[te]),
             "persistence": np.repeat(X[te][:, -1, 0][:, None], len(LEADS), 1)}
    # Baselines and networks all learn from the training period only (networks early-stop on
    # validation), so every model sees the same data. Damped persistence: per-pixel, per-lead
    # coefficient.
    x0 = X[tr][:, -1, 0]
    phi = np.stack([(x0 * Y[tr][:, j]).sum(0) / np.maximum((x0 ** 2).sum(0), 1e-9) for j in range(len(LEADS))])
    preds["damped_persistence"] = phi[None] * X[te][:, -1, 0][:, None]
    # Per-pixel linear regression on the last T_IN weeks (ridge, alpha=1).
    Xl = X[tr][:, :, 0].reshape(tr.sum(), T_IN, -1)                           # (n, T, P)
    Xt = X[te][:, :, 0].reshape(te.sum(), T_IN, -1)
    lin = np.zeros((te.sum(), len(LEADS), H * W), dtype="float32")
    for j in range(len(LEADS)):
        Yl = Y[tr][:, j].reshape(tr.sum(), -1)
        A = np.einsum("ntp,nsp->pts", Xl, Xl) + np.eye(T_IN)[None]
        b = np.einsum("ntp,np->pt", Xl, Yl)
        coef = np.linalg.solve(A, b[..., None])[..., 0]                         # (P, T)
        lin[:, j] = np.einsum("ntp,pt->np", Xt, coef)
    preds["linear_per_pixel"] = lin.reshape(te.sum(), len(LEADS), H, W)

    ConvLSTMNet, UNetLite = stm.build_models()
    train_info = []
    # Width 16 keeps training to minutes per model on an M1 GPU (32 channels took ~20 s/epoch).
    for name, ctor in (("convlstm", lambda: ConvLSTMNet(2, len(LEADS), (16, 16))),
                       ("unet", lambda: UNetLite(2 * T_IN, len(LEADS), 16))):
        seed_preds = []
        for seed in SEEDS:
            t0 = time.time()
            # Train on 1993–2015 with early stopping on 2016–2019; the selected model predicts
            # the test set once (no refit, to bound GPU time on an 8 GB M1).
            final, best, ep = stm.train_field_model(ctor(), X[tr], Y[tr], X[va], Y[va], sea, epochs=100,
                                                    patience=10, seed=seed, device=device, log=log)
            seed_preds.append(stm.predict_fields(final, X[te], device=device))
            train_info.append({"model": name, "seed": seed, "val_mse": best, "best_epoch": ep,
                               "params": sum(p.numel() for p in final.parameters()),
                               "seconds": round(time.time() - t0)})
            log.info("%s seed %d: %ds", name, seed, time.time() - t0)
        for seed, p in zip(SEEDS, seed_preds):
            preds[f"{name}_seed{seed}"] = p
        preds[name] = np.mean(seed_preds, 0)
    pd.DataFrame(train_info).to_csv(TAB / "dl_training.csv", index=False)

    # Evaluation.
    w = np.cos(np.deg2rad(lat))[:, None] * np.ones(W)
    regions = {"all_sea": sea, **{k: v & sea for k, v in coarse_regions(lat, lon, s).items()}}
    Yte = Y[te]
    rows, maps = [], {}
    for name, P in preds.items():
        for j, lead in enumerate(LEADS):
            err2 = (P[:, j] - Yte[:, j]) ** 2
            for reg, m in regions.items():
                ww = w * m
                rmse = float(np.sqrt((err2 * ww).sum() / (ww.sum() * len(err2))))
                # Temporal ACC averaged over cells in the region.
                yy, pp = Yte[:, j][:, m], P[:, j][:, m]
                with np.errstate(all="ignore"):
                    acc = np.nanmean([np.corrcoef(yy[:, c], pp[:, c])[0, 1] if pp[:, c].std() > 0 else np.nan
                                      for c in range(yy.shape[1])])
                rows.append({"model": name, "lead_weeks": lead, "region": reg, "RMSE": rmse, "ACC": acc,
                             "n_weeks": len(err2)})
            if not name.endswith(tuple(f"seed{s_}" for s_ in SEEDS)):
                mse_map = err2.mean(0)
                maps[f"mse_{name}_lead{lead}"] = (("latitude", "longitude"), np.where(sea, mse_map, np.nan))
    res = pd.DataFrame(rows)
    pers = res[res.model == "persistence"].set_index(["lead_weeks", "region"])["RMSE"]
    clim_r = res[res.model == "climatology"].set_index(["lead_weeks", "region"])["RMSE"]
    res["MSSS_vs_persistence"] = 1 - (res.RMSE / res.set_index(["lead_weeks", "region"]).index.map(pers)) ** 2
    res["MSSS_vs_climatology"] = 1 - (res.RMSE / res.set_index(["lead_weeks", "region"]).index.map(clim_r)) ** 2
    res.round(4).to_csv(TAB / "field_forecast_test.csv", index=False)
    xr.Dataset(maps, coords={"latitude": lat, "longitude": lon}).to_netcdf(
        ROOT / "data" / "processed" / "paper3_mse_maps.nc")
    # Spread across seeds.
    sd = (res[res.model.str.contains("_seed")].assign(base=lambda d: d.model.str.replace(r"_seed\d", "", regex=True))
          .groupby(["base", "lead_weeks", "region"])["RMSE"].agg(["mean", "std"]).reset_index())
    sd.round(4).to_csv(TAB / "dl_seed_spread.csv", index=False)
    # Example test week for figures.
    k = int(np.argmax(np.abs(Yte[:, 0]).sum((1, 2))))
    ex = {f"{n}": (("lead", "latitude", "longitude"), np.where(sea, preds[n][k], np.nan))
          for n in ("persistence", "damped_persistence", "linear_per_pixel", "convlstm", "unet")}
    ex["truth"] = (("lead", "latitude", "longitude"), np.where(sea, Yte[k], np.nan))
    xr.Dataset(ex, coords={"lead": list(LEADS), "latitude": lat, "longitude": lon},
               attrs={"origin_week": str(starts[origins[te]][k].date())}).to_netcdf(
        ROOT / "data" / "processed" / "paper3_example_week.nc")
    log.info("Part B done")
    return {"device": device, "n_train": int(tr.sum()), "n_val": int(va.sum()), "n_test": int(te.sum())}


def main() -> None:
    import torch
    s = utils.load_settings()
    utils.set_seed(s["project"]["random_seed"])
    log = utils.setup_logging(s, "marine.paper3")
    TAB.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    # `--part a` runs only the EOF/decorrelation analysis (monthly cache, offline);
    # `--part b` only the deep-learning experiments (needs the daily 0.25° cache).
    part = sys.argv[sys.argv.index("--part") + 1] if "--part" in sys.argv else "ab"
    if "a" in part:
        part_a(s, log)
    if "b" not in part:
        log.info("Paper 3 part A finished in %ss", round(time.time() - t0))
        return
    info = part_b(s, log)
    meta = {"seconds": round(time.time() - t0), "torch": torch.__version__, **info, "T_in_weeks": T_IN,
            "leads_weeks": LEADS, "seeds": SEEDS,
            "splits": {"train": f"..{TRAIN_END}", "validation": f"{VAL_START}..{VAL_END}",
                       "test": f"{TEST_START}..{TEST_END}"}}
    (TAB / "run_meta.json").write_text(json.dumps(meta, indent=2))
    log.info("Paper 3 finished in %ss", meta["seconds"])


if __name__ == "__main__":
    main()
