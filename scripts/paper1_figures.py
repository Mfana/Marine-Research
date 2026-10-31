"""Figures for paper 1 from papers/paper1/tables and data/processed/paper1_*."""

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import TwoSlopeNorm

from paper_figures_common import CREDIT, ZONE_LABEL, study_area_map  # noqa: E402
from src import utils, visualisation  # noqa: E402

ROOT = utils.PROJECT_ROOT
TAB, FIG = ROOT / "papers/paper1/tables", ROOT / "papers/paper1/figures"
ZONES = list(ZONE_LABEL)
MODELS = ["persistence", "ridge", "random_forest", "xgboost", "lstm"]
MLABEL = {"climatology": "Climatology", "persistence": "Persistence", "ridge": "Ridge",
          "damped_persistence": "Damped persistence", "random_forest": "Random Forest", "xgboost": "XGBoost", "lstm": "LSTM"}
MCOLOR = {"climatology": "0.5", "persistence": "0.25", "ridge": "#2563eb", "random_forest": "#16a34a",
          "damped_persistence": "#0891b2", "xgboost": "#ea580c", "lstm": "#9333ea"}


def skill_heatmaps(r):
    fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
    for ax, h in zip(axes, (1, 3, 6)):
        d = r[(r.h == h) & r.features.isin(["-", "multizone"])].pivot_table(index="zone", columns="model",
                                                                               values="MSSS_clim")
        d = d.loc[ZONES, ["persistence", "ridge", "random_forest", "xgboost", "lstm"]]
        norm=TwoSlopeNorm(vmin=min(-0.01,r.MSSS_clim.min()),vcenter=0,vmax=max(.01,r.MSSS_clim.max()))
        im = ax.imshow(d.values, cmap="RdBu", norm=norm, aspect="auto")
        for i in range(d.shape[0]):
            for j in range(d.shape[1]):
                shade=norm(d.iloc[i,j])
                ax.text(j, i, f"{d.iloc[i, j]:+.2f}", ha="center", va="center", fontsize=10,
                        color="white" if shade<.2 or shade>.8 else "black")
        ax.set_xticks(range(d.shape[1]), [MLABEL[c] for c in d.columns], rotation=25, ha="right", fontsize=10)
        ax.set_yticks(range(len(ZONES)), [ZONE_LABEL[z] for z in ZONES], fontsize=10)
        ax.set_title(f"Lead {h} month{'s' if h > 1 else ''}")
        ax.grid(False)
    fig.subplots_adjust(left=.27,right=.88,bottom=.13,hspace=.35)
    fig.colorbar(im, ax=axes, label="MSSS vs climatology", shrink=0.85)
    visualisation.save(fig, "fig_skill_heatmap", FIG)
    plt.close(fig)


def rmse_vs_lead(r):
    fig, axes = plt.subplots(2, 3, figsize=(7.5, 5.5), sharex=True)
    for ax, z in zip(axes.ravel(), ZONES):
        d = r[(r.zone == z) & r.features.isin(["-", "multizone"])]
        for m in ["climatology", "damped_persistence"] + MODELS:
            dm = d[d.model == m].sort_values("h")
            ax.plot(dm.h, dm.RMSE, marker="o", ms=3, color=MCOLOR[m], label=MLABEL[m],
                    ls="--" if m in ("climatology", "persistence") else "-")
        ax.set_title(ZONE_LABEL[z], fontsize=10)
        ax.set_xticks([1, 3, 6])
    for ax in axes[:, 0]:
        ax.set_ylabel("Test RMSE (°C)")
    for ax in axes[-1]:
        ax.set_xlabel("Lead (months)")
    fig.legend(*axes[0,0].get_legend_handles_labels(),fontsize=9,ncol=3,loc="upper center")
    visualisation.source_note(axes[-1, -1], CREDIT)
    fig.tight_layout(rect=[0,0,1,.85])
    visualisation.save(fig, "fig_rmse_vs_lead", FIG)
    plt.close(fig)


def forecast_timeseries(preds):
    fig, axes = plt.subplots(2, 1, figsize=(7, 5.5), sharex=True)
    for ax, z in zip(axes, ("benguela_nearshore", "agulhas_offshore")):
        d = preds[(preds.zone == z) & (preds.h == 1) & preds.features.eq("multizone")]
        obs = d[d.model == "ridge"].set_index("target_time")
        ax.plot(obs.index, obs.y_anom, color="k", lw=1.6, label="Observed")
        ax.axhline(0,color=MCOLOR["climatology"],ls=":",label="Climatology")
        for m in ("damped_persistence", "ridge", "xgboost", "lstm"):
            dm = d[d.model == m].set_index("target_time")
            ax.plot(dm.index, dm.pred_anom, color=MCOLOR[m], lw=1, label=MLABEL[m],
                    ls="--" if m == "persistence" else "-")
        ax.set(title=f"{ZONE_LABEL[z]}: 1-month-ahead retrospective forecasts", ylabel="SST anomaly (°C)")
    axes[0].legend(fontsize=9, ncol=3)
    visualisation.source_note(axes[1], CREDIT)
    fig.tight_layout()
    visualisation.save(fig, "fig_test_forecasts", FIG)
    plt.close(fig)


def shap_figures():
    import shap
    with open(ROOT / "data/processed/paper1_shap.pkl", "rb") as f:
        store = pickle.load(f)
    imp = pd.read_csv(TAB / "shap_importance.csv")
    fig, axes = plt.subplots(2, 1, figsize=(7, 7))
    for ax, z in zip(axes, ("benguela_nearshore", "agulhas_offshore")):
        d = imp[(imp.zone == z) & (imp.h == 1)].sort_values("mean_abs_shap").tail(10)
        ax.barh(d.feature.str.replace("_", " "), d.mean_abs_shap, color="#ea580c")
        ax.set(title=f"{ZONE_LABEL[z]} (XGBoost, lead 1)", xlabel="mean |SHAP| (°C)")
        ax.tick_params(axis="y", labelsize=10)
    fig.tight_layout()
    visualisation.save(fig, "fig_shap_importance", FIG)
    plt.close(fig)
    vals, base, X = store[("benguela_nearshore", 1)]
    plt.figure()
    shap.summary_plot(vals, X, max_display=10, show=False, plot_size=(7, 4))
    visualisation.save(plt.gcf(), "fig_shap_beeswarm_benguela_nearshore", FIG)
    plt.close("all")


def ablation(r):
    d = r[r.features.isin(["univariate", "multizone"])].groupby(["model", "features", "h"])["MSSS_clim"].mean().unstack("features")
    fig, axes = plt.subplots(1, 3, figsize=(7.8, 3.4), sharey=True)
    for ax, h in zip(axes, (1, 3, 6)):
        dd = d.xs(h, level="h").loc[["ridge", "random_forest", "xgboost", "lstm"]]
        x = np.arange(len(dd))
        ax.bar(x - 0.2, dd["univariate"], 0.4, label="Own history", color="#94a3b8")
        ax.bar(x + 0.2, dd["multizone"], 0.4, label="+ other zones", color="#1e3a8a")
        ax.set_xticks(x, [MLABEL[m] for m in dd.index], rotation=35, ha="right", fontsize=10)
        ax.set_title(f"Lead {h}")
        ax.axhline(0, color="0.3", lw=0.8)
    axes[0].set_ylabel("Mean MSSS vs climatology")
    axes[0].legend(fontsize=9)
    fig.tight_layout()
    visualisation.save(fig, "fig_feature_ablation", FIG)
    plt.close(fig)


def supplementary_figures():
    d=pd.read_csv(TAB / "lstm_learning_curves.csv")
    fig,axes=plt.subplots(2,1,figsize=(7,6),sharey=True)
    for ax,(z,g) in zip(axes,d.groupby("zone",sort=False)):
        for (seed,s),color in zip(g.groupby("seed"),("#2563eb","#ea580c","#16a34a")):
            ax.plot(s.epoch,s.training_mse_standardised,color=color,ls="--",alpha=.7,label=f"Train seed {seed}")
            ax.plot(s.epoch,s.validation_mse_standardised,color=color,label=f"Val. seed {seed}")
        ax.set(title=ZONE_LABEL[z],xlabel="Epoch",ylabel="Standardised MSE")
    axes[0].legend(fontsize=9,ncol=2)
    fig.tight_layout();visualisation.save(fig,"fig_lstm_learning_curves",FIG);plt.close(fig)

    d=pd.read_csv(TAB/"ablation_uncertainty.csv")
    d=d[(d.h==1)&(d.block_months==12)&(d.zone!="mean")]
    fig,axes=plt.subplots(2,2,figsize=(7.8,6.5),sharex=True)
    for ax,m in zip(axes.ravel(),("ridge","random_forest","xgboost","lstm")):
        g=d[d.model==m].set_index("zone").loc[ZONES]
        # Percentile intervals need not contain the point estimate, so draw the
        # interval separately instead of forcing nonnegative errorbar distances.
        y=np.arange(len(g))
        ax.hlines(y,g.lower,g.upper,color=MCOLOR[m],lw=1.5)
        ax.plot(g.delta_MSSS,y,"o",color=MCOLOR[m],ms=4)
        ax.axvline(0,color="0.4",ls="--",lw=.7)
        ax.set_yticks(y,[ZONE_LABEL[z] for z in ZONES],fontsize=10)
        ax.set(title=MLABEL[m],xlabel="Change in MSSS against climatology")
    fig.tight_layout();visualisation.save(fig,"fig_ablation_intervals",FIG);plt.close(fig)

    d=pd.read_csv(TAB/"ridge_coefficients.csv")
    d=d[(d.h==1)&(d.features=="multizone")]
    fig,axes=plt.subplots(2,1,figsize=(7,6.5))
    for ax,z in zip(axes,("benguela_nearshore","agulhas_offshore")):
        g=d[d.zone==z].pivot(index="feature",columns="fit",values="coefficient")
        keep=g.refit.abs().nlargest(10).index
        g=g.loc[keep].iloc[::-1];y=np.arange(len(g))
        ax.plot(g.initial,y,"o",label="Initial fit",color="#94a3b8",ms=5)
        ax.plot(g.refit,y,"s",label="Final refit",color="#2563eb",ms=4)
        ax.set_yticks(y,[f.replace('_',' ') for f in g.index],fontsize=10)
        ax.axvline(0,color="0.4",ls="--",lw=.7)
        ax.set(title=ZONE_LABEL[z],xlabel="Coefficient in original feature units")
    axes[0].legend(fontsize=9)
    fig.tight_layout();visualisation.save(fig,"fig_ridge_coefficients",FIG);plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    visualisation.apply_style()
    r = pd.read_csv(TAB / "test_scores.csv")
    preds = pd.read_csv(ROOT / "data/processed/paper1_test_predictions.csv", parse_dates=["origin", "target_time"])
    study_area_map(FIG)
    skill_heatmaps(r)
    rmse_vs_lead(r)
    forecast_timeseries(preds)
    shap_figures()
    ablation(r)
    supplementary_figures()
    print("figures:", sorted(p.name for p in FIG.glob("*.pdf")))


if __name__ == "__main__":
    main()
