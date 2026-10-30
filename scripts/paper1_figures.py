"""Figures for paper 1 from papers/paper1/tables and data/processed/paper1_*."""

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from paper_figures_common import CREDIT, ZONE_LABEL, study_area_map  # noqa: E402
from src import utils, visualisation  # noqa: E402

ROOT = utils.PROJECT_ROOT
TAB, FIG = ROOT / "papers/paper1/tables", ROOT / "papers/paper1/figures"
ZONES = list(ZONE_LABEL)
MODELS = ["persistence", "ridge", "random_forest", "xgboost", "lstm"]
MLABEL = {"climatology": "Climatology", "persistence": "Persistence", "ridge": "Ridge",
          "random_forest": "Random Forest", "xgboost": "XGBoost", "lstm": "LSTM"}
MCOLOR = {"climatology": "0.5", "persistence": "0.25", "ridge": "#2563eb", "random_forest": "#16a34a",
          "xgboost": "#ea580c", "lstm": "#9333ea"}


def skill_heatmaps(r):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
    for ax, h in zip(axes, (1, 3, 6)):
        d = r[(r.h == h) & r.features.isin(["-", "multizone"])].pivot_table(index="zone", columns="model",
                                                                               values="MSSS_clim")
        d = d.loc[ZONES, ["persistence", "ridge", "random_forest", "xgboost", "lstm"]]
        im = ax.imshow(d.values, cmap="RdBu", vmin=-0.6, vmax=0.6, aspect="auto")
        for i in range(d.shape[0]):
            for j in range(d.shape[1]):
                ax.text(j, i, f"{d.iloc[i, j]:+.2f}", ha="center", va="center", fontsize=7)
        ax.set_xticks(range(d.shape[1]), [MLABEL[c] for c in d.columns], rotation=35, ha="right", fontsize=8)
        ax.set_yticks(range(len(ZONES)), [ZONE_LABEL[z] for z in ZONES], fontsize=8)
        ax.set_title(f"Lead {h} month{'s' if h > 1 else ''}")
        ax.grid(False)
    fig.colorbar(im, ax=axes, label="MSSS vs climatology", shrink=0.85)
    visualisation.save(fig, "fig_skill_heatmap", FIG)
    plt.close(fig)


def rmse_vs_lead(r):
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.4), sharex=True)
    for ax, z in zip(axes.ravel(), ZONES):
        d = r[(r.zone == z) & r.features.isin(["-", "multizone"])]
        for m in ["climatology"] + MODELS:
            dm = d[d.model == m].sort_values("h")
            ax.plot(dm.h, dm.RMSE, marker="o", ms=3, color=MCOLOR[m], label=MLABEL[m],
                    ls="--" if m in ("climatology", "persistence") else "-")
        ax.set_title(ZONE_LABEL[z], fontsize=9)
        ax.set_xticks([1, 3, 6])
    for ax in axes[:, 0]:
        ax.set_ylabel("Test RMSE (°C)")
    for ax in axes[-1]:
        ax.set_xlabel("Lead (months)")
    axes[0, 0].legend(fontsize=7, ncol=2)
    visualisation.source_note(axes[-1, -1], CREDIT)
    fig.tight_layout()
    visualisation.save(fig, "fig_rmse_vs_lead", FIG)
    plt.close(fig)


def forecast_timeseries(preds):
    fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    for ax, z in zip(axes, ("benguela_nearshore", "agulhas_offshore")):
        d = preds[(preds.zone == z) & (preds.h == 1) & preds.features.eq("multizone")]
        obs = d[d.model == "ridge"].set_index("target_time")
        ax.plot(obs.index, obs.y_anom + obs.clim, color="k", lw=1.6, label="Observed")
        for m in ("persistence", "ridge", "xgboost", "lstm"):
            dm = d[d.model == m].set_index("target_time")
            ax.plot(dm.index, dm.pred_anom + dm.clim, color=MCOLOR[m], lw=1, label=MLABEL[m],
                    ls="--" if m == "persistence" else "-")
        ax.set(title=f"{ZONE_LABEL[z]}: 1-month-ahead forecasts, test period", ylabel="SST (°C)")
    axes[0].legend(fontsize=7, ncol=5)
    visualisation.source_note(axes[1], CREDIT)
    fig.tight_layout()
    visualisation.save(fig, "fig_test_forecasts", FIG)
    plt.close(fig)


def shap_figures():
    import shap
    with open(ROOT / "data/processed/paper1_shap.pkl", "rb") as f:
        store = pickle.load(f)
    imp = pd.read_csv(TAB / "shap_importance.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, z in zip(axes, ("benguela_nearshore", "agulhas_offshore")):
        d = imp[(imp.zone == z) & (imp.h == 1)].sort_values("mean_abs_shap").tail(10)
        ax.barh(d.feature.str.replace("_", " "), d.mean_abs_shap, color="#ea580c")
        ax.set(title=f"{ZONE_LABEL[z]} (XGBoost, lead 1)", xlabel="mean |SHAP| (°C)")
        ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    visualisation.save(fig, "fig_shap_importance", FIG)
    plt.close(fig)
    vals, base, X = store[("benguela_nearshore", 1)]
    plt.figure()
    shap.summary_plot(vals, X, max_display=10, show=False, plot_size=(7, 4))
    visualisation.save(plt.gcf(), "fig_shap_beeswarm_benguela_nearshore", FIG)
    plt.close("all")


def ablation(r):
    d = r[r.features.isin(["univariate", "multizone"])].groupby(["model", "features", "h"])["MSSS_pers"].mean().unstack("features")
    fig, axes = plt.subplots(1, 3, figsize=(10, 3), sharey=True)
    for ax, h in zip(axes, (1, 3, 6)):
        dd = d.xs(h, level="h").loc[["ridge", "random_forest", "xgboost", "lstm"]]
        x = np.arange(len(dd))
        ax.bar(x - 0.2, dd["univariate"], 0.4, label="Own history", color="#94a3b8")
        ax.bar(x + 0.2, dd["multizone"], 0.4, label="+ other zones", color="#1e3a8a")
        ax.set_xticks(x, [MLABEL[m] for m in dd.index], rotation=30, ha="right", fontsize=8)
        ax.set_title(f"Lead {h}")
        ax.axhline(0, color="0.3", lw=0.8)
    axes[0].set_ylabel("Mean MSSS vs persistence")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    visualisation.save(fig, "fig_feature_ablation", FIG)
    plt.close(fig)


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
    print("figures:", sorted(p.name for p in FIG.glob("*.pdf")))


if __name__ == "__main__":
    main()
