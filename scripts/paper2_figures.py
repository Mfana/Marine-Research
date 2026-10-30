"""Figures for paper 2 from papers/paper2/tables and data/processed/paper2_*."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402

from paper_figures_common import CREDIT, ZONE_COLORS, ZONE_LABEL, study_area_map  # noqa: E402
from src import utils, visualisation  # noqa: E402

ROOT = utils.PROJECT_ROOT
TAB, FIG = ROOT / "papers/paper2/tables", ROOT / "papers/paper2/figures"
ZONES = list(ZONE_LABEL)
MLABEL = {"base_rate": "Base rate", "persistence": "Persistence", "logistic": "Logistic",
          "random_forest": "Random Forest", "xgboost": "XGBoost"}
MCOLOR = {"base_rate": "0.6", "persistence": "0.25", "logistic": "#2563eb", "random_forest": "#16a34a",
          "xgboost": "#ea580c"}


def example_event_panel():
    """Daily SST, climatology and threshold around the largest Benguela-nearshore event."""
    from src import mhw
    daily = pd.read_csv(ROOT / "data/cache/sst_satellite_daily_zones_1993_2024.csv", index_col=0, parse_dates=True)
    ev = pd.read_csv(TAB / "events.csv", parse_dates=["start", "end", "peak"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
    for ax, z in zip(axes, ("benguela_nearshore", "agulhas_offshore")):
        d, _ = mhw.detect(daily[z], baseline=("1993-01-01", "2022-12-31"))
        e = ev[ev.zone == z].sort_values("intensity_cumulative").iloc[-1]
        win = d.loc[e.start - pd.Timedelta(days=60):e.end + pd.Timedelta(days=60)]
        ax.plot(win.index, win.sst, color="k", lw=1, label="SST")
        ax.plot(win.index, win.seas, color="#2563eb", lw=1, label="Climatology")
        ax.plot(win.index, win.thresh, color="#16a34a", lw=1, ls="--", label="90th percentile")
        ax.fill_between(win.index, win.thresh, win.sst, where=win.in_mhw, color="#dc2626", alpha=0.5, label="MHW")
        ax.set(title=f"{ZONE_LABEL[z]}: largest event ({e.start:%b %Y}, {int(e.duration_days)} d)", ylabel="SST (°C)")
        ax.tick_params(axis="x", labelrotation=30)
    axes[0].legend(fontsize=7)
    visualisation.source_note(axes[1], CREDIT)
    fig.tight_layout()
    visualisation.save(fig, "fig_example_events", FIG)
    plt.close(fig)


def annual_mhw_days():
    a = pd.read_csv(TAB / "annual_metrics.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
    for ax, region in zip(axes, ("benguela", "agulhas")):
        for z in [z for z in ZONES if z.startswith(region)]:
            d = a[a.zone == z]
            ax.plot(d.year, d.mhw_days, marker="o", ms=2.5, color=ZONE_COLORS[z], label=ZONE_LABEL[z])
        ax.set(title=f"{region.title()}: annual MHW days", xlabel="Year")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("MHW days per year")
    visualisation.source_note(axes[1], CREDIT)
    fig.tight_layout()
    visualisation.save(fig, "fig_annual_mhw_days", FIG)
    plt.close(fig)


def grid_maps():
    g = xr.open_dataset(ROOT / "data/processed/paper2_mhw_grid.nc")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.3), sharey=True)
    m1 = g.mean_annual_mhw_days.plot(ax=axes[0], cmap="YlOrRd", add_colorbar=False)
    fig.colorbar(m1, ax=axes[0], label="MHW days per year")
    lim = float(np.nanpercentile(np.abs(g.trend_mhw_days_per_decade), 98))
    m2 = g.trend_mhw_days_per_decade.plot(ax=axes[1], cmap="RdBu_r", vmin=-lim, vmax=lim, add_colorbar=False)
    fig.colorbar(m2, ax=axes[1], label="Trend (days per year per decade)")
    axes[0].set(title="Mean annual MHW days, 1993–2024", xlabel="Longitude (°E)", ylabel="Latitude (°N)")
    axes[1].set(title="Trend in annual MHW days", xlabel="Longitude (°E)", ylabel="")
    for ax in axes:
        ax.axvline(20, color="k", ls="--", lw=0.6)
        ax.grid(False)
    visualisation.source_note(axes[1], CREDIT + " (0.25° daily)")
    fig.tight_layout()
    visualisation.save(fig, "fig_mhw_maps", FIG)
    plt.close(fig)


def unsupervised():
    u = pd.read_csv(TAB / "unsupervised_detection_test.csv")
    d = u.pivot_table(index="zone", columns="detector", values="AUPRC").loc[ZONES]
    cols = ["zscore_run5", "isolation_forest", "isolation_forest_warm", "autoencoder", "autoencoder_warm"]
    lab = {"zscore_run5": "z-score rule", "isolation_forest": "Isolation Forest", "isolation_forest_warm": "IF (warm-only)",
           "autoencoder": "Autoencoder", "autoencoder_warm": "AE (warm-only)"}
    base = u.groupby("zone")["base_rate"].first().loc[ZONES]
    fig, ax = plt.subplots(figsize=(9, 3.4))
    x = np.arange(len(ZONES))
    w = 0.15
    for i, c in enumerate(cols):
        ax.bar(x + (i - 2) * w, d[c], w, label=lab[c])
    ax.scatter(x, base, marker="_", s=400, color="k", label="Base rate (no skill)", zorder=3)
    ax.set_xticks(x, [ZONE_LABEL[z] for z in ZONES], rotation=20, ha="right", fontsize=8)
    ax.set(ylabel="Test AUPRC", title="Unsupervised detection of Hobday MHW days, 2020–2024")
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    visualisation.save(fig, "fig_unsupervised_detection", FIG)
    plt.close(fig)


def early_warning():
    e = pd.read_csv(TAB / "early_warning_test.csv")
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.4), sharex=True, sharey=True)
    for ax, z in zip(axes.ravel(), ZONES):
        d = e[e.zone == z]
        for m in ("persistence", "logistic", "random_forest", "xgboost"):
            dm = d[d.model == m].sort_values("lead_days")
            ax.plot(dm.lead_days, dm.AUPRC, marker="o", ms=3, color=MCOLOR[m], label=MLABEL[m],
                    ls="--" if m == "persistence" else "-")
        b = d[d.model == "base_rate"].sort_values("lead_days")
        ax.plot(b.lead_days, b.base_rate, color="0.6", ls=":", label="Base rate")
        ax.set_title(ZONE_LABEL[z], fontsize=9)
        ax.set_xticks([1, 3, 7, 14])
    for ax in axes[:, 0]:
        ax.set_ylabel("Test AUPRC")
    for ax in axes[-1]:
        ax.set_xlabel("Lead (days)")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    visualisation.save(fig, "fig_early_warning", FIG)
    plt.close(fig)


def shap_bar():
    s = pd.read_csv(TAB / "early_warning_shap.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, (z, k) in zip(axes, (("benguela_nearshore", 7), ("agulhas_offshore", 7))):
        d = s[(s.zone == z) & (s.lead == k)].sort_values("mean_abs_shap").tail(10)
        ax.barh(d.feature.str.replace("_", " "), d.mean_abs_shap, color="#ea580c")
        ax.set(title=f"{ZONE_LABEL[z]}, lead {k} d (XGBoost)", xlabel="mean |SHAP| (log-odds)")
        ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    visualisation.save(fig, "fig_early_warning_shap", FIG)
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    visualisation.apply_style()
    study_area_map(FIG)
    example_event_panel()
    annual_mhw_days()
    grid_maps()
    unsupervised()
    early_warning()
    shap_bar()
    print("figures:", sorted(p.name for p in FIG.glob("*.pdf")))


if __name__ == "__main__":
    main()
