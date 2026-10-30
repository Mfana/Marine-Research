"""Figures for paper 3 from papers/paper3/tables and data/processed/paper3_*."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402

from paper_figures_common import CREDIT, study_area_map  # noqa: E402
from src import utils, visualisation  # noqa: E402

ROOT = utils.PROJECT_ROOT
TAB, FIG = ROOT / "papers/paper3/tables", ROOT / "papers/paper3/figures"
P = ROOT / "data/processed"
MLABEL = {"climatology": "Climatology", "persistence": "Persistence", "damped_persistence": "Damped persistence",
          "linear_per_pixel": "Per-pixel linear", "convlstm": "ConvLSTM", "unet": "U-Net"}
SHORT = {"linear_per_pixel": "linear", "persistence": "persistence", "unet": "U-Net"}
MCOLOR = {"climatology": "0.6", "persistence": "0.25", "damped_persistence": "#64748b",
          "linear_per_pixel": "#2563eb", "convlstm": "#9333ea", "unet": "#ea580c"}


def eof_patterns():
    pats = xr.open_dataset(P / "paper3_eof_patterns.nc")
    var = pd.read_csv(TAB / "eof_variance.csv")
    pcs = pd.read_csv(P / "paper3_eof_pcs.csv", index_col=0, parse_dates=True)
    fig, axes = plt.subplots(2, 3, figsize=(13, 5.6), gridspec_kw={"height_ratios": [1.2, 1]})
    for j, k in enumerate((1, 2, 3)):
        ax = axes[0, j]
        p = pats["domain_patterns"].sel(mode=k)
        lim = float(np.nanpercentile(np.abs(p), 99))
        im = p.plot(ax=ax, cmap="RdBu_r", vmin=-lim, vmax=lim, add_colorbar=False)
        fig.colorbar(im, ax=ax, label="°C per s.d.", shrink=0.85)
        f = var[(var.domain == "domain") & (var["mode"] == k)].iloc[0]
        ax.set(title=f"EOF{k}: {100 * f.variance_frac:.1f}% ± {100 * f.north_err:.1f}%", xlabel="", ylabel="")
        ax.axvline(20, color="k", ls="--", lw=0.6)
        ax.grid(False)
        ax2 = axes[1, j]
        pc = pcs[f"domain_pc{k}"]
        ax2.plot(pc.index, pc, color="0.6", lw=0.6)
        ax2.plot(pc.rolling(12, center=True).mean(), color="k", lw=1.4)
        ax2.axhline(0, color="0.3", lw=0.6)
        ax2.set(title=f"PC{k} (12-month mean in black)", ylabel="s.d.")
    visualisation.source_note(axes[1, 2], CREDIT)
    fig.tight_layout()
    visualisation.save(fig, "fig_eof_domain", FIG)
    plt.close(fig)


def decorrelation():
    c = pd.read_csv(P / "paper3_decorrelation_curves.csv", index_col=0)
    L = pd.read_csv(TAB / "decorrelation_lengths.csv").set_index("site")
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for site in c.columns:
        ax.plot(c.index, c[site], marker="o", ms=2, label=f"{site.replace('_', ' ')} (L = {L.loc[site, 'efold_km']:.0f} km)")
    ax.axhline(np.exp(-1), color="k", ls=":", lw=0.8)
    ax.set(xlabel="Distance (km)", ylabel="Correlation of monthly anomalies",
           title="Spatial decorrelation of SST anomalies")
    ax.legend(fontsize=6.5)
    visualisation.source_note(ax, CREDIT)
    fig.tight_layout()
    visualisation.save(fig, "fig_decorrelation", FIG)
    plt.close(fig)


def skill_by_lead():
    r = pd.read_csv(TAB / "field_forecast_test.csv")
    r = r[~r.model.str.contains("_seed")]
    regions = ["all_sea", "benguela_coastal", "benguela_offshore", "agulhas_coastal", "agulhas_offshore"]
    fig, axes = plt.subplots(1, 5, figsize=(14, 3.2), sharey=True)
    for ax, reg in zip(axes, regions):
        d = r[r.region == reg]
        for m in MLABEL:
            dm = d[d.model == m].sort_values("lead_weeks")
            ax.plot(dm.lead_weeks, dm.RMSE, marker="o", ms=3, color=MCOLOR[m], label=MLABEL[m],
                    ls="--" if m in ("climatology", "persistence") else "-")
        ax.set_title(reg.replace("_", " "), fontsize=9)
        ax.set_xticks([1, 2, 4])
        ax.set_xlabel("Lead (weeks)")
    axes[0].set_ylabel("Test RMSE (°C)")
    axes[0].legend(fontsize=6.5)
    fig.tight_layout()
    visualisation.save(fig, "fig_field_skill", FIG)
    plt.close(fig)


def skill_maps():
    m = xr.open_dataset(P / "paper3_mse_maps.nc")
    fig, axes = plt.subplots(2, 3, figsize=(13, 5.2), sharex=True, sharey=True)
    # Row 1: the best statistical baseline against persistence; row 2: U-Net against that baseline,
    # which shows directly where deep learning adds skill.
    for i, (model, ref, vmax) in enumerate((("linear_per_pixel", "persistence", 0.6), ("unet", "linear_per_pixel", 0.3))):
        for j, lead in enumerate((1, 2, 4)):
            ax = axes[i, j]
            ss = 1 - m[f"mse_{model}_lead{lead}"] / m[f"mse_{ref}_lead{lead}"]
            im = ss.plot(ax=ax, cmap="RdBu", vmin=-vmax, vmax=vmax, add_colorbar=False)
            ax.set(title=f"{SHORT[model]} vs {SHORT[ref]}, {lead} wk", xlabel="", ylabel="")
            ax.axvline(20, color="k", ls="--", lw=0.6)
            ax.grid(False)
        fig.colorbar(im, ax=axes[i], label="MSSS", shrink=0.9)
    visualisation.save(fig, "fig_skill_maps", FIG)
    plt.close(fig)


def example_week():
    e = xr.open_dataset(P / "paper3_example_week.nc")
    names = ["truth", "persistence", "linear_per_pixel", "convlstm", "unet"]
    fig, axes = plt.subplots(1, 5, figsize=(15, 2.6), sharey=True)
    lim = float(np.nanpercentile(np.abs(e["truth"].sel(lead=1)), 99))
    for ax, n in zip(axes, names):
        im = e[n].sel(lead=1).plot(ax=ax, cmap="RdBu_r", vmin=-lim, vmax=lim, add_colorbar=False)
        ax.set(title="Observed" if n == "truth" else MLABEL[n], xlabel="", ylabel="")
        ax.grid(False)
    fig.colorbar(im, ax=axes, label="Anomaly (°C)", shrink=0.85)
    fig.suptitle(f"1-week-ahead anomaly field, origin week {e.attrs['origin_week']}", y=1.04, fontsize=10)
    visualisation.save(fig, "fig_example_week", FIG)
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    visualisation.apply_style()
    study_area_map(FIG)
    eof_patterns()
    decorrelation()
    skill_by_lead()
    skill_maps()
    example_week()
    print("figures:", sorted(p.name for p in FIG.glob("*.pdf")))


if __name__ == "__main__":
    main()
