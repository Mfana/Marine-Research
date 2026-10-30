"""Publication-quality figures (PNG + PDF)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .utils import PROJECT_ROOT

FIG_DIR = PROJECT_ROOT / "outputs" / "figures"

STYLE = {
    "figure.dpi": 120, "savefig.dpi": 300, "font.size": 10, "axes.titlesize": 11,
    "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "legend.frameon": False,
}


def apply_style() -> None:
    plt.rcParams.update(STYLE)


def save(fig, name: str, folder: Path = FIG_DIR) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    paths = [folder / f"{name}.{ext}" for ext in ("png", "pdf")]
    # Source notes are excluded from tight_layout; include them explicitly in the saved bounds.
    extra = fig.get_default_bbox_extra_artists() + [t for ax in fig.axes for t in ax.texts]
    for p in paths:
        fig.savefig(p, bbox_inches="tight", bbox_extra_artists=extra)
    return paths


def source_note(ax, text: str = "Data: E.U. Copernicus Marine Service (doi:10.48670/moi-00021)") -> None:
    # Placed just below the measured x-axis decorations (ticks, multi-line labels, x-label),
    # and kept out of tight_layout so a long note does not push neighbouring axes apart.
    fig = ax.figure
    renderer = fig.canvas.get_renderer()
    axis_bottom = ax.xaxis.get_tightbbox(renderer).y0
    offset = (ax.get_window_extent(renderer).y0 - axis_bottom) * 72 / fig.dpi + 3
    note = ax.annotate(text, xy=(1, 0), xycoords="axes fraction", xytext=(0, -offset),
                       textcoords="offset points", ha="right", va="top", fontsize=7, color="0.4")
    note.set_in_layout(False)


def plot_timeseries(s: pd.Series, title: str, ylabel: str = "Temperature (°C)", ax=None):
    apply_style()
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(7, 3))
    ax.plot(s.index, s.values, lw=1.2, marker="o" if len(s) < 40 else None, ms=3)
    ax.set(title=title, ylabel=ylabel, xlabel="")
    source_note(ax)
    fig.tight_layout()
    return fig, ax


def plot_seasonality(clim: pd.Series, title: str, ylabel: str = "Temperature (°C)", ax=None):
    apply_style()
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(5, 3))
    ax.plot(clim.index, clim.values, marker="o", lw=1.2)
    ax.set(title=title, ylabel=ylabel, xlabel="Month", xticks=range(1, 13))
    source_note(ax)
    fig.tight_layout()
    return fig, ax


def plot_map(da, title: str, cbar_label: str = "Temperature (°C)", ax=None):
    apply_style()
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(6, 3.5))
    m = da.plot(ax=ax, cmap="RdYlBu_r", add_colorbar=True, cbar_kwargs={"label": cbar_label})
    ax.set(title=title, xlabel="Longitude (°E)", ylabel="Latitude (°N)")
    ax.grid(False)
    fig.tight_layout()
    return fig, ax
