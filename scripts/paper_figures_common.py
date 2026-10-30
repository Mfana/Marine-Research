"""Figure helpers shared by the three papers (study-area map with zones)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from src import data_access, utils, visualisation  # noqa: E402

ZONE_COLORS = {"benguela_nearshore": "#0b2559", "benguela_shelf": "#2b6cb0", "benguela_offshore": "#90cdf4",
               "agulhas_nearshore": "#7c2d12", "agulhas_shelf": "#dd6b20", "agulhas_offshore": "#fbd38d"}
ZONE_LABEL = {z: z.replace("_", " ").title() for z in ZONE_COLORS}
CREDIT = "Data: ESA CCI/C3S SST via Copernicus Marine"


def study_area_map(out_dir: Path, name: str = "fig_study_area") -> None:
    """Mean satellite SST (left) and zone definitions (right)."""
    from build_satellite_validation import zone_masks_on
    s = utils.load_settings()
    sat = data_access.load_monthly_grid("satellite", s)
    static = data_access.open_static(s)
    sea = sat.notnull().any("time")
    masks = zone_masks_on(sea, static["deptho"], s)
    visualisation.apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
    m = sat.mean("time").plot(ax=axes[0], cmap="RdYlBu_r", add_colorbar=False)
    fig.colorbar(m, ax=axes[0], label="Mean SST 1993–2024 (°C)", shrink=0.9)
    static["deptho"].plot.contour(ax=axes[0], levels=[200], colors="k", linewidths=0.5)
    axes[0].set(title="Mean SST and 200 m isobath", xlabel="Longitude (°E)", ylabel="Latitude (°N)")
    zid = np.full(sea.shape, np.nan)
    for i, (k, mk) in enumerate(masks.items()):
        order = list(ZONE_COLORS).index(k)
        zid = np.where(mk.values & np.isnan(zid), order, zid)
    # Nearshore overrides shelf/offshore for display (zones overlap by definition).
    for k in ("benguela_nearshore", "agulhas_nearshore"):
        zid = np.where(masks[k].values, list(ZONE_COLORS).index(k), zid)
    axes[1].pcolormesh(sat.longitude, sat.latitude, zid, cmap=ListedColormap(list(ZONE_COLORS.values())),
                       vmin=-0.5, vmax=5.5, shading="auto")
    axes[1].axvline(20, color="k", ls="--", lw=0.8)
    axes[1].set(title="Zones (nearshore ≤ 50 km overlaps shelf)", xlabel="Longitude (°E)")
    axes[1].legend(handles=[Patch(color=c, label=ZONE_LABEL[k]) for k, c in ZONE_COLORS.items()],
                   fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5))
    for ax in axes:
        ax.grid(False)
    visualisation.source_note(axes[1], CREDIT)
    fig.tight_layout()
    visualisation.save(fig, name, out_dir)
    plt.close(fig)
