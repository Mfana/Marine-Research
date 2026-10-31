"""LaTeX tables for paper 1, generated from the result CSVs (no hand-copied numbers)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src import utils  # noqa: E402

ROOT = utils.PROJECT_ROOT
TAB = ROOT / "papers/paper1/tables"
ZONES = ["benguela_nearshore", "benguela_shelf", "benguela_offshore",
         "agulhas_nearshore", "agulhas_shelf", "agulhas_offshore"]
ZL = {z: z.replace("_", " ").title() for z in ZONES}
ML = {"climatology": "Climatology", "persistence": "Persistence", "ridge": "Ridge",
      "damped_persistence": "Damped", "random_forest": "RF", "xgboost": "XGBoost", "lstm": "LSTM"}


def zone_table():
    sat = pd.read_csv(ROOT / "data/processed/sst_satellite_zones_monthly.csv", index_col=0, parse_dates=True)
    qc = json.load(open(ROOT / "outputs/tables/satellite_validation_qc.json"))
    r = pd.read_csv(TAB / "test_scores.csv")
    sd = r[(r.model == "climatology") & (r.h == 1)].set_index("zone")["RMSE"]
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Zone & Cells (0.05\textdegree) & Mean SST (\si{\celsius}) & Anomaly s.d. (\si{\celsius}) & Evaluation anomaly RMS (\si{\celsius}) \\",
             r"\midrule"]
    for z in ZONES:
        s = sat[z]
        clim = s.loc[:"2014"].groupby(s.loc[:"2014"].index.month).mean()
        anom = s - clim.reindex(s.index.month).set_axis(s.index)
        lines.append(f"{ZL[z]} & {qc['satellite']['zone_cells'][z]:,} & {s.mean():.2f} & {anom.std():.2f} & {sd[z]:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "tab_zones.tex").write_text("\n".join(lines) + "\n")


def lead1_table():
    r = pd.read_csv(TAB / "test_scores.csv")
    d = r[(r.h == 1) & r.features.isin(["-", "multizone"])]
    models = ["climatology", "persistence", "damped_persistence", "ridge", "random_forest", "xgboost", "lstm"]
    rm = d.pivot_table(index="zone", columns="model", values="RMSE").loc[ZONES, models]
    lines = [r"\begin{tabular}{l" + "r" * len(models) + "}", r"\toprule",
             "Zone & " + " & ".join(ML[m] for m in models) + r" \\", r"\midrule"]
    for z in ZONES:
        row = rm.loc[z]
        best = row[["ridge", "random_forest", "xgboost", "lstm"]].round(3).min()
        cells = [(rf"\textbf{{{v:.3f}}}" if (m in ("ridge", "random_forest", "xgboost", "lstm") and round(v,3) == best) else f"{v:.3f}")
                 for m, v in row.items()]
        lines.append(f"{ZL[z]} & " + " & ".join(cells) + r" \\")
    ms = d.groupby("model")["MSSS_clim"].mean()
    mp = d.groupby("model")["MSSS_pers"].mean()
    lines.append(r"\midrule")
    lines.append(r"Mean MSSS$_\mathrm{clim}$ & " + " & ".join(f"{ms[m]:+.2f}" for m in models) + r" \\")
    lines.append(r"Mean MSSS$_\mathrm{pers}$ & " + " & ".join(f"{mp[m]:+.2f}" for m in models) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "tab_lead1.tex").write_text("\n".join(lines) + "\n")


def lead_summary_table():
    r = pd.read_csv(TAB / "test_scores.csv")
    rows = []
    for h in (1, 3, 6):
        for fs in ("univariate", "multizone"):
            d = r[(r.h == h) & (r.features == fs)]
            rows.append((h, fs, d.groupby("model")["MSSS_clim"].mean()))
    base = r[r.features == "-"].groupby(["h", "model"])["MSSS_clim"].mean()
    models = ["ridge", "random_forest", "xgboost", "lstm"]
    lines = [r"\begin{tabular}{llrrrrr}", r"\toprule",
             r"Lead & Inputs & Persistence & " + " & ".join(ML[m] for m in models) + r" \\", r"\midrule"]
    for h, fs, s in rows:
        p = base.loc[(h, "persistence")]
        lines.append(f"{h} & {'Own history' if fs == 'univariate' else '+ other zones'} & {p:+.2f} & "
                     + " & ".join(f"{s[m]:+.2f}" for m in models) + r" \\")
        if fs == "multizone" and h != 6:
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "tab_lead_summary.tex").write_text("\n".join(lines) + "\n")


def dm_table():
    dm = pd.read_csv(TAB / "diebold_mariano.csv")
    d = dm[dm.features == "multizone"]
    lines = [r"\begin{tabular}{llrrr}", r"\toprule",
             r"Model & Reference & Lead 1 & Lead 3 & Lead 6 \\", r"\midrule"]
    for ref in ("persistence", "climatology", "damped_persistence"):
        for m in ("ridge", "random_forest", "xgboost", "lstm"):
            x = d[(d.model == m) & (d.reference == ref)]
            cells = []
            for h in (1, 3, 6):
                xh = x[x.h == h]
                better = int((xh.dm_stat < 0).sum())
                sig = int((xh.significant_fdr & (xh.dm_stat < 0)).sum())
                cells.append(f"{better}/6 ({sig})")
            lines.append(f"{ML[m]} & {ML[ref]} & " + " & ".join(cells) + r" \\")
        lines.append(r"\addlinespace")
    x = d[(d.model == "xgboost") & (d.reference == "lstm")]
    cells = [f"{int((x[x.h == h].dm_stat < 0).sum())}/6 ({int((x[x.h == h].significant_fdr & (x[x.h == h].dm_stat < 0)).sum())})"
             for h in (1, 3, 6)]
    lines.append("XGBoost & LSTM & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "tab_dm.tex").write_text("\n".join(lines) + "\n")


def main():
    zone_table()
    lead1_table()
    lead_summary_table()
    dm_table()
    for f in sorted(TAB.glob("tab_*.tex")):
        print(f"== {f.name}\n{f.read_text()}")


if __name__ == "__main__":
    main()
