"""LaTeX tables for paper 2, generated from the result CSVs (no hand-copied numbers)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src import utils  # noqa: E402

TAB = utils.PROJECT_ROOT / "papers/paper2/tables"
ZONES = ["benguela_nearshore", "benguela_shelf", "benguela_offshore",
         "agulhas_nearshore", "agulhas_shelf", "agulhas_offshore"]
ZL = {z: z.replace("_", " ").title() for z in ZONES}


def summary_table():
    s = pd.read_csv(TAB / "zone_summary.csv", index_col=0).loc[ZONES]
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"Zone & Events & Days/yr & Mean dur. & Max dur. & Mean peak \\",
             r" & & & (d) & (d) & (\si{\celsius}) \\", r"\midrule"]
    for z in ZONES:
        r = s.loc[z]
        lines.append(f"{ZL[z]} & {int(r.n_events)} & {r.mhw_days_per_year:.1f} & "
                     f"{r.duration_mean:.1f} & {int(r.duration_max)} & {r.intensity_max_mean:.2f}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "tab_summary.tex").write_text("\n".join(lines) + "\n")


def unsup_table():
    f = TAB / "unsupervised_detection_test.csv"
    if not f.exists():
        return
    u = pd.read_csv(f)
    dets = ["zscore_run5", "isolation_forest", "isolation_forest_warm", "autoencoder", "autoencoder_warm"]
    lab = {"zscore_run5": "z-score rule", "isolation_forest": "Isolation Forest", "isolation_forest_warm": "IF (warm)",
           "autoencoder": "Autoencoder", "autoencoder_warm": "AE (warm)"}
    a = u.pivot_table(index="zone", columns="detector", values="AUPRC").loc[ZONES, dets]
    f1 = u.pivot_table(index="zone", columns="detector", values="F1").loc[ZONES, dets]
    base = u.groupby("zone")["base_rate"].first().loc[ZONES]
    lines = [r"\begin{tabular}{lr" + "r" * len(dets) + "}", r"\toprule",
             r"Zone & Base rate & " + " & ".join(lab[d] for d in dets) + r" \\", r"\midrule"]
    for z in ZONES:
        best = a.loc[z].max()
        cells = [(rf"\textbf{{{a.loc[z, d]:.2f}}}" if a.loc[z, d] == best else f"{a.loc[z, d]:.2f}") + f" ({f1.loc[z, d]:.2f})"
                 for d in dets]
        lines.append(f"{ZL[z]} & {base[z]:.2f} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "tab_unsupervised.tex").write_text("\n".join(lines) + "\n")


def ew_table():
    f = TAB / "early_warning_test.csv"
    if not f.exists():
        return
    e = pd.read_csv(f)
    models = ["persistence", "logistic", "random_forest", "xgboost"]
    lab = {"persistence": "Persist.", "logistic": "Logistic", "random_forest": "RF", "xgboost": "XGBoost"}
    lines = [r"\begin{tabular}{lr" + "rr" * 4 + "}", r"\toprule",
             r"Model & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{Lead {k} d}}" for k in (1, 3, 7, 14)) + r" \\",
             r" & " + " & ".join(["AUPRC & BSS"] * 4) + r" \\", r"\midrule"]
    lines[0] = r"\begin{tabular}{l" + "rr" * 4 + "}"
    for m in models:
        d = e[e.model == m].groupby("lead_days")[["AUPRC", "BSS_vs_base_rate"]].mean()
        lines.append(f"{lab[m]} & " + " & ".join(f"{d.loc[k, 'AUPRC']:.2f} & {d.loc[k, 'BSS_vs_base_rate']:+.2f}"
                                                for k in (1, 3, 7, 14)) + r" \\")
    b = e[e.model == "base_rate"].groupby("lead_days")["base_rate"].mean()
    lines.append(r"\midrule")
    lines.append("Base rate & " + " & ".join(f"{b.loc[k]:.2f} & --" for k in (1, 3, 7, 14)) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "tab_early_warning.tex").write_text("\n".join(lines) + "\n")


def main():
    summary_table()
    unsup_table()
    ew_table()
    for f in sorted(TAB.glob("tab_*.tex")):
        print(f"== {f.name}\n{f.read_text()}")


if __name__ == "__main__":
    main()
