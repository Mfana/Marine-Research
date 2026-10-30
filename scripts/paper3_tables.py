"""LaTeX tables for paper 3, generated from the result CSVs (no hand-copied numbers)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src import utils  # noqa: E402

TAB = utils.PROJECT_ROOT / "papers/paper3/tables"
DOM = {"domain": "Whole domain", "benguela": "Benguela", "agulhas": "Agulhas"}
ML = {"climatology": "Climatology", "persistence": "Persistence", "damped_persistence": "Damped persistence",
      "linear_per_pixel": "Per-pixel linear", "convlstm": "ConvLSTM", "unet": "U-Net"}
REG = ["all_sea", "benguela_coastal", "benguela_offshore", "agulhas_coastal", "agulhas_offshore"]
RL = {"all_sea": "All", "benguela_coastal": "Benguela coastal", "benguela_offshore": "Benguela offshore",
      "agulhas_coastal": "Agulhas coastal", "agulhas_offshore": "Agulhas offshore"}


def eof_table():
    v = pd.read_csv(TAB / "eof_variance.csv")
    t = pd.read_csv(TAB / "eof_pc_trends.csv").set_index("pc")
    lines = [r"\begin{tabular}{llrrr}", r"\toprule",
             r"Domain & Mode & Variance (\%) & Distinct & PC trend (s.d./decade) \\", r"\midrule"]
    for d in DOM:
        for _, r in v[v.domain == d].iterrows():
            tr = t.loc[f"{d}_pc{int(r['mode'])}"]
            star = "$^{*}$" if tr.p_value < 0.05 else ""
            lines.append(f"{DOM[d] if r['mode'] == 1 else ''} & {int(r['mode'])} & "
                         f"{100 * r.variance_frac:.1f} $\\pm$ {100 * r.north_err:.1f} & "
                         f"{'yes' if r.separated_from_next else 'no'} & "
                         f"{tr.trend_sd_per_decade:+.2f} $\\pm$ {tr.ci95:.2f}{star} \\\\")
        if d != "agulhas":
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "tab_eof.tex").write_text("\n".join(lines) + "\n")


def dl_table():
    f = TAB / "field_forecast_test.csv"
    if not f.exists():
        return
    r = pd.read_csv(f)
    r = r[~r.model.str.contains("_seed")]
    sd = pd.read_csv(TAB / "dl_seed_spread.csv")
    lines = [r"\begin{tabular}{ll" + "r" * len(REG) + "}", r"\toprule",
             "Lead & Model & " + " & ".join(RL[g] for g in REG) + r" \\", r"\midrule"]
    for lead in (1, 2, 4):
        d = r[r.lead_weeks == lead].pivot_table(index="model", columns="region", values="RMSE")
        best = d.loc[[m for m in d.index if m not in ("climatology", "persistence")], REG].min()
        for m in ML:
            cells = []
            for g in REG:
                v = d.loc[m, g]
                s = f"{v:.3f}"
                if m in ("convlstm", "unet"):
                    x = sd[(sd.base == m) & (sd.lead_weeks == lead) & (sd.region == g)]
                    s += f"$_{{\\pm{x['std'].iloc[0]:.3f}}}$" if len(x) else ""
                cells.append(rf"\textbf{{{s}}}" if (m not in ("climatology", "persistence") and v == best[g]) else s)
            lines.append(f"{lead if m == 'climatology' else ''} & {ML[m]} & " + " & ".join(cells) + r" \\")
        if lead != 4:
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "tab_dl.tex").write_text("\n".join(lines) + "\n")


def main():
    eof_table()
    dl_table()
    for f in sorted(TAB.glob("tab_*.tex")):
        print(f"== {f.name}\n{f.read_text()}")


if __name__ == "__main__":
    main()
