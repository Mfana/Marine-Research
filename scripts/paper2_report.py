"""Paper 2: test-event counts, the extended forecasting table and the added figures, from result tables.

The manuscript prose is written separately from the same tables.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src import utils,visualisation
from paper_figures_common import ZONE_LABEL
from paper2_mhw import ZONES,LEADS

ROOT=utils.PROJECT_ROOT;PAPER=ROOT/"papers/paper2";TAB=PAPER/"tables";FIG=PAPER/"figures"


def event_counts():
    """Test-period MHW runs and positive one-day targets per zone (training-baseline labels)."""
    from src import mhw
    from paper2_mhw import train_labels_features
    probs=pd.read_csv(ROOT/"data/processed/paper2_ew_test_probs_extended.csv")
    tests=probs[probs.lead_days==1].groupby("zone").y.sum()
    daily=pd.read_csv(ROOT/"data/cache/sst_satellite_daily_zones_1993_2024.csv",index_col=0,parse_dates=True)
    lab,_,_=train_labels_features(daily)
    counts={z:len(mhw._runs(lab[z].loc["2020":].to_numpy().astype(bool))) for z in ZONES}
    pd.DataFrame({"zone":ZONES,"test_event_runs":[counts[z] for z in ZONES],
                  "test_positive_days":[tests[z] for z in ZONES]}).to_csv(TAB/"test_event_counts.csv",index=False)


def table():
    s=pd.read_csv(TAB/"forecast_scores_extended.csv").groupby(["model","lead_days"])[["AUPRC","BSS_vs_base_rate"]].mean()
    names={"threshold_distance":"TD score","univariate_logistic":"Univariate LR","state_logistic":"State LR",
        "logistic":"Multi LR","xgboost":"XGBoost","random_forest":"Balanced RF",
        "rf_unweighted":"Natural RF","random_forest_sigmoid":"Sigmoid RF","persistence":"Binary persistence"}
    lines=[r"\begin{tabular}{lrrrrrr}",r"\toprule",r"Model & AP 1 d & AP 3 d & AP 7 d & AP 14 d & BSS 7 d & BSS 14 d \\",r"\midrule"]
    for model,label in names.items():
        g=s.loc[model]
        cells=[f"{g.loc[k,'AUPRC']:.2f}" for k in LEADS]+[f"{g.loc[k,'BSS_vs_base_rate']:+.2f}" if np.isfinite(g.loc[k,'BSS_vs_base_rate']) else "--" for k in (7,14)]
        lines.append(label+" & "+" & ".join(cells)+r" \\")
    lines.extend([r"\bottomrule",r"\end{tabular}"])
    (TAB/"tab_early_warning.tex").write_text("\n".join(lines)+"\n")


def figures():
    visualisation.apply_style()
    u=pd.read_csv(TAB/"early_warning_uncertainty.csv")
    u=u[(u.zone=="mean")&(u.metric=="AUPRC")&(u.comparison=="logistic-threshold_distance")]
    fig,ax=plt.subplots(figsize=(8,3.8))
    for block,offset in ((30,-.24),(60,-.08),(180,.08),(365,.24)):
        d=u[u.block_days==block].set_index("lead_days").loc[list(LEADS)]
        x=np.arange(4)+offset
        ax.errorbar(x,d["diff"],yerr=[d["diff"]-d.lo,d.hi-d["diff"]],fmt="o",capsize=3,label=f"{block}-day blocks")
    ax.axhline(0,color=".5",lw=1);ax.set_xticks(range(4),[str(k) for k in LEADS]);ax.set(xlabel="Lead (days)",ylabel="AP difference",title="Multivariate logistic minus current threshold distance")
    ax.legend(fontsize=9,ncol=2);fig.tight_layout();visualisation.save(fig,"fig_forecast_differences",FIG);plt.close(fig)
    bins=pd.read_csv(TAB/"reliability_bins.csv");ci=pd.read_csv(TAB/"calibration_uncertainty.csv")
    fig,axes=plt.subplots(2,3,figsize=(10,5.7),sharex=True,sharey=True)
    for ax,z in zip(axes.ravel(),ZONES):
        g=bins[(bins.model=="logistic")&(bins.lead_days==14)&(bins.zone==z)]
        c=ci[(ci.block_days==30)&(ci.lead_days==14)&(ci.zone==z)]
        g=g.merge(c,on=["zone","lead_days","model","bin"])
        ax.plot([0,1],[0,1],color=".65",ls="--")
        ax.vlines(g.mean_prediction,g.lo,g.hi,color="#2563eb",lw=1)
        ax.scatter(g.mean_prediction,g.observed_fraction,s=10+np.sqrt(g["count"])*3,color="#2563eb")
        ax.set_title(ZONE_LABEL[z],fontsize=9);ax.set(xlim=(0,1),ylim=(0,1))
    for ax in axes[:,0]:ax.set_ylabel("Observed frequency")
    for ax in axes[-1]:ax.set_xlabel("Predicted probability")
    fig.tight_layout();visualisation.save(fig,"fig_reliability",FIG);plt.close(fig)


if __name__=="__main__":
    event_counts();table();figures()
