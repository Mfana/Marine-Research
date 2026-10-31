"""Audit regenerated paper-1 outputs, including chronology and numeric provenance."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
TAB=ROOT/"papers/paper1/tables"


def main():
    scores=pd.read_csv(TAB/"test_scores.csv")
    predictions=pd.read_csv(ROOT/"data/processed/paper1_test_predictions.csv",parse_dates=["origin","target_time"])
    assert len(scores)==198
    assert (scores.n==60).all()
    max_error=0
    for _,r in scores.iterrows():
        fs="multizone" if r.features=="-" else r.features
        d=predictions[(predictions.zone==r.zone)&(predictions.h==r.h)&
                      (predictions.features==fs)&(predictions.model==r.model)]
        assert len(d)==60 and d.target_time.nunique()==60
        rmse=np.sqrt(np.mean((d.pred_anom-d.y_anom)**2))
        max_error=max(max_error,abs(rmse-r.RMSE))
        assert abs(rmse-r.RMSE)<=0.000051
        assert (d.target_time==d.origin+pd.DateOffset(months=int(r.h))).all()
    tuned=pd.read_csv(TAB/"tuned_hyperparameters.csv")
    for _,r in tuned.iterrows():
        p=json.loads(r.params)
        assert p["validation_end"]<=p["first_test_origin"]
        assert p["refit_end"]<=p["first_test_origin"]
        if r.model=="lstm":
            assert len(p["epochs"])==3 and all(e>=1 for e in p["epochs"])
    dm=pd.read_csv(TAB/"diebold_mariano.csv")
    assert len(dm)==576 and dm.p_value.between(0,1).all()
    assert (dm.p_fdr>=dm.p_value-0.00011).all()
    assert not dm.duplicated(["zone","h","features","model","reference"]).any()
    for name in ("paired_uncertainty.csv","aggregate_uncertainty.csv","ablation_uncertainty.csv"):
        d=pd.read_csv(TAB/name)
        assert np.isfinite(d[["lower","upper"]].to_numpy()).all()
        assert (d.lower<=d.upper).all()
    overlap=pd.read_csv(TAB/"zone_overlap.csv")
    assert overlap.fraction_target_area_shared.between(0,1.000001).all()
    print(f"Verified {len(scores)} score rows, 576 comparisons, 144 selection records; maximum RMSE rounding difference {max_error:.8f} C")


if __name__=="__main__":main()
