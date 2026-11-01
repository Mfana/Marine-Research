"""Scientific edge cases identified in the paper-2 review."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from src import mhw
from paper2_revision import onset_target, causal_inactive


def test_category_uses_maximum_normalized_intensity():
    dates = pd.date_range("2001-01-01", periods=7)
    seas = np.zeros(366)
    threshold = np.ones(366)
    threshold[0] = 3
    series = pd.Series([5, 4.5, 2, 2, 2, 0, 0], index=dates)
    daily, events = mhw.detect(series, clim=(seas, threshold))
    assert events.iloc[0].category == 4
    assert daily.category.iloc[0] == 1
    assert daily.category.iloc[1] == 4
    assert daily.event_category.iloc[0] == 4


def test_climatology_ignores_observations_outside_baseline():
    dates = pd.date_range("1993-01-01", "2001-01-10")
    values = np.sin(np.arange(len(dates)) / 58)
    before = mhw.climatology(values, dates, ("1993-01-01", "2000-12-31"))
    values[dates > "2000-12-31"] = 100000
    after = mhw.climatology(values, dates, ("1993-01-01", "2000-12-31"))
    np.testing.assert_array_equal(before, after)


def test_onset_window_excludes_today_and_censors_incomplete_horizon():
    labels = pd.Series([0, 0, 1, 1, 1, 0, 0, 1, 1, 1])
    target = onset_target(labels, 2)
    assert target.iloc[0] == 1
    assert target.iloc[2] == 0
    assert target.iloc[5] == 1
    assert target.iloc[-2:].isna().all()


def test_merged_gap_label_needs_six_future_days():
    exceed = np.array([1]*5 + [0]*2 + [1]*5, dtype=bool)
    assert mhw.event_mask(exceed)[5]
    # Until the second five-day run is complete, the gap cannot yet be labelled.
    assert not mhw.event_mask(exceed[:-1])[5]


def test_onset_eligibility_tracks_confirmed_events_causally():
    run = pd.Series([0, 1, 2, 3, 4, 5, 0, 0, 0, 1, 2, 3])
    eligible = causal_inactive(run)
    assert eligible.iloc[0]
    assert not eligible.iloc[6:8].any()
    assert eligible.iloc[8]
    changed = run.copy()
    changed.iloc[9:] = [5, 6, 7]
    np.testing.assert_array_equal(eligible.iloc[:9], causal_inactive(changed).iloc[:9])
