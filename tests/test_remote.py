"""Tiny live connectivity test. Requires `copernicusmarine login`.

Run with:  pytest -m remote
"""

import numpy as np
import pytest

pytestmark = pytest.mark.remote


def test_tiny_point_subset():
    from src import data_access, preprocessing, utils
    s = utils.load_settings()
    c = s["connectivity_test"]
    ds = data_access.open_point(c["lon"], c["lat"], c["start"], c["end"], s)
    assert "thetao" in ds
    assert ds.sizes["time"] == 3
    series = preprocessing.to_series(ds["thetao"].load())
    assert series.notna().all()
    assert np.all((series > 5) & (series < 30))
