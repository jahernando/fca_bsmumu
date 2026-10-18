import numpy as np
import pytest

import search


def test_model_peak_holds_its_events():
    """With no background, the Bs peak spreads n_bs events over the bins."""
    x, _ = search.histogram([])
    counts = search.model(x, n_bkg=0, slope=1, n_b0=0, n_bs=100, sigma=0.022)
    assert counts.sum() == pytest.approx(100, rel=0.01)
