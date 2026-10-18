import numpy as np
import pandas as pd
import pytest

import dimuon
import mc
from calibration import Calibration


@pytest.fixture
def control():
    return mc.generate('control', n_bkg=8000, seed=3)[0]


def test_unit_scales_change_nothing(control):
    pd.testing.assert_frame_equal(Calibration(1.0, 1.0).apply(control), control,
                                  check_dtype=False)


def test_apply_does_not_modify_its_input(control):
    before = control.copy()
    Calibration(2.0, 2.0).apply(control)
    assert control.equals(before)


def test_apply_divides_by_the_scale_of_each_region(control):
    corrected = Calibration(2.0, 4.0).apply(control)
    expected = control['mu1_pt'] / np.where(control['mu1_eta'] < 3.5, 2.0, 4.0)
    assert np.allclose(corrected['mu1_pt'], expected)


def test_fit_on_mc_gives_unit_scales(control):
    cal = Calibration().fit(dimuon.select(control))
    assert cal.low == pytest.approx(1, abs=0.002)
    assert cal.high == pytest.approx(1, abs=0.002)
