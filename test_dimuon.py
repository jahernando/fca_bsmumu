import numpy as np
import pandas as pd
import pytest

import dimuon
import mc


def pair(pt1, eta1, phi1, pt2, eta2, phi2):
    return pd.DataFrame({'mu1_pt': [pt1], 'mu1_eta': [eta1], 'mu1_phi': [phi1],
                         'mu2_pt': [pt2], 'mu2_eta': [eta2], 'mu2_phi': [phi2]})


def test_back_to_back():
    """Two muons of momentum p in opposite directions: m = 2 E."""
    p = 2.0
    m = dimuon.invariant_mass(pair(p, 0, 0, p, 0, np.pi))
    assert m.iloc[0] == pytest.approx(2 * np.sqrt(p**2 + dimuon.M_MU**2))


def test_jpsi_peak_in_mc():
    """The J/psi of the (perfectly calibrated) MC is at its mass."""
    df, _ = mc.generate('control', n_bkg=0, seed=1)
    assert np.median(dimuon.invariant_mass(df)) == pytest.approx(3.0969, abs=0.005)


def test_select_removes_missing_values():
    df, _ = mc.generate('control', n_bkg=4000, seed=2)
    assert not dimuon.select(df).isna().any().any()
