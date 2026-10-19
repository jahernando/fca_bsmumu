import numpy as np
import pytest
from scipy import stats

import search

SIGMA = 0.023       # GeV, width of the peaks in the toys


def toy(n_b0, n_bs, n_bkg=3000, seed=1):
    """Masses made with numpy: exponential background plus B0 and Bs peaks."""
    rng = np.random.default_rng(seed)
    bkg = 4.9 + rng.exponential(1 / 1.2, n_bkg)
    return np.concatenate([bkg[bkg < 5.9], rng.normal(search.M_B0, SIGMA, n_b0),
                           rng.normal(search.M_BS, SIGMA, n_bs)])


def test_model_peak_holds_its_events():
    """With no background, the Bs peak spreads n_bs events over the bins."""
    x, _ = search.histogram([])
    counts = search.model(x, n_bkg=0, slope=1, n_b0=0, n_bs=100, sigma=0.022)
    assert counts.sum() == pytest.approx(100, rel=0.01)


def test_fit_returns_parameters_errors_and_chi2():
    params, errors, chi2 = search.fit(toy(30, 100), SIGMA)
    assert len(params) == len(errors) == 4          # n_bkg, slope, n_b0, n_bs
    assert np.all(np.asarray(errors) > 0)
    assert chi2 > 0


def test_fit_recovers_the_bs_of_a_toy():
    params, errors, _ = search.fit(toy(30, 200), SIGMA)
    assert abs(params[3] - 200) < 3 * errors[3]


def test_fit_without_bs_has_three_parameters_and_a_larger_chi2():
    m = toy(30, 100)
    params, errors, chi2_h0 = search.fit(m, SIGMA, with_bs=False)
    assert len(params) == len(errors) == 3
    assert chi2_h0 > search.fit(m, SIGMA)[2]


def test_pvalue_small_with_signal_large_without():
    p, z = search.pvalue(toy(30, 100), SIGMA)
    assert p < 1e-3 and z > 3
    p, z = search.pvalue(toy(30, 0), SIGMA)
    assert p > 0.01


def test_pvalue_and_significance_agree():
    p, z = search.pvalue(toy(30, 100), SIGMA)
    assert p == pytest.approx(stats.chi2.sf(z**2, 1), rel=1e-6)
