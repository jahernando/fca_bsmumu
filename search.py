"""Search for Bs -> mu mu: fit of the mass spectrum and p-value."""

import numpy as np
from scipy import optimize, stats

M_B0 = 5.27966          # GeV, PDG
M_BS = 5.36692
BINS = np.linspace(4.9, 5.9, 51)        # 20 MeV bins


def histogram(m, bins=BINS):
    """Bin centres and counts of the masses m."""
    counts, edges = np.histogram(m, bins)
    return 0.5 * (edges[1:] + edges[:-1]), counts


def model(x, n_bkg, slope, n_b0, n_bs, sigma):
    """Expected counts per bin at masses x: exponential background, B0 and Bs peaks.

    n_b0 and n_bs are numbers of events; n_bkg is the background per bin at 5 GeV.
    """
    width = BINS[1] - BINS[0]
    peak = lambda mass: width * stats.norm.pdf(x, mass, sigma)
    return n_bkg * np.exp(-slope * (x - 5)) + n_b0 * peak(M_B0) + n_bs * peak(M_BS)


def fit(m, sigma, with_bs=True):
    """Fit the model to the histogram of m with curve_fit, the width sigma fixed.

    with_bs=False fixes n_bs = 0 (the background-only hypothesis, H0).
    Returns the parameters, their errors and the chi2 of the fit.
    """
    raise NotImplementedError       # TODO (D)


def pvalue(m, sigma):
    """p-value of the background-only hypothesis (no Bs), and its significance in sigmas."""
    raise NotImplementedError       # TODO (D)
