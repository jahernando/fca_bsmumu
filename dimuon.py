"""Muon pairs: invariant mass and selection."""

from functools import reduce

import numpy as np

M_MU = 0.1056584        # GeV


def invariant_mass(df):
    """Invariant mass (GeV) of the two muons of each row of df.

    Uses the columns mu1_pt, mu1_eta, mu1_phi, mu2_pt, mu2_eta, mu2_phi.
    Vectorised with numpy: no loops over the rows.
    """
    raise NotImplementedError       # TODO (A)


# A cut is a function: DataFrame -> boolean Series (True = keep the row)

def has_trigger(df):
    return df['trigger']


# TODO (C): more cuts: opposite charges (mu1_q, mu2_q), and on mu1_pid, mu2_pid and vtx_chi2

CUTS = [has_trigger]


def select(df, cuts=CUTS):
    """Rows of df without missing values that pass all the cuts.

    Combine the cuts with functools.reduce.
    """
    raise NotImplementedError       # TODO (C)
