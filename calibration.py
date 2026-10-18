"""Momentum calibration of the muons, from the J/psi peak."""

import json

import numpy as np
from scipy import optimize

import dimuon

M_JPSI = 3.0969         # GeV, PDG
ETA_SPLIT = 3.5         # boundary between the 'low' and 'high' eta regions


def fit_peak(m, lo=2.85, hi=3.25, bins=80):
    """Fit a Gaussian plus a straight line to the histogram of m in (lo, hi).

    Returns the mean, its error and the width of the Gaussian.
    """
    raise NotImplementedError       # TODO (B)


class Calibration:
    """Momentum scale of the muons in two eta regions: measured pT = scale * true pT."""

    def __init__(self, low=1.0, high=1.0):
        self.low, self.high = low, high

    def __repr__(self):
        return f'Calibration(low={self.low:.4f}, high={self.high:.4f})'

    def save(self, path='calibration.json'):
        with open(path, 'w') as f:
            json.dump({'low': self.low, 'high': self.high}, f)

    @classmethod
    def load(cls, path='calibration.json'):
        with open(path) as f:
            return cls(**json.load(f))

    def apply(self, df):
        """Return a copy of df with mu1_pt and mu2_pt corrected (divided by their scale).

        The interface agreed by the two teams. For now a dummy: it returns the data unchanged.
        """
        return df.copy()            # TODO (B): divide each pT by the scale of its eta region

    def fit(self, control):
        """Fit the J/psi peak in events with both muons in the same region; set the scales.

        control: selected events of the control sample. Returns self.
        """
        raise NotImplementedError   # TODO (B)
