"""Internal to the MC: generator of muon-pair samples. Do not edit.

Each event is a muon pair: (pt, eta, phi, charge, pid) per muon, the vertex chi2 and a
trigger bit. Components: resonances decaying to two real muons of opposite charge, and a
combinatorial background whose true mass follows an exponential and whose charges are
independent (half of the pairs have the same sign).
"""

import numpy as np
import pandas as pd

import _kinematics as kin

RESOLUTION = 0.006              # relative pT resolution of a muon
ETA_SPLIT = 3.5                 # boundary of the two calibration regions
WINDOWS = {'control': (2.7, 3.5), 'search': (4.7, 6.2)}   # true-mass range of the background
SLOPES = {'control': 1.0, 'search': 1.2}                  # 1/GeV, exponential background


def _exponential(n, lo, hi, slope, rng):
    """Masses in (lo, hi) with density ~ exp(-slope m), by inversion."""
    u = rng.uniform(size=n)
    a, b = np.exp(-slope * lo), np.exp(-slope * hi)
    return -np.log(a - u * (a - b)) / slope


def _component(mass, signal, rng):
    """One component: muon kinematics plus selection variables, only accepted events."""
    while True:     # generate a little more than needed, to compensate the acceptance
        mu, ok = kin.decay(np.resize(mass, 3 * len(mass)), rng)
        if ok.sum() >= len(mass):
            break
    df = pd.DataFrame(mu)[ok].iloc[:len(mass)].reset_index(drop=True)
    n = len(df)
    for i in (1, 2):
        real = np.ones(n, bool) if signal else rng.uniform(size=n) < 0.5
        df[f'mu{i}_pid'] = np.where(real, rng.beta(8, 1.2, n), rng.beta(1.2, 3, n))
    df['vtx_chi2'] = rng.chisquare(1, n) if signal else rng.exponential(6, n)
    df['trigger'] = rng.uniform(size=n) < (0.9 if signal else 0.6)
    return df


def generate(sample, yields, scales=(1.0, 1.0), seed=None):
    """Generate a sample ('control' or 'search') with yields {'bkg': n, 'jpsi': n, ...}.

    scales = (low-eta, high-eta) multiply the pT of each muon according to its eta: the
    miscalibration. Returns the shuffled DataFrame.
    """
    rng = np.random.default_rng(seed)
    masses = {'jpsi': kin.M_JPSI, 'b0': kin.M_B0, 'bs': kin.M_BS}
    parts, signal = [], []
    for name, n in yields.items():
        if n == 0:
            continue
        if name == 'bkg':
            m = _exponential(n, *WINDOWS[sample], SLOPES[sample], rng)
        else:
            m = np.full(n, masses[name])
        parts.append(_component(m, name != 'bkg', rng))
        signal.append(np.full(len(parts[-1]), name != 'bkg'))
    df = pd.concat(parts, ignore_index=True)
    signal = np.concatenate(signal)

    # charges from their own stream, so that the other columns do not depend on them
    qrng = np.random.default_rng(None if seed is None else [int(seed), 7])
    q1 = qrng.choice(np.array([-1, 1], dtype='int8'), len(df))
    q2 = np.where(signal, -q1, qrng.choice(np.array([-1, 1], dtype='int8'), len(df)))
    df.insert(df.columns.get_loc('mu1_phi') + 1, 'mu1_q', q1)
    df.insert(df.columns.get_loc('mu2_phi') + 1, 'mu2_q', q2.astype('int8'))

    for i in (1, 2):
        scale = np.where(df[f'mu{i}_eta'] < ETA_SPLIT, *scales)
        df[f'mu{i}_pt'] *= scale * (1 + RESOLUTION * rng.standard_normal(len(df)))

    # failed reconstruction: some missing values
    for col, frac in [('mu1_pid', 0.01), ('mu2_pid', 0.01), ('vtx_chi2', 0.01)]:
        df.loc[rng.uniform(size=len(df)) < frac, col] = np.nan

    df = df.sample(frac=1, random_state=rng.integers(2**31)).reset_index(drop=True)
    return df.round(5)
