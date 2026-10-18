"""Monte Carlo for the B -> mu mu exercise: muon pairs in the same format as the data.

The MC is perfectly calibrated. The number of B0 and Bs decays is random, and returned in
`truth`, so that you can check that your fit recovers what was generated.
"""

import numpy as np

import _generator as gen


def generate(sample='search', n_bkg=20_000, n_b0=None, n_bs=None, seed=None):
    """Return (DataFrame, truth) for sample 'search' (B region) or 'control' (J/psi).

    In 'search', the numbers of B0 and Bs decays are random unless given (n_b0, n_bs).
    n_bkg=0 gives signal only. In 'control', n_bkg // 4 J/psi decays (5000 if n_bkg=0).
    """
    rng = np.random.default_rng(seed)
    if sample == 'control':
        truth = {'jpsi': n_bkg // 4 or 5000}
    else:
        truth = {'b0': int(rng.integers(0, 30)) if n_b0 is None else n_b0,
                 'bs': int(rng.integers(0, 100)) if n_bs is None else n_bs}
    df = gen.generate(sample, {'bkg': n_bkg, **truth}, seed=rng.integers(2**31))
    df = df.astype({c: 'float32' for c in df.columns if df[c].dtype == 'float64'})   # as the data
    return df, truth
