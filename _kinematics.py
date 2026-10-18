"""Internal to the MC: two-body decays into a muon pair (masses in GeV). Do not edit."""

import numpy as np

M_MU = 0.1056584
M_JPSI = 3.096900
M_B0 = 5.27966
M_BS = 5.36692

ETA_RANGE = (2.0, 5.0)          # LHCb acceptance
PT_MIN = 0.5                    # GeV, minimum muon pT


def _boost(e, px, py, pz, bx, by, bz):
    """Boost the four-vectors (e, p) by the velocity (bx, by, bz)."""
    b2 = bx**2 + by**2 + bz**2
    gamma = 1 / np.sqrt(1 - b2)
    bp = bx * px + by * py + bz * pz
    k = (gamma - 1) * bp / b2 + gamma * e
    return gamma * (e + bp), px + k * bx, py + k * by, pz + k * bz


def _to_pt_eta_phi(px, py, pz):
    pt = np.hypot(px, py)
    return pt, np.arcsinh(pz / pt), np.arctan2(py, px)


def decay(mass, rng):
    """Decay parents of the given masses into two muons, isotropic in the rest frame.

    The parents have pT ~ Gamma(2, 2.5 GeV) and eta uniform in (1.5, 5.5). Returns a dict
    of muon arrays (pt, eta, phi for each muon) and the acceptance mask.
    """
    n = len(mass)
    pt = rng.gamma(2.0, 2.5, n)
    eta = rng.uniform(1.5, 5.5, n)
    phi = rng.uniform(-np.pi, np.pi, n)
    px, py, pz = pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)
    e = np.sqrt(px**2 + py**2 + pz**2 + mass**2)

    pstar = np.sqrt(mass**2 / 4 - M_MU**2)
    cos_t = rng.uniform(-1, 1, n)
    sin_t = np.sqrt(1 - cos_t**2)
    phi_s = rng.uniform(-np.pi, np.pi, n)
    qx, qy, qz = pstar * sin_t * np.cos(phi_s), pstar * sin_t * np.sin(phi_s), pstar * cos_t
    estar = np.sqrt(pstar**2 + M_MU**2)

    bx, by, bz = px / e, py / e, pz / e
    out = {}
    for i, sign in [(1, 1), (2, -1)]:
        _, mx, my, mz = _boost(estar, sign * qx, sign * qy, sign * qz, bx, by, bz)
        out[f'mu{i}_pt'], out[f'mu{i}_eta'], out[f'mu{i}_phi'] = _to_pt_eta_phi(mx, my, mz)

    accepted = np.ones(n, dtype=bool)
    for i in (1, 2):
        accepted &= (out[f'mu{i}_eta'] > ETA_RANGE[0]) & (out[f'mu{i}_eta'] < ETA_RANGE[1])
        accepted &= out[f'mu{i}_pt'] > PT_MIN
    return out, accepted
