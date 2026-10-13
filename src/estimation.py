"""Roughness estimation (Gatheral, Jaisson & Rosenbaum 2018, Section 2).

For a log-volatility series x_t = log sigma_t = 0.5 * log(variance), define

    m(q, D) = mean_t |x_{t+D} - x_t|^q .

If x has stationary increments with x_{t+D} - x_t ~ nu * D^H * Z, then
m(q, D) = nu^q E|Z|^q D^{qH}, so log m(q, D) is linear in log D with slope
zeta_q = q H. H is the slope of zeta_q against q (regression through 0).
"""
from __future__ import annotations

import numpy as np
from scipy import stats

QS = np.array([0.5, 1.0, 1.5, 2.0, 3.0])
LAGS = np.arange(1, 51)


def log_vol(logv: np.ndarray) -> np.ndarray:
    """log sigma from log variance."""
    return 0.5 * np.asarray(logv, dtype=float)


def m_q_delta(x: np.ndarray, qs=QS, lags=LAGS) -> np.ndarray:
    """Matrix m[i, j] = mean |x_{t+lags[j]} - x_t|^{qs[i]}.

    Lags are in trading days (consecutive valid observations).
    """
    x = np.asarray(x, dtype=float)
    out = np.empty((len(qs), len(lags)))
    for j, d in enumerate(lags):
        inc = np.abs(x[d:] - x[:-d])
        for i, q in enumerate(qs):
            out[i, j] = np.mean(inc**q)
    return out


def zeta_from_m(m: np.ndarray, lags=LAGS):
    """OLS of log m(q, D) on log D for every q. Returns (zeta_q, intercepts)."""
    ld = np.log(lags)
    X = np.column_stack([np.ones_like(ld), ld])
    coef, *_ = np.linalg.lstsq(X, np.log(m).T, rcond=None)
    return coef[1], coef[0]


def h_from_zeta(zeta: np.ndarray, qs=QS) -> float:
    """Slope of zeta_q on q through the origin."""
    return float(np.dot(qs, zeta) / np.dot(qs, qs))


def estimate_h(x: np.ndarray, qs=QS, lags=LAGS) -> dict:
    """Full estimate: m, zeta_q, H (all q), H_2 = zeta_2 / 2, and nu.

    nu is the scale of log-vol increments: m(2, D) = nu^2 D^{2H}.
    """
    m = m_q_delta(x, qs, lags)
    zeta, icpt = zeta_from_m(m, lags)
    i2 = int(np.where(np.isclose(qs, 2.0))[0][0])
    return {"m": m, "zeta": zeta, "H": h_from_zeta(zeta, qs),
            "H2": zeta[i2] / 2.0, "nu": float(np.exp(icpt[i2] / 2.0))}


def h_nu_q2(x: np.ndarray, lags=LAGS) -> tuple[float, float]:
    """Fast (H, nu) from q = 2 only; used inside the forecasting loop."""
    x = np.asarray(x, dtype=float)
    lm = np.array([np.log(np.mean((x[d:] - x[:-d]) ** 2)) for d in lags])
    ld = np.log(lags)
    slope, icpt = np.polyfit(ld, lm, 1)
    return float(slope / 2.0), float(np.exp(icpt / 2.0))


def block_bootstrap_h(x: np.ndarray, block: int = 250, n_boot: int = 500,
                      qs=QS, lags=LAGS, seed: int = 0) -> np.ndarray:
    """Moving-block bootstrap distribution of H.

    The series is cut into non-overlapping blocks of `block` days. For every
    block we store the sums of |increment|^q and the pair counts per lag,
    using only pairs inside the block (so no artificial increments across
    block joins). A bootstrap draw resamples blocks with replacement and
    re-aggregates the sums into m(q, D), then H. Volatility clustering
    within a year is preserved by the 250-day blocks.
    """
    x = np.asarray(x, dtype=float)
    nb = len(x) // block
    sums = np.zeros((nb, len(qs), len(lags)))
    cnts = np.zeros((nb, len(lags)))
    for b in range(nb):
        xb = x[b * block:(b + 1) * block]
        for j, d in enumerate(lags):
            inc = np.abs(xb[d:] - xb[:-d])
            cnts[b, j] = len(inc)
            for i, q in enumerate(qs):
                sums[b, i, j] = np.sum(inc**q)
    rng = np.random.default_rng(seed)
    hs = np.empty(n_boot)
    for k in range(n_boot):
        idx = rng.integers(0, nb, nb)
        m = sums[idx].sum(0) / cnts[idx].sum(0)
        zeta, _ = zeta_from_m(m, lags)
        hs[k] = h_from_zeta(zeta, qs)
    return hs


def increment_moments(x: np.ndarray, lags=(1, 5, 25, 125)) -> dict:
    """Skewness and excess kurtosis of log-vol increments at several lags.

    A Gaussian has 0 and 0. Also returns the Jarque-Bera statistic, which
    rejects at almost any lag in long samples, so the moments are the
    informative numbers.
    """
    x = np.asarray(x, dtype=float)
    res = {}
    for d in lags:
        inc = x[d:] - x[:-d]
        res[d] = {"skew": float(stats.skew(inc)),
                  "exkurt": float(stats.kurtosis(inc)),
                  "sd": float(np.std(inc))}
    return res


# ---------------------------------------------------------------------------
# Simulation (used for the smoothing-bias check)
# ---------------------------------------------------------------------------

def fgn_davies_harte(n: int, H: float, rng: np.random.Generator) -> np.ndarray:
    """Exact fractional Gaussian noise (unit variance) by circulant embedding."""
    k = np.arange(n + 1)
    gamma = 0.5 * ((k + 1.0) ** (2 * H) - 2 * k ** (2 * H) + np.abs(k - 1.0) ** (2 * H))
    row = np.concatenate([gamma, gamma[-2:0:-1]])
    lam = np.fft.fft(row).real
    if np.any(lam < 0):
        lam = np.clip(lam, 0, None)
    m = len(row)
    w = rng.standard_normal(m) + 1j * rng.standard_normal(m)
    z = np.fft.fft(np.sqrt(lam / m) * w)
    return z.real[:n]


def simulate_rfsv(n_days: int, H: float, nu: float, steps_per_day: int = 50,
                  mean_logvar: float = np.log(1e-4), seed: int = 0):
    """Simulate log sigma_t = nu * W^H_t on an intraday grid (time unit: days).

    Returns (spot daily log-vol sampled at day ends, log of the daily
    integrated variance) so the two can be compared.
    """
    rng = np.random.default_rng(seed)
    n = n_days * steps_per_day
    dt = 1.0 / steps_per_day
    fgn = fgn_davies_harte(n, H, rng) * dt**H
    x = 0.5 * mean_logvar + nu * np.cumsum(fgn)  # log sigma on the fine grid
    var_fine = np.exp(2 * x)
    iv = var_fine.reshape(n_days, steps_per_day).mean(1)  # daily integrated var
    spot = x[steps_per_day - 1::steps_per_day]
    return spot, 0.5 * np.log(iv)
