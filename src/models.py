"""Real-time (walk-forward) volatility forecasts.

Conventions used by every function here
---------------------------------------
* Arrays are indexed by trading day t = 0..N-1 of one index.
  y[t] = log variance (log RV or log Garman-Klass), v[t] = exp(y[t]),
  r[t] = daily return.
* A forecast "at origin t for horizon h" targets y[t+h] and may only use
  y[0..t], r[0..t]. Every function returns arrays ``f_log`` and ``f_var``
  of length N where element t is the forecast made at origin t
  (NaN when not enough history).
* Parameters are re-estimated every ``refit`` days (default 20) on data up
  to the refit day only. Regression coefficients for horizon h at origin t
  use only pairs (x_s, y_{s+h}) with s + h <= t, i.e. targets already seen.

No function looks at y[t+1:] when forming the forecast at t. The function
``evaluation.check_no_lookahead`` verifies this by
perturbing the future and checking that past forecasts do not change.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import lfilter
from scipy.special import gamma as G

from .estimation import h_nu_q2

REFIT = 20
MIN_TRAIN_PAIRS = 250


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def trailing_mean(x: np.ndarray, w: int) -> np.ndarray:
    """mean(x[t-w+1..t]) at every t (NaN for t < w-1)."""
    c = np.concatenate([[0.0], np.cumsum(x)])
    out = np.full(len(x), np.nan)
    out[w - 1:] = (c[w:] - c[:-w]) / w
    return out


def expanding_direct_ols(X: np.ndarray, y: np.ndarray, h: int, min_pairs=MIN_TRAIN_PAIRS):
    """Direct h-step OLS refitted at every origin on all past pairs.

    Pair s is (X[s], y[s+h]). At origin t the usable pairs are s <= t - h.
    Uses running sums of X'X, X'y, y'y so each origin costs O(k^3).
    Returns (forecast at every origin, in-sample residual variance).
    """
    N, k = X.shape
    valid = np.isfinite(X).all(1)
    tgt = np.full(N, np.nan)
    tgt[:N - h] = y[h:]
    pair_ok = valid & np.isfinite(tgt)
    Xz = np.where(pair_ok[:, None], X, 0.0)
    tz = np.where(pair_ok, tgt, 0.0)
    Sxx = np.cumsum(Xz[:, :, None] * Xz[:, None, :], axis=0)
    Sxy = np.cumsum(Xz * tz[:, None], axis=0)
    Syy = np.cumsum(tz**2)
    n = np.cumsum(pair_ok)
    f = np.full(N, np.nan)
    s2 = np.full(N, np.nan)
    for t in range(h, N):
        s = t - h  # last usable pair index
        if n[s] < min_pairs or not valid[t]:
            continue
        A, b = Sxx[s], Sxy[s]
        try:
            beta = np.linalg.solve(A + 1e-10 * np.eye(k), b)
        except np.linalg.LinAlgError:
            continue
        f[t] = X[t] @ beta
        rss = Syy[s] - 2 * beta @ b + beta @ A @ beta
        s2[t] = max(rss, 0.0) / max(n[s] - k, 1)
    return f, s2


# ---------------------------------------------------------------------------
# (a) random walk
# ---------------------------------------------------------------------------

def random_walk(y: np.ndarray, h: int):
    """Tomorrow (and day t+h) equals today."""
    return y.copy(), np.exp(y)


# ---------------------------------------------------------------------------
# (b) HAR on log variance (Corsi 2009), with optional extra regressors
# ---------------------------------------------------------------------------

def har_design(y: np.ndarray, extra: list[np.ndarray] | None = None) -> np.ndarray:
    cols = [np.ones_like(y), y, trailing_mean(y, 5), trailing_mean(y, 22)]
    if extra:
        cols += list(extra)
    return np.column_stack(cols)


def leverage_terms(r: np.ndarray) -> list[np.ndarray]:
    """Negative-return terms of the LHAR model (Corsi & Reno 2012)."""
    neg = np.minimum(r, 0.0)
    return [neg, trailing_mean(neg, 5), trailing_mean(neg, 22)]


def har(y: np.ndarray, h: int, extra=None):
    """Direct h-step HAR. Level forecast uses the log-normal correction
    exp(f + s^2 / 2) with s^2 the in-sample residual variance."""
    f, s2 = expanding_direct_ols(har_design(y, extra), y, h)
    return f, np.exp(f + 0.5 * s2)


# ---------------------------------------------------------------------------
# (c) GARCH(1,1) on daily returns
# ---------------------------------------------------------------------------

def _garch_fit(r: np.ndarray):
    """QMLE of a zero-mean Gaussian GARCH(1,1) on percent returns."""
    from arch import arch_model

    res = arch_model(100 * r, mean="Zero", vol="GARCH", p=1, q=1,
                     rescale=False).fit(disp="off", show_warning=False)
    p = res.params
    return p["omega"] / 1e4, p["alpha[1]"], p["beta[1]"]


def garch(y: np.ndarray, r: np.ndarray, hs, start: int, refit: int = REFIT):
    """GARCH(1,1) h-step variance forecasts for every h in `hs`, calibrated.

    Returns-based variance and realized/range variance differ in level
    (overnight returns, microstructure, estimator bias), and E[log RV]
    differs from log E[RV]. To be fair to GARCH we calibrate on past data
    only, using the in-sample one-step conditional variances sigma2_{s|s-1}:
      log forecast = log sigma2_{t+h|t} + mean_{s<=t0}(y_s - log sigma2_s)
      var forecast = sigma2_{t+h|t} * mean_{s<=t0}(v_s / sigma2_s)
    where t0 <= t is the last refit day. Returns {h: (f_log, f_var)}.
    """
    N = len(y)
    out = {h: (np.full(N, np.nan), np.full(N, np.nan)) for h in hs}
    r2 = r**2
    v = np.exp(y)
    for t0 in range(start, N, refit):
        om, a, b = _garch_fit(r[: t0 + 1])
        # sigma2[s] = conditional variance of r[s] given r[..s-1]; causal filter
        init = np.mean(r2[: min(t0 + 1, 250)])
        base = om / (1 - b)
        sig2 = lfilter([0.0, a], [1.0, -b], r2) + base + b ** np.arange(N) * (init - base)
        sig2 = np.maximum(sig2, 1e-12)
        nxt = om + a * r2 + b * sig2  # forecast made at s for day s+1 (uses r[..s])
        p = a + b
        off_log = np.mean(y[1: t0 + 1] - np.log(sig2[1: t0 + 1]))
        off_var = np.mean(v[1: t0 + 1] / sig2[1: t0 + 1])
        sl = slice(t0, min(t0 + refit, N))
        for h in hs:
            const = om * np.sum(p ** np.arange(h - 1)) if h > 1 else 0.0
            fh = const + p ** (h - 1) * nxt[sl]
            out[h][0][sl] = np.log(fh) + off_log
            out[h][1][sl] = fh * off_var
    return out


# ---------------------------------------------------------------------------
# (d) RFSV predictor (Gatheral, Jaisson & Rosenbaum 2018, Section 5)
# ---------------------------------------------------------------------------

def rfsv_weights(H: float, h: int, L: int = 500) -> np.ndarray:
    """Discretised kernel of the fBm conditional expectation.

    E[log v_{t+h} | F_t] = cos(H pi)/pi * h^{H+1/2}
                           * int_0^inf log v_{t-u} / ((u+h) u^{H+1/2}) du.
    Cell k covers u in [k, k+1): we integrate u^{-(H+1/2)} exactly on the
    cell and evaluate 1/(u+h) at the cell midpoint. Weights are normalised
    to sum to one (the continuous kernel integrates to one; truncation at L
    days and discretisation would otherwise bias the level).
    """
    a = H + 0.5
    k = np.arange(L, dtype=float)
    cell = ((k + 1) ** (1 - a) - k ** (1 - a)) / (1 - a)
    w = cell / (k + 0.5 + h)
    return w / w.sum()


def rfsv_c(H: float) -> float:
    """c(H) = Gamma(3/2-H) / (Gamma(H+1/2) Gamma(2-2H)); c(1/2) = 1."""
    return G(1.5 - H) / (G(H + 0.5) * G(2 - 2 * H))


def rfsv_params(y: np.ndarray, start: int, refit: int = REFIT,
                window: int | None = None, min_obs: int = 250):
    """(H, nu) for log-vol x = y/2, re-estimated every `refit` days from
    the expanding past (window=None) or the last `window` days."""
    N = len(y)
    H = np.full(N, np.nan)
    nu = np.full(N, np.nan)
    x = 0.5 * y
    for t0 in range(start, N, refit):
        lo = 0 if window is None else max(0, t0 + 1 - window)
        if t0 + 1 - lo < min_obs:
            continue
        hh, nn = h_nu_q2(x[lo: t0 + 1])
        hh = float(np.clip(hh, 0.01, 0.49))
        H[t0: t0 + refit] = hh
        nu[t0: t0 + refit] = nn
    return H, nu


def rfsv(y: np.ndarray, h: int, H: np.ndarray, nu: np.ndarray, L: int = 500):
    """RFSV forecasts of log variance and of variance.

    log forecast  f_t = sum_k w_k(H_t, h) y_{t-k}
    var forecast  exp(f_t + 2 c(H) nu^2 h^{2H})   (paper's eq. for E[v_{t+h}|F_t];
                  nu is the vol-of-log-vol of log sigma, so the conditional
                  variance of log v = 2 log sigma is 4 c nu^2 h^{2H}).
    """
    N = len(y)
    f = np.full(N, np.nan)
    fv = np.full(N, np.nan)
    from numpy.lib.stride_tricks import sliding_window_view

    if N < L:
        return f, fv
    # past[t] = [y_t, y_{t-1}, ..., y_{t-L+1}]
    past = sliding_window_view(y, L)[:, ::-1]  # row j corresponds to t = j+L-1
    ok = np.isfinite(H)
    for Hval in np.unique(H[ok]):
        idx = np.where(ok & (H == Hval))[0]
        idx = idx[idx >= L - 1]
        if len(idx) == 0:
            continue
        w = rfsv_weights(Hval, h, L)
        f[idx] = past[idx - (L - 1)] @ w
    var_corr = 2 * np.vectorize(lambda a: rfsv_c(a) if np.isfinite(a) else np.nan)(H) * nu**2 * h ** (2 * H)
    fv = np.exp(f + var_corr)
    return f, fv


# ---------------------------------------------------------------------------
# Improvements
# ---------------------------------------------------------------------------

def combine(y: np.ndarray, h: int, fa: np.ndarray, fb: np.ndarray,
            min_pairs: int = MIN_TRAIN_PAIRS):
    """Real-time combination w*fa + (1-w)*fb of two log forecasts.

    At origin t, w minimises the past squared error over forecasts whose
    targets are already observed (origin s with s + h <= t); w in [0, 1].
    Level forecast exp(f + s^2/2) with s^2 the past combined MSE.
    Returns (f_log, f_var, w).
    """
    N = len(y)
    tgt = np.full(N, np.nan)
    tgt[:N - h] = y[h:]
    ok = np.isfinite(fa) & np.isfinite(fb) & np.isfinite(tgt)
    d = np.where(ok, fa - fb, 0.0)
    e = np.where(ok, tgt - fb, 0.0)
    Sdd, Sde, See, n = (np.cumsum(d * d), np.cumsum(d * e), np.cumsum(e * e), np.cumsum(ok))
    f = np.full(N, np.nan)
    fv = np.full(N, np.nan)
    wv = np.full(N, np.nan)
    for t in range(h, N):
        s = t - h
        if n[s] < min_pairs or not (np.isfinite(fa[t]) and np.isfinite(fb[t])):
            continue
        w = float(np.clip(Sde[s] / Sdd[s], 0.0, 1.0)) if Sdd[s] > 0 else 0.5
        mse = (See[s] - 2 * w * Sde[s] + w * w * Sdd[s]) / n[s]
        f[t] = w * fa[t] + (1 - w) * fb[t]
        fv[t] = np.exp(f[t] + 0.5 * mse)
        wv[t] = w
    return f, fv, wv
