"""Walk-forward forecasting competition, losses, Diebold-Mariano tests,
validation-based model selection, and a look-ahead check.

Sample split (by date, identical for every index)
-------------------------------------------------
* burn-in    : first T0 = 1000 trading days of each index (no forecasts scored)
* validation : forecasts whose *target* date is <= 2014-12-31
* test       : forecasts whose *origin* date is >= 2015-01-01
Improved models are chosen on validation only; the test period is used once.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import models as M

HS = (1, 5, 20)
T0 = 1000
L_RFSV = 500
VAL_END = pd.Timestamp("2014-12-31")
TEST_START = pd.Timestamp("2015-01-01")
ROLL_WINDOWS = (250, 500, 1000)

BASELINES = ["RW", "HAR", "GARCH", "RFSV"]
CANDIDATES = ["RFSV-roll250", "RFSV-roll500", "RFSV-roll1000", "HAR-L",
              "HAR-RFSV", "HAR-L-RFSV", "COMB(RFSV,HAR)", "COMB(RFSV,HAR-L)"]


# ---------------------------------------------------------------------------
# forecasts
# ---------------------------------------------------------------------------

def forecast_series(df: pd.DataFrame, hs=HS, with_garch: bool = True) -> dict:
    """All models, all horizons, for one index. Returns {(model, h): (f_log, f_var)}."""
    y = df["logv"].to_numpy(float)
    r = df["ret"].to_numpy(float)
    start = L_RFSV - 1
    out = {}
    H, nu = M.rfsv_params(y, start)
    Hroll = {w: M.rfsv_params(y, start, window=w) for w in ROLL_WINDOWS}
    lev = M.leverage_terms(r)
    g = M.garch(y, r, hs, start=T0) if with_garch else {}
    for h in hs:
        out[("RW", h)] = M.random_walk(y, h)
        out[("HAR", h)] = M.har(y, h)
        if with_garch:
            out[("GARCH", h)] = g[h]
        rf = M.rfsv(y, h, H, nu, L_RFSV)
        out[("RFSV", h)] = rf
        for w, (Hw, nuw) in Hroll.items():
            out[(f"RFSV-roll{w}", h)] = M.rfsv(y, h, Hw, nuw, L_RFSV)
        out[("HAR-L", h)] = M.har(y, h, extra=lev)
        out[("HAR-RFSV", h)] = M.har(y, h, extra=[rf[0]])
        out[("HAR-L-RFSV", h)] = M.har(y, h, extra=lev + [rf[0]])
        out[("COMB(RFSV,HAR)", h)] = M.combine(y, h, rf[0], out[("HAR", h)][0])[:2]
        out[("COMB(RFSV,HAR-L)", h)] = M.combine(y, h, rf[0], out[("HAR-L", h)][0])[:2]
    return out


def to_long(name: str, df: pd.DataFrame, fc: dict) -> pd.DataFrame:
    """Stack forecasts into a long frame with targets, losses and period labels."""
    y = df["logv"].to_numpy(float)
    dates = df.index.to_numpy()
    N = len(y)
    rows = []
    for (model, h), (fl, fv) in fc.items():
        t = np.arange(T0, N - h)
        tgt = y[t + h]
        od, td = dates[t], dates[t + h]
        period = np.where(td <= VAL_END.to_datetime64(), "val",
                          np.where(od >= TEST_START.to_datetime64(), "test", "gap"))
        rows.append(pd.DataFrame({
            "index": name, "model": model, "h": h, "origin": od, "target": td,
            "period": period, "y": tgt.astype("float32"),
            "f_log": fl[t].astype("float32"), "f_var": fv[t].astype("float32"),
        }))
    d = pd.concat(rows, ignore_index=True)
    d = d[d.period != "gap"]
    d["se_log"] = (d.y - d.f_log) ** 2
    ratio = np.exp(d.y.astype(float)) / d.f_var.astype(float)
    d["qlike"] = ratio - np.log(ratio) - 1.0
    return d


# ---------------------------------------------------------------------------
# losses and tests
# ---------------------------------------------------------------------------

def nw_lag(T: int, h: int) -> int:
    """Newey-West truncation lag: at least h (h-step errors overlap)."""
    return int(max(h, np.floor(4 * (T / 100.0) ** (2.0 / 9.0))))


def dm_test(d: np.ndarray, h: int) -> tuple[float, float]:
    """Diebold-Mariano test of E[d] = 0 with a Bartlett HAC variance.

    d = loss(model A) - loss(model B); negative mean => A better.
    Returns (statistic, two-sided p-value from N(0,1)).
    """
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    T = len(d)
    if T < 50:
        return np.nan, np.nan
    u = d - d.mean()
    L = nw_lag(T, h)
    lrv = u @ u / T
    for k in range(1, L + 1):
        lrv += 2 * (1 - k / (L + 1)) * (u[k:] @ u[:-k]) / T
    if lrv <= 0:
        return np.nan, np.nan
    stat = d.mean() / np.sqrt(lrv / T)
    return float(stat), float(2 * (1 - stats.norm.cdf(abs(stat))))


def common_sample(long: pd.DataFrame) -> pd.DataFrame:
    """Keep only (index, h, origin) cells where every model has a forecast,
    so that all models are scored on exactly the same days."""
    ok = long.dropna(subset=["f_log", "f_var"])
    n_models = long.model.nunique()
    cnt = ok.groupby(["index", "h", "origin"]).model.transform("count")
    return ok[cnt == n_models]


def loss_table(long: pd.DataFrame) -> pd.DataFrame:
    """Mean losses per index, horizon, model and period."""
    g = long.groupby(["period", "index", "h", "model"])
    return g.agg(n=("se_log", "size"), mse_log=("se_log", "mean"),
                 qlike=("qlike", "mean")).reset_index()


def ratio_summary(lt: pd.DataFrame, period: str, ref: str = "HAR") -> pd.DataFrame:
    """Loss of each model relative to `ref`, per index; then the geometric
    mean across indices and the number of indices where the model wins."""
    d = lt[lt.period == period].copy()
    base = d[d.model == ref].set_index(["index", "h"])[["mse_log", "qlike"]]
    d = d.join(base, on=["index", "h"], rsuffix="_ref")
    d["r_mse"] = d.mse_log / d.mse_log_ref
    d["r_qlike"] = d.qlike / d.qlike_ref
    s = d.groupby(["h", "model"]).agg(
        gm_ratio_mse=("r_mse", lambda x: float(np.exp(np.mean(np.log(x))))),
        wins_mse=("r_mse", lambda x: int((x < 1).sum())),
        gm_ratio_qlike=("r_qlike", lambda x: float(np.exp(np.mean(np.log(x))))),
        wins_qlike=("r_qlike", lambda x: int((x < 1).sum())),
        n_index=("r_mse", "size"),
    ).reset_index()
    return s


def select_models(lt: pd.DataFrame) -> dict[int, str]:
    """Best candidate per horizon on the VALIDATION period: lowest geometric
    mean (across indices) of the log-MSE ratio to HAR."""
    s = ratio_summary(lt, "val")
    s = s[s.model.isin(CANDIDATES)]
    return {int(h): g.sort_values("gm_ratio_mse").model.iloc[0] for h, g in s.groupby("h")}


def dm_table(long: pd.DataFrame, pairs: list[tuple[str, str]], period: str = "test") -> pd.DataFrame:
    """DM tests for every index, horizon and (A, B) pair, both losses."""
    d = long[long.period == period]
    rows = []
    for (idx, h), g in d.groupby(["index", "h"]):
        piv_se = g.pivot(index="origin", columns="model", values="se_log")
        piv_ql = g.pivot(index="origin", columns="model", values="qlike")
        for a, b in pairs:
            if a not in piv_se or b not in piv_se:
                continue
            s1, p1 = dm_test((piv_se[a] - piv_se[b]).to_numpy(), h)
            s2, p2 = dm_test((piv_ql[a] - piv_ql[b]).to_numpy(), h)
            rows.append({"index": idx, "h": h, "A": a, "B": b,
                         "dm_mse": s1, "p_mse": p1, "dm_qlike": s2, "p_qlike": p2})
    return pd.DataFrame(rows)


def dm_counts(dm: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Number of indices where A is significantly better / worse than B."""
    def c(g, stat, p):
        return pd.Series({"A_better": int(((g[stat] < 0) & (g[p] < alpha)).sum()),
                          "A_worse": int(((g[stat] > 0) & (g[p] < alpha)).sum()),
                          "n": len(g)})
    a = dm.groupby(["A", "B", "h"]).apply(lambda g: c(g, "dm_mse", "p_mse"), include_groups=False)
    b = dm.groupby(["A", "B", "h"]).apply(lambda g: c(g, "dm_qlike", "p_qlike"), include_groups=False)
    return a.join(b, lsuffix="_mse", rsuffix="_qlike").reset_index()


# ---------------------------------------------------------------------------
# look-ahead check
# ---------------------------------------------------------------------------

def check_no_lookahead(df: pd.DataFrame, cut: int, hs=HS, seed: int = 0) -> bool:
    """Perturb every observation after `cut` and verify that no forecast made
    at an origin <= cut changes. Raises AssertionError on failure."""
    rng = np.random.default_rng(seed)
    base = forecast_series(df, hs)
    df2 = df.copy()
    n_after = len(df2) - cut - 1
    df2.iloc[cut + 1:, df2.columns.get_loc("logv")] += rng.normal(0, 1.0, n_after)
    df2.iloc[cut + 1:, df2.columns.get_loc("ret")] *= rng.uniform(0.2, 5.0, n_after)
    pert = forecast_series(df2, hs)
    for key in base:
        for a, b in zip(base[key], pert[key]):
            np.testing.assert_allclose(a[: cut + 1], b[: cut + 1], rtol=1e-9, atol=1e-12,
                                       err_msg=f"look-ahead in {key}")
    return True
