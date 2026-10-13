"""Generate notebooks/rough_vol.ipynb (the notebook is then executed with
`jupyter nbconvert --execute`). Kept as a script so the notebook source is
reviewable as plain Python."""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s.strip()))


def code(s):
    cells.append(nbf.v4.new_code_cell(s.strip()))


md(r"""
# Volatility is rough: replication and forecasting

Replication of Gatheral, Jaisson & Rosenbaum, *Volatility is Rough* (Quantitative Finance, 2018)
on 31 indices of the Oxford-Man Realized Library (2000-2022), an extension to 2026 with
range-based (Garman-Klass) variance from Yahoo Finance for 20 indices, and a walk-forward
forecasting competition (random walk, HAR, GARCH(1,1), the paper's RFSV predictor and
validation-selected improvements).

**Two data types are never mixed.** `rv` = 5-minute realized variance (Oxford-Man),
`gk` = Garman-Klass range variance (Yahoo). Every table says which one it uses.

Raw downloads are treated as untrusted data: they are parsed only by
`scripts/build_data.py`, which this notebook runs in a separate `python -I` process.
""")

code(r"""
import os, sys, json, subprocess, time
from pathlib import Path
ROOT = Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd().resolve()
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from src.data import load_processed
from src import estimation as ES, evaluation as E, models as M

RUN_PIPELINE = True        # False = reuse data/processed and results/ from a previous run
FIG, RES = ROOT / "figures", ROOT / "results"
FIG.mkdir(exist_ok=True); RES.mkdir(exist_ok=True)
pd.set_option("display.width", 200); pd.set_option("display.max_rows", 120)

# chart style: recessive grid, thin marks; categorical slots blue / orange / aqua
C1, C2, C3, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#9a9993"
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
    "axes.edgecolor": "#8f8e88", "axes.labelcolor": "#2b2b29", "xtick.color": "#52514e",
    "ytick.color": "#52514e", "lines.linewidth": 1.6, "font.size": 9.5, "legend.frameon": False})

def run(script, *args):
    t = time.time()
    out = subprocess.run([sys.executable, "-I", str(ROOT / "scripts" / script), *args],
                         capture_output=True, text=True, cwd=str(ROOT / "scripts"))
    print(out.stdout[-3000:], out.stderr[-2000:], f"[{script}: {time.time()-t:.0f}s]")
    out.check_returncode()
""")

md("## 1. Data")

code(r"""
if RUN_PIPELINE or not (ROOT / "data/processed/rv.csv").exists():
    if not (ROOT / "data/raw/oxfordman.zip").exists():
        run("download_data.py")
    run("build_data.py")
rv = load_processed(ROOT / "data/processed/rv.csv")
gk = load_processed(ROOT / "data/processed/gk.csv")
qa = pd.read_csv(ROOT / "data/processed/data_quality.csv", index_col=0)
print(f"Oxford-Man RV: {len(rv)} indices; Garman-Klass (Yahoo): {len(gk)} indices")
qa
""")

code(r"""
fig, ax = plt.subplots(figsize=(9, 3.2))
ax.plot(gk["SPX"].index, 0.5 * gk["SPX"].logv, color=C2, lw=0.6, alpha=0.8, label="Garman-Klass (Yahoo OHLC)")
ax.plot(rv["SPX"].index, 0.5 * rv["SPX"].logv, color=C1, lw=0.6, label="5-min realized variance (Oxford-Man)")
ax.set_ylabel(r"log $\sigma_t$ (daily)"); ax.set_title("S&P 500 daily log-volatility, two proxies")
ax.legend(loc="upper left", ncol=2)
fig.tight_layout(); fig.savefig(FIG / "spx_logvol.png"); plt.show()
""")

md(r"""
## 2. Replication: smoothness of log-volatility

$m(q,\Delta) = \langle |\log\sigma_{t+\Delta} - \log\sigma_t|^q \rangle$ for $q\in\{0.5,1,1.5,2,3\}$,
$\Delta = 1..50$ trading days, with $\log\sigma_t = \tfrac12 \log \mathrm{RV}_t$.
If $\log\sigma$ behaves like fBm, $m(q,\Delta) \propto \Delta^{\zeta_q}$ with $\zeta_q = qH$.
$H$ = slope of $\zeta_q$ on $q$ through the origin. Standard errors: block bootstrap with
250-day blocks (500 draws), increments only inside blocks.
""")

code(r"""
def h_table(series, tag):
    rows = []
    for name, df in series.items():
        x = ES.log_vol(df.logv.to_numpy())
        e = ES.estimate_h(x)
        boot = ES.block_bootstrap_h(x, block=250, n_boot=500, seed=1)
        r = {"data": tag, "index": name, "n": len(x), "start": df.index[0].date(), "end": df.index[-1].date(),
             "H": e["H"], "se": boot.std(ddof=1), "ci_lo": np.quantile(boot, .025), "ci_hi": np.quantile(boot, .975),
             "H_q2": e["H2"], "nu": e["nu"]}
        r.update({f"zeta/q (q={q:g})": z / q for q, z in zip(ES.QS, e["zeta"])})
        rows.append(r)
    return pd.DataFrame(rows)

H_rv, H_gk = h_table(rv, "rv"), h_table(gk, "gk")
Htab = pd.concat([H_rv, H_gk], ignore_index=True)
Htab.to_csv(RES / "h_table.csv", index=False)
Htab.round(3)
""")

code(r"""
def describe_h(t):
    return pd.Series({"indices": len(t), "median H": t.H.median(), "min H": t.H.min(), "max H": t.H.max(),
                      "median se": t.se.median(), "share in [0.10,0.15]": ((t.H >= .10) & (t.H <= .15)).mean(),
                      "median nu": t.nu.median()})
Hsum = pd.DataFrame({"rv (Oxford-Man, 2000-2022)": describe_h(H_rv), "gk (Yahoo, 2000-2026)": describe_h(H_gk)})
Hsum.round(3)
""")

code(r"""
spx = ES.estimate_h(ES.log_vol(rv["SPX"].logv.to_numpy()))
blues = ["#a9c8f0", "#7aaae6", "#4d8bdc", "#2a6fc4", "#164f96"]   # sequential: light -> dark = small -> large q
fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.6))
ld = np.log(ES.LAGS)
zeta, icpt = ES.zeta_from_m(spx["m"])
for i, q in enumerate(ES.QS):
    axs[0].plot(ld, np.log(spx["m"][i]), "o", ms=2.5, color=blues[i])
    axs[0].plot(ld, icpt[i] + zeta[i] * ld, "-", lw=1, color=blues[i])
    axs[0].text(-0.08, icpt[i], f"q={q:g}", va="center", ha="right", fontsize=8, color="#52514e")
axs[0].set_xlabel(r"log $\Delta$"); axs[0].set_ylabel(r"log $m(q,\Delta)$"); axs[0].set_title("SPX (Oxford-Man RV)")
axs[0].set_xlim(-0.75, 4.05)
for name, df in rv.items():
    z = H_rv.set_index("index").loc[name]
    axs[1].plot(ES.QS, [z[f"zeta/q (q={q:g})"] * q for q in ES.QS], color=GREY, lw=0.6, alpha=0.6)
axs[1].plot(ES.QS, zeta, "o-", color=C1, label=f"SPX, H = {spx['H']:.3f}")
axs[1].plot(ES.QS, spx["H"] * ES.QS, "--", color=C2, lw=1, label=r"$qH$ line")
axs[1].set_xlabel("q"); axs[1].set_ylabel(r"$\zeta_q$"); axs[1].set_title(r"$\zeta_q$ vs $q$ (grey: other 30 indices)")
axs[1].legend()
fig.tight_layout(); fig.savefig(FIG / "scaling_spx.png"); plt.show()
""")

code(r"""
t = Htab.copy()
order = H_rv.sort_values("H")["index"].tolist()
order += [n for n in H_gk["index"] if n not in order]
fig, ax = plt.subplots(figsize=(9.5, 4))
for k, (tag, col, off) in enumerate([("rv", C1, -0.17), ("gk", C2, 0.17)]):
    s = t[t.data == tag].set_index("index").reindex(order)
    y = np.arange(len(order)) + off
    ax.errorbar(y, s.H, yerr=[s.H - s.ci_lo, s.ci_hi - s.H], fmt="o", ms=4, color=col, lw=1,
                label={"rv": "5-min RV (Oxford-Man, 2000-22)", "gk": "Garman-Klass (Yahoo, 2000-26)"}[tag])
ax.axhspan(0.10, 0.15, color="#e4e3df", alpha=0.6, lw=0)
ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=70, fontsize=8)
ax.set_ylabel("H (95% block-bootstrap CI)"); ax.set_title("Estimated Hurst exponent of log-volatility, every index")
ax.legend(loc="upper left")
fig.tight_layout(); fig.savefig(FIG / "h_by_index.png"); plt.show()
""")

md(r"""
### Are log-vol increments Gaussian at every lag?
The paper (Fig. 4) shows the distribution of $\log\sigma_{t+\Delta}-\log\sigma_t$ is close to
Gaussian for all $\Delta$. We show SPX and the skewness / excess kurtosis for every index.
""")

code(r"""
x = ES.log_vol(rv["SPX"].logv.to_numpy())
lags = (1, 5, 25, 125)
fig, axs = plt.subplots(1, 4, figsize=(10, 2.6))
from scipy import stats
for ax, d in zip(axs, lags):
    inc = x[d:] - x[:-d]; z = (inc - inc.mean()) / inc.std()
    ax.hist(z, bins=60, density=True, color=C1, alpha=0.75, edgecolor="white", linewidth=0.3)
    g = np.linspace(-5, 5, 200); ax.plot(g, stats.norm.pdf(g), color=C2, lw=1.2)
    ax.set_title(f"Δ = {d} days", fontsize=9); ax.set_xlim(-5, 5); ax.set_yticks([])
axs[0].set_ylabel("density (standardised)")
fig.suptitle("SPX log-vol increments vs N(0,1)", fontsize=10); fig.tight_layout()
fig.savefig(FIG / "gauss_spx.png"); plt.show()

rows = []
for tag, ser in (("rv", rv), ("gk", gk)):
    for name, df in ser.items():
        mom = ES.increment_moments(ES.log_vol(df.logv.to_numpy()), lags)
        for d in lags:
            rows.append({"data": tag, "index": name, "lag": d, **mom[d]})
gauss = pd.DataFrame(rows); gauss.to_csv(RES / "increment_moments.csv", index=False)
gauss.groupby(["data", "lag"])[["skew", "exkurt"]].describe().loc[:, (slice(None), ["25%", "50%", "75%"])].round(2)
""")

md(r"""
### Smoothing bias
Realized variance measures the *integrated* variance over a day, not the spot variance.
We simulate $\log\sigma_t = \nu W^H_t$ ($\nu = 0.3$, 50 steps per day, 5000 days), and estimate $H$
(i) from spot log-vol at day ends, (ii) from the log of daily integrated variance,
(iii) from (ii) plus i.i.d. measurement noise with s.d. 0.25 on log variance (a crude stand-in for RV
estimation error; range estimators are noisier). Three seeds each.
""")

code(r"""
rows = []
for Htrue in (0.05, 0.10, 0.15, 0.20, 0.30, 0.50):
    for seed in range(3):
        spot, liv = ES.simulate_rfsv(5000, Htrue, 0.3, steps_per_day=50, seed=seed)
        noisy = liv + 0.5 * np.random.default_rng(100 + seed).normal(0, 0.25, len(liv))
        rows.append({"H_true": Htrue, "seed": seed, "H_spot": ES.estimate_h(spot)["H"],
                     "H_integrated": ES.estimate_h(liv)["H"], "H_integrated+noise": ES.estimate_h(noisy)["H"]})
sim = pd.DataFrame(rows); sim.to_csv(RES / "smoothing_bias.csv", index=False)
sim_s = sim.groupby("H_true")[["H_spot", "H_integrated", "H_integrated+noise"]].mean()
sim_s.round(3)
""")

md(r"""
## 3. Forecasting competition

Walk-forward, one forecast per day per index; burn-in of 1000 days; parameters re-estimated
every 20 days on past data only (H and ν of RFSV; GARCH; calibration offsets), HAR-type
regressions refitted every day on all past (feature, target) pairs whose target is already observed.
Horizons h = 1, 5, 20; target $\log v_{t+h}$ (point target, as in the paper).

* **validation**: targets dated ≤ 2014-12-31 — used to pick one improved model per horizon.
* **test**: origins dated ≥ 2015-01-01 (to 2022-02-25 for `rv`, to 2026-10 for `gk`) — used once.

Losses: MSE of the log-variance forecast (ratios are identical for log-vol) and QLIKE on the
variance forecast. All models are scored on the same days.
""")

code(r"""
if RUN_PIPELINE or not (RES / "forecasts_rv.parquet").exists():
    run("run_forecasts.py")
print((RES / "lookahead_check.txt").read_text())
long = {tag: pd.read_parquet(RES / f"forecasts_{tag}.parquet") for tag in ("rv", "gk")}
lt = {tag: E.loss_table(d) for tag, d in long.items()}
pd.concat([v.assign(data=k) for k, v in lt.items()]).to_csv(RES / "loss_table.csv", index=False)
""")

md("### 3.1 Validation period: which improvement is selected (by log-MSE ratio to HAR, geometric mean over indices)")

code(r"""
sel = {tag: E.select_models(lt[tag]) for tag in lt}
print("selected on validation:", sel)
for tag in lt:
    v = E.ratio_summary(lt[tag], "val")
    print(f"\n[{tag}] validation, loss / HAR loss (geometric mean over indices)")
    print(v.pivot(index="model", columns="h", values=["gm_ratio_mse", "gm_ratio_qlike"]).round(3))
""")

md("### 3.2 Held-out test period")

code(r"""
def headline(tag):
    s = E.ratio_summary(lt[tag], "test")
    d = lt[tag][lt[tag].period == "test"]
    rows = []
    for h in E.HS:
        best = sel[tag][h]
        for lab, m in [("RW", "RW"), ("GARCH", "GARCH"), ("HAR", "HAR"), ("RFSV (paper)", "RFSV"), (f"Selected: {best}", best)]:
            r = s[(s.h == h) & (s.model == m)].iloc[0]
            mm = d[(d.h == h) & (d.model == m)]
            rows.append({"data": tag, "h": h, "model": lab, "mean MSE(log v)": mm.mse_log.mean(),
                         "MSE / HAR": r.gm_ratio_mse, "beats HAR (MSE)": f"{r.wins_mse}/{r.n_index}",
                         "mean QLIKE": mm.qlike.mean(), "QLIKE / HAR": r.gm_ratio_qlike,
                         "beats HAR (QLIKE)": f"{r.wins_qlike}/{r.n_index}"})
    return pd.DataFrame(rows)
head = pd.concat([headline("rv"), headline("gk")], ignore_index=True)
head.to_csv(RES / "test_headline.csv", index=False)
head.round(3)
""")

code(r"""
for tag in lt:
    s = E.ratio_summary(lt[tag], "test")
    print(f"\n[{tag}] TEST, all models incl. non-selected candidates (for transparency only; selection was made on validation)")
    print(s.pivot(index="model", columns="h", values=["gm_ratio_mse", "gm_ratio_qlike"]).round(3))
""")

md(r"""
### 3.3 Diebold-Mariano tests (test period, per index)
HAC (Bartlett, lag = max(h, 4(T/100)^{2/9})) DM statistic on loss differentials; 5% level.
Counts = number of indices where model A is significantly better / worse than B.
A Model Confidence Set was not run (cost/complexity); pairwise DM tests are reported instead.
""")

code(r"""
dms = []
for tag in long:
    d = long[tag][long[tag].period == "test"]
    pairs = [("RFSV", "HAR"), ("RFSV", "RW"), ("RFSV", "GARCH"), ("HAR", "GARCH")]
    pairs += [(m, "HAR") for m in sorted(set(sel[tag].values()))] + [(m, "RFSV") for m in sorted(set(sel[tag].values()))]
    dm = E.dm_table(d, pairs, "test").assign(data=tag)
    dms.append(dm)
dm = pd.concat(dms, ignore_index=True); dm.to_csv(RES / "dm_tests.csv", index=False)
cnt = pd.concat([E.dm_counts(dm[dm.data == t]).assign(data=t) for t in long], ignore_index=True)
# keep only the selected model rows at their own horizon
keep = cnt.A.isin(["RFSV", "HAR"]) | cnt.apply(lambda r: sel[r.data].get(int(r.h)) == r.A, axis=1)
cnt = cnt[keep]; cnt.to_csv(RES / "dm_counts.csv", index=False)
cnt.set_index(["data", "A", "B", "h"])
""")

code(r"""
# per-index test-period table: log-MSE ratio to HAR for RFSV and the selected model
rows = []
for tag in lt:
    d = lt[tag][lt[tag].period == "test"].set_index(["index", "h", "model"])
    for (idx, h), _ in d.groupby(level=[0, 1]):
        har = d.loc[(idx, h, "HAR")]
        r = {"data": tag, "index": idx, "h": h}
        for lab, m in [("RFSV", "RFSV"), ("selected", sel[tag][h])]:
            r[f"{lab} MSE/HAR"] = d.loc[(idx, h, m)].mse_log / har.mse_log
            r[f"{lab} QLIKE/HAR"] = d.loc[(idx, h, m)].qlike / har.qlike
        q = dm[(dm.data == tag) & (dm["index"] == idx) & (dm.h == h)].set_index(["A", "B"])
        r["p DM RFSV-HAR (MSE)"] = q.loc[("RFSV", "HAR")].p_mse
        r["p DM sel-HAR (MSE)"] = q.loc[(sel[tag][h], "HAR")].p_mse
        rows.append(r)
per_index = pd.DataFrame(rows); per_index.to_csv(RES / "test_per_index.csv", index=False)
per_index.pivot_table(index=["data", "index"], columns="h",
                      values=["RFSV MSE/HAR", "selected MSE/HAR"]).round(3)
""")

code(r"""
fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.6), sharey=False)
for ax, tag in zip(axs, ("rv", "gk")):
    p = per_index[per_index.data == tag]
    for j, h in enumerate(E.HS):
        q = p[p.h == h]
        jit = np.random.default_rng(j).uniform(-0.08, 0.08, len(q))
        ax.plot(j - 0.15 + jit, q["RFSV MSE/HAR"], "o", ms=3.5, color=C1, alpha=0.8, label="RFSV (paper)" if j == 0 else None)
        ax.plot(j + 0.15 + jit, q["selected MSE/HAR"], "o", ms=3.5, color=C2, alpha=0.8, label="validation-selected model" if j == 0 else None)
    ax.axhline(1, color="#52514e", lw=0.8)
    ax.set_xticks(range(3)); ax.set_xticklabels([f"h = {h}\n{sel[tag][h]}" for h in E.HS], fontsize=8)
    ax.set_title({"rv": "Oxford-Man RV, test 2015-2022", "gk": "Garman-Klass, test 2015-2026"}[tag], fontsize=9.5)
axs[0].set_ylabel("test log-MSE / HAR log-MSE (each dot = one index)")
fig.legend(*axs[0].get_legend_handles_labels(), loc="upper center", ncol=2, fontsize=8.5)
fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(FIG / "test_ratios.png"); plt.show()
""")

md(r"""
### 3.4 Calibration of the variance forecasts (why QLIKE ranks differ from MSE ranks)
For each index: mean of $v_{t+h}/\hat v_{t+h}$ on the test period (1 = unbiased level, < 1 = forecast too high);
the table shows the median across indices.
""")

code(r"""
rows = []
for tag, d in long.items():
    d = d[d.period == "test"].assign(ratio=lambda x: np.exp(x.y.astype(float)) / x.f_var.astype(float))
    per = d.groupby(["index", "h", "model"]).ratio.mean().groupby(["h", "model"]).median()
    rows.append(per.unstack(0).assign(data=tag))
bias = pd.concat(rows); bias.to_csv(RES / "variance_calibration.csv")
bias.round(3)
""")

code(r"""
# SPX example: 5-day-ahead log-vol forecasts around 2020
d = long["rv"]; d = d[(d["index"] == "SPX") & (d.h == 5)]
piv = d.pivot(index="target", columns="model", values="f_log").loc["2019-10-01":"2020-12-31"]
yv = d[d.model == "HAR"].set_index("target").y.loc["2019-10-01":"2020-12-31"]
fig, ax = plt.subplots(figsize=(9, 3.2))
ax.plot(yv.index, 0.5 * yv, color=GREY, lw=0.8, label="realised log-vol (target)")
ax.plot(piv.index, 0.5 * piv.HAR, color=C1, lw=1.2, label="HAR")
ax.plot(piv.index, 0.5 * piv.RFSV, color=C2, lw=1.2, label="RFSV")
ax.set_ylabel(r"log $\sigma$"); ax.set_title("SPX, 5-day-ahead forecasts (Oxford-Man RV)"); ax.legend(ncol=3, loc="upper left")
fig.tight_layout(); fig.savefig(FIG / "spx_forecast_2020.png"); plt.show()
""")

code(r"""
# key numbers for README / report / CV
summary = {
    "n_indices_rv": len(rv), "n_indices_gk": len(gk),
    "H_rv_median": float(H_rv.H.median()), "H_rv_min": float(H_rv.H.min()), "H_rv_max": float(H_rv.H.max()),
    "H_gk_median": float(H_gk.H.median()), "H_spx_rv": float(H_rv.set_index("index").loc["SPX", "H"]),
    "H_spx_rv_se": float(H_rv.set_index("index").loc["SPX", "se"]),
    "selected": {k: {str(h): m for h, m in v.items()} for k, v in sel.items()},
    "headline": head.to_dict(orient="records"),
}
(RES / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
print(json.dumps({k: v for k, v in summary.items() if k != "headline"}, indent=1, default=str))
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
(ROOT / "notebooks").mkdir(exist_ok=True)
nbf.write(nb, ROOT / "notebooks" / "rough_vol.ipynb")
print("written")
