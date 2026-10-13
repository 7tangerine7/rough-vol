"""Write the LaTeX tables of report/report.tex from results/*.csv, so every
number in the report comes straight from the pipeline output."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "report" / "tables"
OUT.mkdir(parents=True, exist_ok=True)


def esc(s: str) -> str:
    return str(s).replace("_", r"\_").replace("&", r"\&")


def h_table():
    h = pd.read_csv(RES / "h_table.csv")
    rv = h[h.data == "rv"].set_index("index")
    gk = h[h.data == "gk"].set_index("index")
    names = list(rv.index)
    half = (len(names) + 1) // 2
    lines = [r"\begin{tabular}{lrrr@{\hskip 18pt}lrrr}", r"\toprule",
             r"Index & $H_{RV}$ (s.e.) & $H_{GK}$ (s.e.) & $\nu_{RV}$ & Index & $H_{RV}$ (s.e.) & $H_{GK}$ (s.e.) & $\nu_{RV}$\\",
             r"\midrule"]

    def cell(n):
        a = f"{rv.loc[n, 'H']:.3f} ({rv.loc[n, 'se']:.3f})"
        b = f"{gk.loc[n, 'H']:.3f} ({gk.loc[n, 'se']:.3f})" if n in gk.index else "--"
        return f"{esc(n)} & {a} & {b} & {rv.loc[n, 'nu']:.2f}"

    for i in range(half):
        left = cell(names[i])
        right = cell(names[i + half]) if i + half < len(names) else "& & &"
        lines.append(f"{left} & {right}\\\\")
    lines += [r"\midrule",
              f"\\multicolumn{{8}}{{l}}{{Median $H$: RV {rv.H.median():.3f} (31 indices, range {rv.H.min():.3f}--{rv.H.max():.3f}); "
              f"GK {gk.H.median():.3f} (20 indices, range {gk.H.min():.3f}--{gk.H.max():.3f}).}}\\\\",
              r"\bottomrule", r"\end{tabular}"]
    (OUT / "h_table.tex").write_text("\n".join(lines))


def sim_table():
    s = pd.read_csv(RES / "smoothing_bias.csv").groupby("H_true")[["H_spot", "H_integrated", "H_integrated+noise"]].mean()
    lines = [r"\begin{tabular}{rrrr}", r"\toprule",
             r"$H$ true & spot & integrated & integrated + noise\\", r"\midrule"]
    for H, r in s.iterrows():
        lines.append(f"{H:.2f} & {r.H_spot:.3f} & {r.H_integrated:.3f} & {r['H_integrated+noise']:.3f}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "sim_table.tex").write_text("\n".join(lines))


def gauss_table():
    g = pd.read_csv(RES / "increment_moments.csv")
    m = g.groupby(["data", "lag"])[["skew", "exkurt"]].median()
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r" & \multicolumn{2}{c}{RV (31)} & \multicolumn{2}{c}{GK (20)}\\",
             r"Lag $\Delta$ & skew & exc. kurt. & skew & exc. kurt.\\", r"\midrule"]
    for lag in sorted(g.lag.unique()):
        a, b = m.loc[("rv", lag)], m.loc[("gk", lag)]
        lines.append(f"{lag} & {a["skew"]:.2f} & {a["exkurt"]:.2f} & {b["skew"]:.2f} & {b["exkurt"]:.2f}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "gauss_table.tex").write_text("\n".join(lines))


def ratio_table(period: str, fname: str):
    lt = pd.read_csv(RES / "loss_table.csv")
    import sys
    sys.path.insert(0, str(ROOT))
    from src.evaluation import ratio_summary
    order = ["RW", "GARCH", "HAR", "RFSV", "RFSV-roll250", "RFSV-roll500", "RFSV-roll1000",
             "HAR-L", "HAR-RFSV", "HAR-L-RFSV", "COMB(RFSV,HAR)", "COMB(RFSV,HAR-L)"]
    sel = json.loads((RES / "summary.json").read_text())["selected"]
    lines = [r"\begin{tabular}{lrrrrrr@{\hskip 14pt}rrrrrr}", r"\toprule",
             r" & \multicolumn{6}{c}{Oxford-Man RV (31 indices)} & \multicolumn{6}{c}{Garman--Klass (20 indices)}\\",
             r" & \multicolumn{3}{c}{MSE / HAR} & \multicolumn{3}{c}{QLIKE / HAR} & \multicolumn{3}{c}{MSE / HAR} & \multicolumn{3}{c}{QLIKE / HAR}\\",
             r"Model & 1 & 5 & 20 & 1 & 5 & 20 & 1 & 5 & 20 & 1 & 5 & 20\\", r"\midrule"]
    s = {t: ratio_summary(lt[lt.data == t], period).set_index(["model", "h"]) for t in ("rv", "gk")}
    for m in order:
        cells = []
        for t in ("rv", "gk"):
            for col in ("gm_ratio_mse", "gm_ratio_qlike"):
                for h in (1, 5, 20):
                    v = s[t].loc[(m, h), col]
                    txt = f"{v:.3f}"
                    if col == "gm_ratio_mse" and sel[t][str(h)] == m:
                        txt = r"\textbf{" + txt + "}"
                    cells.append(txt)
        lines.append(esc(m) + " & " + " & ".join(cells) + r"\\")
        if m in ("RFSV", "RFSV-roll1000", "HAR-L-RFSV"):
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / fname).write_text("\n".join(lines))


def dm_tab():
    c = pd.read_csv(RES / "dm_counts.csv")
    sel = json.loads((RES / "summary.json").read_text())["selected"]
    lines = [r"\begin{tabular}{llrcccc}", r"\toprule",
             r"Data & Comparison (A vs B) & $h$ & \multicolumn{2}{c}{log-MSE: A better / worse} & \multicolumn{2}{c}{QLIKE: A better / worse}\\",
             r"\midrule"]
    for t in ("rv", "gk"):
        rows = []
        for h in (1, 5, 20):
            rows.append(("RFSV", "HAR", h))
        for h in (1, 5, 20):
            rows.append((sel[t][str(h)], "HAR", h))
            rows.append((sel[t][str(h)], "RFSV", h))
        for a, b, h in rows:
            r = c[(c.data == t) & (c.A == a) & (c.B == b) & (c.h == h)].iloc[0]
            lines.append(f"{t.upper()} & {esc(a)} vs {b} & {h} & {r.A_better_mse} & {r.A_worse_mse} & "
                         f"{r.A_better_qlike} & {r.A_worse_qlike}\\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    (OUT / "dm_table.tex").write_text("\n".join(lines))


if __name__ == "__main__":
    h_table(); sim_table(); gauss_table()
    ratio_table("test", "test_table.tex"); ratio_table("val", "val_table.tex"); dm_tab()
    print("tables written to", OUT)
