"""Parse data/raw/ into data/processed/{rv,gk}.csv plus a quality table.

Run with `python -I scripts/build_data.py` (raw files are untrusted data).
"""
from __future__ import annotations

import site
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, site.getusersitepackages())
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data import load_oxford_man, load_yfinance_dir, save_processed  # noqa: E402

RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"


def main() -> None:
    PROC.mkdir(parents=True, exist_ok=True)
    rv = load_oxford_man(RAW / "oxfordman.zip")
    gk = load_yfinance_dir(RAW / "yfinance")
    save_processed(rv, PROC / "rv.csv")
    save_processed(gk, PROC / "gk.csv")

    rows = []
    for name in sorted(set(rv) | set(gk)):
        r = {"index": name}
        if name in rv:
            d = rv[name]
            r.update(rv_start=d.index[0].date(), rv_end=d.index[-1].date(), rv_n=len(d))
        if name in gk:
            g = gk[name]
            raw_n = len(pd.read_csv(RAW / "yfinance" / f"{name}.csv"))
            r.update(gk_start=g.index[0].date(), gk_end=g.index[-1].date(), gk_n=len(g),
                     gk_dropped=raw_n - len(g),
                     corr_gk_park=np.corrcoef(np.log(g["var"]), np.log(g["park"]))[0, 1])
        if name in rv and name in gk:
            j = rv[name][["logv"]].join(gk[name][["logv"]], rsuffix="_gk", how="inner")
            r["corr_logrv_loggk"] = j.corr().iloc[0, 1]
            r["overlap_n"] = len(j)
        rows.append(r)
    qa = pd.DataFrame(rows).set_index("index")
    qa.to_csv(PROC / "data_quality.csv")
    pd.set_option("display.width", 200)
    print(qa.round(3).to_string())


if __name__ == "__main__":
    main()
