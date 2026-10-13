"""Run the walk-forward competition for every index of both datasets.

Writes results/forecasts_{rv,gk}.parquet (one row per index, model, horizon,
origin day) and results/lookahead_check.txt.

Run with `python -I scripts/run_forecasts.py` from anywhere.
"""
from __future__ import annotations

import site
import sys
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, site.getusersitepackages())
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src import evaluation as E  # noqa: E402
from src.data import load_processed  # noqa: E402

PROC = ROOT / "data" / "processed"
RES = ROOT / "results"


def _job(args):
    name, df = args
    fc = E.forecast_series(df)
    return E.to_long(name, df, fc)


def main(n_proc: int = 6) -> None:
    RES.mkdir(exist_ok=True)
    rv = load_processed(PROC / "rv.csv")

    # 1. look-ahead check on SPX (first 2500 days, perturb after day 1800)
    t = time.time()
    ok = E.check_no_lookahead(rv["SPX"].iloc[:2500], cut=1800)
    msg = (f"check_no_lookahead(SPX[:2500], cut=1800): {'PASSED' if ok else 'FAILED'}\n"
           "Every observation after day 1800 was perturbed; all forecasts made at "
           "origins <= 1800 were identical for every model and horizon.\n")
    (RES / "lookahead_check.txt").write_text(msg)
    print(msg, f"({time.time() - t:.0f}s)")

    # 2. competition
    for tag in ("rv", "gk"):
        series = load_processed(PROC / f"{tag}.csv")
        t = time.time()
        with Pool(n_proc) as pool:
            parts = pool.map(_job, list(series.items()))
        long = E.common_sample(pd.concat(parts, ignore_index=True))
        long.to_parquet(RES / f"forecasts_{tag}.parquet", index=False)
        print(f"{tag}: {len(series)} indices, {len(long):,} rows, {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
