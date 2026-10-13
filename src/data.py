"""Data loading and volatility proxies.

Two kinds of daily variance series are used and are never mixed:

* ``"rv"``  -- 5-minute realized variance (``rv5``) from the Oxford-Man
  Institute Realized Library, 2000-01-03 .. 2022-02-25 (31 indices).
  This is the data used in Gatheral, Jaisson & Rosenbaum (2018).
* ``"gk"``  -- range-based Garman-Klass variance computed from Yahoo Finance
  daily OHLC, 2000 .. 2026. Parkinson is computed as a cross-check.

Every loader returns, per index, a DataFrame indexed by date with columns
``var`` (daily variance, > 0), ``logv`` (= log var) and ``ret`` (daily log
return used by GARCH and the leverage term: open-to-close for Oxford-Man,
close-to-close for Yahoo).
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

OXFORD_MAN_URL = (
    "https://web.archive.org/web/20220301022212id_/"
    "https://realized.oxford-man.ox.ac.uk/images/oxfordmanrealizedvolatilityindices.zip"
)

# Oxford-Man symbol -> short name
OM_NAMES = {
    ".AEX": "AEX", ".AORD": "AORD", ".BFX": "BFX", ".BSESN": "BSESN",
    ".BVLG": "BVLG", ".BVSP": "BVSP", ".DJI": "DJI", ".FCHI": "FCHI",
    ".FTMIB": "FTMIB", ".FTSE": "FTSE", ".GDAXI": "GDAXI", ".GSPTSE": "GSPTSE",
    ".HSI": "HSI", ".IBEX": "IBEX", ".IXIC": "IXIC", ".KS11": "KS11",
    ".KSE": "KSE", ".MXX": "MXX", ".N225": "N225", ".NSEI": "NSEI",
    ".OMXC20": "OMXC20", ".OMXHPI": "OMXHPI", ".OMXSPI": "OMXSPI",
    ".OSEAX": "OSEAX", ".RUT": "RUT", ".SMSI": "SMSI", ".SPX": "SPX",
    ".SSEC": "SSEC", ".SSMI": "SSMI", ".STI": "STI", ".STOXX50E": "STOXX50E",
}

# Short name -> Yahoo ticker for the range-based extension to 2026
YF_TICKERS = {
    "SPX": "^GSPC", "DJI": "^DJI", "IXIC": "^IXIC", "RUT": "^RUT",
    "STOXX50E": "^STOXX50E", "FTSE": "^FTSE", "GDAXI": "^GDAXI",
    "FCHI": "^FCHI", "AEX": "^AEX", "SSMI": "^SSMI", "IBEX": "^IBEX",
    "N225": "^N225", "HSI": "^HSI", "KS11": "^KS11", "AORD": "^AORD",
    "GSPTSE": "^GSPTSE", "BVSP": "^BVSP", "MXX": "^MXX", "BSESN": "^BSESN",
    "NSEI": "^NSEI",
}

MIN_OBS = 2000  # indices with fewer valid days are dropped (none in practice)


def load_oxford_man(zip_path: str | Path, measure: str = "rv5") -> dict[str, pd.DataFrame]:
    """Read the Oxford-Man zip (CSV inside is parsed as data only)."""
    with zipfile.ZipFile(zip_path) as z:
        raw = pd.read_csv(io.BytesIO(z.read(z.namelist()[0])))
    raw = raw.rename(columns={raw.columns[0]: "date"})
    raw["date"] = pd.to_datetime(raw["date"].str[:10])
    out = {}
    for sym, g in raw.groupby("Symbol"):
        name = OM_NAMES.get(sym, sym.lstrip("."))
        g = g.set_index("date").sort_index()
        df = pd.DataFrame({
            "var": pd.to_numeric(g[measure], errors="coerce"),
            "ret": pd.to_numeric(g["open_to_close"], errors="coerce"),
        })
        df = df[(df["var"] > 0) & np.isfinite(df["var"])].copy()
        df["ret"] = df["ret"].fillna(0.0)
        df["logv"] = np.log(df["var"])
        df = df[~df.index.duplicated()]
        if len(df) >= MIN_OBS:
            out[name] = df
    return dict(sorted(out.items()))


def garman_klass(o, h, l, c):
    """Garman-Klass (1980) daily variance from OHLC (open-to-close session)."""
    hl = np.log(h / l)
    co = np.log(c / o)
    return 0.5 * hl**2 - (2 * np.log(2) - 1) * co**2


def parkinson(h, l):
    """Parkinson (1980) daily variance from the high-low range."""
    return np.log(h / l) ** 2 / (4 * np.log(2))


def load_yfinance_dir(dir_path: str | Path) -> dict[str, pd.DataFrame]:
    """Build Garman-Klass / Parkinson series from the downloaded OHLC CSVs.

    Cleaning: drop rows with missing prices, non-positive prices, high < low,
    open or close outside [low, high], and zero-range days (H == L), which
    are stale quotes. GK can be <= 0 on rare days; those are dropped too.
    """
    out = {}
    for f in sorted(Path(dir_path).glob("*.csv")):
        d = pd.read_csv(f, index_col=0, parse_dates=True)
        d = d[["Open", "High", "Low", "Close"]].apply(pd.to_numeric, errors="coerce").dropna()
        d = d[(d > 0).all(axis=1)]
        ok = ((d.High > d.Low) & (d.Open <= d.High) & (d.Open >= d.Low)
              & (d.Close <= d.High) & (d.Close >= d.Low))
        ret = np.log(d.Close).diff()  # close-to-close, before row filtering
        d = d[ok].copy()
        d["ret"] = ret.reindex(d.index).fillna(0.0)
        d["var"] = garman_klass(d.Open, d.High, d.Low, d.Close)
        d["park"] = parkinson(d.High, d.Low)
        d = d[d["var"] > 0]
        d["logv"] = np.log(d["var"])
        d.index = pd.to_datetime(d.index).tz_localize(None).normalize()
        d = d[~d.index.duplicated()]
        if len(d) >= MIN_OBS:
            out[f.stem] = d[["var", "logv", "ret", "park"]]
    return out


def save_processed(series: dict[str, pd.DataFrame], path: str | Path) -> None:
    """Stack a dict of per-index frames into one long CSV."""
    frames = [df.assign(index_name=k) for k, df in series.items()]
    pd.concat(frames).rename_axis("date").to_csv(path)


def load_processed(path: str | Path) -> dict[str, pd.DataFrame]:
    d = pd.read_csv(path, index_col=0, parse_dates=True)
    return {k: g.drop(columns="index_name") for k, g in d.groupby("index_name")}
