"""Download the raw data into data/raw/.

1. Oxford-Man Institute Realized Library (final public version, discontinued
   in 2022) from the Internet Archive's copy of the official zip.
2. Daily OHLC for major indices from Yahoo Finance through yfinance.

Downloaded files are treated as untrusted data: this script only writes them
to disk; it never executes anything from them. Run it with `python -I`.
"""
from __future__ import annotations

import site
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, site.getusersitepackages())  # -I disables the user site
sys.path.insert(0, str(ROOT))

from src.data import OXFORD_MAN_URL, YF_TICKERS  # noqa: E402

RAW = ROOT / "data" / "raw"


def download_oxford_man(dest: Path) -> None:
    if dest.exists():
        print(f"exists: {dest}")
        return
    print(f"downloading {OXFORD_MAN_URL}")
    urllib.request.urlretrieve(OXFORD_MAN_URL, dest)


def download_yfinance(dest_dir: Path, start: str = "2000-01-01",
                      end: str | None = None) -> None:
    import yfinance as yf

    dest_dir.mkdir(parents=True, exist_ok=True)
    for name, ticker in YF_TICKERS.items():
        out = dest_dir / f"{name}.csv"
        if out.exists():
            print(f"exists: {out}")
            continue
        df = yf.download(ticker, start=start, end=end, auto_adjust=False,
                         progress=False, multi_level_index=False)
        if df is None or df.empty:
            print(f"no data for {ticker}")
            continue
        df[["Open", "High", "Low", "Close"]].to_csv(out)
        print(f"{ticker}: {len(df)} rows, {df.index.min().date()} .. {df.index.max().date()}")


if __name__ == "__main__":
    RAW.mkdir(parents=True, exist_ok=True)
    download_oxford_man(RAW / "oxfordman.zip")
    download_yfinance(RAW / "yfinance", end=sys.argv[1] if len(sys.argv) > 1 else None)
