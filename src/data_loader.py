"""
Downloads and caches S&P 500 OHLCV data from Yahoo Finance.
Uses parquet for fast local caching — subsequent loads take <1s.
"""

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

PRICE_FILE = DATA_DIR / "prices.parquet"
META_FILE = DATA_DIR / "metadata.parquet"


def get_sp500_tickers() -> list[str]:
    """Scrape current S&P 500 constituents from Wikipedia."""
    tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    df = tables[0]
    tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
    return sorted(tickers)


def download_prices(
    tickers: list[str],
    start: str = "2014-01-01",
    end: str | None = None,
    batch_size: int = 50,
) -> pd.DataFrame:
    """
    Download adjusted close + volume for all tickers.
    Returns MultiIndex DataFrame: columns = (field, ticker).
    Batches requests to avoid Yahoo rate limits.
    """
    all_data = []
    failed = []

    batches = [tickers[i : i + batch_size] for i in range(0, len(tickers), batch_size)]

    for batch in tqdm(batches, desc="Downloading price data"):
        try:
            raw = yf.download(
                batch,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            if isinstance(raw.columns, pd.MultiIndex):
                all_data.append(raw)
            else:
                # Single ticker returned flat columns
                ticker = batch[0]
                raw.columns = pd.MultiIndex.from_tuples(
                    [(col, ticker) for col in raw.columns]
                )
                all_data.append(raw)
        except Exception as e:
            print(f"  Batch failed: {batch[:3]}... — {e}")
            failed.extend(batch)
        time.sleep(0.5)

    if not all_data:
        raise RuntimeError("All downloads failed.")

    prices = pd.concat(all_data, axis=1)
    prices = prices.sort_index(axis=1)

    if failed:
        print(f"\nFailed tickers ({len(failed)}): {failed[:10]}...")

    return prices


def load_or_download(
    start: str = "2014-01-01",
    force_refresh: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns (close, volume, high_low_range) as per-ticker DataFrames (date x ticker).
    Refreshes cache if data is older than 1 day or force_refresh=True.
    """
    cache_stale = (
        force_refresh
        or not PRICE_FILE.exists()
        or (
            pd.Timestamp.now() - pd.Timestamp(PRICE_FILE.stat().st_mtime, unit="s")
        ).days >= 1
    )

    if not cache_stale:
        print("Loading from cache...")
        prices = pd.read_parquet(PRICE_FILE)
    else:
        print("Fetching S&P 500 tickers...")
        tickers = get_sp500_tickers()
        print(f"Found {len(tickers)} tickers. Downloading {start} → today...")
        prices = download_prices(tickers, start=start)
        prices.to_parquet(PRICE_FILE)
        print(f"Saved to {PRICE_FILE}")

    # Extract clean per-field DataFrames
    fields = prices.columns.get_level_values(0).unique()

    close = prices["Close"] if "Close" in fields else prices["Adj Close"]
    volume = prices["Volume"]
    high = prices["High"]
    low = prices["Low"]
    open_ = prices["Open"]

    # Drop tickers with too many missing values (>20% of rows)
    threshold = 0.80
    close = close.loc[:, close.notna().mean() >= threshold]
    tickers_kept = close.columns
    volume = volume[tickers_kept]
    high = high[tickers_kept]
    low = low[tickers_kept]
    open_ = open_[tickers_kept]

    print(f"Loaded {len(tickers_kept)} tickers × {len(close)} days")
    return close, volume, high, low, open_
