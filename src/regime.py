"""
Market regime filter.

Classifies each day as bull / neutral / bear using two independent signals:
  1. SPY trend  — is the market above its 200-day moving average?
  2. VIX level  — how fearful is the market? (uses live intraday quote)

Position scalar:
  bull    (SPY > SMA200, VIX < 20)  → 1.0  — full exposure
  neutral (SPY > SMA200, VIX 20-30) → 0.6  — reduce size
  caution (SPY < SMA200, VIX 20-30) → 0.3  — minimal exposure
  bear    (VIX > 30 OR SPY < SMA200 by >5%) → 0.0  — cash
"""

import numpy as np
import pandas as pd
import yfinance as yf


def _live_quote(ticker: str) -> float | None:
    """Fetch the current intraday price for a ticker. Returns None if unavailable."""
    try:
        info = yf.Ticker(ticker).fast_info
        price = getattr(info, "last_price", None) or getattr(info, "regular_market_price", None)
        return float(price) if price else None
    except Exception:
        return None


def fetch_regime_data(start: str = "2013-01-01") -> tuple[pd.Series, pd.Series]:
    """
    Download SPY and VIX historical closes, then patch in the live intraday
    quote so today's regime reflects current market conditions, not yesterday's close.
    """
    raw = yf.download(["SPY", "^VIX"], start=start, auto_adjust=True, progress=False)
    spy = raw["Close"]["SPY"].ffill()
    vix = raw["Close"]["^VIX"].ffill()

    today = pd.Timestamp.now().normalize()

    # Patch with live quotes if market is open or recently closed
    live_spy = _live_quote("SPY")
    live_vix = _live_quote("^VIX")

    if live_spy:
        spy[today] = live_spy
        spy = spy.sort_index()
    if live_vix:
        vix[today] = live_vix
        vix = vix.sort_index()

    return spy, vix


def compute_regime_series(
    spy: pd.Series,
    vix: pd.Series,
    sma_window: int = 200,
    vix_bull: float = 20.0,
    vix_bear: float = 30.0,
) -> pd.DataFrame:
    """
    Compute daily regime label and position scalar.
    Returns DataFrame with columns: [regime, scalar, spy_sma200, spy_vs_sma, vix].
    """
    sma200 = spy.rolling(sma_window).mean()
    spy_vs_sma = (spy - sma200) / sma200

    vix_aligned = vix.reindex(spy.index).ffill()

    regime = pd.Series("bull", index=spy.index)
    scalar = pd.Series(1.0, index=spy.index)

    mask_neutral = (spy > sma200) & (vix_aligned >= vix_bull) & (vix_aligned < vix_bear)
    regime[mask_neutral] = "neutral"
    scalar[mask_neutral] = 0.6

    mask_caution = (spy < sma200) & (vix_aligned < vix_bear)
    regime[mask_caution] = "caution"
    scalar[mask_caution] = 0.3

    mask_bear = (vix_aligned >= vix_bear) | (spy_vs_sma < -0.05)
    regime[mask_bear] = "bear"
    scalar[mask_bear] = 0.0

    return pd.DataFrame({
        "regime": regime,
        "scalar": scalar,
        "spy_sma200": sma200,
        "spy_vs_sma": spy_vs_sma,
        "vix": vix_aligned,
    })


def get_current_regime(spy: pd.Series, vix: pd.Series) -> dict:
    """Return today's regime for use in daily_signals.py."""
    df = compute_regime_series(spy, vix)
    latest = df.iloc[-1]
    return {
        "regime": latest["regime"],
        "scalar": latest["scalar"],
        "vix": round(latest["vix"], 1),
        "spy_vs_sma200": round(latest["spy_vs_sma"] * 100, 1),
        "date": df.index[-1].date(),
    }
