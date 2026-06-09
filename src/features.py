"""
Feature engineering for cross-sectional equity prediction.

Two categories of features:
  1. Time-series features per ticker (momentum, mean-reversion, volatility, volume)
  2. Cross-sectional ranks — neutralizes market-wide moves so the model
     predicts relative outperformance, not absolute direction.
"""

import numpy as np
import pandas as pd
from scipy.stats import rankdata


# ─────────────────────────────────────────────
# Time-series features
# ─────────────────────────────────────────────

def _pct_change(s: pd.Series, n: int) -> pd.Series:
    return s.pct_change(n)


def _rolling_std(s: pd.Series, n: int) -> pd.Series:
    return s.pct_change().rolling(n).std()


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd_signal(close: pd.Series) -> pd.Series:
    """MACD histogram (MACD line minus signal line)."""
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return (macd - signal) / close  # normalised by price


def _bb_position(close: pd.Series, n: int = 20) -> pd.Series:
    """Position within Bollinger Bands: 0 = lower band, 1 = upper band."""
    sma = close.rolling(n).mean()
    std = close.rolling(n).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    return (close - lower) / (upper - lower + 1e-10)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    tr = pd.concat(
        [
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(n).mean() / close  # normalised ATR


def _stochastic(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    lowest_low = low.rolling(n).min()
    highest_high = high.rolling(n).max()
    return (close - lowest_low) / (highest_high - lowest_low + 1e-10)


def _volume_ratio(volume: pd.Series, n: int = 20) -> pd.Series:
    return volume / volume.rolling(n).mean()


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def _distance_from_high(close: pd.Series, n: int = 52 * 5) -> pd.Series:
    """Distance from n-day high (negative = below high)."""
    rolling_high = close.rolling(n).max()
    return (close - rolling_high) / rolling_high


def _trend_strength(close: pd.Series, n: int = 20) -> pd.Series:
    """Linear regression slope normalised by price."""
    x = np.arange(n)
    def slope(y):
        if np.isnan(y).any():
            return np.nan
        return np.polyfit(x, y, 1)[0]
    return close.rolling(n).apply(slope, raw=True) / close


def build_features_for_ticker(
    ticker: str,
    close: pd.Series,
    volume: pd.Series,
    high: pd.Series,
    low: pd.Series,
) -> pd.DataFrame:
    """Compute all time-series features for a single ticker."""
    c = close.copy()
    v = volume.copy()
    h = high.copy()
    l = low.copy()

    # Ensure inputs are Series (yfinance single-ticker downloads return DataFrames)
    if isinstance(c, pd.DataFrame):
        c = c.squeeze()
    if isinstance(v, pd.DataFrame):
        v = v.squeeze()
    if isinstance(h, pd.DataFrame):
        h = h.squeeze()
    if isinstance(l, pd.DataFrame):
        l = l.squeeze()

    feat = pd.DataFrame(index=c.index)

    # Momentum — multiple lookbacks
    for n in [1, 2, 3, 5, 10, 21, 42, 63, 126, 252]:
        feat[f"ret_{n}d"] = _pct_change(c, n)

    # Skip-1 month momentum (avoid short-term reversal contaminating medium momentum)
    feat["mom_12_1"] = _pct_change(c, 252) - _pct_change(c, 21)

    # Volatility
    for n in [5, 21, 63]:
        feat[f"vol_{n}d"] = _rolling_std(c, n)

    feat["atr_14"] = _atr(h, l, c, 14)

    # Mean reversion signals
    feat["rsi_14"] = _rsi(c, 14)
    feat["rsi_28"] = _rsi(c, 28)
    feat["bb_pos_20"] = _bb_position(c, 20)
    feat["stoch_14"] = _stochastic(h, l, c, 14)
    feat["macd_hist"] = _macd_signal(c)

    # Trend
    feat["trend_20"] = _trend_strength(c, 20)

    # Price level features
    feat["dist_52w_high"] = _distance_from_high(c, 252)
    feat["dist_20d_high"] = _distance_from_high(c, 20)

    # Price vs moving averages
    for n in [10, 21, 50, 200]:
        sma = c.rolling(n).mean()
        feat[f"price_vs_sma{n}"] = (c - sma) / sma

    # Volume
    feat["vol_ratio_10"] = _volume_ratio(v, 10)
    feat["vol_ratio_20"] = _volume_ratio(v, 20)

    # OBV momentum
    obv = _obv(c, v)
    feat["obv_ret_5"] = _pct_change(obv, 5)
    feat["obv_ret_21"] = _pct_change(obv, 21)

    # Gap features — overnight price jumps carry strong signal
    # open_ not passed in, approximate overnight gap from close-to-close vs intraday
    feat["intraday_ret"] = (c - l) / (h - l + 1e-10)          # where in day's range close sits
    feat["hl_range"] = (h - l) / c                             # normalised daily range (fear indicator)
    feat["body_size"] = ((c - c.shift()) / c.shift()).abs()    # absolute daily move

    # Mean reversion over short windows
    feat["ret_1d_sq"] = feat["ret_1d"] ** 2                    # squared return (shock signal)
    feat["reversal_5d"] = -feat["ret_5d"]                      # short-term mean reversion counter-signal

    # Trend consistency — what fraction of last N days were positive
    for n in [5, 10, 21]:
        feat[f"up_days_{n}"] = (c.diff() > 0).rolling(n).mean()

    # Volume momentum interaction
    feat["vol_price_confirm"] = feat["ret_5d"] * feat["vol_ratio_10"]  # strong move + high vol = confirmed

    # Volatility regime
    feat["vol_ratio_21_63"] = feat["vol_21d"] / (feat["vol_63d"] + 1e-10)  # expanding vs contracting vol

    return feat


# ─────────────────────────────────────────────
# Cross-sectional normalisation
# ─────────────────────────────────────────────

def cross_sectional_rank(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert each feature to cross-sectional rank [0, 1] on each date.
    This neutralises market-wide effects — the model predicts relative
    performance, not absolute direction.
    """
    def rank_row(row):
        valid = row.notna()
        if valid.sum() < 5:
            return row
        ranks = rankdata(row[valid], method="average")
        row[valid] = (ranks - 1) / (valid.sum() - 1)
        return row

    return df.apply(rank_row, axis=1)


def build_feature_matrix(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    target_horizon: int = 1,
) -> pd.DataFrame:
    """
    Build a stacked (date × ticker) feature matrix with target variable.

    Target: forward return over `target_horizon` days, ranked cross-sectionally.
    Returns a DataFrame with a MultiIndex (date, ticker).
    """
    from tqdm import tqdm

    tickers = close.columns.tolist()
    all_features = []

    print(f"Engineering features for {len(tickers)} tickers...")
    for ticker in tqdm(tickers):
        try:
            feat = build_features_for_ticker(
                ticker,
                close[ticker],
                volume[ticker],
                high[ticker],
                low[ticker],
            )
            feat["ticker"] = ticker
            all_features.append(feat)
        except Exception as e:
            print(f"  Skipping {ticker}: {e}")

    stacked = pd.concat(all_features)
    stacked = stacked.reset_index().rename(columns={"index": "date", "Date": "date"})
    stacked = stacked.set_index(["date", "ticker"]).sort_index()

    # fwd_return = ALWAYS 1-day return — used by the backtest for daily P&L.
    # Keeping this as 1-day regardless of target_horizon prevents the backtest
    # from counting multi-day returns multiple times during a holding period.
    fwd_1d = close.pct_change(1).shift(-1)
    fwd_1d_stacked = fwd_1d.stack()
    fwd_1d_stacked.index.names = ["date", "ticker"]
    stacked["fwd_return"] = fwd_1d_stacked

    # target = horizon-day return rank — what the model is trained to predict.
    # A longer horizon smooths out noise and suits the intended holding period.
    print("Computing cross-sectional ranks...")
    if target_horizon > 1:
        fwd_horizon = close.pct_change(target_horizon).shift(-target_horizon)
        fwd_horizon_stacked = fwd_horizon.stack()
        fwd_horizon_stacked.index.names = ["date", "ticker"]
        target_base = fwd_horizon_stacked
    else:
        target_base = fwd_1d_stacked

    stacked["target"] = (
        target_base
        .groupby(level="date")
        .rank(pct=True)
    )

    # Cross-sectional rank all features
    feature_cols = [c for c in stacked.columns if c not in ("fwd_return", "target")]
    stacked[feature_cols] = (
        stacked[feature_cols]
        .groupby(level="date")
        .rank(pct=True)
    )

    return stacked
