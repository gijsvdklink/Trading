"""
Daily signal generator — run this each morning before market open.

Loads the trained model, downloads latest price data, engineers features
for the most recent date, and outputs ranked buy signals.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def generate_signals(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    model,
    feature_cols: list[str],
    top_n: int = 10,
    min_price: float = 5.0,
    min_avg_volume: float = 500_000,
) -> pd.DataFrame:
    """
    Generate ranked buy signals for today.

    Filters:
        - Price >= min_price (avoid penny stocks/ETF tracking issues)
        - 20-day avg volume >= min_avg_volume shares (liquidity filter)
        - Only tickers with a full feature vector (no excessive NaNs)

    Returns DataFrame sorted by predicted rank (highest first).
    """
    from src.features import build_features_for_ticker
    from tqdm import tqdm

    latest_date = close.index[-1]
    tickers = close.columns.tolist()

    signals = []
    for ticker in tqdm(tickers, desc="Computing signals", ncols=80):
        try:
            c = close[ticker].dropna()
            v = volume[ticker].dropna()
            h = high[ticker].dropna()
            l = low[ticker].dropna()

            # Liquidity and price filters
            current_price = c.iloc[-1]
            avg_volume = v.tail(20).mean()
            if current_price < min_price or avg_volume < min_avg_volume:
                continue

            feat = build_features_for_ticker(ticker, c, v, h, l)
            latest_feat = feat.iloc[[-1]][feature_cols]

            nan_pct = latest_feat.isna().mean(axis=1).iloc[0]
            if nan_pct > 0.3:  # Skip if >30% features are NaN
                continue

            # Fill remaining NaN with 0.5 (neutral cross-sectional rank)
            X = latest_feat.fillna(0.5).values
            pred = model.predict(X)[0]

            signals.append(
                {
                    "ticker": ticker,
                    "predicted_rank": pred,
                    "price": current_price,
                    "avg_volume_20d": avg_volume,
                    "rsi_14": feat["rsi_14"].iloc[-1] if "rsi_14" in feat.columns else np.nan,
                    "ret_1d": feat["ret_1d"].iloc[-1] if "ret_1d" in feat.columns else np.nan,
                    "ret_5d": feat["ret_5d"].iloc[-1] if "ret_5d" in feat.columns else np.nan,
                    "ret_21d": feat["ret_21d"].iloc[-1] if "ret_21d" in feat.columns else np.nan,
                    "vol_21d": feat["vol_21d"].iloc[-1] if "vol_21d" in feat.columns else np.nan,
                }
            )
        except Exception:
            continue

    if not signals:
        print("No signals generated — check data or filters.")
        return pd.DataFrame()

    df = pd.DataFrame(signals).sort_values("predicted_rank", ascending=False)
    df["rank"] = range(1, len(df) + 1)

    # Cross-sectional rank normalisation
    n = len(df)
    df["predicted_rank_pct"] = (n - df["rank"] + 1) / n

    top = df.head(top_n).copy()
    top["signal_date"] = latest_date
    top["equal_weight"] = 1.0 / top_n

    print(f"\nTop {top_n} signals as of {latest_date.date()}:")
    print("=" * 70)
    print(
        top[["rank", "ticker", "predicted_rank_pct", "price", "rsi_14", "ret_5d", "ret_21d"]]
        .to_string(index=False, float_format=lambda x: f"{x:.3f}")
    )
    print("=" * 70)
    print(f"\nEqual weight per position: {1/top_n:.1%}")
    print("Note: This is a ranking model — it predicts RELATIVE outperformance,")
    print("not absolute direction. Always use a stop-loss.\n")

    return top


def generate_risk_tiers(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    model,
    feature_cols: list[str],
    top_n_per_tier: int = 5,
) -> dict[str, pd.DataFrame]:
    """
    Generate signals split by risk tier.
    Useful when your universe contains a mix of instruments.
    """
    from src.universe import get_high_risk_tickers, get_sp500_tickers, get_etf_universe, get_crypto_tickers

    sp500     = set(get_sp500_tickers())
    high_risk = set(get_high_risk_tickers())
    etfs      = set(get_etf_universe())
    crypto    = set(get_crypto_tickers())

    available = set(close.columns)

    tiers = {
        "medium_risk_sp500": list(available & sp500 - high_risk),
        "high_risk":         list(available & high_risk),
        "etfs":              list(available & etfs - high_risk),
        "crypto":            list(available & crypto),
    }

    # Volume filter per tier (crypto has no volume concept in the same units)
    min_volume = {
        "medium_risk_sp500": 500_000,
        "high_risk":         100_000,
        "etfs":              200_000,
        "crypto":            0,  # yfinance reports crypto volume in coin units — skip filter
    }

    results = {}
    for tier_name, tier_tickers in tiers.items():
        if not tier_tickers:
            continue
        tier_close = close[tier_tickers]
        tier_vol   = volume[tier_tickers]
        tier_high  = high[tier_tickers]
        tier_low   = low[tier_tickers]

        print(f"\n--- {tier_name.upper()} ({len(tier_tickers)} instruments) ---")
        signals = generate_signals(
            tier_close, tier_vol, tier_high, tier_low,
            model, feature_cols,
            top_n=top_n_per_tier,
            min_avg_volume=min_volume[tier_name],
            min_price=0.01 if tier_name == "crypto" else 5.0,
        )
        results[tier_name] = signals

    return results
