"""
Run this every week to retrain and get fresh stock picks.

    python main.py
"""

import warnings
import time
warnings.filterwarnings("ignore")

from pathlib import Path
from datetime import date

import yfinance as yf
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()


def main():
    print(f"\n{'='*58}")
    print(f"  TRADING SYSTEM — {date.today().strftime('%A %d %b %Y')}")
    print(f"{'='*58}\n")

    # ── Step 1: Market regime ─────────────────────────────────────────────────
    print("Step 1/4 — Checking market regime...")
    from src.regime import fetch_regime_data, get_current_regime
    spy, vix = fetch_regime_data(start="2023-01-01")
    regime = get_current_regime(spy, vix)

    icons = {"bull": "🟢", "neutral": "🟡", "caution": "🟠", "bear": "🔴"}
    icon  = icons.get(regime["regime"], "⚪")
    print(f"  {icon}  {regime['regime'].upper()}  |  VIX: {regime['vix']}  "
          f"|  SPY vs SMA200: {regime['spy_vs_sma200']:+.1f}%")
    print(f"  Deploy {regime['scalar']:.0%} of your budget today\n")

    # ── Step 2: Download latest prices ────────────────────────────────────────
    print("Step 2/4 — Downloading latest price data...")
    from src.universe import get_universe, REVOLUT_UNAVAILABLE

    tickers = get_universe("all")
    raw = yf.download(tickers, period="400d", auto_adjust=True,
                      progress=False, threads=True)

    fields = raw.columns.get_level_values(0).unique()
    close  = raw["Close"]  if "Close"  in fields else raw["Adj Close"]
    volume = raw["Volume"]
    high   = raw["High"]
    low    = raw["Low"]

    valid = close.notna().sum() >= 200
    close, volume, high, low = (
        close.loc[:, valid], volume.loc[:, valid],
        high.loc[:, valid],  low.loc[:, valid],
    )
    print(f"  {len(close.columns)} instruments  ×  {len(close)} trading days\n")

    # ── Step 3: Retrain model on latest data ──────────────────────────────────
    print("Step 3/4 — Retraining model on most recent data...")
    from src.features import build_feature_matrix
    from src.model import train_final_model

    feature_matrix = build_feature_matrix(
        close, volume, high, low, target_horizon=7
    )
    model, feature_cols = train_final_model(feature_matrix)
    print()

    # ── Step 4: Generate top 5 picks ─────────────────────────────────────────
    print("Step 4/4 — Generating picks for the next 7 days...")
    from src.signals import generate_risk_tiers

    tier_signals = generate_risk_tiers(
        close, volume, high, low, model, feature_cols,
        top_n_per_tier=8,
    )
    tier_signals.pop("crypto", None)

    tier_label = {
        "medium_risk_sp500": "S&P 500",
        "high_risk":         "High Risk",
        "etfs":              "ETF",
    }

    # Merge all tiers, sort by model score, take top 5 Revolut-available picks
    frames = []
    for tier, df in tier_signals.items():
        if df.empty:
            continue
        df = df.copy()
        df["tier_label"] = tier_label.get(tier, tier)
        frames.append(df)

    if not frames:
        print("  No signals generated.")
        return

    all_picks = (
        pd.concat(frames)
        .sort_values("predicted_rank", ascending=False)
        .pipe(lambda d: d[~d["ticker"].isin(REVOLUT_UNAVAILABLE)])
        .head(5)
        .reset_index(drop=True)
    )

    budget     = 100.0
    effective  = budget * regime["scalar"]
    per_pos    = effective / len(all_picks) if len(all_picks) else 0
    sell_date  = _next_weekday(days=7)

    # ── Print results ─────────────────────────────────────────────────────────
    print(f"\n{'='*58}")
    print(f"  TOP 5 PICKS FOR THE WEEK  —  sell by {sell_date}")
    print(f"  Budget: €{effective:.0f} of €{budget:.0f}  "
          f"(regime scalar {regime['scalar']:.0%})")
    print(f"{'='*58}")
    print(f"\n  {'#':<3} {'Ticker':<8} {'Tier':<12} {'Price':>8} "
          f"{'Buy €':>6} {'Stop':>8} {'RSI':>5} {'5d%':>6}")
    print(f"  {'─'*56}")

    for i, row in all_picks.iterrows():
        price     = float(row["price"])
        stop      = price * 0.95
        rsi       = row.get("rsi_14", float("nan"))
        ret5      = row.get("ret_5d", float("nan"))
        rsi_str   = f"{rsi:.0f}"   if rsi == rsi   else "—"
        ret5_str  = f"{ret5*100:+.1f}%" if ret5 == ret5 else "—"
        print(f"  {i+1:<3} {row['ticker']:<8} {row['tier_label']:<12} "
              f"{price:>8.2f} {per_pos:>6.0f} {stop:>8.2f} "
              f"{rsi_str:>5} {ret5_str:>6}")

    print(f"\n{'='*58}")
    print(f"  HOW TO TRADE THIS ON REVOLUT")
    print(f"{'='*58}")
    for i, row in all_picks.iterrows():
        price = float(row["price"])
        stop  = price * 0.95
        print(f"  {i+1}. Buy {row['ticker']:<6}  →  €{per_pos:.0f}  "
              f"  set stop-loss at ${stop:.2f}")
    print(f"\n  Sell everything on or after {sell_date}.")
    print(f"  Set stop-losses immediately — never skip this step.\n")

    if regime["scalar"] == 0.0:
        print("  ⚠  BEAR MARKET — recommend staying in cash this week.\n")
    elif regime["scalar"] < 1.0:
        print(f"  ⚠  Caution mode — only deploying {regime['scalar']:.0%} "
              f"of budget (€{effective:.0f}).\n")

    # ── Register buys ─────────────────────────────────────────────────────────
    print(f"  After buying, register each position:")
    for _, row in all_picks.iterrows():
        print(f"  python portfolio.py buy {row['ticker']} {per_pos:.0f} <price>")
    print()


def _next_weekday(days: int) -> str:
    from datetime import timedelta
    d = date.today()
    added = 0
    while added < days:
        d += timedelta(days=1)
        if d.weekday() < 5:
            added += 1
    return d.strftime("%A %d %b")


if __name__ == "__main__":
    main()
