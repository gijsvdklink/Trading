"""
Run every morning before 15:30 CET (US market open).

Outputs:
  1. Current market regime — tells you whether to trade at all today
  2. Ranked buy signals per risk tier
  3. Exact position sizes for your €100 budget

Usage:
    python daily_signals.py                       # Default: all tiers, €100 budget
    python daily_signals.py --budget 150          # Different budget
    python daily_signals.py --top-n 5             # Top 5 per tier
    python daily_signals.py --stop-loss 0.05      # 5% stop-loss (default)
    python daily_signals.py --export signals.csv  # Save to CSV
"""

import argparse
import warnings

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=float, default=100.0, help="Total budget in EUR/USD")
    parser.add_argument("--top-n", type=int, default=5, help="Top N per risk tier")
    parser.add_argument("--stop-loss", type=float, default=0.05, help="Stop-loss % (default 5%)")
    parser.add_argument("--export", type=str, default=None)
    parser.add_argument("--universe", default="all",
                        choices=["sp500", "nasdaq100", "high_risk", "etfs", "all"])
    args = parser.parse_args()

    # ── Load model ────────────────────────────────────────────────────────────
    from src.model import load_model
    model, feature_cols = load_model()

    # ── Market regime check ───────────────────────────────────────────────────
    print("\nChecking market regime...")
    from src.regime import fetch_regime_data, get_current_regime
    spy, vix = fetch_regime_data(start="2023-01-01")
    regime = get_current_regime(spy, vix)

    print(f"\n{'='*55}")
    print(f"  MARKET REGIME — {regime['date']}")
    print(f"{'='*55}")
    print(f"  Regime:        {regime['regime'].upper()}")
    print(f"  VIX:           {regime['vix']}")
    print(f"  SPY vs SMA200: {regime['spy_vs_sma200']:+.1f}%")
    print(f"  Position size: {regime['scalar']:.0%} of normal")

    if regime["scalar"] == 0.0:
        print("\n  ⚠  BEAR MARKET — Stay in cash today. No trades recommended.")
        print(f"{'='*55}\n")
        return

    if regime["scalar"] < 1.0:
        print(f"\n  ⚠  Caution — Reduce position sizes to {regime['scalar']:.0%}")

    print(f"{'='*55}\n")

    # ── Download recent data ──────────────────────────────────────────────────
    from src.universe import get_universe
    tickers = get_universe(args.universe)

    print(f"Downloading recent data for {len(tickers)} instruments...")
    raw = yf.download(tickers, period="300d", auto_adjust=True, progress=False, threads=True)

    if raw.empty:
        print("Download failed.")
        return

    fields = raw.columns.get_level_values(0).unique()
    close  = raw["Close"] if "Close" in fields else raw["Adj Close"]
    volume = raw["Volume"]
    high   = raw["High"]
    low    = raw["Low"]

    valid = close.notna().sum() >= 200
    close, volume, high, low = close.loc[:, valid], volume.loc[:, valid], high.loc[:, valid], low.loc[:, valid]
    print(f"Valid instruments: {len(close.columns)}\n")

    # ── Generate signals by tier ──────────────────────────────────────────────
    from src.signals import generate_risk_tiers
    tier_signals = generate_risk_tiers(
        close, volume, high, low,
        model, feature_cols,
        top_n_per_tier=args.top_n,
    )

    if not tier_signals:
        print("No signals generated.")
        return

    # ── Position sizing ───────────────────────────────────────────────────────
    import os
    using_t212 = bool(os.environ.get("T212_API_KEY"))
    if using_t212:
        tier_signals.pop("crypto", None)
        tier_budget_pct = {"medium_risk_sp500": 0.57, "etfs": 0.17, "high_risk": 0.26}
    else:
        tier_budget_pct = {"medium_risk_sp500": 0.50, "etfs": 0.15, "high_risk": 0.20, "crypto": 0.15}

    all_signals = []
    print(f"\n{'='*65}")
    print(f"  TODAY'S TRADING PLAN  |  Budget: €{args.budget:.0f}  |  Stop-loss: {args.stop_loss:.0%}")
    print(f"{'='*65}")

    for tier, signals in tier_signals.items():
        if signals.empty:
            continue

        budget_pct = tier_budget_pct.get(tier, 0.20)
        tier_budget = args.budget * budget_pct * regime["scalar"]
        n = len(signals)
        per_position = tier_budget / n if n > 0 else 0

        print(f"\n  {tier.upper().replace('_', ' ')}  (€{tier_budget:.0f} total, €{per_position:.0f}/position)")
        print(f"  {'Ticker':<8} {'Price':>7} {'Buy €':>6} {'Shares':>7} {'Stop':>7} {'RSI':>5} {'5d%':>6}")
        print(f"  {'-'*55}")

        for _, row in signals.iterrows():
            price = row["price"]
            shares = per_position / price if price > 0 else 0
            stop_price = price * (1 - args.stop_loss)
            rsi = row.get("rsi_14", np.nan)
            ret5 = row.get("ret_5d", np.nan) * 100 if not np.isnan(row.get("ret_5d", np.nan)) else np.nan

            print(f"  {row['ticker']:<8} {price:>7.2f} {per_position:>6.0f} {shares:>7.3f} "
                  f"{stop_price:>7.2f} {rsi:>5.1f} {ret5:>+6.1f}%")

            signals_row = row.to_dict()
            signals_row["tier"] = tier
            signals_row["budget_eur"] = per_position
            signals_row["shares_to_buy"] = shares
            signals_row["stop_loss_price"] = stop_price
            all_signals.append(signals_row)

    print(f"\n{'='*65}")
    print(f"  INSTRUCTIONS:")
    print(f"  1. Open Revolut → Stocks")
    print(f"  2. Buy each ticker at market price with the € amount shown")
    print(f"  3. Set a stop-loss at the 'Stop' price immediately after buying")
    print(f"  4. Come back in 5 trading days, sell everything, then run this again")
    print(f"{'='*65}")
    print(f"\n  Risk reminders:")
    print(f"  • High-risk tier (leveraged ETFs) can move ±10% in one day")
    print(f"  • Never invest more than you can afford to lose")
    print(f"  • This model predicts relative performance, not absolute direction")
    print(f"  • A market-wide crash will hurt all positions regardless of rank\n")

    if args.export:
        pd.DataFrame(all_signals).to_csv(args.export, index=False)
        print(f"Signals exported to {args.export}\n")


if __name__ == "__main__":
    main()
