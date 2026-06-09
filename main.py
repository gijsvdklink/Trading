"""
python main.py

Retrains the model on the latest data and prints exactly
what to buy this week on Revolut with a €100 budget.
"""

import warnings
warnings.filterwarnings("ignore")

from datetime import date, timedelta
from dotenv import load_dotenv
load_dotenv()

BUDGET       = 100.0
STOP_LOSS    = 0.05   # 5%
HORIZON_DAYS = 7


def _sell_date() -> str:
    d, count = date.today(), 0
    while count < HORIZON_DAYS:
        d += timedelta(days=1)
        if d.weekday() < 5:
            count += 1
    return d.strftime("%A %d %b")


def main():
    import yfinance as yf
    import pandas as pd

    print("\n" + "═" * 54)
    print("  📈  WEEKLY STOCK PICKS  —  " + date.today().strftime("%d %b %Y"))
    print("═" * 54)

    # ── 1. Regime ─────────────────────────────────────────────────────────────
    from src.regime import fetch_regime_data, get_current_regime
    spy, vix = fetch_regime_data(start="2023-01-01")
    regime = get_current_regime(spy, vix)

    icons = {"bull": "🟢", "neutral": "🟡", "caution": "🟠", "bear": "🔴"}
    icon  = icons.get(regime["regime"], "⚪")
    scalar = regime["scalar"]

    print(f"\n  Market:  {icon} {regime['regime'].upper()}  "
          f"(VIX {regime['vix']}  |  SPY vs 200d: {regime['spy_vs_sma200']:+.1f}%)")

    if scalar == 0.0:
        print("  ⚠  BEAR MARKET — stay in cash this week.\n")
        return

    effective = BUDGET * scalar
    per_pos   = effective / 5
    print(f"  Budget:  €{effective:.0f} deployed  "
          f"(€{BUDGET:.0f} × {scalar:.0%} caution factor)\n")

    # ── 2. Download ───────────────────────────────────────────────────────────
    from src.universe import get_universe
    tickers = get_universe("all")

    print(f"  Downloading price data...")
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

    # ── 3. Retrain ────────────────────────────────────────────────────────────
    print(f"  Retraining model on latest data...")
    from src.features import build_feature_matrix
    from src.model import train_final_model

    fm = build_feature_matrix(close, volume, high, low, target_horizon=HORIZON_DAYS)
    model, feature_cols = train_final_model(fm)

    # ── 4. Signals ────────────────────────────────────────────────────────────
    from src.signals import generate_risk_tiers
    from src.universe import REVOLUT_UNAVAILABLE

    tier_signals = generate_risk_tiers(
        close, volume, high, low, model, feature_cols, top_n_per_tier=8,
    )
    tier_signals.pop("crypto", None)

    tier_label = {
        "medium_risk_sp500": "S&P 500",
        "high_risk":         "High Risk",
        "etfs":              "ETF",
    }

    frames = []
    for tier, df in tier_signals.items():
        if df.empty: continue
        df = df.copy()
        df["tier_label"] = tier_label.get(tier, tier)
        frames.append(df)

    if not frames:
        print("  No signals generated.\n")
        return

    picks = (
        pd.concat(frames)
        .sort_values("predicted_rank", ascending=False)
        .pipe(lambda d: d[~d["ticker"].isin(REVOLUT_UNAVAILABLE)])
        .head(5)
        .reset_index(drop=True)
    )

    # ── 5. Print results ──────────────────────────────────────────────────────
    sell_on = _sell_date()

    print(f"\n{'═' * 54}")
    print(f"  BUY THESE 5 STOCKS ON REVOLUT NOW")
    print(f"  Sell on: {sell_on}  |  Stop-loss: {STOP_LOSS:.0%} below entry")
    print(f"{'═' * 54}\n")
    print(f"  {'#':<3} {'Ticker':<7} {'Tier':<12} {'Price':>8} "
          f"{'Invest':>8} {'Stop-loss':>10}")
    print(f"  {'─' * 52}")

    for i, row in picks.iterrows():
        price = float(row["price"])
        stop  = price * (1 - STOP_LOSS)
        print(f"  {i+1:<3} {row['ticker']:<7} {row['tier_label']:<12} "
              f"${price:>7.2f}  €{per_pos:>5.0f}    ${stop:>7.2f}")

    print(f"\n{'═' * 54}")
    print(f"  STEP-BY-STEP INSTRUCTIONS")
    print(f"{'═' * 54}\n")

    for i, row in picks.iterrows():
        price = float(row["price"])
        stop  = price * (1 - STOP_LOSS)
        shares = per_pos / price
        print(f"  {i+1}. Open Revolut → search '{row['ticker']}'")
        print(f"     Buy €{per_pos:.0f}  (~{shares:.3f} shares at ${price:.2f})")
        print(f"     Set stop-loss at ${stop:.2f}  (protects against big loss)")
        print()

    print(f"  ✅ Come back on {sell_on} and sell everything.")
    print(f"  ✅ Then run  python main.py  again for next week's picks.\n")
    print(f"{'═' * 54}\n")


if __name__ == "__main__":
    main()
