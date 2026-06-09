"""
Daily morning email — buy/sell recommendations for Revolut.

Sends every weekday morning at 7:00 CET via GitHub Actions.
Tracks your current Revolut positions via data/portfolio_state.json.

Usage:
    python morning_email.py              # generate + send email
    python morning_email.py --no-email   # print to terminal only
    python morning_email.py --budget 150 # override budget
"""

import argparse
import os
import warnings
from datetime import date, timedelta

import numpy as np
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore")

HOLDING_DAYS  = int(os.environ.get("HOLDING_DAYS", 5))
STOP_LOSS_PCT = float(os.environ.get("STOP_LOSS_PCT", 0.05))
BUDGET        = float(os.environ.get("BUDGET_EUR", 100))
TOP_N         = 5   # total picks across all tiers combined


def _trading_days_since(date_str: str) -> int:
    start = date.fromisoformat(date_str)
    today = date.today()
    count = 0
    cur = start
    while cur < today:
        cur += timedelta(days=1)
        if cur.weekday() < 5:
            count += 1
    return count


def _next_rebalance(buy_date_str: str) -> date:
    cur = date.fromisoformat(buy_date_str)
    count = 0
    while count < HOLDING_DAYS:
        cur += timedelta(days=1)
        if cur.weekday() < 5:
            count += 1
    return cur


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget",   type=float, default=BUDGET)
    parser.add_argument("--no-email", action="store_true")
    args = parser.parse_args()

    # ── Model ─────────────────────────────────────────────────────────────────
    print("Loading model...")
    from src.model import load_model
    model, feature_cols = load_model()

    # ── Regime ────────────────────────────────────────────────────────────────
    print("Checking market regime...")
    from src.regime import fetch_regime_data, get_current_regime
    spy, vix = fetch_regime_data(start="2023-01-01")
    regime = get_current_regime(spy, vix)

    # ── Current positions ─────────────────────────────────────────────────────
    from src.portfolio_state import get_state
    positions = get_state().get("positions", {})

    # ── Download data ─────────────────────────────────────────────────────────
    from src.universe import get_universe
    tickers = get_universe("all")
    all_tickers = sorted(set(tickers) | set(positions.keys()))

    print(f"Downloading price data for {len(all_tickers)} instruments...")
    raw = yf.download(all_tickers, period="300d", auto_adjust=True,
                      progress=False, threads=True)

    fields = raw.columns.get_level_values(0).unique()
    close  = raw["Close"] if "Close" in fields else raw["Adj Close"]
    volume = raw["Volume"]
    high   = raw["High"]
    low    = raw["Low"]

    valid = close.notna().sum() >= 200
    close, volume, high, low = (
        close.loc[:, valid], volume.loc[:, valid],
        high.loc[:, valid],  low.loc[:, valid],
    )

    # ── Signals ───────────────────────────────────────────────────────────────
    from src.signals import generate_risk_tiers
    from src.universe import REVOLUT_UNAVAILABLE

    tier_signals = generate_risk_tiers(
        close, volume, high, low, model, feature_cols,
        top_n_per_tier=10,   # get more candidates, we'll pick best 5 total
    )
    tier_signals.pop("crypto", None)   # crypto not available as stocks on Revolut

    # ── Flat ranked list — best 5 picks across all tiers ─────────────────────
    import pandas as pd
    tier_label = {
        "medium_risk_sp500": "S&P 500",
        "high_risk":         "High Risk",
        "etfs":              "ETF",
    }
    tier_budget_pct = {
        "medium_risk_sp500": 0.57,
        "high_risk":         0.26,
        "etfs":              0.17,
    }

    all_candidates = []
    for tier, df in tier_signals.items():
        if df.empty:
            continue
        df = df.copy()
        df["tier"] = tier
        df["tier_label"] = tier_label.get(tier, tier)
        all_candidates.append(df)

    if all_candidates:
        candidates = pd.concat(all_candidates).sort_values("predicted_rank", ascending=False)
        # Remove Revolut-unavailable and already-held tickers
        held = set(positions.keys())
        candidates = candidates[
            ~candidates["ticker"].isin(REVOLUT_UNAVAILABLE) &
            ~candidates["ticker"].isin(held)
        ]
        top_picks = candidates.head(TOP_N)
    else:
        top_picks = pd.DataFrame()

    # ── Evaluate current positions: SELL or HOLD ──────────────────────────────
    top_ticker_set = set(top_picks["ticker"].tolist()) if not top_picks.empty else set()
    sell_list, hold_list = [], []

    for ticker, pos in positions.items():
        days_held    = _trading_days_since(pos["buy_date"])
        entry_price  = float(pos.get("entry_price", 0))
        stop_price   = float(pos.get("stop_price", entry_price * (1 - STOP_LOSS_PCT)))
        notional     = float(pos.get("notional", 0))

        current_price = None
        if ticker in close.columns:
            current_price = float(close[ticker].dropna().iloc[-1])

        pnl_pct = ((current_price - entry_price) / entry_price * 100
                   if current_price and entry_price else None)

        action, reason = "HOLD", ""
        if days_held >= HOLDING_DAYS:
            action = "SELL"
            reason = f"5-day cycle complete (held {days_held} days)"
        elif current_price and current_price <= stop_price:
            action = "SELL"
            reason = f"stop-loss hit ({pnl_pct:+.1f}% from entry)"
        elif ticker not in top_ticker_set and days_held >= 3:
            action = "SELL"
            reason = "dropped out of top signals"
        else:
            reason = f"still top-ranked — hold until {_next_rebalance(pos['buy_date'])}"

        record = dict(
            ticker=ticker, action=action, reason=reason,
            days_held=days_held, entry_price=entry_price,
            current_price=current_price, pnl_pct=pnl_pct,
            notional=notional, tier=pos.get("tier", ""),
            buy_date=pos["buy_date"],
        )
        (sell_list if action == "SELL" else hold_list).append(record)

    # ── Build buy list ────────────────────────────────────────────────────────
    effective_budget = args.budget * regime["scalar"]
    per_position     = effective_budget / TOP_N if TOP_N > 0 else 0

    buy_list = []
    for _, row in top_picks.iterrows():
        price = float(row["price"])
        buy_list.append(dict(
            tier=row["tier"], tier_label=row["tier_label"],
            ticker=row["ticker"], price=price,
            budget_eur=per_position,
            shares=per_position / price if price > 0 else 0,
            stop_price=price * (1 - STOP_LOSS_PCT),
            rsi=float(row.get("rsi_14", np.nan)),
            ret_5d=float(row.get("ret_5d", np.nan)),
        ))

    # ── Print ─────────────────────────────────────────────────────────────────
    _print_report(regime, sell_list, hold_list, buy_list, args.budget, effective_budget)

    # ── Email ─────────────────────────────────────────────────────────────────
    if not args.no_email:
        from src.notifier import send_morning_email
        send_morning_email(regime, sell_list, hold_list, buy_list,
                           args.budget, effective_budget)


def _print_report(regime, sell_list, hold_list, buy_list, budget, effective_budget):
    today = date.today().strftime("%A %d %b %Y")
    r = regime["regime"].upper()
    print(f"\n{'='*62}")
    print(f"  MORNING REPORT — {today}")
    print(f"  Regime: {r}  |  VIX: {regime['vix']}  |  "
          f"Deploy: {regime['scalar']:.0%} (€{effective_budget:.0f} of €{budget:.0f})")
    print(f"{'='*62}")

    if regime["scalar"] == 0.0:
        print("\n  BEAR MARKET — Stay in cash. Sell everything if you can.")
        return

    if sell_list:
        print(f"\n  SELL TODAY")
        print(f"  {'─'*55}")
        for s in sell_list:
            pnl = f"  [{s['pnl_pct']:+.1f}%]" if s["pnl_pct"] is not None else ""
            print(f"  ❌  {s['ticker']:<7} {s['reason']}{pnl}")

    if hold_list:
        print(f"\n  HOLD — keep these, do not sell yet")
        print(f"  {'─'*55}")
        for h in hold_list:
            pnl = f"  [{h['pnl_pct']:+.1f}%]" if h["pnl_pct"] is not None else ""
            print(f"  ✅  {h['ticker']:<7} day {h['days_held']}/5{pnl}  {h['reason']}")

    if not sell_list and not hold_list:
        print("\n  No positions tracked yet.")
        print("  After buying, register each position:")
        print("  python portfolio.py buy TICKER AMOUNT PRICE")

    if buy_list:
        print(f"\n  BUY TODAY  (€{effective_budget:.0f} total, ~€{effective_budget/len(buy_list):.0f}/position)")
        print(f"  {'─'*55}")
        print(f"  {'#':<3} {'Ticker':<8} {'Tier':<12} {'Price':>8} {'Buy €':>6} {'Stop':>8} {'RSI':>5}")
        print(f"  {'─'*55}")
        for i, b in enumerate(buy_list, 1):
            rsi = f"{b['rsi']:.0f}" if not np.isnan(b["rsi"]) else " —"
            print(f"  {i:<3} {b['ticker']:<8} {b['tier_label']:<12} "
                  f"{b['price']:>8.2f} {b['budget_eur']:>6.0f} "
                  f"{b['stop_price']:>8.2f} {rsi:>5}")

    print(f"\n  After trading on Revolut, update your positions:")
    print(f"  python portfolio.py buy  TICKER AMOUNT PRICE")
    print(f"  python portfolio.py sell TICKER PRICE")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()
