"""
Autonomous trading agent.

Runs once per day at US market open. Handles the full cycle:
  1. Check market regime — go to cash if bear
  2. On rebalance day: sell all positions, generate new signals, buy top picks
  3. Set stop-losses on all new positions
  4. Send email summary
  5. Track all state in data/portfolio_state.json

Setup:
  1. Copy .env.example to .env and fill in your Alpaca keys + Gmail details
  2. Run once manually to test: python agent.py --dry-run
  3. Schedule to run daily at 15:35 CET (5 min after US market open)

Scheduling (Mac):
  crontab -e
  35 15 * * 1-5 cd /Users/gijsvdklink/Documents/Gijs/trading && .venv/bin/python agent.py >> logs/agent.log 2>&1

Usage:
    python agent.py              # Live run (uses .env settings)
    python agent.py --dry-run    # Simulate without placing orders
    python agent.py --force      # Force rebalance even if not due yet
    python agent.py --status     # Print current positions and performance
"""

import argparse
import os
import sys
import traceback
from datetime import date, timedelta
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

Path("logs").mkdir(exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Simulate without placing real orders")
    parser.add_argument("--force", action="store_true",
                        help="Force rebalance even if hold period not complete")
    parser.add_argument("--status", action="store_true",
                        help="Print current portfolio status and exit")
    parser.add_argument("--budget", type=float,
                        default=float(os.environ.get("BUDGET_EUR", 100)))
    parser.add_argument("--top-n", type=int,
                        default=int(os.environ.get("TOP_N_PER_TIER", 5)))
    parser.add_argument("--holding-days", type=int,
                        default=int(os.environ.get("HOLDING_DAYS", 5)))
    parser.add_argument("--stop-loss", type=float,
                        default=float(os.environ.get("STOP_LOSS_PCT", 0.05)))
    args = parser.parse_args()

    try:
        _run(args)
    except Exception as e:
        msg = traceback.format_exc()
        print(f"\nAGENT ERROR:\n{msg}")
        if not args.dry_run:
            from src.notifier import send_error_alert
            send_error_alert(msg)
        sys.exit(1)


def _run(args):
    import yfinance as yf
    from src.regime import fetch_regime_data, get_current_regime
    from src.portfolio_state import (
        get_state, should_rebalance, get_next_rebalance_date,
        record_buy, clear_positions, get_performance_summary,
    )

    # ── Status mode ───────────────────────────────────────────────────────────
    if args.status:
        state = get_state()
        positions = state["positions"]
        perf = get_performance_summary()

        print("\n=== CURRENT PORTFOLIO ===")
        if positions:
            for t, p in positions.items():
                print(f"  {t:<8} bought {p['buy_date']}  stop: ${p['stop_price']:.2f}  tier: {p['tier']}")
        else:
            print("  No open positions.")

        next_rb = get_next_rebalance_date(args.holding_days)
        print(f"\n  Next rebalance: {next_rb}")

        print("\n=== PERFORMANCE ===")
        for k, v in perf.items():
            print(f"  {k}: {v:.1%}" if "rate" in k or "return" in k or "trade" not in k else f"  {k}: {v}")
        return

    # ── Regime check ──────────────────────────────────────────────────────────
    print(f"\n[{date.today()}] Trading agent starting...")
    print("Checking market regime...")
    spy, vix = fetch_regime_data(start="2023-01-01")
    regime = get_current_regime(spy, vix)

    print(f"  Regime: {regime['regime'].upper()}  |  VIX: {regime['vix']}  "
          f"|  SPY vs SMA200: {regime['spy_vs_sma200']:+.1f}%  "
          f"|  Position scalar: {regime['scalar']:.0%}")

    # ── Account info ──────────────────────────────────────────────────────────
    trades_placed = []
    trades_closed = []
    account = {"cash": args.budget, "portfolio_value": args.budget, "paper": True}

    if not args.dry_run:
        from src.broker import get_account
        account = get_account()
        mode = "PAPER" if account["paper"] else "LIVE"
        print(f"  Account ({mode}): ${account['portfolio_value']:.2f} portfolio, "
              f"${account['cash']:.2f} cash")

    # ── Bear market — go to cash ──────────────────────────────────────────────
    if regime["scalar"] == 0.0:
        print("\n  BEAR MARKET — staying in cash. No trades today.")
        if not args.dry_run:
            from src.broker import close_all_positions, cancel_all_stop_orders
            if get_state()["positions"]:
                print("  Closing all positions...")
                cancel_all_stop_orders()
                trades_closed = close_all_positions()
                clear_positions()
        _send_summary(regime, trades_placed, trades_closed, account, args)
        return

    # ── Check if rebalance is due ─────────────────────────────────────────────
    rebalance_due = args.force or should_rebalance(args.holding_days)
    if not rebalance_due:
        next_rb = get_next_rebalance_date(args.holding_days)
        print(f"\n  Hold period not complete. Next rebalance: {next_rb}. Nothing to do.")
        _send_summary(regime, [], [], account, args)
        return

    # ── Close existing positions ──────────────────────────────────────────────
    state = get_state()
    if state["positions"]:
        print(f"\n  Closing {len(state['positions'])} existing positions...")
        if not args.dry_run:
            from src.broker import close_all_positions, cancel_all_stop_orders
            cancel_all_stop_orders()
            trades_closed = close_all_positions()
        else:
            trades_closed = [{"ticker": t, "status": "dry-run"} for t in state["positions"]]
        clear_positions()

    # ── Generate new signals ──────────────────────────────────────────────────
    print("\n  Generating signals...")
    from src.model import load_model
    from src.universe import get_universe

    model, feature_cols = load_model()
    tickers = get_universe("all")

    raw = yf.download(tickers, period="300d", auto_adjust=True, progress=False, threads=True)
    fields = raw.columns.get_level_values(0).unique()
    close  = raw["Close"] if "Close" in fields else raw["Adj Close"]
    volume = raw["Volume"]
    high   = raw["High"]
    low    = raw["Low"]

    valid = close.notna().sum() >= 200
    close, volume, high, low = (
        close.loc[:, valid], volume.loc[:, valid],
        high.loc[:, valid], low.loc[:, valid],
    )

    from src.signals import generate_risk_tiers
    tier_signals = generate_risk_tiers(
        close, volume, high, low,
        model, feature_cols,
        top_n_per_tier=args.top_n,
    )

    # ── Allocate budget ───────────────────────────────────────────────────────
    # Trading212 doesn't support crypto via API — skip that tier and
    # redistribute its 15% budget proportionally to the other three tiers.
    using_t212 = bool(os.environ.get("T212_API_KEY"))
    if using_t212:
        tier_signals.pop("crypto", None)
        tier_budget_pct = {"medium_risk_sp500": 0.57, "etfs": 0.17, "high_risk": 0.26}
    else:
        tier_budget_pct = {"medium_risk_sp500": 0.50, "etfs": 0.15, "high_risk": 0.20, "crypto": 0.15}
    available_cash = account["cash"] if not args.dry_run else args.budget
    effective_budget = available_cash * regime["scalar"]

    print(f"\n  Effective budget: ${effective_budget:.2f} "
          f"(${available_cash:.2f} × {regime['scalar']:.0%} regime scalar)")

    # ── Place orders ──────────────────────────────────────────────────────────
    print("\n  Placing orders:")
    for tier, signals in tier_signals.items():
        if signals.empty:
            continue

        budget_pct = tier_budget_pct.get(tier, 0.20)
        tier_budget = effective_budget * budget_pct
        per_position = tier_budget / len(signals)

        if per_position < 1.0:
            print(f"    {tier}: skipping (${per_position:.2f}/position is below $1 minimum)")
            continue

        for _, row in signals.iterrows():
            ticker = row["ticker"]
            price  = float(row["price"])
            stop   = price * (1 - args.stop_loss)

            print(f"    BUY  {ticker:<8} ${per_position:.2f}  stop: ${stop:.2f}  [{tier}]")

            if not args.dry_run:
                from src.broker import place_buy_order, place_stop_loss, get_latest_price
                try:
                    order = place_buy_order(ticker, per_position)
                    live_price = get_latest_price(ticker) or price
                    # Recalculate stop from live price so it's always below entry
                    live_stop = live_price * (1 - args.stop_loss)
                    qty_approx = per_position / live_price
                    place_stop_loss(ticker, qty_approx, live_stop)
                    record_buy(ticker, per_position, live_price, live_stop, tier)
                    trades_placed.append({
                        "ticker": ticker, "notional": per_position,
                        "stop_price": stop, "tier": tier,
                    })
                except Exception as e:
                    print(f"    ERROR placing {ticker}: {e}")
            else:
                trades_placed.append({
                    "ticker": ticker, "notional": per_position,
                    "stop_price": stop, "tier": tier,
                })
                record_buy(ticker, per_position, price, stop, tier)

    print(f"\n  {len(trades_placed)} orders placed.")

    # ── Performance summary ───────────────────────────────────────────────────
    perf = get_performance_summary()
    if perf["total_trades"] > 0:
        print(f"\n  Lifetime performance: {perf['total_trades']} trades, "
              f"win rate {perf['win_rate']:.1%}, avg return {perf['avg_return']:.2%}")

    _send_summary(regime, trades_placed, trades_closed, account, args)


def _send_summary(regime, trades_placed, trades_closed, account, args):
    from src.portfolio_state import get_next_rebalance_date
    from src.notifier import send_trade_summary

    next_rb = get_next_rebalance_date(args.holding_days)
    if not args.dry_run:
        send_trade_summary(regime, trades_placed, trades_closed, account, str(next_rb))
    else:
        print(f"\n  [dry-run] Email suppressed. Next rebalance: {next_rb}")

    print("\nAgent run complete.\n")


if __name__ == "__main__":
    main()
