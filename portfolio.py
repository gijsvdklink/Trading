"""
Track your Revolut positions manually.

Run this after every trade on Revolut so the morning email
knows what you hold and can give personalised sell/hold advice.

Usage:
    python portfolio.py buy AAPL 10 79.68     # bought €10 of AAPL at $79.68
    python portfolio.py sell AAPL 85.00       # sold AAPL at $85.00
    python portfolio.py sell AAPL             # sold (price unknown)
    python portfolio.py status                # show current holdings
    python portfolio.py history               # show past trades
"""

import sys
from datetime import date, timedelta

from dotenv import load_dotenv
load_dotenv()


def _trading_days_since(date_str: str) -> int:
    start = date.fromisoformat(date_str)
    today = date.today()
    count, cur = 0, start
    while cur < today:
        cur += timedelta(days=1)
        if cur.weekday() < 5:
            count += 1
    return count


def _guess_tier(ticker: str) -> str:
    from src.universe import get_high_risk_tickers, get_etf_universe, get_crypto_tickers
    if ticker in get_high_risk_tickers(): return "high_risk"
    if ticker in get_crypto_tickers():    return "crypto"
    if ticker in get_etf_universe():      return "etfs"
    return "medium_risk_sp500"


def cmd_buy(ticker: str, amount: float, price: float):
    from src.portfolio_state import record_buy
    ticker = ticker.upper()
    stop   = round(price * 0.95, 2)
    tier   = _guess_tier(ticker)
    record_buy(ticker, amount, price, stop, tier)
    print(f"  ✅  Recorded BUY  {ticker}  €{amount:.2f} at ${price:.2f}  "
          f"(stop-loss: ${stop:.2f}  tier: {tier})")
    print(f"\n  Don't forget to commit and push so GitHub Actions sees this:")
    print(f"  git add data/portfolio_state.json && git commit -m 'buy {ticker}' && git push")


def cmd_sell(ticker: str, price: float | None = None):
    from src.portfolio_state import record_sell, get_state
    ticker = ticker.upper()
    state  = get_state()
    if ticker not in state["positions"]:
        print(f"  {ticker} is not in your tracked positions.")
        _list_positions(state)
        return
    pos = state["positions"][ticker]
    record_sell(ticker, price, reason="manual")
    entry = pos.get("entry_price", 0)
    if price and entry:
        pnl = (price - entry) / entry * 100
        print(f"  ✅  Recorded SELL {ticker} at ${price:.2f}  "
              f"(entry ${entry:.2f}  P&L {pnl:+.1f}%)")
    else:
        print(f"  ✅  Recorded SELL {ticker}")
    print(f"\n  Don't forget to commit and push:")
    print(f"  git add data/portfolio_state.json && git commit -m 'sell {ticker}' && git push")


def cmd_status():
    from src.portfolio_state import get_state, get_performance_summary
    state = get_state()
    _list_positions(state)
    perf = get_performance_summary()
    if perf.get("total_trades", 0) > 0:
        print(f"  History: {perf['total_trades']} trades  "
              f"win rate {perf['win_rate']:.0%}  "
              f"avg return {perf['avg_return']:+.1%}\n")


def _list_positions(state: dict):
    positions = state.get("positions", {})
    print(f"\n  Current positions: {len(positions)}")
    if not positions:
        print("  (none)\n")
        return
    print(f"  {'Ticker':<8} {'Date':<12} {'Days':>4}  {'Entry':>8}  {'Stop':>8}  {'€':>6}  Tier")
    print(f"  {'─'*60}")
    for t, p in positions.items():
        days = _trading_days_since(p["buy_date"])
        print(f"  {t:<8} {p['buy_date']:<12} {days:>3}d  "
              f"${p.get('entry_price',0):>6.2f}  "
              f"${p.get('stop_price',0):>6.2f}  "
              f"€{p.get('notional',0):>4.0f}  {p.get('tier','')}")
    print()


def cmd_history():
    from src.portfolio_state import get_state
    history = get_state().get("trade_history", [])
    if not history:
        print("\n  No closed trades yet.\n")
        return
    print(f"\n  Trade history (last 20):")
    print(f"  {'Ticker':<8} {'Bought':<12} {'Sold':<12} {'Return':>8}  Reason")
    print(f"  {'─'*58}")
    for t in reversed(history[-20:]):
        ret = f"{t['return_pct']:+.1%}" if "return_pct" in t else "   n/a"
        print(f"  {t['ticker']:<8} {t.get('buy_date',''):<12} "
              f"{t.get('sell_date',''):<12} {ret:>8}  {t.get('reason','')}")
    print()


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        return

    cmd = sys.argv[1].lower()

    if cmd == "buy":
        if len(sys.argv) < 5:
            print("Usage: python portfolio.py buy TICKER AMOUNT PRICE")
            print("Example: python portfolio.py buy AAPL 10 79.68")
            return
        cmd_buy(sys.argv[2], float(sys.argv[3]), float(sys.argv[4]))

    elif cmd == "sell":
        if len(sys.argv) < 3:
            print("Usage: python portfolio.py sell TICKER [PRICE]")
            return
        price = float(sys.argv[3]) if len(sys.argv) >= 4 else None
        cmd_sell(sys.argv[2], price)

    elif cmd == "status":
        cmd_status()

    elif cmd == "history":
        cmd_history()

    else:
        print(f"Unknown command '{cmd}'. Options: buy, sell, status, history")


if __name__ == "__main__":
    main()
