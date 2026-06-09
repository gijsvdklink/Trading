"""
Tracks the agent's portfolio state between runs.

Persists to a JSON file so the agent knows:
  - What it's currently holding
  - When it bought each position (to determine 5-day hold expiry)
  - Trade history for performance tracking
"""

import json
from datetime import date, datetime, timedelta
from pathlib import Path

STATE_FILE = Path(__file__).parent.parent / "data" / "portfolio_state.json"


def _load() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"positions": {}, "trade_history": [], "last_rebalance": None}


def _save(state: dict) -> None:
    STATE_FILE.parent.mkdir(exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def get_state() -> dict:
    return _load()


def record_buy(ticker: str, notional: float, price: float, stop_price: float, tier: str) -> None:
    state = _load()
    state["positions"][ticker] = {
        "ticker": ticker,
        "notional": notional,
        "entry_price": price,
        "stop_price": stop_price,
        "tier": tier,
        "buy_date": date.today().isoformat(),
    }
    state["last_rebalance"] = date.today().isoformat()
    _save(state)


def record_sell(ticker: str, exit_price: float | None = None, reason: str = "scheduled") -> None:
    state = _load()
    if ticker in state["positions"]:
        pos = state["positions"].pop(ticker)
        pos["exit_price"] = exit_price
        pos["sell_date"] = date.today().isoformat()
        pos["reason"] = reason
        if exit_price and pos.get("entry_price"):
            pos["return_pct"] = (exit_price - pos["entry_price"]) / pos["entry_price"]
        state["trade_history"].append(pos)
    _save(state)


def clear_positions() -> None:
    state = _load()
    # Move all to history before clearing
    for ticker, pos in state["positions"].items():
        pos["sell_date"] = date.today().isoformat()
        pos["reason"] = "rebalance"
        state["trade_history"].append(pos)
    state["positions"] = {}
    state["last_rebalance"] = date.today().isoformat()
    _save(state)


def should_rebalance(holding_days: int = 5) -> bool:
    """Returns True if holding_days trading days have passed since last rebalance."""
    state = _load()
    last = state.get("last_rebalance")
    if not last:
        return True
    last_date = date.fromisoformat(last)
    # Count trading days elapsed (rough: skip weekends)
    elapsed = 0
    current = last_date
    today = date.today()
    while current < today:
        current += timedelta(days=1)
        if current.weekday() < 5:  # Mon–Fri only
            elapsed += 1
    return elapsed >= holding_days


def get_next_rebalance_date(holding_days: int = 5) -> date:
    state = _load()
    last = state.get("last_rebalance")
    if not last:
        return date.today()
    start = date.fromisoformat(last)
    count = 0
    current = start
    while count < holding_days:
        current += timedelta(days=1)
        if current.weekday() < 5:
            count += 1
    return current


def get_performance_summary() -> dict:
    state = _load()
    history = state["trade_history"]
    if not history:
        return {"total_trades": 0}
    returns = [t["return_pct"] for t in history if "return_pct" in t]
    return {
        "total_trades": len(history),
        "completed_trades": len(returns),
        "win_rate": sum(1 for r in returns if r > 0) / len(returns) if returns else 0,
        "avg_return": sum(returns) / len(returns) if returns else 0,
        "best_trade": max(returns) if returns else 0,
        "worst_trade": min(returns) if returns else 0,
    }
