"""
Trading212 broker abstraction.

Uses the Trading212 REST API v0 directly via requests.
Set T212_DEMO=false in .env to switch from demo (paper) to live trading.

API reference: https://docs.trading212.com/api
"""

import json
import os
from pathlib import Path
from typing import Optional

import requests

DEMO_BASE = "https://demo.trading212.com/api/v0"
LIVE_BASE  = "https://live.trading212.com/api/v0"

# Disk-cached ticker map so we only hit the instruments endpoint once
_TICKER_CACHE = Path("data/t212_ticker_map.json")
_ticker_map: dict[str, str] = {}   # yfinance_ticker → t212_ticker


# ─── HTTP helpers ─────────────────────────────────────────────────────────────

def _base() -> str:
    demo = os.environ.get("T212_DEMO", "true").lower() != "false"
    return DEMO_BASE if demo else LIVE_BASE


def _headers() -> dict:
    return {
        "Authorization": os.environ["T212_API_KEY"],
        "Content-Type": "application/json",
    }


def _get(path: str):
    r = requests.get(f"{_base()}/{path}", headers=_headers(), timeout=15)
    r.raise_for_status()
    return r.json()


def _post(path: str, body: dict) -> dict:
    r = requests.post(f"{_base()}/{path}", headers=_headers(), json=body, timeout=15)
    r.raise_for_status()
    return r.json()


def _delete(path: str) -> None:
    r = requests.delete(f"{_base()}/{path}", headers=_headers(), timeout=15)
    r.raise_for_status()


# ─── Ticker mapping ───────────────────────────────────────────────────────────

def _load_ticker_map() -> dict[str, str]:
    """
    Build a yfinance→T212 ticker map from the instruments endpoint.
    Cached to disk because the endpoint is rate-limited to 1 req/50s.
    """
    global _ticker_map
    if _ticker_map:
        return _ticker_map

    if _TICKER_CACHE.exists():
        with open(_TICKER_CACHE) as f:
            _ticker_map = json.load(f)
        return _ticker_map

    try:
        instruments = _get("equity/metadata/instruments")
        for inst in instruments:
            t212 = inst.get("ticker", "")      # e.g. "AAPL_US_EQ"
            short = inst.get("shortName", "")   # e.g. "AAPL" for many US stocks

            # Derive yfinance key by stripping trailing _XX_EQ suffix
            parts = t212.rsplit("_", 2)
            if len(parts) == 3 and parts[2] == "EQ":
                yf_key = parts[0]               # "AAPL" from "AAPL_US_EQ"
            else:
                yf_key = t212

            _ticker_map[yf_key] = t212
            if short and short != yf_key:
                _ticker_map[short] = t212

        _TICKER_CACHE.parent.mkdir(exist_ok=True)
        with open(_TICKER_CACHE, "w") as f:
            json.dump(_ticker_map, f)

        print(f"  T212 instrument map built: {len(_ticker_map)} entries cached")
    except Exception as e:
        print(f"  Warning: could not build T212 ticker map ({e}), using fallback")

    return _ticker_map


def _to_t212(ticker: str) -> str:
    """Convert yfinance ticker (e.g. 'AAPL') to Trading212 format (e.g. 'AAPL_US_EQ')."""
    mapping = _load_ticker_map()
    if ticker in mapping:
        return mapping[ticker]
    # Fallback: nearly all US equities and ETFs follow this pattern
    return f"{ticker.replace('-', '_')}_US_EQ"


def _from_t212(t212_ticker: str) -> str:
    """Convert T212 ticker back to yfinance format."""
    mapping = _load_ticker_map()
    reverse = {v: k for k, v in mapping.items()}
    if t212_ticker in reverse:
        return reverse[t212_ticker]
    # Strip the suffix
    parts = t212_ticker.rsplit("_", 2)
    return parts[0] if len(parts) == 3 and parts[2] == "EQ" else t212_ticker


# ─── Public interface (same as Alpaca broker) ─────────────────────────────────

def get_account() -> dict:
    """Return account cash, portfolio value, and demo status."""
    data = _get("equity/account/cash")
    demo = os.environ.get("T212_DEMO", "true").lower() != "false"
    return {
        "cash": float(data.get("free", 0)),
        "portfolio_value": float(data.get("total", 0)),
        "buying_power": float(data.get("free", 0)),
        "paper": demo,
    }


def get_positions() -> list[dict]:
    """Return all open positions."""
    portfolio = _get("equity/portfolio")
    result = []
    for p in portfolio:
        qty = float(p["quantity"])
        avg = float(p["averagePrice"])
        cur = float(p["currentPrice"])
        ppl = float(p["ppl"])
        result.append({
            "ticker": _from_t212(p["ticker"]),
            "qty": qty,
            "avg_price": avg,
            "market_value": qty * cur,
            "unrealized_pnl": ppl,
            "unrealized_pnl_pct": ppl / max(qty * avg, 1e-6),
        })
    return result


def get_latest_price(ticker: str) -> Optional[float]:
    """
    Get the current price for a ticker.
    T212 has no standalone quote endpoint, so we use yfinance.
    """
    try:
        import yfinance as yf
        price = yf.Ticker(ticker).fast_info.last_price
        if price and float(price) > 0:
            return float(price)
    except Exception:
        pass
    return None


def place_buy_order(ticker: str, notional_usd: float) -> dict:
    """
    Buy approximately `notional_usd` worth of `ticker` at market price.
    T212 requires quantity (not notional), so we divide by current price.
    Fractional quantities are supported for most instruments.
    """
    price = get_latest_price(ticker)
    if not price:
        raise ValueError(f"Could not get price for {ticker}")

    qty = round(notional_usd / price, 6)
    t212_ticker = _to_t212(ticker)

    result = _post("equity/orders/market", {"ticker": t212_ticker, "quantity": qty})
    return {
        "id": str(result.get("id", "")),
        "ticker": ticker,
        "notional": notional_usd,
        "status": result.get("status", "submitted"),
    }


def place_stop_loss(ticker: str, qty: float, stop_price: float) -> dict:
    """Place a Good-Till-Cancelled stop-loss sell order."""
    t212_ticker = _to_t212(ticker)
    result = _post("equity/orders/stop", {
        "ticker": t212_ticker,
        "quantity": -abs(round(qty, 6)),   # negative = sell
        "stopPrice": round(stop_price, 2),
        "timeValidity": "GOOD_TILL_CANCEL",
    })
    return {
        "id": str(result.get("id", "")),
        "ticker": ticker,
        "stop_price": stop_price,
    }


def close_position(ticker: str) -> dict:
    """Close a single position by placing a full market sell."""
    try:
        t212_ticker = _to_t212(ticker)
        portfolio = _get("equity/portfolio")
        pos = next((p for p in portfolio if p["ticker"] == t212_ticker), None)
        if not pos or float(pos["quantity"]) <= 0:
            return {"ticker": ticker, "status": "no_position"}
        qty = float(pos["quantity"])
        result = _post("equity/orders/market", {
            "ticker": t212_ticker,
            "quantity": -qty,   # negative = sell all
        })
        return {"ticker": ticker, "status": "closed", "order_id": str(result.get("id", ""))}
    except Exception as e:
        return {"ticker": ticker, "status": "error", "error": str(e)}


def close_all_positions() -> list[dict]:
    """Close every open position with a market sell."""
    portfolio = _get("equity/portfolio")
    results = []
    for pos in portfolio:
        t212_ticker = pos["ticker"]
        qty = float(pos["quantity"])
        if qty <= 0:
            continue
        try:
            result = _post("equity/orders/market", {
                "ticker": t212_ticker,
                "quantity": -qty,
            })
            results.append({
                "ticker": _from_t212(t212_ticker),
                "status": "closed",
                "order_id": str(result.get("id", "")),
            })
        except Exception as e:
            results.append({
                "ticker": _from_t212(t212_ticker),
                "status": "error",
                "error": str(e),
            })
    return results


def cancel_all_stop_orders() -> int:
    """Cancel all pending stop-loss orders before rebalancing."""
    orders = _get("equity/orders")
    cancelled = 0
    for order in orders:
        if order.get("type") in ("STOP", "STOP_LIMIT"):
            try:
                _delete(f"equity/orders/{order['id']}")
                cancelled += 1
            except Exception:
                pass
    return cancelled
