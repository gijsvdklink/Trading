"""
End-to-end test suite. Runs without Alpaca keys or email config.

Tests:
  1. Universe fetching (S&P 500 scraping + crypto)
  2. Data download (small sample)
  3. Feature engineering (correctness + no lookahead)
  4. Regime filter (live VIX quote)
  5. Model load + signal generation
  6. Portfolio state (rebalance timing)
  7. Agent dry-run (full pipeline)

Usage:
    python test.py
    python test.py --quick   # skip slow data download
"""

import argparse
import sys
import traceback
import warnings
from datetime import date, timedelta

warnings.filterwarnings("ignore")

PASS = "  PASS"
FAIL = "  FAIL"
results = []


def test(name: str, fn):
    try:
        fn()
        print(f"{PASS}  {name}")
        results.append((name, True, None))
    except Exception as e:
        print(f"{FAIL}  {name}")
        print(f"       {e}")
        results.append((name, False, str(e)))


# ── 1. Universe ───────────────────────────────────────────────────────────────

def test_sp500():
    from src.universe import get_sp500_tickers
    tickers = get_sp500_tickers()
    assert len(tickers) >= 400, f"Only got {len(tickers)} S&P 500 tickers"


def test_crypto_universe():
    from src.universe import get_crypto_tickers
    tickers = get_crypto_tickers()
    assert "BTC-USD" in tickers
    assert "ETH-USD" in tickers
    assert len(tickers) >= 10


def test_all_universe():
    from src.universe import get_universe
    tickers = get_universe("all")
    assert len(tickers) > 500, f"All universe too small: {len(tickers)}"


# ── 2. Data download ──────────────────────────────────────────────────────────

def test_data_download():
    import yfinance as yf
    raw = yf.download(["AAPL", "BTC-USD"], period="30d", auto_adjust=True, progress=False)
    assert not raw.empty
    assert "Close" in raw.columns.get_level_values(0)
    assert raw["Close"]["AAPL"].notna().sum() > 15
    assert raw["Close"]["BTC-USD"].notna().sum() > 15


# ── 3. Features ───────────────────────────────────────────────────────────────

def test_features_no_lookahead():
    import yfinance as yf
    import numpy as np
    from src.features import build_feature_matrix

    raw = yf.download(["AAPL", "MSFT"], period="2y", auto_adjust=True, progress=False)
    close = raw["Close"]; volume = raw["Volume"]
    high = raw["High"]; low = raw["Low"]

    fm = build_feature_matrix(close, volume, high, low, target_horizon=5)

    # fwd_return must always be 1-day (not 5-day) for correct P&L
    dates = fm.index.get_level_values("date").unique().sort_values()
    for ticker in ["AAPL", "MSFT"]:
        actual_1d = close[ticker].pct_change(1).shift(-1)
        for d in dates[-10:]:
            if d in actual_1d.index:
                stored = fm.loc[(d, ticker), "fwd_return"] if (d, ticker) in fm.index else np.nan
                expected = actual_1d[d]
                if not (np.isnan(stored) or np.isnan(expected)):
                    assert abs(stored - expected) < 1e-6, \
                        f"fwd_return mismatch on {d}: stored={stored:.6f}, expected={expected:.6f}"


def test_feature_count():
    import yfinance as yf
    from src.features import build_features_for_ticker

    raw = yf.download("AAPL", period="2y", auto_adjust=True, progress=False)
    feat = build_features_for_ticker(
        "AAPL", raw["Close"], raw["Volume"], raw["High"], raw["Low"]
    )
    assert feat.shape[1] >= 40, f"Expected 40+ features, got {feat.shape[1]}"
    # Latest row should have no NaNs (enough history)
    nan_count = feat.iloc[-1].isna().sum()
    assert nan_count == 0, f"Latest row has {nan_count} NaNs"


# ── 4. Regime filter ──────────────────────────────────────────────────────────

def test_regime_live():
    from src.regime import fetch_regime_data, get_current_regime
    spy, vix = fetch_regime_data(start="2024-01-01")
    assert len(spy) > 100
    assert len(vix) > 100
    r = get_current_regime(spy, vix)
    assert r["regime"] in ("bull", "neutral", "caution", "bear")
    assert 0.0 <= r["scalar"] <= 1.0
    assert r["vix"] > 0


def test_regime_bear_detected():
    """Verify bear regime is correctly identified during known crash periods."""
    import pandas as pd
    from src.regime import compute_regime_series
    from src.regime import fetch_regime_data

    spy, vix = fetch_regime_data(start="2020-01-01")
    df = compute_regime_series(spy, vix)

    # March 2020 (COVID crash) should be bear
    march_2020 = df.loc["2020-03-15":"2020-03-31"]
    bear_days = (march_2020["regime"] == "bear").sum()
    assert bear_days >= 5, f"Expected bear regime in March 2020, got {bear_days} bear days"


# ── 5. Model ──────────────────────────────────────────────────────────────────

def test_model_loads():
    from src.model import load_model
    model, feature_cols = load_model()
    assert model is not None
    assert len(feature_cols) >= 40


def test_signal_generation():
    import yfinance as yf
    import numpy as np
    from src.model import load_model
    from src.signals import generate_signals

    model, feature_cols = load_model()
    raw = yf.download(["AAPL", "MSFT", "GOOGL", "NVDA", "META"],
                      period="300d", auto_adjust=True, progress=False)
    close = raw["Close"]; volume = raw["Volume"]
    high = raw["High"]; low = raw["Low"]

    signals = generate_signals(close, volume, high, low, model, feature_cols, top_n=3)
    assert not signals.empty, "No signals generated"
    assert len(signals) <= 3
    assert "predicted_rank" in signals.columns
    assert (signals["predicted_rank"] >= 0).all()


# ── 6. Portfolio state ────────────────────────────────────────────────────────

def test_portfolio_state_rebalance_timing():
    from src.portfolio_state import should_rebalance, get_next_rebalance_date
    import src.portfolio_state as ps
    import json

    # Temporarily override state with a known last_rebalance
    original = ps.STATE_FILE
    ps.STATE_FILE = ps.STATE_FILE.parent / "portfolio_state_test.json"

    try:
        # Rebalanced today → should NOT rebalance again
        test_state = {"positions": {}, "trade_history": [], "last_rebalance": date.today().isoformat()}
        with open(ps.STATE_FILE, "w") as f:
            json.dump(test_state, f)
        assert not should_rebalance(5), "Should not rebalance same day"

        # Rebalanced 6 trading days ago → should rebalance
        past = date.today()
        trading_days = 0
        while trading_days < 6:
            past -= timedelta(days=1)
            if past.weekday() < 5:
                trading_days += 1
        test_state["last_rebalance"] = past.isoformat()
        with open(ps.STATE_FILE, "w") as f:
            json.dump(test_state, f)
        assert should_rebalance(5), "Should rebalance after 6 trading days"

    finally:
        ps.STATE_FILE.unlink(missing_ok=True)
        ps.STATE_FILE = original


# ── 7. Full agent dry-run ─────────────────────────────────────────────────────

def test_agent_dry_run():
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, "agent.py", "--dry-run", "--budget", "100", "--top-n", "2", "--force"],
        capture_output=True, text=True, timeout=180
    )
    assert result.returncode == 0, f"Agent crashed:\n{result.stderr[-1000:]}"
    assert "orders placed" in result.stdout, "No orders in agent output"
    assert "Agent run complete" in result.stdout


# ── Runner ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Skip slow tests")
    args = parser.parse_args()

    print("\nRunning tests...\n")

    test("S&P 500 universe fetch",      test_sp500)
    test("Crypto universe",             test_crypto_universe)
    test("All universe size",           test_all_universe)

    if not args.quick:
        test("Data download (AAPL + BTC)", test_data_download)
        test("Feature count (41+)",        test_feature_count)
        test("No lookahead in fwd_return", test_features_no_lookahead)

    test("Regime filter live",          test_regime_live)
    test("Bear regime detection",       test_regime_bear_detected)
    test("Model loads",                 test_model_loads)

    if not args.quick:
        test("Signal generation",       test_signal_generation)

    test("Portfolio rebalance timing",  test_portfolio_state_rebalance_timing)

    if not args.quick:
        test("Agent dry-run",           test_agent_dry_run)

    passed = sum(1 for _, ok, _ in results if ok)
    total  = len(results)
    failed = [(n, e) for n, ok, e in results if not ok]

    print(f"\n{'─'*40}")
    print(f"  {passed}/{total} tests passed")
    if failed:
        print(f"\n  Failed:")
        for name, err in failed:
            print(f"    ✗ {name}: {err}")
    print(f"{'─'*40}\n")

    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
