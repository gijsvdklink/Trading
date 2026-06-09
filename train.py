"""
Full training pipeline. Run this once to:
  1. Download all price data
  2. Engineer features
  3. Run walk-forward cross-validation (produces backtest predictions)
  4. Train a final model on the most recent data
  5. Print performance metrics and save a backtest chart

Usage:
    python train.py                          # Default 'all' universe, 5-day holding
    python train.py --universe sp500         # S&P 500 only
    python train.py --universe high_risk     # High-risk instruments only
    python train.py --top-n 10               # Hold top 10 positions
    python train.py --holding-days 1         # Daily rebalancing (original behaviour)
    python train.py --no-regime              # Disable regime filter
    python train.py --refresh                # Force re-download price data
"""

import argparse
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", default="all",
                        choices=["sp500", "nasdaq100", "high_risk", "etfs", "crypto", "all"])
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--holding-days", type=int, default=5,
                        help="Days between rebalances (5 = weekly, reduces costs)")
    parser.add_argument("--no-regime", action="store_true",
                        help="Disable market regime filter")
    parser.add_argument("--refresh", action="store_true",
                        help="Force re-download price and feature data")
    parser.add_argument("--start", default="2014-01-01")
    parser.add_argument("--skip-backtest", action="store_true")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  TRAINING PIPELINE — Universe: {args.universe}")
    print(f"  Holding period: {args.holding_days}d | Regime filter: {not args.no_regime}")
    print(f"{'='*60}\n")

    # ── 1. Price data ─────────────────────────────────────────────────────────
    import yfinance as yf
    import time
    import numpy as np
    from tqdm import tqdm
    from src.universe import get_universe

    tickers = get_universe(args.universe)
    cache_file = DATA_DIR / f"prices_{args.universe}.parquet"

    if args.refresh or not cache_file.exists():
        print(f"Downloading {len(tickers)} tickers from {args.start}...")
        batch_size = 50
        all_data = []
        for batch in tqdm([tickers[i:i+batch_size] for i in range(0, len(tickers), batch_size)],
                          desc="Downloading"):
            try:
                raw = yf.download(batch, start=args.start, auto_adjust=True,
                                  progress=False, threads=True)
                if not raw.empty:
                    all_data.append(raw)
            except Exception as e:
                print(f"  Batch error: {e}")
            time.sleep(0.3)

        prices = pd.concat(all_data, axis=1) if all_data else pd.DataFrame()
        prices = prices.sort_index(axis=1)
        prices.to_parquet(cache_file)
        print(f"Saved to {cache_file}")
    else:
        print(f"Loading from cache: {cache_file}")
        prices = pd.read_parquet(cache_file)

    fields = prices.columns.get_level_values(0).unique()
    close  = prices["Close"] if "Close" in fields else prices["Adj Close"]
    volume = prices["Volume"]
    high   = prices["High"]
    low    = prices["Low"]

    valid = close.notna().mean() >= 0.80
    close = close.loc[:, valid]
    keep  = close.columns
    volume, high, low = volume[keep], high[keep], low[keep]
    print(f"Clean universe: {len(keep)} instruments × {len(close)} trading days\n")

    # ── 2. Features ───────────────────────────────────────────────────────────
    feat_cache = DATA_DIR / f"features_{args.universe}.parquet"

    if feat_cache.exists() and not args.refresh:
        print(f"Loading features from cache: {feat_cache}")
        feature_matrix = pd.read_parquet(feat_cache)
    else:
        from src.features import build_feature_matrix
        feature_matrix = build_feature_matrix(
            close, volume, high, low, target_horizon=args.holding_days
        )
        feature_matrix.to_parquet(feat_cache)
        print(f"Features saved to {feat_cache}")

    print(f"Feature matrix: {feature_matrix.shape}")
    feature_cols_preview = [c for c in feature_matrix.columns if c not in ("fwd_return", "target")]
    print(f"Features ({len(feature_cols_preview)} total): {feature_cols_preview[:8]}...\n")

    # ── 3. Regime data ────────────────────────────────────────────────────────
    regime_series = None
    if not args.no_regime:
        print("Computing market regime filter (SPY + VIX)...")
        from src.regime import fetch_regime_data, compute_regime_series
        spy, vix = fetch_regime_data(start=args.start)
        regime_series = compute_regime_series(spy, vix)
        bear_days = (regime_series["scalar"] == 0).sum()
        print(f"  Bear days (cash): {bear_days} / {len(regime_series)} ({bear_days/len(regime_series):.1%})\n")

    # ── 4. Walk-forward backtest ──────────────────────────────────────────────
    if not args.skip_backtest:
        from src.model import walk_forward_train, evaluate_predictions
        from src.backtest import full_backtest_report

        predictions, models, feature_importance = walk_forward_train(feature_matrix)

        print("\nTop-20 most important features:")
        print(feature_importance.head(20).to_string())

        print("\nModel evaluation (cross-sectional IC):")
        eval_metrics = evaluate_predictions(predictions)
        for k, v in eval_metrics.items():
            print(f"  {k}: {v:.4f}")

        predictions.to_parquet(DATA_DIR / f"predictions_{args.universe}.parquet")
        full_backtest_report(
            predictions, close,
            top_n=args.top_n,
            holding_days=args.holding_days,
            regime_series=regime_series,
        )

    # ── 5. Train final model ──────────────────────────────────────────────────
    from src.model import train_final_model
    model, feature_cols = train_final_model(feature_matrix)

    print("\nDone. Run daily_signals.py to get today's signals.")


if __name__ == "__main__":
    main()
