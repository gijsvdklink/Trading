"""
Realistic walk-forward backtester — v2.

Improvements over v1:
  - Fixed turnover calculation (v1 double-counted via symmetric_difference)
  - Regime filter: scales position size down in neutral/bear markets
  - Configurable holding period (default 5 days, reduces transaction costs ~5x)
  - Volatility-weighted position sizing (smaller allocations to high-vol stocks)

Transaction costs:
  - Spread:     0.05% per trade
  - Slippage:   0.05% per trade
  - Commission: 0 (Revolut zero-commission)
  Round-trip:   ~0.10%
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SPREAD_BPS = 5
SLIPPAGE_BPS = 5
COST_ONE_WAY = (SPREAD_BPS + SLIPPAGE_BPS) / 10_000  # 0.10% round-trip


def _vol_weights(tickers: list, predictions: pd.DataFrame, atr_col: str = "atr_14") -> dict:
    """
    Inverse-volatility weights. Stocks with lower ATR get higher weight.
    Falls back to equal weight if ATR not available.
    """
    if atr_col not in predictions.columns:
        return {t: 1.0 / len(tickers) for t in tickers}

    atrs = {}
    for t in tickers:
        try:
            atrs[t] = predictions.loc[t, atr_col]
        except Exception:
            atrs[t] = np.nan

    vals = np.array(list(atrs.values()), dtype=float)
    vals = np.where(np.isnan(vals) | (vals <= 0), np.nanmedian(vals[vals > 0]), vals)
    inv_vol = 1.0 / vals
    inv_vol /= inv_vol.sum()
    return dict(zip(tickers, inv_vol))


def run_backtest(
    predictions: pd.DataFrame,
    close: pd.DataFrame,
    top_n: int = 20,
    holding_days: int = 5,
    regime_series: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Simulate long-top-N strategy with configurable holding period.

    Rebalances every `holding_days` days — reduces turnover and costs.
    Uses inverse-volatility weighting within the selected portfolio.
    Scales position size by regime scalar (0 = cash in bear markets).

    Returns daily portfolio returns.
    """
    dates = predictions.index.get_level_values("date").unique().sort_values()
    portfolio_returns = []

    current_longs: list = []
    current_weights: dict = {}
    days_held = holding_days  # force rebalance on first day

    for i, date in enumerate(dates):
        day_preds = predictions.loc[date].dropna(subset=["pred", "fwd_return"])
        if len(day_preds) < top_n:
            continue

        rebalance = days_held >= holding_days

        if rebalance:
            new_longs = day_preds.nlargest(top_n, "pred").index.tolist()
            new_weights = _vol_weights(new_longs, day_preds)

            # Transaction cost: only on stocks that changed
            entries = set(new_longs) - set(current_longs)
            exits = set(current_longs) - set(new_longs)
            n_changed = len(entries) + len(exits)
            turnover = n_changed / max(top_n * 2, 1)  # fraction of portfolio traded
            cost = turnover * COST_ONE_WAY

            current_longs = new_longs
            current_weights = new_weights
            days_held = 1
        else:
            cost = 0.0
            days_held += 1

        if not current_longs:
            continue

        # Compute gross return (weighted)
        gross = 0.0
        for t in current_longs:
            try:
                ret = day_preds.loc[t, "fwd_return"]
                w = current_weights.get(t, 1.0 / top_n)
                if not np.isnan(ret):
                    gross += w * ret
            except Exception:
                pass

        # Apply regime scalar
        scalar = 1.0
        if regime_series is not None and date in regime_series.index:
            scalar = regime_series.loc[date, "scalar"]

        net = gross * scalar - cost

        portfolio_returns.append({
            "date": date,
            "gross_return": gross,
            "cost": cost,
            "regime_scalar": scalar,
            "net_return": net,
            "rebalanced": rebalance,
            "n_longs": len(current_longs),
        })

    return pd.DataFrame(portfolio_returns).set_index("date")


def compute_metrics(returns: pd.Series, label: str = "Strategy") -> dict:
    clean = returns.dropna()
    if len(clean) == 0:
        return {}

    ann_ret = clean.mean() * 252
    ann_vol = clean.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    cum = (1 + clean).cumprod()
    peak = cum.cummax()
    drawdown = (cum - peak) / peak
    max_dd = drawdown.min()
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0
    win_rate = (clean > 0).mean()

    return {
        "label": label,
        "ann_return": ann_ret,
        "ann_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "calmar_ratio": calmar,
        "win_rate": win_rate,
        "total_trading_days": len(clean),
    }


def plot_results(
    portfolio: pd.DataFrame,
    benchmark_close: pd.Series | None = None,
    title: str = "Strategy Performance",
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    cum_strat = (1 + portfolio["net_return"]).cumprod()
    axes[0].plot(cum_strat.index, cum_strat.values, label="Strategy (net)", color="steelblue", lw=1.5)

    if benchmark_close is not None:
        bench_ret = benchmark_close.pct_change().reindex(portfolio.index).fillna(0)
        cum_bench = (1 + bench_ret).cumprod()
        axes[0].plot(cum_bench.index, cum_bench.values, label="S&P 500 (SPY)", color="gray", lw=1, alpha=0.8)

    # Shade bear/caution periods
    if "regime_scalar" in portfolio.columns:
        bear_mask = portfolio["regime_scalar"] == 0.0
        for date in portfolio.index[bear_mask]:
            axes[0].axvspan(date, date + pd.Timedelta(days=1), color="red", alpha=0.05)

    axes[0].set_ylabel("Cumulative Return")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(1, color="black", lw=0.5, ls="--")

    cum = (1 + portfolio["net_return"]).cumprod()
    peak = cum.cummax()
    drawdown = (cum - peak) / peak
    axes[1].fill_between(drawdown.index, drawdown.values, 0, color="red", alpha=0.4, label="Drawdown")
    axes[1].set_ylabel("Drawdown")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    colors = np.where(portfolio["net_return"].values >= 0, "steelblue", "tomato")
    axes[2].bar(portfolio.index, portfolio["net_return"].values, color=colors, width=1, alpha=0.7)
    axes[2].set_ylabel("Daily Net Return")
    axes[2].grid(True, alpha=0.3)
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    path = RESULTS_DIR / "backtest_performance.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {path}")
    plt.show()


def full_backtest_report(
    predictions: pd.DataFrame,
    close: pd.DataFrame,
    top_n: int = 20,
    holding_days: int = 5,
    regime_series: pd.DataFrame | None = None,
) -> pd.DataFrame:
    label = f"Top-{top_n}, {holding_days}d hold"
    if regime_series is not None:
        label += ", regime filter"

    print(f"\nRunning backtest ({label})...")
    portfolio = run_backtest(predictions, close, top_n=top_n,
                             holding_days=holding_days, regime_series=regime_series)

    metrics = compute_metrics(portfolio["net_return"], label=label)

    # Days in cash due to bear regime
    if "regime_scalar" in portfolio.columns:
        cash_days = (portfolio["regime_scalar"] == 0).sum()
        cash_pct = cash_days / len(portfolio)
        print(f"  Days in cash (bear regime): {cash_days} ({cash_pct:.1%})")

    print("\n" + "=" * 55)
    print(f"  BACKTEST RESULTS — {metrics['label']}")
    print("=" * 55)
    print(f"  Annualised Return:     {metrics['ann_return']:.1%}")
    print(f"  Annualised Volatility: {metrics['ann_volatility']:.1%}")
    print(f"  Sharpe Ratio:          {metrics['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown:          {metrics['max_drawdown']:.1%}")
    print(f"  Calmar Ratio:          {metrics['calmar_ratio']:.2f}")
    print(f"  Win Rate:              {metrics['win_rate']:.1%}")
    print(f"  Trading Days:          {metrics['total_trading_days']}")
    print("=" * 55 + "\n")

    spy_close = close["SPY"] if "SPY" in close.columns else None
    portfolio.to_parquet(RESULTS_DIR / "portfolio_returns.parquet")
    plot_results(portfolio, benchmark_close=spy_close, title=f"Backtest — {label}")

    return portfolio
