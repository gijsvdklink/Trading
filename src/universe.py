"""
Curated universe of ~90 instruments available on Revolut.

Focused list — fast to download (~20s), high model quality.
Split into three risk tiers for portfolio construction.
"""

# Leveraged/inverse ETFs blocked for EU retail investors under PRIIPs regulation
REVOLUT_UNAVAILABLE = {
    "TQQQ", "SOXL", "TECL", "UPRO", "LABU", "FNGU",
    "WEBL", "DFEN", "NAIL", "WANT", "UVXY", "SVXY",
}

# ── Tier 1: Stable large-cap S&P 500 stocks ───────────────────────────────────
SP500 = [
    # Tech
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "AVGO", "AMD", "INTC", "QCOM",
    "ADBE", "CRM", "NOW", "SNOW", "PANW",
    # Finance
    "JPM", "BAC", "GS", "MS", "V", "MA", "AXP", "BLK", "SCHW",
    # Healthcare
    "LLY", "UNH", "JNJ", "ABBV", "MRK", "PFE", "AMGN", "GILD",
    # Energy
    "XOM", "CVX", "COP", "SLB", "MPC", "VLO", "OXY",
    # Consumer
    "AMZN", "TSLA", "HD", "MCD", "SBUX", "NKE", "TGT", "COST", "WMT",
    # Industrial
    "CAT", "DE", "HON", "RTX", "LMT", "GE",
    # Other
    "BRK-B", "PG", "KO", "PEP", "PM", "TMO", "DHR", "ADM",
]

# ── Tier 2: High-growth / high-risk individual names ─────────────────────────
HIGH_RISK = [
    "NVDA", "TSLA", "COIN", "PLTR", "MSTR", "RKLB",
    "IONQ", "HOOD", "SOFI", "UPST", "AFRM", "AXON",
    "SMCI", "RXRX", "SHOP", "NFLX",
]

# ── Tier 3: ETFs available on Revolut ────────────────────────────────────────
ETFS = [
    # Broad market
    "SPY", "QQQ", "IWM", "DIA",
    # Sectors
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLB",
    # Thematic
    "ARKK", "SOXX", "SMH", "BOTZ",
    "GLD", "SLV",
    "ICLN", "TAN",
    "EEM", "EFA",
]

# ── Combined ─────────────────────────────────────────────────────────────────
ALL_TICKERS = sorted(set(SP500 + HIGH_RISK + ETFS) - REVOLUT_UNAVAILABLE)


def get_universe(name: str = "all") -> list[str]:
    universes = {
        "all":       ALL_TICKERS,
        "sp500":     sorted(set(SP500) - REVOLUT_UNAVAILABLE),
        "high_risk": sorted(set(HIGH_RISK) - REVOLUT_UNAVAILABLE),
        "etfs":      sorted(set(ETFS) - REVOLUT_UNAVAILABLE),
    }
    if name not in universes:
        raise ValueError(f"Unknown universe '{name}'. Options: {list(universes)}")
    tickers = universes[name]
    print(f"  Universe '{name}': {len(tickers)} instruments")
    return tickers


def get_sp500_tickers()     -> list[str]: return get_universe("sp500")
def get_high_risk_tickers() -> list[str]: return get_universe("high_risk")
def get_etf_universe()      -> list[str]: return get_universe("etfs")
def get_crypto_tickers()    -> list[str]: return []
