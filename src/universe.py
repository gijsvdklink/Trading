"""
Stock universe definitions.

Supports multiple risk tiers — mix and match based on your risk appetite.
All tickers are available on Revolut.
"""

from pathlib import Path
import pandas as pd

# Tickers NOT available on Revolut for EU retail investors.
# Leveraged/inverse ETFs are blocked under PRIIPs/KID regulation.
REVOLUT_UNAVAILABLE = {
    # 3× leveraged ETFs
    "TQQQ", "SOXL", "TECL", "UPRO", "LABU", "FNGU",
    "WEBL", "DFEN", "NAIL", "WANT",
    # Volatility products
    "UVXY", "SVXY",
}


def get_sp500_tickers() -> list[str]:
    """S&P 500 constituents — large-cap, liquid."""
    try:
        import requests
        headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}
        resp = requests.get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers=headers,
            timeout=10,
        )
        resp.raise_for_status()
        from io import StringIO
        tables = pd.read_html(StringIO(resp.text))
        tickers = tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
        if len(tickers) > 400:
            return sorted(tickers)
        raise ValueError(f"Only got {len(tickers)} tickers — table format may have changed")
    except Exception as e:
        print(f"  Wikipedia scrape failed ({e}), using static S&P 500 list")
        return _SP500_FALLBACK


def get_nasdaq100_tickers() -> list[str]:
    """Nasdaq-100 — high-growth tech heavy."""
    try:
        import requests
        from io import StringIO
        headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}
        resp = requests.get("https://en.wikipedia.org/wiki/Nasdaq-100", headers=headers, timeout=10)
        resp.raise_for_status()
        tables = pd.read_html(StringIO(resp.text))
        for t in tables:
            if "Ticker" in t.columns:
                return sorted(t["Ticker"].str.replace(".", "-", regex=False).tolist())
        for t in tables:
            for col in t.columns:
                if "symbol" in col.lower() or "ticker" in col.lower():
                    return sorted(t[col].str.replace(".", "-", regex=False).dropna().tolist())
    except Exception:
        pass
    return _NASDAQ100_FALLBACK


def get_high_risk_tickers() -> list[str]:
    """
    High-risk / high-reward universe:
    - Leveraged ETFs (2-3x market/sector exposure)
    - High-growth individual names
    - Small-cap ETFs
    All available on Revolut.
    """
    leveraged_etfs = [
        "TQQQ",   # 3x Nasdaq
        "SOXL",   # 3x Semiconductors
        "TECL",   # 3x Tech
        "UPRO",   # 3x S&P 500
        "LABU",   # 3x Biotech
        "FNGU",   # 3x FANGs
        "WEBL",   # 3x Internet
        "DFEN",   # 3x Aerospace & Defense
        "NAIL",   # 3x Homebuilders
        "WANT",   # 3x Consumer Discretionary
    ]
    small_cap_etfs = [
        "IWM",    # Russell 2000
        "IJR",    # S&P 600 Small Cap
        "VBK",    # Vanguard Small-Cap Growth
        "SCHA",   # Schwab Small-Cap
    ]
    high_growth = [
        "NVDA", "AMD", "MSTR", "COIN", "SMCI",
        "PLTR", "RKLB", "IONQ", "RXRX", "ARKG",
        "TSLA", "NFLX", "META", "SHOP", "SQ",
        "HOOD", "SOFI", "UPST", "AFRM", "AXON",
    ]
    return sorted(set(leveraged_etfs + small_cap_etfs + high_growth))


def get_crypto_tickers() -> list[str]:
    """
    Major cryptocurrencies via Yahoo Finance suffix (-USD).
    These are tradeable via Alpaca's crypto API.
    High volatility — treat as a separate allocation.
    """
    return sorted([
        "BTC-USD",   # Bitcoin
        "ETH-USD",   # Ethereum
        "SOL-USD",   # Solana
        "BNB-USD",   # BNB
        "XRP-USD",   # Ripple
        "ADA-USD",   # Cardano
        "AVAX-USD",  # Avalanche
        "DOGE-USD",  # Dogecoin
        "LINK-USD",  # Chainlink
        "DOT-USD",   # Polkadot
        "POL-USD",   # Polygon (rebranded from MATIC)
        "AAVE-USD",  # Aave
        "LTC-USD",   # Litecoin
        "BCH-USD",   # Bitcoin Cash
        "ATOM-USD",  # Cosmos
    ])


def get_etf_universe() -> list[str]:
    """
    Liquid sector + thematic ETFs — good for momentum plays.
    All available on Revolut.
    """
    return sorted([
        # US Broad
        "SPY", "QQQ", "IWM", "DIA", "MDY",
        # Sectors
        "XLK", "XLF", "XLE", "XLV", "XLI", "XLB", "XLP", "XLU", "XLRE", "XLY",
        # Thematic
        "ARKK", "ARKG", "ARKW", "ARKF", "ARKX",
        "BOTZ", "ROBO", "SOXX", "SMH",
        "CIBR", "BUG",                  # Cybersecurity
        "ICLN", "QCLN",                 # Clean energy
        "GLD", "SLV", "GDX", "GDXJ",   # Commodities
        "TAN", "URNM",                  # Solar / Uranium
        # Volatility
        "UVXY", "SVXY",
        # International
        "EEM", "EFA", "FXI", "EWJ",
    ])


# ─── Universe compositions ──────────────────────────────────────────────────

UNIVERSES = {
    "sp500": {
        "description": "S&P 500 — large-cap, safest universe for the model",
        "risk_level": "medium",
        "get_tickers": get_sp500_tickers,
    },
    "nasdaq100": {
        "description": "Nasdaq-100 — tech-heavy, higher beta",
        "risk_level": "medium-high",
        "get_tickers": get_nasdaq100_tickers,
    },
    "high_risk": {
        "description": "Leveraged ETFs + high-growth names — maximum volatility",
        "risk_level": "high",
        "get_tickers": get_high_risk_tickers,
    },
    "etfs": {
        "description": "Sector + thematic ETFs — good for momentum",
        "risk_level": "medium",
        "get_tickers": get_etf_universe,
    },
    "crypto": {
        "description": "Major cryptocurrencies — very high volatility",
        "risk_level": "very-high",
        "get_tickers": get_crypto_tickers,
    },
    "all": {
        "description": "All universes combined",
        "risk_level": "mixed",
        "get_tickers": lambda: sorted(set(
            get_sp500_tickers()
            + get_nasdaq100_tickers()
            + get_high_risk_tickers()
            + get_etf_universe()
            + get_crypto_tickers()
        )),
    },
}


def get_universe(name: str) -> list[str]:
    if name not in UNIVERSES:
        raise ValueError(f"Unknown universe '{name}'. Options: {list(UNIVERSES)}")
    tickers = UNIVERSES[name]["get_tickers"]()
    print(f"Universe '{name}': {len(tickers)} instruments ({UNIVERSES[name]['description']})")
    return tickers


# ─── Fallbacks in case Wikipedia scraping fails ─────────────────────────────

_NASDAQ100_FALLBACK = [
    "AAPL","MSFT","AMZN","NVDA","META","TSLA","GOOGL","GOOG","AVGO","COST",
    "NFLX","AMD","ADBE","QCOM","INTU","AMAT","MELI","ISRG","LRCX","AZN",
    "CSCO","TXN","AMGN","BKNG","MU","PANW","REGN","KLAC","SNPS","CDNS",
    "ASML","CRWD","ADI","CEG","FTNT","MRVL","ABNB","ORLY","TEAM","MNST",
    "NXPI","WDAY","CTAS","ADSK","CPRT","PCAR","PAYX","MAR","DASH","ROST",
    "AEP","ODFL","FAST","DXCM","CHTR","GEHC","TTD","FANG","EXC","IDXX",
    "VRSK","EA","CTSH","ILMN","ON","BKR","LULU","SGEN","GFS","ZS",
    "ANSS","DLTR","BIIB","ALGN","ENPH","DDOG","LCID","ZM","MTCH","RIVN",
]

_SP500_FALLBACK = [
    # Mega-cap / top 50
    "AAPL","MSFT","NVDA","AMZN","GOOGL","GOOG","META","BRK-B","LLY","AVGO",
    "JPM","TSLA","UNH","V","XOM","COST","MA","PG","JNJ","HD",
    "ABBV","MRK","CVX","BAC","NFLX","CRM","PEP","KO","AMD","WMT",
    "TMO","ADBE","ACN","MCD","LIN","ABT","TXN","CSCO","DHR","PM",
    "VZ","NEE","NKE","RTX","UPS","BMY","HON","LOW","CMCSA","AMGN",
    # 51-150
    "QCOM","IBM","GS","CAT","SPGI","BLK","GE","AXP","INTU","ISRG",
    "NOW","PLD","DE","SYK","MDLZ","MO","BKNG","GILD","TJX","ZTS",
    "PGR","SCHW","CI","REGN","MMM","DUK","SO","COP","EOG","SLB",
    "ADP","ITW","CME","AON","ICE","ZBH","MCO","EQIX","PSA","AMT",
    "SBUX","T","CHTR","TMUS","ATVI","EA","TTWO","NUAN","CTSH","FISV",
    "HCA","ELV","MCK","ABS","CVS","WBA","HUM","CNC","MOH","ANTM",
    "FIS","PYPL","SQ","ADSK","WDAY","SNPS","CDNS","ANSS","DOCU","ZM",
    "CRWD","PANW","FTNT","OKTA","NET","DDOG","SPLK","VEEV","RCL","CCL",
    "MAR","HLT","MGM","WYNN","LVS","NWL","CLX","CHD","EL","CL",
    # 151-250
    "PFE","MRNA","BIIB","ILMN","VRTX","REGN","ALNY","RARE","SGEN","NKTR",
    "BAX","BDX","BSX","EW","IDXX","IQV","MTRX","PDCO","XRAY","ABC",
    "GPN","V","MA","AXP","DFS","COF","SYF","ALLY","CBOE","NDAQ",
    "BK","STT","TROW","BEN","IVZ","AMG","NTRS","RF","CFG","HBAN",
    "KEY","FITB","MTB","WFC","C","USB","PNC","MS","GS","JPM",
    "LMT","NOC","GD","BA","RTX","TXT","HII","L","TDG","CARR",
    "OTIS","PH","EMR","ETN","ROK","AME","FTV","GNRC","HUBB","IR",
    "XYL","RXN","IEX","TT","JCI","A","KEYS","TRMB","LDOS","SAIC",
    "DXC","HPE","HPQ","NTAP","STX","WDC","SNDK","SWKS","QRVO","MCHP",
    "MPWR","NXPI","ADI","TXN","AMAT","LRCX","KLAC","ASML","INTC","MU",
    # 251-350
    "F","GM","STLA","TM","HMC","TSLA","RIVN","LCID","FSR","NKLA",
    "CAT","DE","CMI","PCAR","WAB","ALV","BWA","VC","LEA","MGA",
    "LYB","DOW","DD","EMN","PPG","SHW","RPM","AVY","PKG","SEE",
    "IP","WRK","GPK","BERY","SON","OI","GEF","CLW","KS","RKT",
    "NEM","FCX","AA","X","NUE","STLD","RS","CMC","MP","LAC",
    "DVN","PXD","FANG","OXY","COP","CVX","XOM","PSX","VLO","MPC",
    "HES","APA","HAL","BKR","SLB","FTI","NOV","CHK","RRC","EQT",
    "CNX","AR","CQP","LNG","TELL","NFE","NEW","ET","KMI","WMB",
    "OKE","TRGP","MMP","EPD","PAA","MPLX","HES","DCP","ENLC","CEQP",
    "D","EXC","AEP","SO","DUK","NEE","XEL","WEC","ES","ETR",
    # 351-500
    "AWK","CWT","SJW","MSEX","ARTNA","YORW","CTWS","GWRS","SRR","PRMW",
    "O","SPG","EQR","AVB","ESS","MAA","UDR","CPT","AIV","NHI",
    "VTR","WELL","HR","DOC","PEAK","OHI","SBRA","LTC","CSR","LXP",
    "WPC","STORE","EPRT","ADC","NNN","NETL","NTST","WHLR","NXRT","IRT",
    "AMH","INVH","TRICON","SFR","RESI","NREF","BRSP","KREF","BXMT","GPMT",
    "CMG","YUM","MCD","QSR","JACK","WEN","SHAK","TXRH","BJRI","CAKE",
    "DRI","EAT","CBRL","BLMN","DENN","NATH","RUTH","RRGB","HABT","PZZA",
    "KR","ACI","SFM","WMT","COST","TGT","BJ","CASY","TSCO","ORLY",
    "AZO","AAP","MNRO","GPC","LKQ","COPART","KMX","AN","PAG","LAD",
    "ABG","SAH","RUSHA","RUSHB","CVNA","VRM","OPEN","OPAD","CARG","TRU",
]
