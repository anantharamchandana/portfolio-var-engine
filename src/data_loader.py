"""
data_loader.py
--------------
Responsible for fetching, validating, and caching historical price data
from Yahoo Finance. All other modules depend on clean output from here.

Design principles:
- Fails loudly on bad tickers or missing data (no silent NaN propagation)
- Caches raw data locally so repeated runs don't hit the API
- Returns adjusted closing prices only (accounts for splits and dividends)
"""

import os
import logging
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path

# Configure logger for this module
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Project root and cache directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "data"
CACHE_DIR.mkdir(exist_ok=True)


def fetch_prices(
    tickers: list[str],
    start_date: str,
    end_date: str,
    use_cache: bool = True
) -> pd.DataFrame:
    """
    Fetch adjusted closing prices for a list of tickers over a date range.

    Parameters
    ----------
    tickers : list[str]
        List of valid Yahoo Finance ticker symbols e.g. ['AAPL', 'MSFT', 'JPM']
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    use_cache : bool
        If True, checks local cache before hitting Yahoo Finance API

    Returns
    -------
    pd.DataFrame
        DataFrame of adjusted closing prices.
        Index: DatetimeIndex (trading days only)
        Columns: one column per ticker

    Raises
    ------
    ValueError
        If tickers list is empty, dates are invalid, or data is missing
        for any requested ticker after download.
    """
    _validate_inputs(tickers, start_date, end_date)

    cache_path = _build_cache_path(tickers, start_date, end_date)

    if use_cache and cache_path.exists():
        logger.info(f"Loading prices from cache: {cache_path.name}")
        prices = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        logger.info(f"Loaded {len(prices)} trading days from cache.")
        return prices

    logger.info(f"Fetching prices from Yahoo Finance for: {tickers}")
    raw = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,   # gives adjusted prices directly
        progress=False
    )

    # yfinance returns multi-level columns when multiple tickers are passed
    # We want only the 'Close' level
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        # Single ticker returns flat columns
        prices = raw[["Close"]]
        prices.columns = tickers

    _validate_output(prices, tickers)

    # Persist to cache
    prices.to_csv(cache_path)
    logger.info(f"Cached price data to: {cache_path.name}")

    return prices


def compute_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily log returns from price data.
    """
    if prices.empty:
        raise ValueError("prices DataFrame is empty — cannot compute returns.")

    import numpy as np
    returns = np.log(prices / prices.shift(1)).dropna()

    bad_cols = returns.columns[returns.isnull().all()].tolist()
    if bad_cols:
        logger.warning(f"Dropping tickers with all-NaN returns: {bad_cols}")
        returns = returns.drop(columns=bad_cols)

    logger.info(
        f"Computed log returns: {len(returns)} trading days, "
        f"{len(returns.columns)} tickers."
    )
    return returns


# ---------------------------------------------------------------------------
# Internal helpers — not part of public API
# ---------------------------------------------------------------------------

def _validate_inputs(tickers: list, start_date: str, end_date: str) -> None:
    """Validate inputs before hitting the API."""
    if not tickers:
        raise ValueError("tickers list cannot be empty.")

    if not all(isinstance(t, str) and t.strip() for t in tickers):
        raise ValueError("All tickers must be non-empty strings.")

    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError:
        raise ValueError("Dates must be in 'YYYY-MM-DD' format.")

    if start >= end:
        raise ValueError(
            f"start_date ({start_date}) must be before end_date ({end_date})."
        )

    if end > datetime.today():
        raise ValueError("end_date cannot be in the future.")

    min_days = 252  # one trading year minimum for meaningful VaR
    if (end - start).days < min_days:
        logger.warning(
            f"Date range is only {(end - start).days} calendar days. "
            "VaR estimates are unreliable with less than one year of data. "
            "Recommend at least 3 years (start_date ~3 years back)."
        )


def _validate_output(prices: pd.DataFrame, tickers: list) -> None:
    """Ensure downloaded data is usable — fail loudly if not."""
    if prices.empty:
        raise ValueError(
            "Yahoo Finance returned empty data. "
            "Check ticker symbols and date range."
        )

    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        raise ValueError(
            f"No price data returned for tickers: {missing}. "
            "Verify these are valid Yahoo Finance symbols."
        )

    # Warn about tickers with significant gaps (>5% missing trading days)
    for ticker in prices.columns:
        null_pct = prices[ticker].isnull().mean()
        if null_pct > 0.05:
            logger.warning(
                f"{ticker} has {null_pct:.1%} missing values. "
                "This may indicate a data quality issue or a ticker "
                "that wasn't publicly traded for the full date range."
            )


def _build_cache_path(tickers: list, start_date: str, end_date: str) -> Path:
    """Build a deterministic cache filename from the request parameters."""
    ticker_str = "_".join(sorted(tickers))
    filename = f"{ticker_str}_{start_date}_{end_date}.csv"
    return CACHE_DIR / filename


# ---------------------------------------------------------------------------
# Quick smoke test — run this file directly to verify setup
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Sample portfolio: two banks, one tech stock, three years of data
    TICKERS = ["JPM", "GS", "MSFT"]
    END = datetime.today().strftime("%Y-%m-%d")
    START = (datetime.today() - timedelta(days=3 * 365)).strftime("%Y-%m-%d")

    print(f"\nFetching prices for {TICKERS} from {START} to {END}...\n")
    prices = fetch_prices(TICKERS, START, END)
    print(prices.tail())

    print("\nComputing log returns...\n")
    returns = compute_daily_returns(prices)
    print(returns.tail())
    print(f"\nReturns shape: {returns.shape}")
    print("\ndata_loader.py working correctly.")