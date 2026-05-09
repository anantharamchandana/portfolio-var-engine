"""
portfolio.py
------------
Handles portfolio construction and weighted return computation.

Responsibilities:
- Accept a set of tickers and capital weights
- Validate that weights are properly formed (sum to 1.0, no negatives)
- Compute weighted portfolio returns from individual asset log returns
- Provide portfolio summary statistics used by both simulation modules

Design principles:
- Weights are explicit — never assumed to be equal unless caller says so
- Fails loudly if weights and tickers don't align
- All outputs are labeled (no raw numpy arrays passed between modules)
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Portfolio:
    """
    Represents a weighted portfolio of assets.

    Attributes
    ----------
    tickers : list[str]
        Ordered list of ticker symbols in the portfolio.
    weights : list[float]
        Capital allocation weights corresponding to each ticker.
        Must sum to 1.0. Must all be non-negative (long-only portfolio).
    initial_value : float
        Total portfolio value in dollars. Used to convert percentage
        Value at Risk into dollar Value at Risk. Default is $1,000,000.

    Example
    -------
    >>> p = Portfolio(
    ...     tickers=["JPM", "GS", "MSFT"],
    ...     weights=[0.5, 0.3, 0.2],
    ...     initial_value=1_000_000
    ... )
    """
    tickers: list[str]
    weights: list[float]
    initial_value: float = 1_000_000.0

    # Derived attribute — built after validation
    weights_array: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        """Run validation immediately on construction. Fail loudly if invalid."""
        self._validate()
        self.weights_array = np.array(self.weights)
        logger.info(
            f"Portfolio constructed: {len(self.tickers)} assets, "
            f"initial value ${self.initial_value:,.0f}"
        )
        for ticker, weight in zip(self.tickers, self.weights):
            logger.info(f"  {ticker}: {weight:.1%}")

    def _validate(self) -> None:
        """
        Validate portfolio inputs before any computation.

        Checks:
        - Tickers and weights have the same length
        - No duplicate tickers
        - All weights are non-negative (short selling not supported here)
        - Weights sum to 1.0 within floating point tolerance
        - Initial value is positive
        """
        if not self.tickers:
            raise ValueError("tickers list cannot be empty.")

        if len(self.tickers) != len(self.weights):
            raise ValueError(
                f"tickers length ({len(self.tickers)}) must match "
                f"weights length ({len(self.weights)})."
            )

        if len(self.tickers) != len(set(self.tickers)):
            duplicates = [t for t in self.tickers if self.tickers.count(t) > 1]
            raise ValueError(f"Duplicate tickers found: {list(set(duplicates))}")

        if any(w < 0 for w in self.weights):
            negatives = [(t, w) for t, w in zip(self.tickers, self.weights) if w < 0]
            raise ValueError(
                f"Negative weights are not supported (long-only portfolio). "
                f"Negative weights found: {negatives}"
            )

        weight_sum = sum(self.weights)
        if not np.isclose(weight_sum, 1.0, atol=1e-6):
            raise ValueError(
                f"Weights must sum to 1.0. Current sum: {weight_sum:.6f}. "
                f"Difference: {abs(weight_sum - 1.0):.6f}. "
                "Adjust your weights so they sum to exactly 1.0."
            )

        if self.initial_value <= 0:
            raise ValueError(
                f"initial_value must be positive. Got: {self.initial_value}"
            )

    @classmethod
    def equal_weight(cls, tickers: list[str], initial_value: float = 1_000_000.0):
        """
        Convenience constructor for an equally weighted portfolio.

        Equal weighting means each asset gets 1/N of total capital,
        where N is the number of assets. This is a common baseline
        in portfolio risk analysis.

        Parameters
        ----------
        tickers : list[str]
            List of ticker symbols.
        initial_value : float
            Total portfolio value in dollars.

        Returns
        -------
        Portfolio
            A Portfolio instance with equal weights across all tickers.

        Example
        -------
        >>> p = Portfolio.equal_weight(["JPM", "GS", "MSFT"])
        >>> p.weights
        [0.3333, 0.3333, 0.3333]
        """
        n = len(tickers)
        if n == 0:
            raise ValueError("Cannot construct equal-weight portfolio with no tickers.")

        equal_weights = [round(1.0 / n, 10) for _ in tickers]
        # Fix floating point: force last weight to make sum exactly 1.0
        equal_weights[-1] = 1.0 - sum(equal_weights[:-1])

        logger.info(f"Constructing equal-weight portfolio across {n} assets.")
        return cls(tickers=tickers, weights=equal_weights, initial_value=initial_value)


def compute_portfolio_returns(
    returns: pd.DataFrame,
    portfolio: Portfolio
) -> pd.Series:
    """
    Compute daily weighted portfolio returns from individual asset returns.

    This is the core operation of this module. Each day's portfolio return
    is the dot product of that day's individual asset returns and the
    portfolio weights vector.

    Mathematically:
        portfolio_return(t) = sum(weight_i * return_i(t)) for all assets i

    Parameters
    ----------
    returns : pd.DataFrame
        Daily log returns from data_loader.compute_daily_returns().
        Index: DatetimeIndex. Columns: ticker symbols.
    portfolio : Portfolio
        Portfolio instance containing tickers and weights.

    Returns
    -------
    pd.Series
        Daily weighted portfolio log returns.
        Index: DatetimeIndex (same as input returns).
        Name: 'portfolio_return'

    Raises
    ------
    ValueError
        If any portfolio ticker is missing from the returns DataFrame,
        or if the returns DataFrame is empty.
    """
    if returns.empty:
        raise ValueError(
            "returns DataFrame is empty. "
            "Ensure data_loader.fetch_prices() ran successfully."
        )

    missing = [t for t in portfolio.tickers if t not in returns.columns]
    if missing:
        raise ValueError(
            f"Portfolio tickers not found in returns data: {missing}. "
            f"Available tickers in returns: {list(returns.columns)}"
        )

    # Align returns columns to portfolio ticker order
    # This ensures the dot product matches weights to correct tickers
    aligned_returns = returns[portfolio.tickers]

    # Dot product: (n_days x n_assets) @ (n_assets,) = (n_days,)
    portfolio_returns = aligned_returns.dot(portfolio.weights_array)
    portfolio_returns.name = "portfolio_return"

    logger.info(
        f"Computed portfolio returns: {len(portfolio_returns)} trading days."
    )

    return portfolio_returns


def compute_portfolio_statistics(portfolio_returns: pd.Series) -> dict:
    """
    Compute descriptive statistics on portfolio return series.

    These statistics are used by both simulation modules and the
    final report. Computing them once here avoids duplication.

    Parameters
    ----------
    portfolio_returns : pd.Series
        Daily weighted portfolio returns from compute_portfolio_returns().

    Returns
    -------
    dict with keys:
        mean_daily_return   : float — average daily return
        std_daily_return    : float — daily return standard deviation (volatility)
        annualized_return   : float — mean daily return scaled to 252 trading days
        annualized_volatility: float — daily std scaled to 252 trading days
        skewness            : float — asymmetry of return distribution
                                      negative = more extreme losses than gains
        kurtosis            : float — tail heaviness of return distribution
                                      positive excess kurtosis = fat tails (common in finance)
        sharpe_ratio        : float — annualized return per unit of annualized volatility
                                      assumes risk-free rate of 0 for simplicity
        total_trading_days  : int   — number of observations in the series
        date_range          : tuple — (start_date, end_date) of the return series
    """
    if portfolio_returns.empty:
        raise ValueError("portfolio_returns is empty — cannot compute statistics.")

    mean = portfolio_returns.mean()
    std = portfolio_returns.std()
    trading_days_per_year = 252

    stats = {
        "mean_daily_return": mean,
        "std_daily_return": std,
        "annualized_return": mean * trading_days_per_year,
        "annualized_volatility": std * np.sqrt(trading_days_per_year),
        "skewness": float(portfolio_returns.skew()),
        "kurtosis": float(portfolio_returns.kurtosis()),  # excess kurtosis (normal = 0)
        "sharpe_ratio": (mean * trading_days_per_year) / (std * np.sqrt(trading_days_per_year))
        if std > 0 else 0.0,
        "total_trading_days": len(portfolio_returns),
        "date_range": (
            portfolio_returns.index.min().strftime("%Y-%m-%d"),
            portfolio_returns.index.max().strftime("%Y-%m-%d")
        )
    }

    logger.info("Portfolio statistics computed:")
    logger.info(f"  Annualized return    : {stats['annualized_return']:.2%}")
    logger.info(f"  Annualized volatility: {stats['annualized_volatility']:.2%}")
    logger.info(f"  Sharpe ratio         : {stats['sharpe_ratio']:.2f}")
    logger.info(f"  Skewness             : {stats['skewness']:.4f}")
    logger.info(f"  Excess kurtosis      : {stats['kurtosis']:.4f}")

    return stats


# ---------------------------------------------------------------------------
# Smoke test — run directly to verify this module works end to end
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Allow running from project root
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    from data_loader import fetch_prices, compute_daily_returns
    from datetime import datetime, timedelta

    TICKERS = ["JPM", "GS", "MSFT"]
    END = datetime.today().strftime("%Y-%m-%d")
    START = (datetime.today() - timedelta(days=3 * 365)).strftime("%Y-%m-%d")

    print("\n--- Fetching price data ---")
    prices = fetch_prices(TICKERS, START, END)

    print("\n--- Computing individual asset returns ---")
    returns = compute_daily_returns(prices)

    print("\n--- Building equal-weight portfolio ---")
    portfolio = Portfolio.equal_weight(TICKERS, initial_value=1_000_000)

    print("\n--- Computing portfolio returns ---")
    port_returns = compute_portfolio_returns(returns, portfolio)
    print(port_returns.tail())

    print("\n--- Portfolio statistics ---")
    stats = compute_portfolio_statistics(port_returns)
    for key, val in stats.items():
        print(f"  {key}: {val}")

    print("\nportfolio.py working correctly.")