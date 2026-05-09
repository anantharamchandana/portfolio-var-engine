"""
historical_simulation.py
------------------------
Computes portfolio Value at Risk (VaR) and Conditional Value at Risk (CVaR)
using the Historical Simulation method.

Method summary:
    - Uses actual historical portfolio returns as the scenario set
    - No distributional assumptions (non-parametric)
    - VaR = loss at the (1 - confidence_level) percentile of return distribution
    - CVaR = average loss in the tail beyond VaR (also called Expected Shortfall)

Strengths of this method:
    - Captures actual historical fat tails and skewness
    - No assumption that returns are normally distributed
    - Intuitive and auditable — every scenario is a real observed day

Weaknesses (important for interviews and model risk documentation):
    - Entirely backward-looking — if a crisis has no historical precedent,
      it won't appear in the scenario set
    - Equally weights all historical days — a return from 5 years ago
      counts the same as last week
    - Requires sufficient history — unreliable with less than 1 year of data
    - Cannot extrapolate beyond the worst observed historical day
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Standard confidence levels used in practice
# 95% is common for internal risk management
# 99% is required under Basel regulatory frameworks
SUPPORTED_CONFIDENCE_LEVELS = [0.90, 0.95, 0.99]


@dataclass
class HistoricalVaRResult:
    """
    Structured container for historical simulation outputs.

    Keeping results in a dataclass rather than a raw dict ensures
    downstream modules (report, dashboard) access fields by name,
    not by fragile string keys.

    Attributes
    ----------
    confidence_level : float
        The confidence level used e.g. 0.95 for 95% VaR
    holding_period_days : int
        Number of days the VaR estimate covers (1-day is standard)
    var_pct : float
        Value at Risk as a percentage of portfolio value (positive number)
        e.g. 0.018 means "expect to lose at least 1.8% on a bad day"
    var_dollar : float
        Value at Risk in dollar terms = var_pct * initial_portfolio_value
    cvar_pct : float
        Conditional Value at Risk (Expected Shortfall) as percentage.
        Average loss given that loss exceeds VaR threshold.
    cvar_dollar : float
        Conditional Value at Risk in dollar terms
    worst_day_pct : float
        Single worst historical day return (as positive loss percentage)
    best_day_pct : float
        Single best historical day return
    tail_returns : pd.Series
        All returns in the loss tail beyond the VaR threshold.
        Used for CVaR computation and tail analysis.
    n_observations : int
        Number of historical return observations used
    n_tail_observations : int
        Number of observations in the loss tail
    """
    confidence_level: float
    holding_period_days: int
    var_pct: float
    var_dollar: float
    cvar_pct: float
    cvar_dollar: float
    worst_day_pct: float
    best_day_pct: float
    tail_returns: pd.Series
    n_observations: int
    n_tail_observations: int

    def summary(self) -> str:
        """Return a formatted summary string for logging and reporting."""
        return (
            f"\n{'='*55}\n"
            f"  HISTORICAL SIMULATION — VALUE AT RISK RESULTS\n"
            f"{'='*55}\n"
            f"  Confidence Level     : {self.confidence_level:.0%}\n"
            f"  Holding Period       : {self.holding_period_days} day(s)\n"
            f"  Observations Used    : {self.n_observations} trading days\n"
            f"{'─'*55}\n"
            f"  VaR  (%)             : {self.var_pct:.4%}\n"
            f"  VaR  ($)             : ${self.var_dollar:>12,.2f}\n"
            f"{'─'*55}\n"
            f"  CVaR (%)             : {self.cvar_pct:.4%}\n"
            f"  CVaR ($)             : ${self.cvar_dollar:>12,.2f}\n"
            f"{'─'*55}\n"
            f"  Worst Historical Day : {self.worst_day_pct:.4%}\n"
            f"  Best Historical Day  : {self.best_day_pct:.4%}\n"
            f"  Tail Observations    : {self.n_tail_observations}\n"
            f"{'='*55}\n"
        )


def compute_historical_var(
    portfolio_returns: pd.Series,
    initial_value: float,
    confidence_level: float = 0.95,
    holding_period_days: int = 1
) -> HistoricalVaRResult:
    """
    Compute Value at Risk and Conditional Value at Risk using
    the Historical Simulation method.

    Parameters
    ----------
    portfolio_returns : pd.Series
        Daily weighted portfolio log returns from portfolio.py.
    initial_value : float
        Total portfolio value in dollars (used for dollar VaR conversion).
    confidence_level : float
        Confidence level for VaR. Standard values: 0.90, 0.95, 0.99.
        A 0.95 VaR means: "we are 95% confident losses won't exceed this."
    holding_period_days : int
        Number of days to scale VaR to. Default is 1 (daily VaR).
        Multi-day VaR is approximated by scaling by sqrt(holding_period_days).
        This is the square root of time rule — standard in risk management.

    Returns
    -------
    HistoricalVaRResult
        Structured result object with VaR, CVaR, and tail statistics.

    Raises
    ------
    ValueError
        If confidence_level is not in supported range, returns are empty,
        or holding_period_days is not positive.

    Notes on the square root of time rule
    --------------------------------------
    Scaling 1-day VaR to n-day VaR by multiplying by sqrt(n) assumes:
    - Returns are independent across days (no autocorrelation)
    - Returns are identically distributed each day
    These assumptions are imperfect for real markets but are the
    regulatory standard under Basel frameworks.
    """
    _validate_var_inputs(portfolio_returns, confidence_level, holding_period_days)

    n = len(portfolio_returns)
    loss_threshold = 1 - confidence_level  # e.g. 0.05 for 95% confidence

    # --- Core VaR computation ---
    # np.percentile(returns, 5) gives the 5th percentile return
    # This is a negative number (a loss) — we negate it to express as positive loss
    var_pct_1day = -np.percentile(portfolio_returns, loss_threshold * 100)

    # Scale to holding period using square root of time rule
    var_pct = var_pct_1day * np.sqrt(holding_period_days)
    var_dollar = var_pct * initial_value

    # --- Conditional VaR (Expected Shortfall) ---
    # Identify all returns worse than (below) the VaR threshold
    var_threshold_return = np.percentile(portfolio_returns, loss_threshold * 100)
    tail_returns = portfolio_returns[portfolio_returns <= var_threshold_return]

    # CVaR = average of all tail losses, negated to express as positive loss
    cvar_pct_1day = -tail_returns.mean()
    cvar_pct = cvar_pct_1day * np.sqrt(holding_period_days)
    cvar_dollar = cvar_pct * initial_value

    # --- Tail statistics ---
    worst_day = float(portfolio_returns.min())
    best_day = float(portfolio_returns.max())

    result = HistoricalVaRResult(
        confidence_level=confidence_level,
        holding_period_days=holding_period_days,
        var_pct=var_pct,
        var_dollar=var_dollar,
        cvar_pct=cvar_pct,
        cvar_dollar=cvar_dollar,
        worst_day_pct=worst_day,
        best_day_pct=best_day,
        tail_returns=tail_returns,
        n_observations=n,
        n_tail_observations=len(tail_returns)
    )

    logger.info(result.summary())
    return result


def compute_var_term_structure(
    portfolio_returns: pd.Series,
    initial_value: float,
    confidence_levels: list[float] = None,
    holding_periods: list[int] = None
) -> pd.DataFrame:
    """
    Compute VaR across multiple confidence levels and holding periods.

    In practice, risk desks report VaR at multiple horizons simultaneously.
    This function builds that full term structure in one call.

    Parameters
    ----------
    portfolio_returns : pd.Series
        Daily weighted portfolio log returns.
    initial_value : float
        Total portfolio value in dollars.
    confidence_levels : list[float]
        List of confidence levels. Defaults to [0.90, 0.95, 0.99].
    holding_periods : list[int]
        List of holding periods in days. Defaults to [1, 5, 10] representing
        1-day, 1-week, and 2-week VaR.

    Returns
    -------
    pd.DataFrame
        VaR term structure table.
        Rows: confidence levels. Columns: holding periods.
        Values: dollar VaR.
    """
    if confidence_levels is None:
        confidence_levels = [0.90, 0.95, 0.99]
    if holding_periods is None:
        holding_periods = [1, 5, 10]

    rows = []
    for cl in confidence_levels:
        row = {}
        for hp in holding_periods:
            result = compute_historical_var(
                portfolio_returns, initial_value,
                confidence_level=cl,
                holding_period_days=hp
            )
            row[f"{hp}d VaR ($)"] = result.var_dollar
        rows.append(row)

    df = pd.DataFrame(rows, index=[f"{cl:.0%}" for cl in confidence_levels])
    df.index.name = "Confidence Level"
    return df


def plot_return_distribution(
    portfolio_returns: pd.Series,
    result: HistoricalVaRResult,
    save_path: str = None
) -> None:
    """
    Plot the historical return distribution with VaR and CVaR marked.

    This visualization is critical for understanding what VaR represents
    and is expected in any professional risk report.

    Parameters
    ----------
    portfolio_returns : pd.Series
        Daily portfolio returns.
    result : HistoricalVaRResult
        Output from compute_historical_var() — used to draw threshold lines.
    save_path : str, optional
        If provided, saves the figure to this path instead of displaying it.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Return distribution
    sns.histplot(
        portfolio_returns,
        bins=80,
        kde=True,
        color="#2c7bb6",
        alpha=0.6,
        ax=ax,
        label="Daily Returns"
    )

    # VaR threshold line
    ax.axvline(
        x=-result.var_pct,
        color="#d7191c",
        linewidth=2,
        linestyle="--",
        label=f"{result.confidence_level:.0%} VaR: {result.var_pct:.2%} (${result.var_dollar:,.0f})"
    )

    # CVaR threshold line
    ax.axvline(
        x=-result.cvar_pct,
        color="#fdae61",
        linewidth=2,
        linestyle=":",
        label=f"CVaR (Expected Shortfall): {result.cvar_pct:.2%} (${result.cvar_dollar:,.0f})"
    )

    # Shade the tail region
    tail_x = np.linspace(portfolio_returns.min(), -result.var_pct, 100)
    ax.fill_betweenx(
        [0, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 50],
        portfolio_returns.min(), -result.var_pct,
        alpha=0.15, color="#d7191c", label="Loss Tail"
    )

    ax.set_xlabel("Daily Log Return", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(
        f"Portfolio Return Distribution — Historical Simulation VaR\n"
        f"({result.n_observations} trading days | "
        f"{result.confidence_level:.0%} confidence | "
        f"{result.holding_period_days}-day holding period)",
        fontsize=13
    )
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=1))
    ax.legend(fontsize=10)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Return distribution plot saved to: {save_path}")
    else:
        plt.show()

    plt.close()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_var_inputs(
    portfolio_returns: pd.Series,
    confidence_level: float,
    holding_period_days: int
) -> None:
    """Validate inputs before VaR computation. Fail loudly."""
    if portfolio_returns.empty:
        raise ValueError(
            "portfolio_returns is empty. "
            "Run portfolio.compute_portfolio_returns() first."
        )

    if not (0 < confidence_level < 1):
        raise ValueError(
            f"confidence_level must be between 0 and 1. Got: {confidence_level}. "
            f"Use 0.95 for 95% VaR, not 95."
        )

    if confidence_level not in SUPPORTED_CONFIDENCE_LEVELS:
        logger.warning(
            f"confidence_level {confidence_level} is non-standard. "
            f"Typical values: {SUPPORTED_CONFIDENCE_LEVELS}. "
            "Proceeding but verify this is intentional."
        )

    if holding_period_days < 1:
        raise ValueError(
            f"holding_period_days must be at least 1. Got: {holding_period_days}."
        )

    min_required_obs = 252
    if len(portfolio_returns) < min_required_obs:
        logger.warning(
            f"Only {len(portfolio_returns)} observations available. "
            f"Historical VaR is unreliable with fewer than {min_required_obs} "
            "trading days. Results should be interpreted cautiously."
        )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path
    from datetime import datetime, timedelta

    sys.path.insert(0, str(Path(__file__).resolve().parent))

    from data_loader import fetch_prices, compute_daily_returns
    from portfolio import Portfolio, compute_portfolio_returns

    TICKERS = ["JPM", "GS", "MSFT"]
    END = datetime.today().strftime("%Y-%m-%d")
    START = (datetime.today() - timedelta(days=3 * 365)).strftime("%Y-%m-%d")

    print("\n--- Building portfolio ---")
    prices = fetch_prices(TICKERS, START, END)
    returns = compute_daily_returns(prices)
    portfolio = Portfolio.equal_weight(TICKERS, initial_value=1_000_000)
    port_returns = compute_portfolio_returns(returns, portfolio)

    print("\n--- Computing Historical VaR at 95% confidence ---")
    result_95 = compute_historical_var(
        port_returns,
        initial_value=portfolio.initial_value,
        confidence_level=0.95,
        holding_period_days=1
    )
    print(result_95.summary())

    print("\n--- Computing Historical VaR at 99% confidence ---")
    result_99 = compute_historical_var(
        port_returns,
        initial_value=portfolio.initial_value,
        confidence_level=0.99,
        holding_period_days=1
    )
    print(result_99.summary())

    print("\n--- VaR Term Structure (multiple confidence levels x holding periods) ---")
    term_structure = compute_var_term_structure(
        port_returns,
        initial_value=portfolio.initial_value
    )
    print(term_structure.to_string())

    print("\n--- Plotting return distribution ---")
    plot_return_distribution(port_returns, result_95)

    print("\nhistorical_simulation.py working correctly.")