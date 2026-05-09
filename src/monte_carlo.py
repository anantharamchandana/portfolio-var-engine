"""
monte_carlo.py
--------------
Computes portfolio Value at Risk (VaR) and Conditional Value at Risk (CVaR)
using the Monte Carlo Simulation method.

Method summary:
    - Fits a statistical distribution to historical portfolio returns
    - Generates thousands of hypothetical future return scenarios
    - VaR and CVaR are read from the simulated distribution

How this differs from Historical Simulation:
    - Historical simulation uses ONLY what actually happened in the past
    - Monte Carlo GENERATES scenarios that could happen, based on the
      statistical properties of past returns (mean, volatility, correlations)
    - Monte Carlo can simulate scenarios more extreme than anything
      observed historically — which is both its strength and its risk

Strengths:
    - Can generate scenarios beyond historical worst case
    - Flexible — can model different distributions and assumptions
    - Scales well to large portfolios with many assets

Weaknesses (critical for model risk documentation):
    - Results are only as good as the distributional assumption
    - Standard implementation assumes normally distributed returns —
      real returns have fat tails (excess kurtosis) that a normal
      distribution underestimates
    - Results vary slightly each run due to randomness (controlled by seed)
    - Computationally heavier than historical simulation
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from dataclasses import dataclass
from scipy import stats

logger = logging.getLogger(__name__)

# Number of simulated scenarios — 10,000 is industry standard minimum
# Higher = more stable results, more compute time
DEFAULT_N_SIMULATIONS = 10_000

# Random seed for reproducibility
# In production, you would document and version-control this seed
DEFAULT_RANDOM_SEED = 42


@dataclass
class MonteCarloVaRResult:
    """
    Structured container for Monte Carlo simulation outputs.

    Attributes
    ----------
    confidence_level : float
        Confidence level used e.g. 0.95
    holding_period_days : int
        Number of days the VaR estimate covers
    n_simulations : int
        Number of scenarios simulated
    var_pct : float
        Value at Risk as percentage of portfolio (positive number)
    var_dollar : float
        Value at Risk in dollars
    cvar_pct : float
        Conditional Value at Risk as percentage (average tail loss)
    cvar_dollar : float
        Conditional Value at Risk in dollars
    simulated_mean : float
        Mean of simulated return distribution
    simulated_std : float
        Standard deviation of simulated return distribution
    simulated_returns : np.ndarray
        Full array of simulated scenario returns (used for plotting)
    fitted_mean : float
        Mean of the fitted normal distribution (from historical returns)
    fitted_std : float
        Std dev of the fitted normal distribution (from historical returns)
    normality_p_value : float
        p-value from Jarque-Bera normality test on historical returns.
        Low p-value (< 0.05) means returns are NOT normally distributed —
        a key model risk disclosure.
    """
    confidence_level: float
    holding_period_days: int
    n_simulations: int
    var_pct: float
    var_dollar: float
    cvar_pct: float
    cvar_dollar: float
    simulated_mean: float
    simulated_std: float
    simulated_returns: np.ndarray
    fitted_mean: float
    fitted_std: float
    normality_p_value: float

    def summary(self) -> str:
        normality_warning = (
            "  *** WARNING: Returns are NOT normally distributed ***\n"
            "  *** Normal distribution assumption underestimates tail risk ***\n"
            if self.normality_p_value < 0.05
            else "  Returns pass normality test at 5% significance level.\n"
        )
        return (
            f"\n{'='*55}\n"
            f"  MONTE CARLO SIMULATION — VALUE AT RISK RESULTS\n"
            f"{'='*55}\n"
            f"  Confidence Level     : {self.confidence_level:.0%}\n"
            f"  Holding Period       : {self.holding_period_days} day(s)\n"
            f"  Simulations Run      : {self.n_simulations:,}\n"
            f"{'─'*55}\n"
            f"  Fitted Mean (daily)  : {self.fitted_mean:.6f}\n"
            f"  Fitted Std  (daily)  : {self.fitted_std:.6f}\n"
            f"  Normality p-value    : {self.normality_p_value:.4f}\n"
            f"{normality_warning}"
            f"{'─'*55}\n"
            f"  VaR  (%)             : {self.var_pct:.4%}\n"
            f"  VaR  ($)             : ${self.var_dollar:>12,.2f}\n"
            f"{'─'*55}\n"
            f"  CVaR (%)             : {self.cvar_pct:.4%}\n"
            f"  CVaR ($)             : ${self.cvar_dollar:>12,.2f}\n"
            f"{'='*55}\n"
        )


def compute_monte_carlo_var(
    portfolio_returns: pd.Series,
    initial_value: float,
    confidence_level: float = 0.95,
    holding_period_days: int = 1,
    n_simulations: int = DEFAULT_N_SIMULATIONS,
    random_seed: int = DEFAULT_RANDOM_SEED
) -> MonteCarloVaRResult:
    """
    Compute VaR and CVaR using Monte Carlo simulation.

    Process:
    1. Fit a normal distribution to historical portfolio returns
       (estimate mean and standard deviation)
    2. Generate n_simulations random return scenarios from that distribution
    3. Sort simulated returns, read VaR at confidence threshold
    4. Compute CVaR as average of tail scenarios

    Parameters
    ----------
    portfolio_returns : pd.Series
        Daily weighted portfolio log returns from portfolio.py.
    initial_value : float
        Total portfolio value in dollars.
    confidence_level : float
        VaR confidence level. Standard: 0.90, 0.95, or 0.99.
    holding_period_days : int
        Holding period. Multi-day returns scaled by sqrt(holding_period_days).
    n_simulations : int
        Number of Monte Carlo scenarios to generate.
        10,000 is the industry standard minimum.
        100,000 gives more stable tail estimates.
    random_seed : int
        Seed for numpy random number generator.
        Always set this for reproducibility — results must be
        replicable for model validation purposes.

    Returns
    -------
    MonteCarloVaRResult
        Structured result object with VaR, CVaR, and simulation metadata.

    Raises
    ------
    ValueError
        If inputs are invalid (empty returns, bad confidence level, etc.)
    """
    _validate_inputs(portfolio_returns, confidence_level, holding_period_days, n_simulations)

    # --- Step 1: Fit normal distribution to historical returns ---
    # Maximum likelihood estimation: mean and std of the return series
    fitted_mean = portfolio_returns.mean()
    fitted_std = portfolio_returns.std()

    logger.info(
        f"Fitting normal distribution: mean={fitted_mean:.6f}, "
        f"std={fitted_std:.6f}"
    )

    # --- Step 2: Test normality assumption ---
    # Jarque-Bera test checks whether the return distribution has the
    # skewness and kurtosis consistent with a normal distribution.
    # Low p-value = evidence against normality = our assumption is violated.
    jb_stat, jb_p_value = stats.jarque_bera(portfolio_returns)
    if jb_p_value < 0.05:
        logger.warning(
            f"Jarque-Bera normality test: p-value={jb_p_value:.4f} < 0.05. "
            "Returns are NOT normally distributed. "
            "Monte Carlo VaR based on normal distribution will UNDERESTIMATE "
            "tail risk due to fat tails in actual return distribution. "
            "This is a known model limitation — document it."
        )
    else:
        logger.info(
            f"Jarque-Bera normality test: p-value={jb_p_value:.4f}. "
            "Returns consistent with normal distribution at 5% level."
        )

    # --- Step 3: Generate simulated scenarios ---
    # Scale mean and std to the holding period
    # Mean scales linearly: holding_period_days * daily_mean
    # Std scales by sqrt: sqrt(holding_period_days) * daily_std
    # This is the same square root of time rule from historical simulation
    scaled_mean = fitted_mean * holding_period_days
    scaled_std = fitted_std * np.sqrt(holding_period_days)

    rng = np.random.default_rng(seed=random_seed)
    simulated_returns = rng.normal(
        loc=scaled_mean,
        scale=scaled_std,
        size=n_simulations
    )

    logger.info(f"Generated {n_simulations:,} simulated scenarios.")

    # --- Step 4: Compute VaR from simulated distribution ---
    loss_threshold = 1 - confidence_level  # e.g. 0.05 for 95%
    var_pct = -np.percentile(simulated_returns, loss_threshold * 100)
    var_dollar = var_pct * initial_value

    # --- Step 5: Compute CVaR (Expected Shortfall) ---
    var_threshold_return = np.percentile(simulated_returns, loss_threshold * 100)
    tail_scenarios = simulated_returns[simulated_returns <= var_threshold_return]
    cvar_pct = -tail_scenarios.mean()
    cvar_dollar = cvar_pct * initial_value

    result = MonteCarloVaRResult(
        confidence_level=confidence_level,
        holding_period_days=holding_period_days,
        n_simulations=n_simulations,
        var_pct=var_pct,
        var_dollar=var_dollar,
        cvar_pct=cvar_pct,
        cvar_dollar=cvar_dollar,
        simulated_mean=float(np.mean(simulated_returns)),
        simulated_std=float(np.std(simulated_returns)),
        simulated_returns=simulated_returns,
        fitted_mean=fitted_mean,
        fitted_std=fitted_std,
        normality_p_value=float(jb_p_value)
    )

    logger.info(result.summary())
    return result


def compare_methods(
    historical_result,
    monte_carlo_result: MonteCarloVaRResult
) -> pd.DataFrame:
    """
    Side-by-side comparison of Historical Simulation vs Monte Carlo results.

    This comparison table is one of the most important outputs of the
    entire project. The gap between the two methods reveals model risk:
    - If Monte Carlo VaR < Historical VaR: normal distribution is
      underestimating tail risk (fat tails in real data)
    - If Monte Carlo VaR > Historical VaR: the historical period may
      not have captured extreme scenarios

    Parameters
    ----------
    historical_result : HistoricalVaRResult
        Output from historical_simulation.compute_historical_var()
    monte_carlo_result : MonteCarloVaRResult
        Output from compute_monte_carlo_var()

    Returns
    -------
    pd.DataFrame
        Comparison table with both methods side by side.
    """
    comparison = {
        "Metric": [
            "Confidence Level",
            "Holding Period (days)",
            "VaR (%)",
            "VaR ($)",
            "CVaR (%)",
            "CVaR ($)",
            "Method"
        ],
        "Historical Simulation": [
            f"{historical_result.confidence_level:.0%}",
            f"{historical_result.holding_period_days}",
            f"{historical_result.var_pct:.4%}",
            f"${historical_result.var_dollar:,.2f}",
            f"{historical_result.cvar_pct:.4%}",
            f"${historical_result.cvar_dollar:,.2f}",
            "Non-parametric"
        ],
        "Monte Carlo": [
            f"{monte_carlo_result.confidence_level:.0%}",
            f"{monte_carlo_result.holding_period_days}",
            f"{monte_carlo_result.var_pct:.4%}",
            f"${monte_carlo_result.var_dollar:,.2f}",
            f"{monte_carlo_result.cvar_pct:.4%}",
            f"${monte_carlo_result.cvar_dollar:,.2f}",
            f"Normal distribution, {monte_carlo_result.n_simulations:,} simulations"
        ]
    }

    df = pd.DataFrame(comparison).set_index("Metric")

    # Compute dollar difference to surface model risk
    dollar_diff = monte_carlo_result.var_dollar - historical_result.var_dollar
    direction = "MC higher" if dollar_diff > 0 else "MC lower"
    logger.info(
        f"VaR difference (MC - Historical): ${dollar_diff:,.2f} ({direction}). "
        f"This gap represents model risk from distributional assumption."
    )

    return df


def plot_simulated_distribution(
    monte_carlo_result: MonteCarloVaRResult,
    historical_result=None,
    save_path: str = None
) -> None:
    """
    Plot the simulated return distribution with VaR marked.
    Optionally overlays the historical VaR for direct comparison.

    Parameters
    ----------
    monte_carlo_result : MonteCarloVaRResult
        Monte Carlo output from compute_monte_carlo_var().
    historical_result : HistoricalVaRResult, optional
        If provided, overlays historical VaR line for comparison.
    save_path : str, optional
        If provided, saves figure to this path.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Simulated return distribution
    ax.hist(
        monte_carlo_result.simulated_returns,
        bins=100,
        color="#1a9641",
        alpha=0.5,
        density=True,
        label=f"Monte Carlo ({monte_carlo_result.n_simulations:,} simulations)"
    )

    # Overlay fitted normal curve
    x = np.linspace(
        monte_carlo_result.simulated_returns.min(),
        monte_carlo_result.simulated_returns.max(),
        500
    )
    fitted_curve = stats.norm.pdf(
        x,
        loc=monte_carlo_result.fitted_mean,
        scale=monte_carlo_result.fitted_std
    )
    ax.plot(x, fitted_curve, color="#1a9641", linewidth=2,
            linestyle="-", label="Fitted Normal Distribution")

    # Monte Carlo VaR line
    ax.axvline(
        x=-monte_carlo_result.var_pct,
        color="#d7191c",
        linewidth=2,
        linestyle="--",
        label=(
            f"MC {monte_carlo_result.confidence_level:.0%} VaR: "
            f"{monte_carlo_result.var_pct:.2%} "
            f"(${monte_carlo_result.var_dollar:,.0f})"
        )
    )

    # Monte Carlo CVaR line
    ax.axvline(
        x=-monte_carlo_result.cvar_pct,
        color="#fdae61",
        linewidth=2,
        linestyle=":",
        label=(
            f"MC CVaR: {monte_carlo_result.cvar_pct:.2%} "
            f"(${monte_carlo_result.cvar_dollar:,.0f})"
        )
    )

    # Overlay historical VaR if provided — key comparison
    if historical_result is not None:
        ax.axvline(
            x=-historical_result.var_pct,
            color="#2c7bb6",
            linewidth=2,
            linestyle="-.",
            label=(
                f"Historical {historical_result.confidence_level:.0%} VaR: "
                f"{historical_result.var_pct:.2%} "
                f"(${historical_result.var_dollar:,.0f})"
            )
        )

    ax.set_xlabel("Portfolio Return", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(
        f"Monte Carlo Simulated Return Distribution\n"
        f"({monte_carlo_result.n_simulations:,} scenarios | "
        f"{monte_carlo_result.confidence_level:.0%} confidence | "
        f"{monte_carlo_result.holding_period_days}-day holding period)",
        fontsize=13
    )
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=1))
    ax.legend(fontsize=9)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Monte Carlo distribution plot saved to: {save_path}")
    else:
        plt.show()

    plt.close()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_inputs(
    portfolio_returns: pd.Series,
    confidence_level: float,
    holding_period_days: int,
    n_simulations: int
) -> None:
    if portfolio_returns.empty:
        raise ValueError("portfolio_returns is empty.")

    if not (0 < confidence_level < 1):
        raise ValueError(
            f"confidence_level must be between 0 and 1. Got: {confidence_level}."
        )

    if holding_period_days < 1:
        raise ValueError(
            f"holding_period_days must be >= 1. Got: {holding_period_days}."
        )

    if n_simulations < 1_000:
        raise ValueError(
            f"n_simulations must be at least 1,000 for reliable VaR estimates. "
            f"Got: {n_simulations}. Recommended minimum: 10,000."
        )

    if n_simulations < 10_000:
        logger.warning(
            f"n_simulations={n_simulations:,} is below the recommended "
            "minimum of 10,000. Tail estimates may be unstable."
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
    from historical_simulation import compute_historical_var

    TICKERS = ["JPM", "GS", "MSFT"]
    END = datetime.today().strftime("%Y-%m-%d")
    START = (datetime.today() - timedelta(days=3 * 365)).strftime("%Y-%m-%d")

    print("\n--- Building portfolio ---")
    prices = fetch_prices(TICKERS, START, END)
    returns = compute_daily_returns(prices)
    portfolio = Portfolio.equal_weight(TICKERS, initial_value=1_000_000)
    port_returns = compute_portfolio_returns(returns, portfolio)

    print("\n--- Historical VaR (95%) ---")
    hist_result = compute_historical_var(
        port_returns,
        initial_value=portfolio.initial_value,
        confidence_level=0.95
    )

    print("\n--- Monte Carlo VaR (95%, 10,000 simulations) ---")
    mc_result = compute_monte_carlo_var(
        port_returns,
        initial_value=portfolio.initial_value,
        confidence_level=0.95,
        n_simulations=10_000
    )
    print(mc_result.summary())

    print("\n--- Monte Carlo VaR (95%, 100,000 simulations) ---")
    mc_result_large = compute_monte_carlo_var(
        port_returns,
        initial_value=portfolio.initial_value,
        confidence_level=0.95,
        n_simulations=100_000
    )
    print(mc_result_large.summary())

    print("\n--- Method Comparison Table ---")
    comparison = compare_methods(hist_result, mc_result)
    print(comparison.to_string())

    print("\n--- Plotting simulated distribution with historical VaR overlay ---")
    plot_simulated_distribution(mc_result, historical_result=hist_result)

    print("\nmonte_carlo.py working correctly.")