"""
stress_testing.py
-----------------
Applies historical crisis scenarios to the current portfolio to measure
potential losses under extreme but plausible market conditions.

Why stress testing exists alongside VaR:
    VaR is calibrated on normal market conditions and recent history.
    It systematically underestimates tail risk during crises because:
    - Crises introduce correlations that don't exist in normal markets
      (assets that normally move independently all drop together)
    - Volatility spikes to multiples of its normal level
    - Liquidity evaporates, making losses worse than models predict

    Stress testing bypasses distributional assumptions entirely.
    It asks: "what if the actual market moves from crisis period X
    happened to our portfolio today?"

Regulatory context:
    Basel III (the international banking regulatory framework post-2008)
    requires banks to conduct regular stress tests using defined scenarios.
    The Federal Reserve's annual DFAST (Dodd-Frank Act Stress Test) and
    CCAR (Comprehensive Capital Analysis and Review) programs mandate
    specific stress scenarios for large banks. Goldman Sachs undergoes
    these tests annually. The systems you would build in risk tech at
    Goldman feed directly into these regulatory stress test frameworks.

Scenarios included:
    1. Global Financial Crisis (GFC) 2008
       Worst period: September - November 2008
       Trigger: Lehman Brothers bankruptcy September 15, 2008
       S&P 500 peak-to-trough: -56.8%

    2. COVID-19 Market Crash 2020
       Worst period: February 20 - March 23, 2020
       Fastest bear market in history: -34% in 33 days
       Unique feature: simultaneous shock to all asset classes

    3. Dot-Com Crash 2000-2002
       Worst period: March 2000 - October 2002
       Peak-to-trough: -49.1% for S&P 500
       Relevant because tech-heavy portfolios were disproportionately hit

    4. Custom scenario
       User-defined percentage shocks per asset
       Used for forward-looking hypothetical stress testing
       e.g. "what if rates rise 200 basis points and equities drop 20%"
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Crisis scenario definitions
# Each scenario captures the worst contiguous period of that crisis
# using actual market dates so we can pull real return data
# ---------------------------------------------------------------------------

CRISIS_SCENARIOS = {
    "GFC_2008": {
        "name": "Global Financial Crisis (2008)",
        "start": "2008-09-01",
        "end": "2008-11-30",
        "description": (
            "Lehman Brothers filed for bankruptcy on September 15, 2008, "
            "triggering a global credit freeze. This 3-month window captures "
            "the acute phase of the crisis. S&P 500 fell approximately 30% "
            "during this period. Correlations between asset classes spiked "
            "toward 1.0 — diversification provided almost no protection."
        )
    },
    "COVID_2020": {
        "name": "COVID-19 Market Crash (2020)",
        "start": "2020-02-20",
        "end": "2020-03-23",
        "description": (
            "The fastest bear market in history. S&P 500 fell 34% in 33 days "
            "as global lockdowns were announced. Unique because the shock hit "
            "all sectors simultaneously with no warning. VIX (volatility index) "
            "reached 82.69 on March 16 — its highest level ever recorded."
        )
    },
    "DOT_COM_2000": {
        "name": "Dot-Com Crash (2000-2002)",
        "start": "2000-03-01",
        "end": "2002-10-31",
        "description": (
            "Technology sector collapse after extreme valuations in the late "
            "1990s. NASDAQ fell 78% peak to trough. S&P 500 fell 49%. "
            "This scenario is most relevant for portfolios with significant "
            "technology exposure and demonstrates how sector concentration "
            "amplifies losses during sector-specific crises."
        )
    }
}


@dataclass
class StressTestResult:
    """
    Structured container for a single stress test scenario result.

    Attributes
    ----------
    scenario_key : str
        Internal key identifying the scenario e.g. 'GFC_2008'
    scenario_name : str
        Human-readable scenario name
    scenario_start : str
        Start date of the crisis period applied
    scenario_end : str
        End date of the crisis period applied
    portfolio_return_pct : float
        Total portfolio return during the crisis period (negative = loss)
    portfolio_loss_dollar : float
        Dollar loss during the crisis period (positive = loss)
    initial_value : float
        Starting portfolio value used for dollar conversion
    daily_returns : pd.Series
        Day-by-day portfolio returns during the scenario
    cumulative_returns : pd.Series
        Cumulative portfolio value through the scenario period
    worst_single_day_pct : float
        Worst single day loss during the scenario
    worst_single_day_date : str
        Date of the worst single day loss
    n_trading_days : int
        Number of trading days in the scenario window
    description : str
        Context and explanation of the scenario
    """
    scenario_key: str
    scenario_name: str
    scenario_start: str
    scenario_end: str
    portfolio_return_pct: float
    portfolio_loss_dollar: float
    initial_value: float
    daily_returns: pd.Series
    cumulative_returns: pd.Series
    worst_single_day_pct: float
    worst_single_day_date: str
    n_trading_days: int
    description: str

    def summary(self) -> str:
        loss_or_gain = "LOSS" if self.portfolio_loss_dollar > 0 else "GAIN"
        return (
            f"\n{'='*60}\n"
            f"  STRESS TEST: {self.scenario_name.upper()}\n"
            f"{'='*60}\n"
            f"  Period               : {self.scenario_start} → {self.scenario_end}\n"
            f"  Trading Days         : {self.n_trading_days}\n"
            f"{'─'*60}\n"
            f"  Portfolio Return     : {self.portfolio_return_pct:.2%}\n"
            f"  Portfolio {loss_or_gain:4s}  ($) : ${abs(self.portfolio_loss_dollar):>12,.2f}\n"
            f"  Ending Value    ($)  : ${self.initial_value - self.portfolio_loss_dollar:>12,.2f}\n"
            f"{'─'*60}\n"
            f"  Worst Single Day     : {self.worst_single_day_pct:.2%} "
            f"on {self.worst_single_day_date}\n"
            f"{'─'*60}\n"
            f"  Context: {self.description[:120]}...\n"
            f"{'='*60}\n"
        )


@dataclass
class CustomStressResult:
    """
    Result from a user-defined percentage shock stress test.

    Used for forward-looking hypothetical scenarios where you specify
    the shock directly rather than replaying historical data.
    e.g. "Equity markets drop 25%, volatility spikes 40%"
    """
    scenario_name: str
    shocks: dict
    portfolio_loss_pct: float
    portfolio_loss_dollar: float
    initial_value: float
    asset_contributions: dict = field(default_factory=dict)

    def summary(self) -> str:
        shock_lines = "\n".join(
            [f"    {ticker}: {shock:+.1%}" for ticker, shock in self.shocks.items()]
        )
        contrib_lines = "\n".join(
            [f"    {ticker}: ${contrib:,.2f}" for ticker, contrib in self.asset_contributions.items()]
        )
        return (
            f"\n{'='*60}\n"
            f"  CUSTOM STRESS TEST: {self.scenario_name.upper()}\n"
            f"{'='*60}\n"
            f"  Applied Shocks:\n{shock_lines}\n"
            f"{'─'*60}\n"
            f"  Portfolio Loss (%)   : {self.portfolio_loss_pct:.2%}\n"
            f"  Portfolio Loss ($)   : ${self.portfolio_loss_dollar:>12,.2f}\n"
            f"  Ending Value    ($)  : ${self.initial_value - self.portfolio_loss_dollar:>12,.2f}\n"
            f"{'─'*60}\n"
            f"  Loss Contributions by Asset:\n{contrib_lines}\n"
            f"{'='*60}\n"
        )


def run_historical_stress_test(
    prices: pd.DataFrame,
    portfolio,
    scenario_key: str,
    initial_value: float = None
) -> StressTestResult:
    """
    Apply a historical crisis scenario to the current portfolio.

    Takes the actual daily returns that occurred during the specified
    crisis period and computes what would happen to the current portfolio
    if those same market moves happened today.

    Parameters
    ----------
    prices : pd.DataFrame
        Full historical price data from data_loader.fetch_prices().
        Must cover the crisis scenario's date range.
    portfolio : Portfolio
        Portfolio instance from portfolio.py (tickers + weights).
    scenario_key : str
        Key from CRISIS_SCENARIOS dict e.g. 'GFC_2008'
    initial_value : float, optional
        Portfolio value in dollars. Defaults to portfolio.initial_value.

    Returns
    -------
    StressTestResult

    Raises
    ------
    ValueError
        If scenario_key is not recognized or price data doesn't cover
        the scenario's date range.
    """
    if scenario_key not in CRISIS_SCENARIOS:
        raise ValueError(
            f"Unknown scenario: '{scenario_key}'. "
            f"Available scenarios: {list(CRISIS_SCENARIOS.keys())}"
        )

    scenario = CRISIS_SCENARIOS[scenario_key]
    iv = initial_value if initial_value is not None else portfolio.initial_value

    logger.info(f"Running stress test: {scenario['name']}")
    logger.info(f"Scenario period: {scenario['start']} → {scenario['end']}")

    # --- Extract crisis period prices ---
    crisis_prices = prices.loc[scenario["start"]:scenario["end"]]

    if crisis_prices.empty:
        raise ValueError(
            f"No price data available for scenario '{scenario_key}' "
            f"({scenario['start']} → {scenario['end']}). "
            "Your price history must extend back to cover this period. "
            "Re-fetch prices with an earlier start_date."
        )

    if len(crisis_prices) < 5:
        logger.warning(
            f"Only {len(crisis_prices)} trading days found for "
            f"scenario '{scenario_key}'. Results may be unreliable. "
            "Check that your tickers were publicly traded during this period."
        )

    # Align to portfolio tickers only
    available_tickers = [t for t in portfolio.tickers if t in crisis_prices.columns]
    missing_tickers = [t for t in portfolio.tickers if t not in crisis_prices.columns]

    if missing_tickers:
        logger.warning(
            f"Tickers not available for this scenario period: {missing_tickers}. "
            "These assets will be excluded from the stress test. "
            "Results represent a partial portfolio stress."
        )

    crisis_prices_aligned = crisis_prices[available_tickers]

    # --- Compute daily log returns during crisis ---
    crisis_returns = np.log(
        crisis_prices_aligned / crisis_prices_aligned.shift(1)
    ).dropna()

    # Recompute weights for available tickers only (renormalize if needed)
    ticker_weight_map = dict(zip(portfolio.tickers, portfolio.weights))
    available_weights = np.array([ticker_weight_map[t] for t in available_tickers])
    if not np.isclose(available_weights.sum(), 1.0):
        logger.warning(
            f"Renormalizing weights for available tickers. "
            f"Original sum: {available_weights.sum():.4f}"
        )
        available_weights = available_weights / available_weights.sum()

    # --- Compute weighted portfolio returns during crisis ---
    portfolio_crisis_returns = crisis_returns.dot(available_weights)
    portfolio_crisis_returns.name = "portfolio_return"

    # --- Cumulative portfolio value through crisis ---
    # Starting from initial_value, compound daily returns
    cumulative_factor = (1 + portfolio_crisis_returns).cumprod()
    cumulative_value = iv * cumulative_factor

    # --- Summary statistics ---
    total_return = float(cumulative_factor.iloc[-1] - 1)
    total_loss_dollar = -total_return * iv  # positive = loss

    worst_day_idx = portfolio_crisis_returns.idxmin()
    worst_day_pct = float(portfolio_crisis_returns.min())
    worst_day_date = worst_day_idx.strftime("%Y-%m-%d")

    result = StressTestResult(
        scenario_key=scenario_key,
        scenario_name=scenario["name"],
        scenario_start=scenario["start"],
        scenario_end=scenario["end"],
        portfolio_return_pct=total_return,
        portfolio_loss_dollar=total_loss_dollar,
        initial_value=iv,
        daily_returns=portfolio_crisis_returns,
        cumulative_returns=cumulative_value,
        worst_single_day_pct=worst_day_pct,
        worst_single_day_date=worst_day_date,
        n_trading_days=len(portfolio_crisis_returns),
        description=scenario["description"]
    )

    logger.info(result.summary())
    return result


def run_custom_stress_test(
    portfolio,
    shocks: dict,
    scenario_name: str = "Custom Scenario",
    initial_value: float = None
) -> CustomStressResult:
    """
    Apply user-defined percentage shocks to the portfolio.

    This is forward-looking stress testing — you specify hypothetical
    market moves and compute the portfolio impact directly.
    Used for scenarios with no historical precedent or for
    regulatory-mandated forward scenarios.

    Parameters
    ----------
    portfolio : Portfolio
        Portfolio instance from portfolio.py.
    shocks : dict
        Dictionary mapping ticker to shock as a decimal.
        e.g. {"JPM": -0.30, "GS": -0.35, "MSFT": -0.20}
        means JPM drops 30%, GS drops 35%, MSFT drops 20%.
    scenario_name : str
        Descriptive name for this scenario.
    initial_value : float, optional
        Portfolio value. Defaults to portfolio.initial_value.

    Returns
    -------
    CustomStressResult

    Example
    -------
    Run a scenario simulating a severe rate shock hitting financials hard:
    >>> shocks = {"JPM": -0.35, "GS": -0.40, "MSFT": -0.15}
    >>> result = run_custom_stress_test(portfolio, shocks, "Rate Shock Scenario")
    """
    iv = initial_value if initial_value is not None else portfolio.initial_value

    logger.info(f"Running custom stress test: {scenario_name}")

    total_portfolio_loss_pct = 0.0
    asset_contributions = {}

    for ticker, weight in zip(portfolio.tickers, portfolio.weights):
        if ticker not in shocks:
            logger.warning(
                f"No shock specified for {ticker}. Assuming 0% shock. "
                "If intentional, this is fine. If not, add it to shocks dict."
            )
            shock = 0.0
        else:
            shock = shocks[ticker]

        # Each asset's contribution to total portfolio loss
        # = weight * shock (negative shock = positive loss contribution)
        contribution_pct = weight * shock
        contribution_dollar = contribution_pct * iv
        total_portfolio_loss_pct += contribution_pct

        asset_contributions[ticker] = -contribution_dollar  # positive = loss

        logger.info(
            f"  {ticker}: weight={weight:.1%}, shock={shock:+.1%}, "
            f"contribution={contribution_pct:+.1%} (${contribution_dollar:,.0f})"
        )

    total_loss_dollar = -total_portfolio_loss_pct * iv

    result = CustomStressResult(
        scenario_name=scenario_name,
        shocks=shocks,
        portfolio_loss_pct=-total_portfolio_loss_pct,
        portfolio_loss_dollar=total_loss_dollar,
        initial_value=iv,
        asset_contributions=asset_contributions
    )

    logger.info(result.summary())
    return result


def run_all_stress_tests(
    prices: pd.DataFrame,
    portfolio,
    custom_scenarios: list[dict] = None
) -> dict:
    """
    Run all built-in historical stress scenarios plus any custom ones.

    Parameters
    ----------
    prices : pd.DataFrame
        Full price history from data_loader (must go back to 2000).
    portfolio : Portfolio
        Portfolio instance.
    custom_scenarios : list[dict], optional
        List of custom scenario dicts, each with keys:
        'name' (str) and 'shocks' (dict of ticker -> shock decimal)

    Returns
    -------
    dict
        Keys: scenario identifiers
        Values: StressTestResult or CustomStressResult objects
    """
    results = {}

    for key in CRISIS_SCENARIOS:
        try:
            results[key] = run_historical_stress_test(prices, portfolio, key)
        except ValueError as e:
            logger.warning(
                f"Skipping scenario '{key}': {e}. "
                "Extend your price history start_date to include this period."
            )

    if custom_scenarios:
        for cs in custom_scenarios:
            result = run_custom_stress_test(
                portfolio,
                shocks=cs["shocks"],
                scenario_name=cs["name"]
            )
            results[cs["name"]] = result

    return results


def build_stress_summary_table(results: dict) -> pd.DataFrame:
    """
    Build a summary table comparing all stress test results.

    This is the table that would appear in a risk report — one row
    per scenario, showing portfolio loss in percentage and dollar terms.

    Parameters
    ----------
    results : dict
        Output from run_all_stress_tests()

    Returns
    -------
    pd.DataFrame
        Summary table sorted by severity (worst loss first).
    """
    rows = []
    for key, result in results.items():
        if isinstance(result, StressTestResult):
            rows.append({
                "Scenario": result.scenario_name,
                "Period": f"{result.scenario_start} → {result.scenario_end}",
                "Trading Days": result.n_trading_days,
                "Portfolio Return": f"{result.portfolio_return_pct:.2%}",
                "Dollar Loss ($)": result.portfolio_loss_dollar,
                "Worst Single Day": f"{result.worst_single_day_pct:.2%} on {result.worst_single_day_date}"
            })
        elif isinstance(result, CustomStressResult):
            rows.append({
                "Scenario": result.scenario_name,
                "Period": "Hypothetical",
                "Trading Days": "N/A",
                "Portfolio Return": f"{result.portfolio_loss_pct:.2%}",
                "Dollar Loss ($)": result.portfolio_loss_dollar,
                "Worst Single Day": "N/A"
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Dollar Loss ($)", ascending=False)
        df["Dollar Loss ($)"] = df["Dollar Loss ($)"].apply(
            lambda x: f"${x:,.2f}" if isinstance(x, float) else x
        )
    return df


def plot_stress_test_paths(
    results: dict,
    save_path: str = None
) -> None:
    """
    Plot cumulative portfolio value through each historical crisis.

    Shows how the portfolio value would have evolved day-by-day
    if each crisis happened to the portfolio today.

    Parameters
    ----------
    results : dict
        Output from run_all_stress_tests()
    save_path : str, optional
        If provided, saves figure to this path.
    """
    historical_results = {
        k: v for k, v in results.items()
        if isinstance(v, StressTestResult)
    }

    if not historical_results:
        logger.warning("No historical stress test results to plot.")
        return

    colors = ["#d7191c", "#2c7bb6", "#1a9641", "#fdae61", "#756bb1"]
    fig, ax = plt.subplots(figsize=(13, 6))

    for (key, result), color in zip(historical_results.items(), colors):
        # Normalize to 100 at scenario start for comparability
        normalized = (result.cumulative_returns / result.initial_value) * 100

        ax.plot(
            normalized.index,
            normalized.values,
            color=color,
            linewidth=2,
            label=(
                f"{result.scenario_name} "
                f"({result.portfolio_return_pct:.1%} total)"
            )
        )

    ax.axhline(y=100, color="black", linewidth=1, linestyle="--",
               alpha=0.5, label="Starting Value (100)")

    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Portfolio Value (indexed to 100 at crisis start)", fontsize=11)
    ax.set_title(
        "Portfolio Value Under Historical Crisis Scenarios\n"
        "(How would this portfolio have performed during each crisis?)",
        fontsize=13
    )
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f'))
    plt.xticks(rotation=30)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Stress test paths plot saved to: {save_path}")
    else:
        plt.show()

    plt.close()


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

    # Must go back to 2000 to cover all three crisis scenarios
    END = datetime.today().strftime("%Y-%m-%d")
    START = "2000-01-01"

    print("\n--- Fetching extended price history (back to 2000) ---")
    prices = fetch_prices(TICKERS, START, END, use_cache=True)
    portfolio = Portfolio.equal_weight(TICKERS, initial_value=1_000_000)

    print("\n--- Running all historical stress tests ---")
    custom_scenarios = [
        {
            "name": "Hypothetical Rate Shock",
            "shocks": {"JPM": -0.35, "GS": -0.40, "MSFT": -0.15}
        }
    ]
    results = run_all_stress_tests(prices, portfolio, custom_scenarios)

    print("\n--- Stress Test Summary Table ---")
    summary = build_stress_summary_table(results)
    print(summary.to_string(index=False))

    print("\n--- Plotting crisis paths ---")
    plot_stress_test_paths(results)

    print("\nstress_testing.py working correctly.")