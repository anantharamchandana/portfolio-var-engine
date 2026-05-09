"""
report.py
---------
Generates a complete risk report from the command line.

Runs the full pipeline end-to-end:
    data_loader → portfolio → historical_simulation →
    monte_carlo → stress_testing → formatted report output

Output:
    - Printed report to terminal (always)
    - Optional text file saved to /reports/ directory

Why this module exists:
    The dashboard is the interactive tool.
    This is the pipeline proof — it shows the entire system
    runs cleanly from a single entry point with no browser,
    no manual steps, and no hidden state. This is what
    engineering teams actually care about in interviews.

Usage:
    # Basic run with defaults (JPM, GS, MSFT — equal weight — $1M)
    python src/report.py

    # Custom portfolio
    python src/report.py --tickers AAPL MSFT GOOGL --weights 0.4 0.4 0.2

    # Custom portfolio value and confidence level
    python src/report.py --tickers JPM GS --weights 0.6 0.4 \
        --value 5000000 --confidence 0.99

    # Save report to file
    python src/report.py --save
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_loader import fetch_prices, compute_daily_returns
from portfolio import Portfolio, compute_portfolio_returns, compute_portfolio_statistics
from historical_simulation import compute_historical_var, compute_var_term_structure
from monte_carlo import compute_monte_carlo_var, compare_methods
from stress_testing import run_all_stress_tests, build_stress_summary_table

# Suppress info logs during report generation — only warnings and errors
logging.basicConfig(level=logging.WARNING)

# Report output directory
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Report sections — each returns a formatted string block
# ---------------------------------------------------------------------------

def _header(portfolio: Portfolio, stats: dict) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tickers_str = "  ".join(
        f"{t} ({w:.1%})"
        for t, w in zip(portfolio.tickers, portfolio.weights)
    )
    return f"""
{'='*65}
  PORTFOLIO RISK REPORT
  Generated: {now}
{'='*65}

  PORTFOLIO COMPOSITION
  {'─'*61}
  {tickers_str}
  Initial Value    : ${portfolio.initial_value:>14,.2f}
  Data Range       : {stats['date_range'][0]}  →  {stats['date_range'][1]}
  Trading Days     : {stats['total_trading_days']:,}

  PORTFOLIO STATISTICS
  {'─'*61}
  Annualized Return     : {stats['annualized_return']:>10.2%}
  Annualized Volatility : {stats['annualized_volatility']:>10.2%}
  Sharpe Ratio          : {stats['sharpe_ratio']:>10.3f}
  Mean Daily Return     : {stats['mean_daily_return']:>10.4%}
  Daily Std Dev         : {stats['std_daily_return']:>10.4%}
  Skewness              : {stats['skewness']:>10.4f}
  Excess Kurtosis       : {stats['kurtosis']:>10.4f}

  {'─'*61}
  DISTRIBUTION INTERPRETATION
  {'─'*61}"""  + _interpret_distribution(stats)


def _interpret_distribution(stats: dict) -> str:
    lines = []

    skew = stats['skewness']
    kurt = stats['kurtosis']

    if skew < -0.5:
        lines.append(
            f"\n  Skewness ({skew:.4f}): NEGATIVELY SKEWED\n"
            "  Losses are more extreme than gains of equivalent frequency.\n"
            "  Typical of equity portfolios — crashes are sharper than rallies."
        )
    elif skew > 0.5:
        lines.append(
            f"\n  Skewness ({skew:.4f}): POSITIVELY SKEWED\n"
            "  Gains tend to be more extreme than losses."
        )
    else:
        lines.append(
            f"\n  Skewness ({skew:.4f}): APPROXIMATELY SYMMETRIC\n"
            "  Gains and losses are roughly balanced in magnitude."
        )

    if kurt > 3.0:
        lines.append(
            f"\n  Excess Kurtosis ({kurt:.4f}): EXTREME FAT TAILS\n"
            "  Extreme daily moves occur far more often than a normal\n"
            "  distribution predicts. This directly causes Monte Carlo\n"
            "  (normal distribution assumption) to underestimate true\n"
            "  tail risk. See CVaR comparison below for evidence."
        )
    elif kurt > 1.0:
        lines.append(
            f"\n  Excess Kurtosis ({kurt:.4f}): FAT TAILS PRESENT\n"
            "  Extreme moves more frequent than normal distribution predicts."
        )
    else:
        lines.append(
            f"\n  Excess Kurtosis ({kurt:.4f}): NEAR-NORMAL TAILS"
        )

    return "\n".join(lines)


def _var_section(
    hist_result,
    mc_result,
    confidence_level: float,
    holding_period: int
) -> str:
    dollar_diff = mc_result.var_dollar - hist_result.var_dollar
    direction = "HIGHER" if dollar_diff > 0 else "LOWER"

    cvar_diff = mc_result.cvar_dollar - hist_result.cvar_dollar
    cvar_direction = "HIGHER" if cvar_diff > 0 else "LOWER"

    normality_flag = (
        "  *** MODEL RISK: Returns FAIL normality test (Jarque-Bera p < 0.05)\n"
        "  *** Normal distribution assumption UNDERESTIMATES true tail risk.\n"
        "  *** CVaR gap below is direct evidence of this.\n"
        if mc_result.normality_p_value < 0.05
        else "  Normality test PASSED at 5% significance level.\n"
    )

    return f"""
{'='*65}
  VALUE AT RISK RESULTS
  Confidence: {confidence_level:.0%}  |  Holding Period: {holding_period} day(s)
{'='*65}

  {'Metric':<35} {'Historical':>12}  {'Monte Carlo':>12}
  {'─'*61}
  {'VaR (%)':<35} {hist_result.var_pct:>12.4%}  {mc_result.var_pct:>12.4%}
  {'VaR ($)':<35} {f'${hist_result.var_dollar:,.2f}':>12}  {f'${mc_result.var_dollar:,.2f}':>12}
  {'CVaR / Expected Shortfall (%)':<35} {hist_result.cvar_pct:>12.4%}  {mc_result.cvar_pct:>12.4%}
  {'CVaR / Expected Shortfall ($)':<35} {f'${hist_result.cvar_dollar:,.2f}':>12}  {f'${mc_result.cvar_dollar:,.2f}':>12}
  {'─'*61}
  {'Observations / Simulations':<35} {hist_result.n_observations:>12,}  {mc_result.n_simulations:>12,}
  {'Tail observations':<35} {hist_result.n_tail_observations:>12}  {'N/A':>12}
  {'Normality p-value':<35} {'N/A':>12}  {mc_result.normality_p_value:>12.4f}

  MODEL RISK ASSESSMENT
  {'─'*61}
{normality_flag}
  VaR  gap (MC - Historical): ${dollar_diff:>10,.2f}  [{direction}]
  CVaR gap (MC - Historical): ${cvar_diff:>10,.2f}  [{cvar_direction}]

  INTERPRETATION:
  The VaR gap measures model risk — how sensitive your risk
  estimate is to the distributional assumption. A larger gap
  means the choice of method materially changes your risk number.

  The CVaR gap is more diagnostic: if Historical CVaR > MC CVaR,
  actual tail losses are worse than the normal distribution predicts.
  This is the empirical signature of fat tails."""


def _term_structure_section(term_structure: pd.DataFrame) -> str:
    lines = [
        f"\n{'='*65}",
        "  VAR TERM STRUCTURE",
        "  Dollar VaR across confidence levels and holding periods",
        f"{'='*65}",
        ""
    ]

    # Header row
    cols = list(term_structure.columns)
    header = f"  {'Confidence':>12}  " + "  ".join(f"{c:>16}" for c in cols)
    lines.append(header)
    lines.append(f"  {'─'*61}")

    for idx, row in term_structure.iterrows():
        row_str = f"  {idx:>12}  " + "  ".join(
            f"${v:>14,.0f}" for v in row.values
        )
        lines.append(row_str)

    lines.append(f"\n  Basel III minimum requirement: 99% confidence, 10-day holding period")
    lines.append(f"  → Your 10d 99% VaR: ${term_structure.loc['99%', '10d VaR ($)']:,.0f}")

    return "\n".join(lines)


def _stress_section(stress_summary: pd.DataFrame, stress_results: dict) -> str:
    from stress_testing import StressTestResult

    lines = [
        f"\n{'='*65}",
        "  STRESS TEST RESULTS",
        "  Portfolio impact under historical crisis scenarios",
        f"{'='*65}",
        ""
    ]

    for _, row in stress_summary.iterrows():
        lines.append(f"  {'─'*61}")
        lines.append(f"  {row['Scenario'].upper()}")
        lines.append(f"  Period        : {row['Period']}")
        lines.append(f"  Trading Days  : {row['Trading Days']}")
        lines.append(f"  Portfolio Rtn : {row['Portfolio Return']}")
        lines.append(f"  Dollar Loss   : {row['Dollar Loss ($)']}")
        lines.append(f"  Worst Day     : {row['Worst Single Day']}")

    lines.append(f"\n  {'─'*61}")
    lines.append("  KEY OBSERVATION:")
    lines.append(
        "  Compare stress test losses to daily VaR above.\n"
        "  Crisis losses routinely exceed daily VaR by 10-20x.\n"
        "  This is why stress testing is required ALONGSIDE VaR\n"
        "  under Basel III — VaR alone does not capture crisis risk."
    )

    return "\n".join(lines)


def _model_risk_summary(hist_result, mc_result, stats: dict) -> str:
    kurt = stats['kurtosis']
    normality_failed = mc_result.normality_p_value < 0.05
    cvar_gap = hist_result.cvar_dollar - mc_result.cvar_dollar

    risk_level = (
        "HIGH" if normality_failed and kurt > 3.0
        else "MODERATE" if normality_failed
        else "LOW"
    )

    return f"""
{'='*65}
  MODEL RISK SUMMARY
{'='*65}

  Overall Model Risk Level : {risk_level}

  FINDINGS:
  {'─'*61}
  1. DISTRIBUTIONAL ASSUMPTION
     Normality test {'FAILED' if normality_failed else 'PASSED'}
     (Jarque-Bera p-value: {mc_result.normality_p_value:.4f})
     Excess Kurtosis: {kurt:.4f}
     → {'Fat tails confirmed. MC normal distribution underestimates' if normality_failed else 'Distribution consistent with normal assumption.'}
     {'  true tail risk by $' + f'{cvar_gap:,.0f} in CVaR terms.' if normality_failed and cvar_gap > 0 else ''}

  2. METHOD SENSITIVITY
     VaR changes by ${abs(mc_result.var_dollar - hist_result.var_dollar):,.0f}
     depending on method choice (Historical vs MC).
     A production risk system would report both and flag
     the gap as a model uncertainty metric.

  3. STRESS TEST VS VAR
     Daily 95% VaR: ${hist_result.var_dollar:,.0f}
     COVID-19 loss is captured in stress tests above.
     Ratio of crisis loss to daily VaR illustrates why
     VaR alone is insufficient for risk governance.

  LIMITATIONS OF THIS MODEL:
  {'─'*61}
  - Historical simulation equally weights all historical days.
    A return from 2001 counts the same as last week.
    Exponentially weighted historical simulation (EWHS) would
    give more weight to recent observations.

  - Monte Carlo assumes normally distributed returns.
    Student-t distribution or GARCH-based simulation would
    better capture fat tails and volatility clustering.

  - Square root of time scaling for multi-day VaR assumes
    return independence across days. Volatility clustering
    in real markets violates this assumption, causing
    multi-day VaR to be underestimated during crises.

  - Long-only portfolio only. Short positions, options,
    and non-linear instruments would require full
    revaluation rather than return-based approximation.

  These limitations are consistent with SR 11-7 model risk
  management documentation requirements."""


def _footer() -> str:
    return f"""
{'='*65}
  END OF REPORT
  Portfolio VaR Engine — github.com/[your-username]/portfolio-var-engine
  Data source: Yahoo Finance  |  For research purposes only
{'='*65}
"""


# ---------------------------------------------------------------------------
# Main report runner
# ---------------------------------------------------------------------------

def generate_report(
    tickers: list[str],
    weights: list[float],
    initial_value: float,
    confidence_level: float,
    holding_period: int,
    n_simulations: int,
    start_date: str,
    end_date: str,
    save: bool = False
) -> str:
    """
    Run the full pipeline and generate a complete risk report.

    Parameters
    ----------
    tickers : list[str]
    weights : list[float]
    initial_value : float
    confidence_level : float
    holding_period : int
    n_simulations : int
    start_date : str  — 'YYYY-MM-DD'
    end_date : str    — 'YYYY-MM-DD'
    save : bool       — if True, writes report to /reports/ directory

    Returns
    -------
    str — complete formatted report
    """
    print("\nFetching price data...")
    prices = fetch_prices(tickers, start_date, end_date, use_cache=True)
    returns = compute_daily_returns(prices)

    print("Building portfolio...")
    portfolio = Portfolio(
        tickers=tickers,
        weights=weights,
        initial_value=initial_value
    )
    port_returns = compute_portfolio_returns(returns, portfolio)
    stats = compute_portfolio_statistics(port_returns)

    print("Computing Historical VaR...")
    hist_result = compute_historical_var(
        port_returns, initial_value,
        confidence_level=confidence_level,
        holding_period_days=holding_period
    )

    print("Running Monte Carlo simulation...")
    mc_result = compute_monte_carlo_var(
        port_returns, initial_value,
        confidence_level=confidence_level,
        holding_period_days=holding_period,
        n_simulations=n_simulations
    )

    print("Computing VaR term structure...")
    term_structure = compute_var_term_structure(port_returns, initial_value)

    print("Running stress tests...")
    stress_results = run_all_stress_tests(prices, portfolio)
    stress_summary = build_stress_summary_table(stress_results)

    print("Assembling report...\n")

    report = (
        _header(portfolio, stats)
        + _var_section(hist_result, mc_result, confidence_level, holding_period)
        + _term_structure_section(term_structure)
        + _stress_section(stress_summary, stress_results)
        + _model_risk_summary(hist_result, mc_result, stats)
        + _footer()
    )

    if save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tickers_str = "_".join(tickers)
        filename = f"risk_report_{tickers_str}_{timestamp}.txt"
        filepath = REPORTS_DIR / filename
        filepath.write_text(report)
        print(f"\nReport saved to: {filepath}")

    return report


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a portfolio risk report from the command line.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/report.py
  python src/report.py --tickers AAPL MSFT GOOGL --weights 0.4 0.4 0.2
  python src/report.py --tickers JPM GS --weights 0.6 0.4 --value 5000000
  python src/report.py --confidence 0.99 --holding-period 10 --save
        """
    )

    parser.add_argument(
        "--tickers", nargs="+",
        default=["JPM", "GS", "MSFT"],
        help="Ticker symbols (default: JPM GS MSFT)"
    )
    parser.add_argument(
        "--weights", nargs="+", type=float,
        default=None,
        help="Portfolio weights summing to 1.0 (default: equal weight)"
    )
    parser.add_argument(
        "--value", type=float,
        default=1_000_000,
        help="Initial portfolio value in dollars (default: 1000000)"
    )
    parser.add_argument(
        "--confidence", type=float,
        default=0.95,
        help="VaR confidence level (default: 0.95)"
    )
    parser.add_argument(
        "--holding-period", type=int,
        default=1,
        help="Holding period in days (default: 1)"
    )
    parser.add_argument(
        "--simulations", type=int,
        default=10_000,
        help="Monte Carlo simulation count (default: 10000)"
    )
    parser.add_argument(
        "--start", type=str,
        default="2000-01-01",
        help="Data start date YYYY-MM-DD (default: 2000-01-01)"
    )
    parser.add_argument(
        "--end", type=str,
        default=datetime.today().strftime("%Y-%m-%d"),
        help="Data end date YYYY-MM-DD (default: today)"
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save report to /reports/ directory"
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    tickers = [t.upper() for t in args.tickers]

    if args.weights is None:
        n = len(tickers)
        weights = [round(1.0 / n, 10)] * n
        weights[-1] = round(1.0 - sum(weights[:-1]), 10)
    else:
        weights = args.weights
        if len(weights) != len(tickers):
            print(
                f"Error: {len(tickers)} tickers but {len(weights)} weights provided. "
                "They must match."
            )
            sys.exit(1)
        if not np.isclose(sum(weights), 1.0, atol=1e-4):
            print(
                f"Error: weights sum to {sum(weights):.4f}. Must equal 1.0."
            )
            sys.exit(1)

    report = generate_report(
        tickers=tickers,
        weights=weights,
        initial_value=args.value,
        confidence_level=args.confidence,
        holding_period=args.holding_period,
        n_simulations=args.simulations,
        start_date=args.start,
        end_date=args.end,
        save=args.save
    )

    print(report)