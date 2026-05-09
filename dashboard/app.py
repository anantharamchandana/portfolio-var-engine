"""
dashboard/app.py
----------------
Streamlit-based interactive risk dashboard for the Portfolio VaR Engine.

How to run:
    From the project root directory:
        streamlit run dashboard/app.py

What this dashboard does:
    - Lets the user configure a portfolio (tickers, weights, value)
    - Computes Value at Risk via Historical Simulation and Monte Carlo
    - Displays a method comparison table
    - Runs all historical stress tests and shows crisis impact
    - Presents everything in a clean, operator-facing layout

Design principle:
    This is a risk tool, not a data science demo.
    Every number shown should be immediately interpretable
    by someone managing a portfolio — not just someone who
    built the model.
"""

import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data_loader import fetch_prices, compute_daily_returns
from portfolio import Portfolio, compute_portfolio_returns, compute_portfolio_statistics
from historical_simulation import (
    compute_historical_var,
    compute_var_term_structure,
)
from monte_carlo import compute_monte_carlo_var, compare_methods
from stress_testing import (
    run_all_stress_tests,
    build_stress_summary_table,
    CRISIS_SCENARIOS,
)

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Portfolio Risk Dashboard",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    .metric-card {
        background-color: #1e2130;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 8px;
    }
    .risk-high { color: #ef4444; font-weight: 700; font-size: 1.4rem; }
    .risk-medium { color: #f97316; font-weight: 700; font-size: 1.4rem; }
    .risk-label { color: #94a3b8; font-size: 0.85rem; text-transform: uppercase;
                  letter-spacing: 0.05em; }
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #e2e8f0;
        border-bottom: 1px solid #334155;
        padding-bottom: 6px;
        margin-bottom: 12px;
    }
    .warning-box {
        background-color: #422006;
        border-left: 4px solid #f97316;
        border-radius: 4px;
        padding: 10px 14px;
        font-size: 0.88rem;
        color: #fed7aa;
    }
    .info-box {
        background-color: #0f2942;
        border-left: 4px solid #3b82f6;
        border-radius: 4px;
        padding: 10px 14px;
        font-size: 0.88rem;
        color: #bfdbfe;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar — portfolio configuration
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("⚙️ Portfolio Configuration")
    st.markdown("---")

    # --- Tickers ---
    st.markdown("**Tickers**")
    ticker_input = st.text_input(
        "Enter tickers separated by commas",
        value="JPM, GS, MSFT",
        help="Use valid Yahoo Finance ticker symbols e.g. JPM, GS, MSFT, AAPL"
    )
    tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

    # --- Weights ---
    st.markdown("**Portfolio Weights**")
    weight_mode = st.radio(
        "Weight method",
        ["Equal weight", "Custom weights"],
        index=0
    )

    weights = []
    if weight_mode == "Equal weight":
        if tickers:
            w = round(1.0 / len(tickers), 6)
            weights = [w] * len(tickers)
            weights[-1] = round(1.0 - sum(weights[:-1]), 6)
            for t, wt in zip(tickers, weights):
                st.caption(f"{t}: {wt:.1%}")
    else:
        st.caption("Weights must sum to 1.0")
        for ticker in tickers:
            w = st.number_input(
                f"{ticker} weight",
                min_value=0.0,
                max_value=1.0,
                value=round(1.0 / len(tickers), 2) if tickers else 0.0,
                step=0.01,
                key=f"weight_{ticker}"
            )
            weights.append(w)
        if weights:
            weight_sum = sum(weights)
            if not np.isclose(weight_sum, 1.0, atol=0.01):
                st.error(f"Weights sum to {weight_sum:.2f}. Must equal 1.0.")

    st.markdown("---")

    # --- Portfolio value ---
    st.markdown("**Initial Portfolio Value**")
    initial_value = st.number_input(
        "Value ($)",
        min_value=10_000,
        max_value=1_000_000_000,
        value=1_000_000,
        step=100_000,
        format="%d"
    )

    st.markdown("---")

    # --- Risk parameters ---
    st.markdown("**Risk Parameters**")
    confidence_level = st.selectbox(
        "VaR Confidence Level",
        options=[0.90, 0.95, 0.99],
        index=1,
        format_func=lambda x: f"{x:.0%}"
    )

    holding_period = st.selectbox(
        "Holding Period",
        options=[1, 5, 10],
        index=0,
        format_func=lambda x: f"{x} day{'s' if x > 1 else ''}"
    )

    n_simulations = st.selectbox(
        "Monte Carlo Simulations",
        options=[10_000, 50_000, 100_000],
        index=0,
        format_func=lambda x: f"{x:,}"
    )

    st.markdown("---")

    # --- Date range ---
    st.markdown("**Historical Data Range**")
    st.caption("Must go back to 2000 for full stress test coverage")
    data_start = st.date_input(
        "Start date",
        value=datetime(2000, 1, 1),
        max_value=datetime.today() - timedelta(days=365)
    )
    data_end = st.date_input(
        "End date",
        value=datetime.today(),
        max_value=datetime.today()
    )

    st.markdown("---")
    run_button = st.button("▶  Run Analysis", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------

st.title("📉 Portfolio Risk Dashboard")
st.caption("Value at Risk · Conditional VaR · Stress Testing · Method Comparison")
st.markdown("---")

if not run_button:
    st.markdown("""
    <div class="info-box">
    Configure your portfolio in the sidebar and click <strong>Run Analysis</strong> to generate risk metrics.
    <br><br>
    This dashboard computes:
    <ul>
        <li><strong>Value at Risk (VaR)</strong> — maximum expected loss at your chosen confidence level</li>
        <li><strong>Conditional VaR</strong> — average loss when losses exceed the VaR threshold</li>
        <li><strong>Method comparison</strong> — Historical Simulation vs Monte Carlo side by side</li>
        <li><strong>Stress tests</strong> — portfolio impact under 2008 GFC, COVID-19, and dot-com crash</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ---------------------------------------------------------------------------
# Validation before running
# ---------------------------------------------------------------------------

errors = []
if not tickers:
    errors.append("No tickers entered.")
if len(tickers) != len(weights):
    errors.append("Number of tickers and weights do not match.")
if weights and not np.isclose(sum(weights), 1.0, atol=0.01):
    errors.append(f"Weights sum to {sum(weights):.3f}. Must equal 1.0.")
if data_start >= data_end.date() if hasattr(data_end, 'date') else data_start >= data_end:
    errors.append("Start date must be before end date.")

if errors:
    for e in errors:
        st.error(e)
    st.stop()

# ---------------------------------------------------------------------------
# Data loading and computation — cached for performance
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def load_and_compute(
    tickers, weights, initial_value,
    confidence_level, holding_period, n_simulations,
    start_str, end_str
):
    """
    Full computation pipeline. Cached so re-renders don't recompute.
    Cache invalidates after 1 hour (ttl=3600).
    """
    prices = fetch_prices(tickers, start_str, end_str, use_cache=True)
    returns = compute_daily_returns(prices)

    portfolio = Portfolio(
        tickers=list(tickers),
        weights=list(weights),
        initial_value=float(initial_value)
    )

    port_returns = compute_portfolio_returns(returns, portfolio)
    stats = compute_portfolio_statistics(port_returns)

    hist_result = compute_historical_var(
        port_returns, initial_value,
        confidence_level=confidence_level,
        holding_period_days=holding_period
    )

    mc_result = compute_monte_carlo_var(
        port_returns, initial_value,
        confidence_level=confidence_level,
        holding_period_days=holding_period,
        n_simulations=n_simulations
    )

    term_structure = compute_var_term_structure(port_returns, initial_value)

    stress_results = run_all_stress_tests(prices, portfolio)
    stress_summary = build_stress_summary_table(stress_results)

    return (
        prices, returns, portfolio, port_returns,
        stats, hist_result, mc_result,
        term_structure, stress_results, stress_summary
    )


with st.spinner("Fetching data and computing risk metrics..."):
    try:
        start_str = data_start.strftime("%Y-%m-%d") if hasattr(data_start, 'strftime') else str(data_start)
        end_str = data_end.strftime("%Y-%m-%d") if hasattr(data_end, 'strftime') else str(data_end)

        (
            prices, returns, portfolio, port_returns,
            stats, hist_result, mc_result,
            term_structure, stress_results, stress_summary
        ) = load_and_compute(
            tuple(tickers), tuple(weights), initial_value,
            confidence_level, holding_period, n_simulations,
            start_str, end_str
        )
    except Exception as e:
        st.error(f"Error during computation: {e}")
        st.caption("Common causes: invalid ticker symbol, insufficient price history, weights not summing to 1.0")
        st.stop()

# ---------------------------------------------------------------------------
# Section 1 — Portfolio overview
# ---------------------------------------------------------------------------

st.markdown("### Portfolio Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Portfolio Value",
        f"${initial_value:,.0f}",
        help="Initial portfolio value used for dollar VaR calculations"
    )

with col2:
    st.metric(
        "Annualized Return",
        f"{stats['annualized_return']:.2%}",
        help="Mean daily return scaled to 252 trading days"
    )

with col3:
    st.metric(
        "Annualized Volatility",
        f"{stats['annualized_volatility']:.2%}",
        help="Daily return std dev scaled to 252 trading days"
    )

with col4:
    st.metric(
        "Sharpe Ratio",
        f"{stats['sharpe_ratio']:.2f}",
        help="Annualized return per unit of annualized volatility (risk-free rate = 0)"
    )

# Portfolio composition bar
st.markdown("**Allocation**")
fig_alloc = go.Figure(go.Bar(
    x=[f"{t} ({w:.1%})" for t, w in zip(tickers, weights)],
    y=weights,
    marker_color=["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6"][:len(tickers)],
    text=[f"{w:.1%}" for w in weights],
    textposition="outside"
))
fig_alloc.update_layout(
    height=220,
    margin=dict(t=10, b=10, l=10, r=10),
    yaxis=dict(tickformat=".0%", showgrid=False),
    xaxis=dict(showgrid=False),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#e2e8f0")
)
st.plotly_chart(fig_alloc, use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------------------------
# Section 2 — VaR results side by side
# ---------------------------------------------------------------------------

st.markdown("### Value at Risk Results")
st.caption(
    f"{confidence_level:.0%} confidence · "
    f"{holding_period}-day holding period · "
    f"${initial_value:,.0f} portfolio"
)

col_hist, col_mc = st.columns(2)

with col_hist:
    st.markdown("#### Historical Simulation")
    st.markdown(f"""
    <div class="metric-card">
        <div class="risk-label">Value at Risk ({confidence_level:.0%})</div>
        <div class="risk-high">${hist_result.var_dollar:,.0f}</div>
        <div style="color:#94a3b8; font-size:0.9rem;">{hist_result.var_pct:.3%} of portfolio</div>
    </div>
    <div class="metric-card">
        <div class="risk-label">Conditional VaR (Expected Shortfall)</div>
        <div class="risk-medium">${hist_result.cvar_dollar:,.0f}</div>
        <div style="color:#94a3b8; font-size:0.9rem;">{hist_result.cvar_pct:.3%} of portfolio</div>
    </div>
    """, unsafe_allow_html=True)
    st.caption(f"Based on {hist_result.n_observations:,} historical trading days")
    st.caption(f"Tail observations (beyond VaR): {hist_result.n_tail_observations}")

with col_mc:
    st.markdown("#### Monte Carlo Simulation")
    st.markdown(f"""
    <div class="metric-card">
        <div class="risk-label">Value at Risk ({confidence_level:.0%})</div>
        <div class="risk-high">${mc_result.var_dollar:,.0f}</div>
        <div style="color:#94a3b8; font-size:0.9rem;">{mc_result.var_pct:.3%} of portfolio</div>
    </div>
    <div class="metric-card">
        <div class="risk-label">Conditional VaR (Expected Shortfall)</div>
        <div class="risk-medium">${mc_result.cvar_dollar:,.0f}</div>
        <div style="color:#94a3b8; font-size:0.9rem;">{mc_result.cvar_pct:.3%} of portfolio</div>
    </div>
    """, unsafe_allow_html=True)
    st.caption(f"Simulations run: {mc_result.n_simulations:,}")

    if mc_result.normality_p_value < 0.05:
        st.markdown("""
        <div class="warning-box">
        ⚠️ <strong>Model Risk Warning</strong><br>
        Returns fail the normality test (Jarque-Bera p-value &lt; 0.05).
        Monte Carlo assumes normally distributed returns.
        Fat tails in actual data mean true tail risk is likely
        higher than the Monte Carlo estimate shows.
        </div>
        """, unsafe_allow_html=True)

# Method difference
dollar_diff = mc_result.var_dollar - hist_result.var_dollar
direction = "higher" if dollar_diff > 0 else "lower"
st.info(
    f"Monte Carlo VaR is **${abs(dollar_diff):,.0f} {direction}** than Historical VaR. "
    f"This gap represents model risk — the sensitivity of your risk estimate "
    f"to the distributional assumption."
)

st.markdown("---")

# ---------------------------------------------------------------------------
# Section 3 — Return distribution chart
# ---------------------------------------------------------------------------

st.markdown("### Return Distribution")

fig_dist = go.Figure()

# Historical return histogram
fig_dist.add_trace(go.Histogram(
    x=port_returns.values,
    nbinsx=80,
    name="Historical Returns",
    marker_color="#3b82f6",
    opacity=0.6,
    histnorm="probability density"
))

# Monte Carlo simulated distribution
fig_dist.add_trace(go.Histogram(
    x=mc_result.simulated_returns,
    nbinsx=100,
    name=f"MC Simulated ({mc_result.n_simulations:,} scenarios)",
    marker_color="#10b981",
    opacity=0.4,
    histnorm="probability density"
))

# VaR lines
fig_dist.add_vline(
    x=-hist_result.var_pct,
    line_dash="dash",
    line_color="#ef4444",
    line_width=2,
    annotation_text=f"Hist VaR {hist_result.var_pct:.2%}",
    annotation_position="top left",
    annotation_font_color="#ef4444"
)

fig_dist.add_vline(
    x=-mc_result.var_pct,
    line_dash="dot",
    line_color="#f97316",
    line_width=2,
    annotation_text=f"MC VaR {mc_result.var_pct:.2%}",
    annotation_position="top right",
    annotation_font_color="#f97316"
)

fig_dist.update_layout(
    height=380,
    xaxis_title="Daily Portfolio Return",
    yaxis_title="Probability Density",
    barmode="overlay",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#e2e8f0"),
    xaxis=dict(tickformat=".1%"),
    margin=dict(t=40, b=40)
)
st.plotly_chart(fig_dist, use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------------------------
# Section 4 — VaR term structure
# ---------------------------------------------------------------------------

st.markdown("### VaR Term Structure")
st.caption("Dollar VaR across multiple confidence levels and holding periods")

# Format term structure for display
ts_display = term_structure.copy()
for col in ts_display.columns:
    ts_display[col] = ts_display[col].apply(lambda x: f"${x:,.0f}")

st.dataframe(
    ts_display,
    use_container_width=True,
    height=140
)

st.markdown("""
<div class="info-box">
Each cell answers: "At this confidence level and holding period, what is the maximum expected loss?"
The 1-day 99% VaR is the number most commonly required under Basel regulatory frameworks.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ---------------------------------------------------------------------------
# Section 5 — Stress tests
# ---------------------------------------------------------------------------

st.markdown("### Stress Test Results")
st.caption("How would this portfolio perform if historical crises happened today?")

# Summary table
st.dataframe(stress_summary, use_container_width=True, hide_index=True)

st.markdown("---")

# Crisis path chart
from stress_testing import StressTestResult

historical_stress = {
    k: v for k, v in stress_results.items()
    if isinstance(v, StressTestResult)
}

if historical_stress:
    st.markdown("**Portfolio Value Through Each Crisis (indexed to 100 at crisis start)**")

    colors = ["#ef4444", "#3b82f6", "#10b981", "#f59e0b"]
    fig_stress = go.Figure()

    for (key, result), color in zip(historical_stress.items(), colors):
        normalized = (result.cumulative_returns / result.initial_value) * 100
        fig_stress.add_trace(go.Scatter(
            x=normalized.index,
            y=normalized.values,
            mode="lines",
            name=(
                f"{result.scenario_name} "
                f"({result.portfolio_return_pct:.1%})"
            ),
            line=dict(color=color, width=2)
        ))

    fig_stress.add_hline(
        y=100,
        line_dash="dash",
        line_color="#64748b",
        line_width=1,
        annotation_text="Starting value",
        annotation_font_color="#64748b"
    )

    fig_stress.update_layout(
        height=420,
        xaxis_title="Date",
        yaxis_title="Portfolio Value (100 = starting value)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"),
        margin=dict(t=40, b=40)
    )
    st.plotly_chart(fig_stress, use_container_width=True)

    # Individual scenario drill-down
    st.markdown("**Drill into a scenario**")
    selected_scenario = st.selectbox(
        "Select scenario for daily return breakdown",
        options=list(historical_stress.keys()),
        format_func=lambda k: historical_stress[k].scenario_name
    )

    selected = historical_stress[selected_scenario]

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric(
            "Total Portfolio Return",
            f"{selected.portfolio_return_pct:.2%}",
            delta=f"${selected.portfolio_loss_dollar:,.0f} loss" if selected.portfolio_loss_dollar > 0 else f"${abs(selected.portfolio_loss_dollar):,.0f} gain"
        )
    with col_b:
        st.metric(
            "Worst Single Day",
            f"{selected.worst_single_day_pct:.2%}",
            delta=selected.worst_single_day_date,
            delta_color="off"
        )
    with col_c:
        st.metric(
            "Trading Days in Scenario",
            selected.n_trading_days
        )

    st.caption(selected.description)

    # Daily return bar chart for selected scenario
    fig_daily = go.Figure(go.Bar(
        x=selected.daily_returns.index,
        y=selected.daily_returns.values,
        marker_color=[
            "#ef4444" if r < 0 else "#10b981"
            for r in selected.daily_returns.values
        ],
        name="Daily Return"
    ))
    fig_daily.update_layout(
        height=280,
        title=f"Daily Returns — {selected.scenario_name}",
        xaxis_title="Date",
        yaxis_title="Daily Return",
        yaxis=dict(tickformat=".1%"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"),
        margin=dict(t=40, b=40)
    )
    st.plotly_chart(fig_daily, use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------------------------
# Section 6 — Portfolio statistics
# ---------------------------------------------------------------------------

with st.expander("📊 Full Portfolio Statistics", expanded=False):
    col1, col2 = st.columns(2)

    stat_items = [
        ("Mean Daily Return", f"{stats['mean_daily_return']:.4%}"),
        ("Std Dev (Daily)", f"{stats['std_daily_return']:.4%}"),
        ("Annualized Return", f"{stats['annualized_return']:.2%}"),
        ("Annualized Volatility", f"{stats['annualized_volatility']:.2%}"),
        ("Sharpe Ratio", f"{stats['sharpe_ratio']:.3f}"),
        ("Skewness", f"{stats['skewness']:.4f}"),
        ("Excess Kurtosis", f"{stats['kurtosis']:.4f}"),
        ("Trading Days", f"{stats['total_trading_days']:,}"),
        ("Data Start", stats['date_range'][0]),
        ("Data End", stats['date_range'][1]),
    ]

    for i, (label, value) in enumerate(stat_items):
        target = col1 if i % 2 == 0 else col2
        target.metric(label, value)

    # Skewness and kurtosis interpretation
    skew = stats['skewness']
    kurt = stats['kurtosis']

    st.markdown("**Distribution Shape Interpretation**")
    if skew < -0.5:
        st.warning(
            f"Skewness: {skew:.3f} — Negatively skewed. "
            "Losses are more extreme than gains of the same frequency. "
            "This is typical of equity portfolios — crashes are sharper than rallies."
        )
    elif skew > 0.5:
        st.info(
            f"Skewness: {skew:.3f} — Positively skewed. "
            "Gains tend to be more extreme than losses."
        )
    else:
        st.info(f"Skewness: {skew:.3f} — Approximately symmetric.")

    if kurt > 1.0:
        st.warning(
            f"Excess Kurtosis: {kurt:.3f} — Fat tails present. "
            "Extreme daily moves (both gains and losses) occur more frequently "
            "than a normal distribution predicts. This is why Monte Carlo VaR "
            "based on a normal distribution underestimates true tail risk."
        )
    else:
        st.info(
            f"Excess Kurtosis: {kurt:.3f} — "
            "Tails are close to normally distributed."
        )

st.markdown("---")
st.caption(
    "Portfolio VaR Engine · "
    "Data: Yahoo Finance · "
    f"Last computed: {datetime.now().strftime('%Y-%m-%d %H:%M')} · "
    "For educational and research purposes only"
)