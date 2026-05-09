# Portfolio VaR Engine

A production-grade portfolio risk engine that computes **Value at Risk (VaR)** and **Conditional VaR (Expected Shortfall)** using two independent methods — Historical Simulation and Monte Carlo — and stress tests the portfolio against three historical market crises.

Built to mirror the risk infrastructure used at financial institutions operating under Basel III regulatory frameworks.

---

## What This Project Does

Most risk tutorials compute a single VaR number and stop. This project goes further:

- Computes VaR two ways and surfaces the **gap between methods as a model risk metric**
- Tests the normality assumption underlying Monte Carlo and flags violations automatically
- Applies **real historical crisis data** (not hypothetical shocks) to measure portfolio behavior under extreme conditions
- Generates a complete **model risk documentation** section consistent with SR 11-7 requirements
- Delivers everything through both a **command-line pipeline** and an **interactive Streamlit dashboard**

---

## Key Findings (JPM / GS / MSFT — Equal Weight — $1M Portfolio)

| Metric | Value |
|---|---|
| Annualized Return | 9.86% |
| Annualized Volatility | 28.64% |
| Sharpe Ratio | 0.34 |
| Excess Kurtosis | 11.51 |
| 1-day 95% VaR (Historical) | $26,682 |
| 1-day 95% VaR (Monte Carlo) | $29,811 |
| Model Risk Gap (VaR) | $3,129 |
| Historical CVaR | $42,279 |
| Monte Carlo CVaR | $37,356 |
| CVaR Gap | $4,923 |

**The CVaR gap is the most important finding.** Historical CVaR exceeds Monte Carlo CVaR by $4,923 — direct empirical evidence that the normal distribution assumption underestimates true tail risk. Excess kurtosis of 11.51 confirms extreme fat tails in the actual return distribution.

---

## Stress Test Results

| Scenario | Period | Trading Days | Portfolio Loss |
|---|---|---|---|
| Dot-Com Crash | Mar 2000 – Oct 2002 | 670 | -53.47% ($534,660) |
| COVID-19 Crash | Feb 20 – Mar 23 2020 | 22 | -40.63% ($406,265) |
| Global Financial Crisis | Sep – Nov 2008 | 62 | -40.29% ($402,895) |

**Key observation:** The COVID crash delivered a 40% loss in 22 trading days. The daily 95% VaR at the time was ~$27,000. The crisis loss was approximately **15x the daily VaR estimate** — illustrating precisely why stress testing is required alongside VaR under Basel III.

---

## Project Structure

```
portfolio-var-engine/
│
├── src/
│   ├── data_loader.py           # Fetches and caches price data from Yahoo Finance
│   ├── portfolio.py             # Portfolio construction and weighted return computation
│   ├── historical_simulation.py # Historical Simulation VaR and CVaR
│   ├── monte_carlo.py           # Monte Carlo VaR with normality testing
│   ├── stress_testing.py        # Crisis scenario stress tests
│   └── report.py                # CLI pipeline — full report from command line
│
├── dashboard/
│   └── app.py                   # Streamlit interactive risk dashboard
│
├── tests/
│   ├── test_historical.py
│   └── test_monte_carlo.py
│
├── data/                        # Cached price data (gitignored)
├── reports/                     # Generated report output (gitignored)
├── requirements.txt
└── README.md
```

---

## Methodology

### Historical Simulation
Uses actual historical daily portfolio returns as the scenario set. No distributional assumptions. VaR is read directly from the empirical return distribution at the chosen confidence threshold.

**Strengths:** Captures real fat tails and skewness. No parametric assumptions.

**Limitations:** Equally weights all historical days. Cannot extrapolate beyond the worst observed historical return. Entirely backward-looking.

### Monte Carlo Simulation
Fits a normal distribution to historical returns (mean, standard deviation) and generates 10,000+ hypothetical future scenarios. VaR and CVaR are computed from the simulated distribution.

**Strengths:** Can generate scenarios more extreme than observed history. Flexible distributional assumptions.

**Limitations:** Results depend entirely on the distributional assumption. Standard implementation assumes normally distributed returns — real financial returns have fat tails (excess kurtosis) that a normal distribution systematically underestimates. The Jarque-Bera normality test fires automatically when this assumption is violated.

### Method Comparison as Model Risk
The gap between Historical and Monte Carlo VaR is surfaced explicitly as a **model risk metric** — the sensitivity of the risk estimate to the distributional assumption. This is consistent with SR 11-7 model risk management guidance, which requires banks to document model limitations and quantify the impact of key assumptions.

### Stress Testing
Applies actual daily market returns from three historical crisis windows directly to the current portfolio. No distribution assumptions — real market moves, real sequencing, real compounding.

Scenarios:
- **Global Financial Crisis (2008):** Sep–Nov 2008. Lehman Brothers bankruptcy. Correlation breakdown across asset classes.
- **COVID-19 Crash (2020):** Feb 20–Mar 23 2020. Fastest bear market in history. VIX hit 82.69.
- **Dot-Com Crash (2000–2002):** Mar 2000–Oct 2002. NASDAQ fell 78% peak-to-trough.

---

## Square Root of Time Rule

Multi-day VaR is computed by scaling 1-day VaR by √(holding period). This is the regulatory standard under Basel III but assumes return independence across days. Volatility clustering in real markets (bad days cluster near other bad days) means multi-day VaR is underestimated during crises — a documented limitation of this approach.

---

## Installation

```bash
git clone https://github.com/anantharamchandana/portfolio-var-engine.git
cd portfolio-var-engine

python -m venv venv
source venv/bin/activate  # Mac/Linux

pip install -r requirements.txt
```

---

## Usage

### Run the full pipeline from command line

```bash
# Default portfolio: JPM, GS, MSFT — equal weight — $1M — 95% confidence
python src/report.py

# Custom portfolio
python src/report.py --tickers AAPL MSFT GOOGL --weights 0.4 0.4 0.2

# Save report to file
python src/report.py --tickers JPM GS --weights 0.6 0.4 --value 5000000 --save

# Basel III regulatory standard: 99% confidence, 10-day holding period
python src/report.py --confidence 0.99 --holding-period 10 --save
```

### Launch the interactive dashboard

```bash
streamlit run dashboard/app.py
```

Opens at `http://localhost:8501`. Configure portfolio, confidence level, holding period, and simulation count from the sidebar. All outputs update dynamically.

---

## Regulatory Context

This project is built around the risk measurement concepts required under **Basel III** — the international banking regulatory framework established after the 2008 financial crisis:

- **VaR** is the primary market risk measure under Basel's Internal Models Approach
- **CVaR (Expected Shortfall)** is required under Basel III's Fundamental Review of the Trading Book (FRTB) as a replacement for VaR in regulatory capital calculations
- **Stress testing** is mandatory under Basel III alongside VaR, specifically because VaR underestimates crisis losses
- **Model risk documentation** consistent with the Federal Reserve's **SR 11-7** guidance — covering methodology, assumptions, limitations, and failure modes

---

## Model Limitations

1. **Historical simulation equally weights all observations.** A return from 2001 counts identically to last week. Exponentially Weighted Historical Simulation (EWHS) would address this.

2. **Monte Carlo assumes normally distributed returns.** Student-t or GARCH-based simulation would better capture fat tails and volatility clustering.

3. **Square root of time scaling assumes return independence.** Violated during crises when volatility clusters. Multi-day VaR is underestimated during stress periods.

4. **Long-only portfolio only.** Options, short positions, and non-linear instruments require full revaluation rather than return-based approximation.

---

## Tech Stack

- **Python 3.10+**
- **pandas / numpy** — data manipulation and numerical computation
- **yfinance** — market data ingestion
- **scipy** — statistical testing (Jarque-Bera normality test)
- **matplotlib / seaborn** — static visualizations
- **plotly** — interactive dashboard charts
- **streamlit** — dashboard framework

---

## Data Source

Historical price data fetched from **Yahoo Finance** via the `yfinance` library. Adjusted closing prices used throughout (accounts for splits and dividends). Data cached locally after first fetch to avoid repeated API calls.

---

*Built as a finance domain project to demonstrate risk infrastructure engineering skills relevant to quantitative risk technology roles.*
