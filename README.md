> **Language / Langue :** **English** | [Passer à la version française (README_FR.md)](README_FR.md)

---

# Multi-Objective Portfolio Optimization Engine & Systematic Backtesting

**Author :** Robin Leblanc

**Stack:** Python 3.8+ | NumPy | Numba JIT | SciPy | Scikit-Learn | Pandas | QuantStats | Pytest | Pydantic

---

## 1. Overview & Architecture

This project implements a quantitative asset allocation and Out-of-Sample backtesting engine combining a multi-objective genetic metaheuristic compiled to machine code via **Numba LLVM (`@njit`)**, classical convex optimization benchmarks (Newton-Raphson, SLSQP), and an asynchronous LLM agent layer for macroeconomic regime detection and sentiment analysis.

## Backtest Reports (QuantStats Tearsheets)

Direct access to interactive QuantStats tearsheets via GitHub Pages:

| Scenario / Universe | Time Horizon | Key Focus | Interactive Report |
| :--- | :--- | :--- | :--- |
| **US MegaTech** | 2015 – 2023 | Tech momentum & high concentration | [View Tearsheet ↗](https://rleblanc92.github.io/portfolio-optimizer/tearsheet_US_MegaTech_2015_2023.html) |
| **Covid Crash & Recovery** | 2020 – 2021 | Extreme volatility & sharp drawdown shock | [View Tearsheet ↗](https://rleblanc92.github.io/portfolio-optimizer/tearsheet_Covid_2020_2021.html) |
| **Global Financial Crisis (GFC)** | 2007 – 2009 | Prolonged bear market & tail-risk behavior | [View Tearsheet ↗](https://rleblanc92.github.io/portfolio-optimizer/tearsheet_GFC_2007_2009.html) |
| **Global Multi-Asset** | 2024 | Cross-asset diversification & macro regime | [View Tearsheet ↗](https://rleblanc92.github.io/portfolio-optimizer/tearsheet_MultiAsset_2024.html) |


Designed to replicate real-world trading desk execution constraints, the engine explicitly models:

* **Daily weight drift** between rebalancing dates.
* **Linear transaction costs (L1) and quadratic permanent market impact (L2)**.
* **Free cash yield** at the evolving risk-free rate and **margin financing drag** for short positions and leverage.
* **Higher-order moment control** (skewness and excess kurtosis) and covariance matrix regularization via **Ledoit-Wolf shrinkage**.

```text
Optimisation de portefeuille AG/
├── config.py                        # Centralized hyperparameters and environment variables
├── market_data.py                   # Data ingestion, arithmetic returns, EMA 40/200 & Ledoit-Wolf
├── classic_optimization.py          # Convex benchmarks (Minimum Variance Newton-Raphson & Max Sharpe SLSQP)
├── main.py                          # Single backtest orchestrator (Rolling Window IS/OOS)
├── general_tests.py                 # Multi-scenario parallel orchestrator (spawn) & QuantStats reporting
├── genetic_algorithm/               # Vectorized evolution core compiled via Numba (@njit)
│   ├── __init__.py
│   ├── evolution.py                 # Evolutionary loop, in-place Fisher-Yates tournament & hyper-mutation
│   ├── crossover.py                 # Vectorized continuous BLX-α crossover
│   ├── mutation.py                  # In-place mass-transfer bump mutation
│   ├── repair.py                    # Bound projection and gross exposure management
│   └── fitness.py                   # 4th-order objective function (Variance, Skewness, Kurtosis, L1/L2 Costs)
├── llm_agents/                      # Semantic and macro layer with strict Pydantic validation
│   ├── __init__.py
│   ├── macro_agent.py               # Market regime classification and dynamic lambda_risk scheduling
│   └── sentiment_agent.py           # Financial news semantic scoring and return expectation modulation
├── tests/
│   └── test_backtest_integration.py # Mathematical audit and system integration test suite (Pytest)
├── outputs/
│   ├── tearsheets/                  # QuantStats Out-of-Sample HTML reports per scenario
│   ├── logs/                        # Deduplicated execution logs (execution_debug.log)
│   └── rebalance_history.parquet    # Detailed log of weights, regimes, transaction fees, and turnover
├── requirements.txt
├── setup.sh / setup.bat
└── README.md

```

---

## 2. Mathematical Modeling & Quantitative Engineering

### 2.1 Return & Covariance Estimators (`market_data.py`)

  * **Cross-Sectional Additivity:** Exclusive use of simple arithmetic returns $R_{i,t} = \frac{P_{i,t} - P_{i,t-1}}{P_{i,t-1}}$ to ensure linear consistency across portfolio weights: $R_{p,t} = \sum_{i=1}^N w_i R_{i,t}$.
  * **Expected Return ($\hat{\mu}$) via EMA 40/200 Trend Filtering:**
  * Smoothing factor: $\alpha = \frac{2}{N+1}$ with a fast window of $40$ days and a slow window of $200$ days (effective phase lag: $\frac{200 - 40}{2} = 80$ days).
  * Daily and annualized estimators:

$$\hat{\mu}_{i,t}^{\text{daily}} = \frac{1}{80}\left(\frac{\text{EMA}_{40}(P_{i,t})}{\text{EMA}_{200}(P_{i,t})} - 1\right), \quad \hat{\mu}_{i,t}^{\text{ann}} = 252 \cdot \hat{\mu}_{i,t}^{\text{daily}}$$




  * **Regularized Covariance Matrix ($\Sigma_{\text{LW}}$):**
  * Estimated via analytical **Ledoit-Wolf shrinkage** (`scikit-learn`) to eliminate finite-sample ill-conditioning ($T = 252$), annualized as $\Sigma_{\text{ann}} = 252 \cdot \Sigma_{\text{daily}}$.
  * Exponential time-weighting shared with fitness moments via a configurable half-life `TIME_WEIGHT_HALFLIFE` (default: $63$ trading days, i.e., one business quarter).



### 2.2 Multi-Moment Objective Function & Market Frictions (`fitness.py`)

For a weight vector $w \in \mathbb{R}^N$, let:

* $L = \sum_{i=1}^N \max(w_i, 0)$ be the long exposure and $S = \sum_{i=1}^N \max(-w_i, 0)$ the short exposure.
* $c = \max(0, 1 - L)$ be the unallocated free cash earning the annualized risk-free rate $R_f$.
* $F = \max(0, L - 1) + S$ be the funded exposure subject to the margin borrowing rate $r_{\text{margin}}$.
* $E = \min(1, L + S)$ be the risky exposure scaling factor.
* $T = \sum_{i=1}^N \vert{}w_i^{(t)} - w_i^{(t-1)}\vert{}$ be the two-way turnover relative to the post-drift portfolio from the previous period.

The exact utility evaluated for each individual across the population is defined as:

$$\text{Fitness}(w) = w^\top \hat{\mu} + c R_f - F \, r_{\text{margin}} - \lambda_{\text{risk}} \, w^\top \Sigma_{\text{LW}} w + \lambda_{\text{skew}} E \, S_3(w) - \lambda_{\text{kurt}} E \, K_4(w) - \lambda_{\text{TO}} \left(T + 0.1 T^2\right)$$

where:

* **Annualized Variance:** $w^\top \Sigma_{\text{LW}} w$ (or empirical variance of daily portfolio returns scaled by $252$ if no precomputed matrix is provided).
* **Standardized Skewness:** $S_3(w) = \frac{M_3(r_p)}{\sigma_p^3}$ weighted by $\lambda_{\text{skew}} = 0.05$ to reward positive upside asymmetry.
* **Excess Kurtosis:** $K_4(w) = \frac{M_4(r_p)}{\sigma_p^4} - 3$ weighted by $\lambda_{\text{kurt}} = 0.01$ to penalize fat-tail risk.
* **L1 Turnover Penalty + L2 Quadratic Impact:** $\lambda_{\text{TO}} = \text{TRANSACTION\_FEE} \times \frac{252}{\text{REBALANCE\_FREQ}}$ captures both proportional execution fees ($T$) and non-linear permanent market impact ($0.1 T^2$).

---

## 3. Low-Level Systems Engineering & Numba Core (`genetic_algorithm/`)

The entire evolutionary loop is compiled to machine code via `@njit(cache=True)`, completely avoiding heap allocations inside performance-critical paths:

1. **Hybrid Initialization & Sparsity Mask (`evolution.py`):**
  * Uniform generation across $[w_{\min}, w_{\max}]$ (`POPULATION_SIZE = 800`).
  * For universes with $N > 4$, a sparsity mask zeros out $60\%$ of genes in the second half of the initial population to accelerate sparse, concentrated portfolio discovery.


2. **Allocation-Free Tournament Selection (`_tournament_selection`):**
  * Employs an **in-place partial Fisher-Yates shuffle** ($k = 3$) on a pre-allocated index buffer, followed by symmetrical swap reversals upon tournament completion. Yields $\mathcal{O}(k)$ time complexity and $\mathcal{O}(1)$ auxiliary memory overhead.


3. **Continuous BLX-$\alpha$ Crossover (`crossover.py`):**
  * Dual independent parent permutations. For each asset, offspring weights are drawn uniformly from $[C_{\min} - \alpha \Delta, \, C_{\max} + \alpha \Delta]$ where $\alpha = 0.5$ and $\Delta = \vert{}w_i^{(1)} - w_i^{(2)}\vert{}$.


4. **In-Place Mass-Transfer Mutation (`mutation.py`):**
  * The `bump_mutation_inplace` operator transfers a stochastic mass $\delta \sim \mathcal{U}(0.002, 0.05)$ from asset $j$ to asset $i$ ($w_i \leftarrow w_i + \delta$, $w_j \leftarrow w_j - \delta$) conditional on bound constraints $[w_{\min}, w_{\max}]$, preserving total exposure without destructive normalization.


5. **Repair Operator & Gross Exposure Constraint (`repair.py`):**
  * Hard clipping onto $[w_{\min}, w_{\max}]$.
  * Gross exposure enforcement: if $\sum_i \vert{}w_i\vert{} > \text{Gross Exposure Target}$ ($1.0$ by default), weights undergo a homothetic contraction: $w_i \leftarrow w_i \cdot \frac{\text{Target}}{\sum_j \vert{}w_j\vert{}}$.
  * In Long-Only setups, if $\sum_i w_i < 1.0$, residual capital $1 - \sum_i w_i$ is systematically assigned to the risk-free cash asset $R_f$ (guaranteed convergence in $< 5$ iterations).


6. **Convergence, Elitism & Adaptive Hyper-Mutation (`evolution.py`):**
  * Direct elitist preservation of the top $5\%$ individuals (`elite_ratio = 0.05`).
  * Early stopping triggered when fitness gains plateau below $\Delta_{\min} = 10^{-7}$ over `patience = 200` generations.
  * Adaptive **hyper-mutation** ($\text{rate} \times 2.5$, capped at $0.50$) engaged after $50$ stagnant generations to break out of non-convex local extrema.



---

## 4. Multi-Agent LLM Layer & Schema Validation (`llm_agents/`)

Upstream of numerical optimization, two agents query large language models (Gemini 1.5 Flash / GPT-4o-mini) with strict output validation via **Pydantic** and deterministic fallback routines during network outages or rate limiting:

* **Macroeconomic Agent (`macro_agent.py`):** Ingests rolling In-Sample metrics (realized volatility, drawdown, cumulative returns) to classify market conditions into 4 regimes (`Bull`, `Bear`, `Stagflation`, `High Vol Recovery`) and dynamically modulate the risk aversion scalar $\lambda_{\text{risk}} \in [0.5, 2.5]$.
* **Sentiment Agent (`sentiment_agent.py`):** Parses raw financial news per asset to output a bounded semantic score $s_i \in [-1.0, 1.0]$, shifting expected returns:

$$\hat{\mu}_{i,\text{adjusted}} = \hat{\mu}_i + \left(\beta_{\text{sentiment}} \cdot s_i \cdot \vert{}\hat{\mu}_i\vert{}\right)$$



A persistent SQLite store combined with an in-memory `_SENTIMENT_CACHE` keyed by `(ticker, date, api_key, use_llm, provider)` ensures backtest idempotency and eliminates redundant API calls.

---

## 5. Backtesting Protocol, Parallel Orchestration & Audit Suite

### 5.1 Rolling Window & Accounting Realism

* **In-Sample Period (`LOOKBACK_WINDOW = 252` days):** Parameter estimation for $\hat{\mu}$, $\Sigma_{\text{LW}}$, regime classification, and simultaneous model optimization (Multi-Objective GA, Minimum Variance Newton-Raphson, Maximum Sharpe SLSQP, and equal-weight $1/N$ benchmark).
* **Out-of-Sample Period (`REBALANCE_FREQ = 21` days):** Application of frozen optimal weights over the subsequent test window. Between rebalancing events, portfolio weights drift daily according to realized asset returns, with daily deductions for margin financing and cash interest accruals. At each rebalancing boundary, transaction fees directly deduct from NAV.
* **Traceability & Reporting (`general_tests.py`):** Multi-process execution (`multiprocessing` via `spawn`) with asynchronous log collection through a `QueueListener` routed to `outputs/logs/execution_debug.log`, full state exports to `outputs/rebalance_history.parquet` (with automatic `.csv` fallback), and **QuantStats** HTML tearsheet generation in `outputs/tearsheets/`.

### 5.2 Mathematical & System Audit Suite (`tests/test_backtest_integration.py`)

The repository contains an isolated Pytest suite partitioned into interface contracts (`@pytest.mark.contract`) and quantitative vulnerability audits (`@pytest.mark.audit_flaw`):

* **Zero Off-by-One & Look-Ahead Bias:** Verifies that the initial Out-of-Sample evaluation index aligns strictly with `dates[config.LOOKBACK_WINDOW]` with zero information leakage from future bars.
* **Strict OOS Risk-Free Rate Isolation:** Asserts that reported Sharpe/Sortino ratios and the QuantStats `rf` argument ingest solely the average risk-free rate across the Out-of-Sample slice (preventing contamination from In-Sample regimes).
* **Non-Linear Frictions Audit:** Validates that under zero asset returns and zero risk-free rates in OOS, any deviation from the base allocation strictly degrades NAV ($V_{\text{final}} < 1.0$) exclusively due to transaction fees and margin costs.
* **Matplotlib Figure Cleanup:** Enforces explicit memory deallocation (`plt.close('all')`) during consecutive QuantStats HTML tearsheet renderings.

---

## 6. Out-of-Sample Empirical Results & Quantitative Post-Mortem

Out-of-Sample performance evaluation (252-day basis, net of all transaction fees and margin borrowing costs) extracted from QuantStats tearsheets across 4 distinct market regimes:

| Scenario (OOS Period) | Risk-Free Rate (Avg. OOS $R_f$) | Cumulative Return (GA vs Bench $1/N$) | CAGR (GA vs Bench) | Sharpe Ratio (GA vs Bench) | Sortino Ratio (GA vs Bench) | Max Drawdown (GA vs Bench) | Ann. Volatility (GA vs Bench) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **`Global_MultiAsset_2024`** *(Jan 2025 – Sep 2026)* | 4.20% | **+31.58%** vs +25.95% | **17.91%** vs 14.91% | **1.36** vs 1.03 | **2.04** vs 1.46 | **-7.83%** vs -10.59% | **10.08%** vs 10.51% |
| **`Global_GFC_2007_2009`** *(Jul 2007 – Dec 2009)* | 3.60% | **+13.00%** vs -28.74% | **5.19%** vs -13.09% | **0.20** vs -0.43 | **0.29** vs -0.60 | **-22.22%** vs -54.99% | **13.57%** vs 30.52% |
| **`FR_Covid_2020_2021`** *(Dec 2019 – Dec 2021)* | 1.20% | +7.75% vs **+40.77%** | 3.89% vs **19.12%** | 0.22 vs **0.79** | 0.29 vs **1.08** | **-36.03%** vs -38.59% | **16.58%** vs 25.21% |
| **`US_MegaTech_2015_2023`** *(Jan 2016 – Dec 2023)* | 2.30% | +312.02% vs **+760.09%** | 19.58% vs **31.19%** | 0.72 vs **1.02** | 1.03 vs **1.50** | **-27.76%** vs -43.72% | **26.95%** vs 29.28% |

### Critical Regime Post-Mortem

1. **Defensive Convexity in Systemic Crises (`Global_GFC_2007_2009`):** During the 2008 liquidity unwind, regime detection coupled with fat-tail kurtosis penalization steered portfolio beta into negative territory (~$-0.45$), delivering **+17.15%** over calendar year 2008 (vs. **-38.14%** for the benchmark) and curbing Max Drawdown by over $32$ percentage points ($-22.22\%$ vs. $-54.99\%$).
2. **Multi-Asset Cross-Sectional Efficiency (`Global_MultiAsset_2024`):** In a positive interest rate regime ($R_f = 4.2\%$) across diversified global assets, Ledoit-Wolf shrinkage paired with quadratic turnover penalties consistently outperformed the naive benchmark on risk-adjusted grounds (Sharpe $1.36$ vs. $1.03$, Calmar $2.29$ vs. $1.41$).
3. **Estimation Lag During Exogenous V-Shaped Shocks (`FR_Covid_2020_2021`):** The sudden February–March 2020 crash unfolded faster than the historical lookback window could process. Once the shock entered the estimation window, elevated variance and kurtosis penalties curtailed risky exposure (annualized volatility cut to $16.58\%$ vs. $25.21\%$), blunting upside participation during the rapid Q2 2020 recovery.
4. **Return vs. Drawdown Trade-Off in Trend Regimes (`US_MegaTech_2015_2023`):** In an aggressive mega-cap bull run, asset weight caps (`MAX_WEIGHT`) and variance aversion clipped raw upside momentum relative to an equal-weighted basket. However, downside protection remained pronounced during the 2022 tech correction, restricting losses to **-22.72%** (vs. **-40.86%** for the benchmark) and reducing overall Max Drawdown from $-43.72\%$ to $-27.76\%$.

---

## 7. Installation & Usage

### Automated Installation (Linux / macOS / Windows)

```bash
# Linux / macOS
chmod +x setup.sh
./setup.sh

# Windows (CMD)
setup.bat

```

### Manual Installation

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate.bat
pip install --upgrade pip
pip install -r requirements.txt

```

### Running Backtests & the Audit Suite

```bash
# 1. Execute primary rolling backtest, generating QuantStats HTML tearsheets and Parquet logs
python main.py

# 2. Run the multi-scenario parallelized stress-test suite
python general_tests.py

# 3. Run the full quantitative audit and integration test suite
pytest tests/test_backtest_integration.py -v

```

### Configuration (`config.py`)

```python
# Genetic Algorithm Parameters (Numba JIT)
POPULATION_SIZE = 800          # Population size
GENERATIONS = 1000             # Maximum evolutionary generations
MUTATION_RATE = 0.10           # Base mutation rate (prior to adaptive hyper-mutation)

# Risk Preferences & Higher-Order Moments
RISK_AVERSION = 0.5            # Nominal Ledoit-Wolf variance penalty
SKEWNESS_PREFERENCE = 0.05     # Positive skewness reward (3rd moment)
KURTOSIS_AVERSION = 0.01       # Excess kurtosis penalty (4th moment)
TIME_WEIGHT_HALFLIFE = 63      # Half-life (days) for exponential time-weighting

# Exposure Constraints & Market Frictions
ALLOW_SHORT = True             # Enable short positions
MAX_WEIGHT = 0.25              # Maximum concentration limit per asset
TRANSACTION_FEE = 0.002        # L1 linear transaction cost (20 bps) + L2 quadratic impact

# Rolling Backtest Windows
LOOKBACK_WINDOW = 252          # In-Sample estimation window (1 trading year)
REBALANCE_FREQ = 21            # Out-of-Sample rebalancing frequency (1 trading month)

```

---

## 8. Academic References

1. **Ledoit, O., & Wolf, M. (2004).** *A well-conditioned estimator for large-dimensional covariance matrices.* Journal of Multivariate Analysis, 88(2), 365–411.
2. **Eshelman, L. J., & Schaffer, J. D. (1993) / Spears, W. M., & Anand, V. (1991).** *Real-Coded Genetic Algorithms and Interval-Schemata (BLX-$\alpha$ Crossover).*
3. **Harvey, C. R., Liechty, J. C., Liechty, M. W., & Müller, P. (2010).** *Portfolio selection with higher moments.* Quantitative Finance, 10(5), 469–485.
4. **Almgren, R., & Chriss, N. (2001).** *Optimal execution of portfolio transactions.* Journal of Risk, 3, 5–40.

---

## Author & License

**Author:** Robin Leblanc — Engineering Student at École Centrale de Lyon & B.Sc. in Economics and Management (Université Lumière Lyon 2).

**License:** Proprietary — All rights reserved.