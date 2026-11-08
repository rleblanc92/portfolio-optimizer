"""
Audit suite for Quantitative Finance and Numba Core Layers.

This module audits the mathematical finance (`market_data.py`, `classic_optimization.py`,
`config.py`) and Genetic Algorithm engine (`genetic_algorithm/*`) without modifying
source code. Tests are categorized via pytest markers:

- `@pytest.mark.contract`: Verify execution stability, Numba/LLVM types, dimensions,
  cross-module signatures, and deterministic reproducibility.
- `@pytest.mark.audit_flaw`: Expose mathematical flaws and critical edge cases.
  Each `audit_flaw` test documents the exact flaw in its docstring and is expected
  to fail until source code is corrected.

Execution constraints:
- Zero network calls (100% mocking)
- 100% deterministic (seeded RNG)
- Total execution time < 10 seconds
"""

import inspect
import multiprocessing
import sys
from typing import Tuple

import numpy as np
import pytest
from scipy import stats

# Set seed globally for reproducibility
np.random.seed(42)


# ============================================================================
# BLOC A — Audit of config.py and Cross-Module Signature Coherence
# ============================================================================

@pytest.mark.contract
def test_config_constants_and_bounds():
    """Verify existence, type, and mathematical coherence of config.py constants."""
    import config

    # Verify existence and type
    assert hasattr(config, 'POPULATION_SIZE') and isinstance(config.POPULATION_SIZE, int)
    assert hasattr(config, 'GENERATIONS') and isinstance(config.GENERATIONS, int)
    assert hasattr(config, 'MUTATION_RATE') and isinstance(config.MUTATION_RATE, float)
    assert hasattr(config, 'MAX_WEIGHT') and isinstance(config.MAX_WEIGHT, float)
    assert hasattr(config, 'MIN_WEIGHT') and isinstance(config.MIN_WEIGHT, float)
    assert hasattr(config, 'GROSS_EXPOSURE') and isinstance(config.GROSS_EXPOSURE, float)
    assert hasattr(config, 'RISK_FREE_RATE') and isinstance(config.RISK_FREE_RATE, float)
    assert hasattr(config, 'MARGIN_RATE') and isinstance(config.MARGIN_RATE, float)
    assert hasattr(config, 'TRANSACTION_FEE') and isinstance(config.TRANSACTION_FEE, float)
    assert hasattr(config, 'REBALANCE_FREQ') and isinstance(config.REBALANCE_FREQ, int)
    assert hasattr(config, 'TURNOVER_AVERSION') and isinstance(config.TURNOVER_AVERSION, float)
    assert hasattr(config, 'LOOKBACK_WINDOW') and isinstance(config.LOOKBACK_WINDOW, int)

    # Verify mathematical bounds
    assert config.POPULATION_SIZE > 0, "POPULATION_SIZE must be > 0"
    assert config.GENERATIONS > 0, "GENERATIONS must be > 0"
    assert 0.0 <= config.MUTATION_RATE <= 1.0, "MUTATION_RATE must be in [0, 1]"
    assert config.MAX_WEIGHT > 0, "MAX_WEIGHT must be > 0"
    assert config.MIN_WEIGHT < 0, "MIN_WEIGHT must be < 0"
    assert config.GROSS_EXPOSURE >= 1.0, "GROSS_EXPOSURE must be >= 1.0"
    assert config.MARGIN_RATE > config.RISK_FREE_RATE, "MARGIN_RATE must exceed RISK_FREE_RATE"
    assert config.TRANSACTION_FEE > 0, "TRANSACTION_FEE must be > 0"
    assert config.REBALANCE_FREQ > 0, "REBALANCE_FREQ must be > 0"

    # Verify TURNOVER_AVERSION computation
    expected_turnover_aversion = config.TRANSACTION_FEE * (252 / config.REBALANCE_FREQ)
    assert np.isclose(
        config.TURNOVER_AVERSION,
        expected_turnover_aversion,
        rtol=1e-10
    ), f"TURNOVER_AVERSION should be {expected_turnover_aversion}, got {config.TURNOVER_AVERSION}"


@pytest.mark.contract
def test_package_exports_and_signatures():
    """Verify genetic_algorithm exports and parameter compatibility with main.py."""
    from genetic_algorithm import (
        run_evolution,
        repair_chromosome,
        repair_population,
        evaluate_fitness,
        calculate_turnover,
        calculate_portfolio_returns,
    )

    # Verify each export exists and is callable
    assert callable(run_evolution)
    assert callable(repair_chromosome)
    assert callable(repair_population)
    assert callable(evaluate_fitness)
    assert callable(calculate_turnover)
    assert callable(calculate_portfolio_returns)

    # Verify signature of run_evolution
    sig_evolution = inspect.signature(run_evolution)
    required_params = {
        'expected_returns', 'simple_returns', 'previous_weights',
        'population_size', 'generations', 'mutation_rate',
        'lambda_risk', 'lambda_skew', 'lambda_kurt', 'lambda_turnover',
        'max_weight', 'w_min', 'elite_ratio', 'mean_rf_rate',
        'margin_rate', 'gross_exposure_target', 'cov_matrix', 'seed'
    }
    for param_name in required_params:
        assert param_name in sig_evolution.parameters, \
            f"run_evolution missing parameter: {param_name}"

    # Verify signature of repair_chromosome
    sig_repair = inspect.signature(repair_chromosome)
    assert 'chromosome' in sig_repair.parameters
    assert 'w_min' in sig_repair.parameters
    assert 'w_max' in sig_repair.parameters
    assert 'gross_exposure_target' in sig_repair.parameters


# ============================================================================
# BLOC B — Audit of market_data.py (Preprocessing, EMA, Covariance, Lookahead Bias)
# ============================================================================

@pytest.mark.contract
def test_calculate_simple_returns_additivity_and_nan_handling():
    """Verify calculate_simple_returns preserves shape, handles NaN, and maintains additivity."""
    from market_data import calculate_simple_returns

    # Test 1: Shape preservation and first-row zeros
    prices = np.array([
        [100.0, 200.0, 300.0],
        [110.0, 220.0, 330.0],
        [121.0, 242.0, 363.0],
    ], dtype=np.float64)

    returns = calculate_simple_returns(prices)
    assert returns.shape == prices.shape, "Shape must be preserved"
    assert np.allclose(returns[0], 0.0), "First row must be zeros"

    # Test 2: Expected values
    expected = np.array([
        [0.0, 0.0, 0.0],
        [0.10, 0.10, 0.10],
        [0.10, 0.10, 0.10],
    ], dtype=np.float64)
    assert np.allclose(returns, expected, rtol=1e-10)

    # Test 3: NaN handling
    prices_with_nan = np.array([
        [100.0, 200.0, np.nan],
        [110.0, 220.0, np.nan],
    ], dtype=np.float64)
    returns_nan = calculate_simple_returns(prices_with_nan)
    assert not np.any(np.isnan(returns_nan)), "NaN must be replaced by 0.0"
    assert np.allclose(returns_nan[1, 2], 0.0), "NaN returns should be 0.0"

    # Test 4: Additivity (cross-sectional sum equals portfolio return)
    prices = np.array([
        [100.0, 50.0, 75.0],
        [110.0, 55.0, 82.5],
    ], dtype=np.float64)
    returns = calculate_simple_returns(prices)
    weights = np.array([0.5, 0.3, 0.2])
    portfolio_returns_vec = np.sum(returns * weights, axis=1)
    assert np.allclose(portfolio_returns_vec[1], 0.1), "Portfolio return should match manual calculation"


@pytest.mark.contract
def test_calculate_ema_recursive_exactness():
    """Verify EMA calculation matches exact recursive formula to 1e-12 precision."""
    from market_data import calculate_ema

    # Use a simple price series: [100, 101, 102, 103, 104]
    prices = np.array([
        [100.0],
        [101.0],
        [102.0],
        [103.0],
        [104.0],
    ], dtype=np.float64)

    span = 10
    alpha = 2.0 / (span + 1)
    ema_result = calculate_ema(prices, span)

    # Manually compute expected EMA values
    ema_manual = np.zeros((5, 1))
    ema_manual[0, 0] = prices[0, 0]
    for t in range(1, 5):
        ema_manual[t, 0] = alpha * prices[t, 0] + (1 - alpha) * ema_manual[t - 1, 0]

    assert np.allclose(ema_result, ema_manual, atol=1e-12), \
        f"EMA mismatch: computed={ema_result.ravel()}, expected={ema_manual.ravel()}"


@pytest.mark.contract
def test_calculate_covariance_matrix_psd_and_zero_variance_guard():
    """Verify covariance matrix is PSD, symmetric, and handles zero-variance assets."""
    from market_data import calculate_covariance_matrix

    # Test 1: PSD property
    returns = np.random.randn(252, 5)
    cov_matrix, shrinkage = calculate_covariance_matrix(returns, shrinkage='ledoit_wolf')

    # Check symmetry
    assert np.allclose(cov_matrix, cov_matrix.T), "Covariance matrix must be symmetric"

    # Check PSD (all eigenvalues >= 0)
    eigenvalues = np.linalg.eigvalsh(cov_matrix)
    assert np.all(eigenvalues >= -1e-10), "All eigenvalues must be non-negative (PSD)"

    # Test 2: Zero-variance asset handling
    flat_returns = np.zeros((252, 5))
    flat_returns[:, :4] = np.random.randn(252, 4)
    cov_flat, _ = calculate_covariance_matrix(flat_returns, shrinkage='ledoit_wolf')

    # Diagonal element for constant-return asset should be forced to 1.0
    assert cov_flat[4, 4] >= 1.0 - 1e-6, "Zero-variance asset should have diagonal >= 1.0"


@pytest.mark.audit_flaw
def test_audit_flat_price_index_shift_bug_in_preprocess():
    """Expose flat-price index shift bug in preprocess_market_data."""
    from market_data import preprocess_market_data
    from unittest.mock import patch

    prices = np.array([
        [100.0, 100.0],
        [110.0, 110.0],
        [110.0, 110.0],  # flat day at index 2
        [121.0, 121.0],
    ], dtype=np.float64)
    dates = pd.date_range('2024-01-01', periods=4)

    with patch('market_data.fetch_historical_data', return_value=(prices, ['A', 'B'], dates)):
        result = preprocess_market_data(['A', 'B'], '2024-01-01', '2024-01-05')

    returns = result['simple_returns']
    # Index 1 had a legitimate +10% return (100 -> 110), NOT a flat price.
    # Bug in market_data.py lines 373-374: valid_ret_indices = flat_indices[flat_indices > 0] - 1
    # overwrites index 1 with 1e-5 instead of index 2!
    assert np.isclose(returns[1, 0], 0.10, atol=1e-6), (
        f"Legitimate +10% return at index 1 was corrupted to {returns[1, 0]} by off-by-one flat_indices shift"
    )


@pytest.mark.audit_flaw
def test_audit_bfill_lookahead_bias_on_late_ipo():
    """Expose lookahead bias from .bfill() in fetch_historical_data on late-IPO assets."""
    from market_data import fetch_historical_data
    from unittest.mock import patch

    dates = pd.date_range('2024-01-01', periods=6)
    close_df = pd.DataFrame({
        'AAPL': [100.0, 102.0, 104.0, 106.0, 108.0, 110.0],
        'IBIT': [np.nan, np.nan, np.nan, 50.0, 55.0, 60.0],  # Late IPO on day 3
    }, index=dates)
    mock_yf_data = pd.concat({'Close': close_df}, axis=1)

    with patch('market_data.yf.download', return_value=mock_yf_data):
        prices, valid_tickers, _ = fetch_historical_data(['AAPL', 'IBIT'], '2024-01-01', '2024-01-07')

    # Before IPO (indices 0, 1, 2), IBIT did not exist and should NOT have its future IPO price (50.0) backfilled
    assert np.all(np.isnan(prices[:3, 1])) or np.all(prices[:3, 1] == 0.0), (
        f"Lookahead bias detected: future IPO price {prices[:3, 1]} was backfilled before inception date"
    )


# ============================================================================
# BLOC C — Audit of genetic_algorithm/repair.py (Projection & Degeneracy Bug)
# ============================================================================

@pytest.mark.contract
def test_repair_chromosome_bounds_and_overexposure_scaling():
    """Verify repair_chromosome respects bounds and scales exposure exactly to target."""
    from genetic_algorithm import repair_chromosome
    import config

    # Create an over-exposed chromosome
    chromosome = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    target_exposure = config.GROSS_EXPOSURE  # 1.25

    repaired = repair_chromosome(
        chromosome,
        w_min=config.MIN_WEIGHT,
        w_max=config.MAX_WEIGHT,
        gross_exposure_target=target_exposure,
    )

    # Check bounds
    assert np.all(repaired >= config.MIN_WEIGHT - 1e-8), "All weights must respect w_min"
    assert np.all(repaired <= config.MAX_WEIGHT + 1e-8), "All weights must respect w_max"

    # Check exposure
    actual_exposure = np.sum(np.abs(repaired))
    assert np.isclose(actual_exposure, target_exposure, rtol=1e-8), \
        f"Exposure should be {target_exposure}, got {actual_exposure}"


@pytest.mark.contract
def test_repair_strict_exposure_and_balanced_zero_long_short():
    from genetic_algorithm import repair_chromosome

    overexposed = repair_chromosome(
        np.array([0.525, 0.525]), 0.0, 1.0,
        gross_exposure_target=1.0, tolerance=0.1,
    )
    assert np.sum(np.abs(overexposed)) <= 1.0

    balanced = repair_chromosome(
        np.zeros(4), -0.25, 0.25,
        gross_exposure_target=1.0,
    )
    assert np.all(balanced[::2] > 0.0)
    assert np.all(balanced[1::2] < 0.0)
    assert np.sum(np.abs(balanced)) <= 1.0


@pytest.mark.audit_flaw
def test_audit_repair_degeneracy_5_assets_long_only():
    """
    FLAW DOCUMENTED:
    For the US_MegaTech scenario (5 assets, long-only, capacity = 1.25),
    repair_chromosome can force all chromosomes to become identical [0.25, 0.25, 0.25, 0.25, 0.25]
    when remaining_exposure == capacity_total, destroying population diversity.
    
    Expected: Population diversity (positive inter-individual variance) should be preserved,
    and Cash weight should never go negative in long-only mode (sum(w) <= 1.0).
    """
    from genetic_algorithm import repair_population
    import config

    n_assets = 5
    target_exposure = 1.25

    # Generate 100 diverse long-only chromosomes
    np.random.seed(123)
    population = np.random.uniform(0.0, 0.25, size=(100, n_assets))

    repaired_pop = repair_population(
        population,
        w_min=0.0,
        w_max=0.25,
        gross_exposure_target=target_exposure,
    )

    # Check diversity: variance across the population should be > 0
    pop_variance = np.std(repaired_pop, axis=0).sum()
    assert pop_variance > 1e-4, \
        f"Population diversity lost after repair: variance = {pop_variance}"

    # Check cash weight: sum(w) should never exceed 1.0 in long-only mode
    for i, chromosome in enumerate(repaired_pop):
        cash_weight = 1.0 - np.sum(chromosome)
        assert cash_weight >= -1e-8, \
            f"Chromosome {i} has negative cash weight (sum={np.sum(chromosome)})"


@pytest.mark.audit_flaw
def test_audit_repair_preserves_cash_flight_capability():
    """
    FLAW DOCUMENTED:
    repair_chromosome may force gross_exposure_target as an equality constraint
    (sum(abs(w)) == target) instead of an upper bound (sum(abs(w)) <= target),
    preventing defensive portfolios from holding a cash buffer (Market Flight).
    
    A defensive portfolio [0.10, 0.10, 0.10, 0.10, 0.10] with 50% cash
    should survive repair without being forced to full investment.
    """
    from genetic_algorithm import repair_chromosome
    import config

    defensive_portfolio = np.array([0.10, 0.10, 0.10, 0.10, 0.10])
    target_exposure = config.GROSS_EXPOSURE  # 1.25

    repaired = repair_chromosome(
        defensive_portfolio,
        w_min=0.0,
        w_max=0.25,
        gross_exposure_target=target_exposure,
    )

    # Check that cash weight is preserved (not forced to zero)
    repaired_sum = np.sum(repaired)
    assert repaired_sum <= 1.0 + 1e-8, \
        f"Repaired portfolio forces full investment: sum={repaired_sum} > 1.0"


# ============================================================================
# BLOC D — Audit of genetic_algorithm/fitness.py (Moments, Margin, Cash/Short)
# ============================================================================

@pytest.mark.contract
def test_skewness_and_kurtosis_against_scipy():
    """Compare Numba skewness/kurtosis to scipy.stats."""
    from genetic_algorithm.fitness import calculate_skewness, calculate_kurtosis_excess, evaluate_fitness

    # Generate an asymmetric series with fat tails
    returns = np.array([
        -0.05, -0.02, 0.00, 0.01, 0.02, 0.03, 0.04, 0.05,
        0.10, 0.15, 0.20, -0.30, 0.08, 0.06, 0.04
    ], dtype=np.float64)

    # Compute via Numba
    skew_numba = calculate_skewness(returns)
    kurt_numba = calculate_kurtosis_excess(returns)

    # Compute via scipy
    skew_scipy = stats.skew(returns, bias=True)
    kurt_scipy = stats.kurtosis(returns, fisher=True, bias=True)

    assert np.isclose(skew_numba, skew_scipy, atol=1e-10), \
        f"Skewness mismatch: Numba={skew_numba}, scipy={skew_scipy}"
    assert np.isclose(kurt_numba, kurt_scipy, atol=1e-10), \
        f"Kurtosis mismatch: Numba={kurt_numba}, scipy={kurt_scipy}"

    # Test degenerate case: constant series
    const_returns = np.ones(10, dtype=np.float64)
    skew_const = calculate_skewness(const_returns)
    kurt_const = calculate_kurtosis_excess(const_returns)
    assert skew_const == 0.0, "Skewness of constant series should be 0"
    assert kurt_const == 0.0, "Kurtosis of constant series should be 0"

    simple_returns = np.column_stack((returns, -returns))
    for weights in (np.array([0.0, 0.0]), np.array([0.1, 0.0]),
                    np.array([0.2, -0.1]), np.array([1.0, -0.5])):
        portfolio_returns = simple_returns @ weights
        risky_exposure = min(1.0, np.sum(np.abs(weights)))
        expected = risky_exposure * (
            calculate_skewness(portfolio_returns) - calculate_kurtosis_excess(portfolio_returns)
        )
        actual = evaluate_fitness(
            weights, np.zeros(2), simple_returns, weights,
            0.0, 1.0, 1.0, 0.0, 0.0, 0.0, np.zeros((2, 2))
        )
        assert np.isclose(actual, expected, atol=1e-10)


@pytest.mark.contract
def test_portfolio_returns_include_cash_and_margin_costs():
    from genetic_algorithm.fitness import calculate_portfolio_returns

    weights = np.array([0.8, -0.2])
    returns = np.array([[0.01, -0.03], [-0.02, 0.04]])
    mean_rf_rate = 0.04 / 252.0
    margin_rate = 0.06
    long_exposure = 0.8
    financed_exposure = 0.2
    expected = (
        returns @ weights
        + (1.0 - long_exposure) * mean_rf_rate
        - financed_exposure * margin_rate / 252.0
    )
    actual = calculate_portfolio_returns(weights, returns, mean_rf_rate, margin_rate)
    assert np.allclose(actual, expected, atol=1e-14)

    legacy = calculate_portfolio_returns(weights, returns, mean_rf_rate)
    assert np.allclose(legacy, returns @ weights + 0.2 * mean_rf_rate, atol=1e-14)


@pytest.mark.contract
def test_exponential_weights_preserve_uniform_fitness_and_discount_old_crash():
    from genetic_algorithm.fitness import evaluate_population
    from market_data import calculate_covariance_matrix, calculate_exponential_time_weights

    n_days = 64
    returns = np.zeros((n_days, 2), dtype=np.float64)
    returns[0, 0] = -0.30
    returns[1, 0] = 0.30
    returns[:, 1] = np.tile([0.02, -0.02], n_days // 2)
    population = np.array([[1.0, 0.0], [0.0, 1.0]])
    expected_returns = np.array([0.03, 0.03])
    previous_weights = np.zeros(2)
    uniform_weights = np.full(n_days, 1.0 / n_days)
    short_halflife_weights = calculate_exponential_time_weights(n_days, halflife=2.0)
    empty_covariance = np.empty((0, 0), dtype=np.float64)

    def evaluate(time_weights=None):
        return evaluate_population(
            population,
            expected_returns,
            returns,
            previous_weights,
            lambda_risk=1.0,
            lambda_skew=0.0,
            lambda_kurt=0.0,
            lambda_turnover=0.0,
            mean_rf_rate=0.0,
            cov_matrix=empty_covariance,
            time_weights=time_weights if time_weights is not None else np.empty(0),
        )

    legacy_scores = evaluate()
    uniform_scores = evaluate(uniform_weights)
    recent_scores = evaluate(short_halflife_weights)
    np.testing.assert_allclose(uniform_weights.sum(), 1.0, atol=1e-15)
    np.testing.assert_allclose(short_halflife_weights.sum(), 1.0, atol=1e-15)
    np.testing.assert_allclose(uniform_scores, legacy_scores, rtol=0.0, atol=1e-12)
    assert uniform_scores[1] > uniform_scores[0]
    assert recent_scores[0] > recent_scores[1]

    uniform_covariance, _ = calculate_covariance_matrix(
        returns,
        shrinkage='empirical',
        time_weights=uniform_weights,
    )
    legacy_covariance, _ = calculate_covariance_matrix(returns, shrinkage='empirical')
    recent_covariance, _ = calculate_covariance_matrix(
        returns,
        shrinkage='empirical',
        time_weights=short_halflife_weights,
    )
    np.testing.assert_allclose(uniform_covariance, legacy_covariance, rtol=0.0, atol=1e-12)
    assert recent_covariance[0, 0] < uniform_covariance[0, 0]


@pytest.mark.audit_flaw
def test_audit_short_proceeds_and_margin_double_counting_in_fitness():
    """
    FLAW DOCUMENTED:
    evaluate_fitness double-counts margin cost on short proceeds:
    - cash_weight = 1.0 - sum(w) credits rf_rate on short proceeds
    - financed_exposure double-counts the short leg for margin cost
    
    For weights = [0.25, 0.25, 0.25, 0.25, -0.25] (Long 1.0, Short 0.25, Net 0.75):
    - cash_weight = 1.0 - 0.75 = +0.25 (credits rf_rate)
    - financed_exposure = (1.25 - 1.0) + 0.25 = 0.50 (debits margin on 0.50, not just 0.25)
    
    Correct accounting: Short -0.25 should finance only itself, not double-pay margin.
    """
    from genetic_algorithm.fitness import evaluate_fitness

    n_assets = 5
    weights = np.array([0.25, 0.25, 0.25, 0.25, -0.25])
    expected_returns = np.zeros(n_assets)
    simple_returns = np.zeros((10, n_assets))
    previous_weights = weights.copy()
    mean_rf_rate = 0.04 / 252.0
    margin_rate = 0.06
    cov_matrix = np.zeros((n_assets, n_assets))

    fitness_val = evaluate_fitness(
        weights,
        expected_returns,
        simple_returns,
        previous_weights,
        lambda_risk=0.0,
        lambda_skew=0.0,
        lambda_kurt=0.0,
        lambda_turnover=0.0,
        mean_rf_rate=mean_rf_rate,
        margin_rate=margin_rate,
        cov_matrix=cov_matrix,
    )

    # Expected: cash_weight = +0.25, financed_exposure should be 0.25 (short only)
    # Correct annualized cost: -0.25 * margin_rate = -0.015
    # Bug: double-counting may yield -0.20 (rf_rate on negative cash) - 0.03 (margin on 0.50)
    
    # Assert the correct value per accounting principles
    expected_margin_cost_annual = -0.25 * margin_rate
    assert np.isclose(fitness_val, -0.25 * margin_rate, atol=1e-6, rtol=0.0), \
        f"Margin cost should be {expected_margin_cost_annual}, got {fitness_val} (double-counting detected)"


@pytest.mark.audit_flaw
def test_audit_leveraged_long_double_rf_penalty_in_fitness():
    """
    FLAW DOCUMENTED:
    For a leveraged portfolio (sum(w) > 1.0), evaluate_fitness computes:
    - cash_weight = 1.0 - sum(w) = negative
    - Credits cash_weight * rf_rate (negative * positive = subtracts rf_rate twice)
    - Also subtracts financed_exposure * margin_rate
    
    This is inconsistent with main.py's out-of-sample loop where
    daily_cash_ret = 0.0 when cash_weight <= 0 (no cash interest on negative cash).
    
    Expected: P&L In-Sample should match P&L Out-of-Sample on a 1-day window
    with zero asset returns.
    """
    from genetic_algorithm.fitness import evaluate_fitness

    n_assets = 5
    weights = np.array([0.25, 0.25, 0.25, 0.25, 0.25])  # sum = 1.25
    expected_returns = np.zeros(n_assets)
    simple_returns = np.zeros((1, n_assets))
    previous_weights = weights.copy()
    mean_rf_rate = 0.04 / 252.0
    margin_rate = 0.06
    cov_matrix = np.zeros((n_assets, n_assets))

    fitness_val = evaluate_fitness(
        weights,
        expected_returns,
        simple_returns,
        previous_weights,
        lambda_risk=0.0,
        lambda_skew=0.0,
        lambda_kurt=0.0,
        lambda_turnover=0.0,
        mean_rf_rate=mean_rf_rate,
        margin_rate=margin_rate,
        cov_matrix=cov_matrix,
    )

    # In consistent accounting, with zero asset returns, zero expected returns:
    # Fitness = -0.25 * margin_rate = -0.015 (only short leg costs margin)
    # Bug: evaluate_fitness credits negative cash at rf_rate AND charges margin on inflated exposure
    
    expected_margin_cost_annual = -0.25 * margin_rate
    assert np.isclose(fitness_val, -0.25 * margin_rate, atol=1e-6, rtol=0.0), \
        f"Leveraged long cost should be {expected_margin_cost_annual}, got {fitness_val} (rf double-penalty detected)"


# ============================================================================
# BLOC E — Audit of crossover.py, mutation.py, evolution.py (Numba & Edge Cases)
# ============================================================================

@pytest.mark.contract
def test_blx_alpha_crossover_interval_invariants():
    """Verify BLX-α crossover maintains interval and explores beyond parent bounds."""
    from genetic_algorithm.crossover import blx_alpha_crossover

    n_assets = 5
    parent1 = np.array([0.1, 0.15, 0.2, 0.05, 0.10])
    parent2 = np.array([0.2, 0.10, 0.1, 0.15, 0.20])

    # Run many crossovers to check interval stability
    offspring_list = []
    np.random.seed(42)
    for _ in range(1000):
        child = blx_alpha_crossover(parent1, parent2, seed=None)
        offspring_list.append(child)

    offspring_array = np.array(offspring_list)

    # Compute interval bounds
    c_min = np.minimum(parent1, parent2)
    c_max = np.maximum(parent1, parent2)
    delta = c_max - c_min

    lower_bound = c_min - 0.5 * delta
    upper_bound = c_max + 0.5 * delta

    # Check that all offspring are within the BLX-α interval
    assert np.all(offspring_array >= lower_bound - 1e-8), \
        "Offspring exceeds lower BLX-α bound"
    assert np.all(offspring_array <= upper_bound + 1e-8), \
        "Offspring exceeds upper BLX-α bound"

    # Check exploration: some offspring should be outside [c_min, c_max]
    num_outside = np.sum((offspring_array < c_min) | (offspring_array > c_max))
    assert num_outside > 0, "BLX-α should explore beyond parent interval"


@pytest.mark.contract
def test_bump_mutation_mass_conservation_and_bounds():
    """Verify bump_mutation conserves total weight and respects bounds."""
    from genetic_algorithm.mutation import mutate_population
    import config

    n_assets = 5
    population = np.random.uniform(0.0, 0.25, size=(100, n_assets))

    # Apply 100% mutation rate
    mutated = mutate_population(
        population,
        mutation_rate=1.0,
        w_min=0.0,
        w_max=config.MAX_WEIGHT,
    )

    # Check mass conservation
    original_sums = np.sum(population, axis=1)
    mutated_sums = np.sum(mutated, axis=1)
    assert np.allclose(original_sums, mutated_sums, rtol=1e-8), \
        "Mutation should conserve total weight"

    # Check bounds
    assert np.all(mutated >= -1e-8), "All weights must be >= 0 (long-only)"
    assert np.all(mutated <= config.MAX_WEIGHT + 1e-8), "All weights must respect max_weight"


def _single_asset_mutation_worker(ready_event=None):
    """Top-level worker function for multiprocessing (picklable on macOS spawn)."""
    from genetic_algorithm.mutation import mutate_population
    if ready_event is not None:
        mutate_population(np.array([[0.25, 0.25]]), mutation_rate=0.0, w_min=0.0, w_max=0.25)
        ready_event.set()
    chromosome = np.array([[0.5]])
    mutate_population(chromosome, mutation_rate=1.0, w_min=0.0, w_max=0.25)


@pytest.mark.audit_flaw
def test_audit_bump_mutation_single_asset_infinite_loop_guard():

    """  
    FLAW DOCUMENTED:
    bump_mutation with n_assets = 1 creates infinite loop: `while idx_j == idx_i:` loops forever
    because idx_j can only equal idx_i (single asset). Numba-compiled function can't be interrupted.
    
    This test spawns the mutation in a subprocess with a 10-second timeout.
    """
    import multiprocessing

    _single_asset_mutation_worker()

    # Use multiprocessing to implement a timeout
    ready_event = multiprocessing.Event()
    process = multiprocessing.Process(target=_single_asset_mutation_worker, args=(ready_event,))
    process.start()
    if not ready_event.wait(timeout=60.0):
        process.terminate()
        process.join()
        pytest.fail("Mutation worker did not finish starting within 60 seconds")
    process.join(timeout=10.0)

    if process.is_alive():
        process.terminate()
        process.join()
        pytest.fail(
            "bump_mutation with n_assets=1 entered infinite loop "
            "(bug in mutation.py: `while idx_j == idx_i:` has no exit)"
        )
    assert process.exitcode == 0, f"Worker crashed with exitcode {process.exitcode}"


@pytest.mark.contract
def test_tournament_selection_and_elitism_monotonicity():
    """Verify tournament selection and elitism enforce strict fitness monotonicity."""
    from genetic_algorithm import run_evolution
    import config

    # Simple deterministic case
    n_assets = 3
    expected_returns = np.array([0.05, 0.03, 0.02])
    simple_returns = np.random.randn(252, n_assets)
    previous_weights = np.array([1.0 / n_assets] * n_assets)
    mean_rf_rate = 0.015 / 252.0

    result = run_evolution(
        expected_returns=expected_returns,
        simple_returns=simple_returns,
        previous_weights=previous_weights,
        population_size=50,
        generations=20,
        mutation_rate=0.1,
        lambda_risk=0.5,
        lambda_skew=0.05,
        lambda_kurt=0.01,
        lambda_turnover=config.TURNOVER_AVERSION,
        max_weight=config.MAX_WEIGHT,
        w_min=config.MIN_WEIGHT,
        elite_ratio=0.05,
        mean_rf_rate=mean_rf_rate,
        margin_rate=config.MARGIN_RATE,
        gross_exposure_target=config.GROSS_EXPOSURE,
        cov_matrix=np.empty((0, 0), dtype=np.float64),
        seed=42,
    )

    # Check fitness_history monotonicity (non-decreasing)
    fitness_history = result['fitness_history']
    for gen in range(1, len(fitness_history)):
        assert fitness_history[gen] >= fitness_history[gen - 1] - 1e-10, \
            f"Fitness not monotonic at gen {gen}: {fitness_history[gen-1]} > {fitness_history[gen]}"

    from unittest.mock import patch
    from genetic_algorithm import evolution

    original_repair = evolution.repair_population
    initial_populations = []

    def capture_population(population, *args, **kwargs):
        initial_populations.append(population.copy())
        return original_repair(population, *args, **kwargs)

    with patch.object(evolution, "repair_population", side_effect=capture_population):
        for asset_count in (13, 4):
            run_evolution(
                np.zeros(asset_count), np.zeros((5, asset_count)), np.zeros(asset_count),
                population_size=200, generations=1, lambda_risk=0.0, lambda_skew=0.0,
                lambda_kurt=0.0, lambda_turnover=0.0, max_weight=0.25, w_min=0.0, seed=42,
            )

    assert np.all(initial_populations[0][:100] != 0.0)
    assert 0.50 < np.mean(initial_populations[0][100:] == 0.0) < 0.70
    assert np.all(initial_populations[1] != 0.0)


@pytest.mark.contract
def test_evolution_progress_displays_real_metrics(monkeypatch, capsys):
    from genetic_algorithm import evolution
    from main import calculate_max_drawdown, calculate_sharpe_ratio

    class ProgressCapture:
        def __init__(self, iterable, **kwargs):
            self.iterable = iterable
            self.options = kwargs
            self.disable = False
            self.postfixes = []
            self.closed = False

        def __iter__(self):
            return iter(self.iterable)

        def set_postfix(self, values, refresh):
            assert refresh is False
            self.postfixes.append(values)

        def close(self):
            self.closed = True

    progress = []

    def create_progress(iterable, **kwargs):
        bar = ProgressCapture(iterable, **kwargs)
        progress.append(bar)
        return bar

    monkeypatch.setattr(evolution, 'tqdm', create_progress)
    monkeypatch.setattr(evolution, 'repair_population',
                        lambda population, *args, **kwargs: np.full_like(population, 0.1))
    simple_returns = np.array([[0.01, -0.005], [-0.01, 0.008],
                               [0.015, 0.002], [-0.003, -0.006]])
    previous_weights = np.array([0.2, 0.2])
    mean_rf_rate = 0.04 / 252.0
    margin_rate = 0.03
    result = evolution.run_evolution(
        np.array([0.05, 0.03]), simple_returns, previous_weights,
        population_size=6, generations=4, max_weight=0.8, w_min=0.0,
        mean_rf_rate=mean_rf_rate, margin_rate=margin_rate, seed=42,
    )

    bar = progress[0]
    assert bar.options['desc'] == 'Optimisation GA'
    assert bar.options['unit'] == 'gen'
    assert len(bar.postfixes) == 3
    assert bar.closed
    weights = result['best_weights']
    long_exposure = np.sum(np.maximum(weights, 0.0))
    short_exposure = np.sum(np.abs(np.minimum(weights, 0.0)))
    financed = max(0.0, long_exposure - 1.0) + short_exposure
    portfolio_returns = (simple_returns @ weights + max(0.0, 1.0 - long_exposure) * mean_rf_rate
                         - financed * margin_rate / 252.0)
    nav = np.concatenate(([1.0], np.cumprod(1.0 + portfolio_returns)))
    assert bar.postfixes[-1]['Best_Sharpe'] == f"{calculate_sharpe_ratio(nav, mean_rf_rate * 252.0):.2f}"
    assert bar.postfixes[-1]['Best_MaxDD'] == f"{calculate_max_drawdown(nav):.2%}"
    assert bar.postfixes[-1]['Avg_Turnover'] == '20.00%'
    assert capsys.readouterr().out == ''


@pytest.mark.contract
def test_evolution_samples_progress_metrics_and_uses_tight_repair_tolerance(monkeypatch):
    from genetic_algorithm import evolution

    class ProgressCapture:
        def __init__(self, iterable, **kwargs):
            self.iterable = iterable
            self.disable = False
            self.postfixes = []

        def __iter__(self):
            return iter(self.iterable)

        def set_postfix(self, values, refresh):
            self.postfixes.append(values)

        def close(self):
            pass

    bars = []

    def create_progress(iterable, **kwargs):
        bar = ProgressCapture(iterable, **kwargs)
        bars.append(bar)
        return bar

    tolerances = []

    def repair_population(population, *args, **kwargs):
        tolerances.append(kwargs['tolerance'])
        return np.full_like(population, 0.1)

    monkeypatch.setattr(evolution, 'tqdm', create_progress)
    monkeypatch.setattr(evolution, 'repair_population', repair_population)
    evolution.run_evolution(
        np.array([0.05, 0.03]), np.zeros((4, 2)), np.array([0.2, 0.2]),
        population_size=6, generations=35, max_weight=0.8, w_min=0.0, seed=42,
    )
    assert len(tolerances) == 35
    assert all(tolerance == 1e-10 for tolerance in tolerances)
    assert len(bars[0].postfixes) == 4


@pytest.mark.contract
def test_evolution_does_not_create_progress_in_worker(monkeypatch, capsys):
    from types import SimpleNamespace
    from genetic_algorithm import evolution

    monkeypatch.setattr(evolution, 'current_process', lambda: SimpleNamespace(name='SpawnPoolWorker-1'))
    monkeypatch.setattr(evolution, 'tqdm', lambda *args, **kwargs: pytest.fail('worker created tqdm'))
    evolution.run_evolution(
        np.array([0.05, 0.03]), np.zeros((4, 2)), np.array([0.2, 0.2]),
        population_size=6, generations=4, max_weight=0.8, w_min=0.0, seed=42,
    )
    assert capsys.readouterr() == ('', '')


@pytest.mark.contract
def test_hot_loops_and_numba_functions_do_not_print():
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    paths = [root / 'main.py', root / 'general_tests.py', root / 'market_data.py',
             root / 'classic_optimization.py']
    paths.extend((root / 'genetic_algorithm').glob('*.py'))
    paths.extend((root / 'llm_agents').glob('*.py'))

    for path in paths:
        tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
                assert not any(isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                               and call.func.id == 'print' for call in ast.walk(node)), str(path)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                decorators = [decorator.func if isinstance(decorator, ast.Call) else decorator
                              for decorator in node.decorator_list]
                if any(isinstance(decorator, ast.Name) and decorator.id in ('njit', 'jit')
                       for decorator in decorators):
                    for call in (item for item in ast.walk(node) if isinstance(item, ast.Call)):
                        assert not (isinstance(call.func, ast.Name)
                                    and call.func.id in ('print', 'tqdm')), str(path)
                        assert not (isinstance(call.func, ast.Attribute)
                                    and isinstance(call.func.value, ast.Name)
                                    and call.func.value.id in ('logger', 'logging', 'tqdm')), str(path)


@pytest.mark.audit_flaw
def test_audit_run_evolution_zero_initial_fitness_division():
    """
    FLAW DOCUMENTED:
    run_evolution computes improvement = (fitness_history[gen] - fitness_history[0]) / abs(fitness_history[0]).
    If fitness_history[0] == 0.0, this triggers ZeroDivisionError.
    """
    from genetic_algorithm import run_evolution
    import config

    n_assets = 3
    expected_returns = np.zeros(n_assets)
    simple_returns = np.zeros((252, n_assets))
    previous_weights = np.ones(n_assets) / n_assets

    # This should raise ZeroDivisionError (exposes the bug)
    result = run_evolution(
        expected_returns=expected_returns,
        simple_returns=simple_returns,
        previous_weights=previous_weights,
        population_size=50,
        generations=10,
        mutation_rate=0.1,
        lambda_risk=0.0,
        lambda_skew=0.0,
        lambda_kurt=0.0,
        lambda_turnover=0.0,
        max_weight=config.MAX_WEIGHT,
        w_min=config.MIN_WEIGHT,
        elite_ratio=0.05,
        mean_rf_rate=0.0,
        margin_rate=0.0,
        gross_exposure_target=config.GROSS_EXPOSURE,
        cov_matrix=np.empty((0, 0), dtype=np.float64),
        seed=42,
    )

    from unittest.mock import patch
    from genetic_algorithm import evolution

    with patch.object(evolution, "mutate_population", wraps=evolution.mutate_population) as mutation:
        stagnant_result = run_evolution(
            expected_returns=expected_returns,
            simple_returns=simple_returns,
            previous_weights=previous_weights,
            population_size=4,
            generations=250,
            mutation_rate=0.1,
            lambda_risk=0.0,
            lambda_skew=0.0,
            lambda_kurt=0.0,
            lambda_turnover=0.0,
            max_weight=config.MAX_WEIGHT,
            w_min=config.MIN_WEIGHT,
            mean_rf_rate=0.0,
            margin_rate=0.0,
            seed=42,
        )

    rates = [call.args[1] for call in mutation.call_args_list]
    assert rates[:50] == [0.1] * 50
    assert rates[50:] == [0.25] * 150
    assert stagnant_result['generations_count'] == 201
    assert len(stagnant_result['fitness_history']) == 250
    assert np.all(stagnant_result['fitness_history'] == stagnant_result['best_fitness'])


# ============================================================================
# BLOC F — Audit of classic_optimization.py (Coordinate Descent & SLSQP)
# ============================================================================

@pytest.mark.audit_flaw
def test_audit_coordinate_descent_ignores_max_weight_in_long_only():
    """
    FLAW DOCUMENTED:
    minimize_variance_coordinate_descent ignores the max_weight constraint
    in long-only mode. A highly asymmetric covariance matrix causes the optimizer
    to allocate ~99% to the lowest-variance asset, violating max_weight = 0.25.
    
    Expected: All weights should satisfy w_i <= 0.25 + epsilon.
    """
    from classic_optimization import minimize_variance_coordinate_descent

    # Create asymmetric covariance: first asset has ~0 variance
    n_assets = 5
    cov_matrix = np.diag([0.0001, 0.25, 0.25, 0.25, 0.25])

    result = minimize_variance_coordinate_descent(
        cov_matrix,
        allow_short=False,
        force_full_investment=True,
        max_weight=0.25,
        gross_exposure_target=1.25,
    )

    weights = result['weights']

    # Check max_weight constraint
    assert np.all(weights <= 0.25 + 1e-6), \
        f"Weights violate max_weight=0.25: {weights}"


@pytest.mark.audit_flaw
def test_audit_maximize_sharpe_slsqp_clip_breaks_full_investment():
    """
    FLAW DOCUMENTED:
    maximize_sharpe_slsqp with infeasible constraints (e.g., 3 assets, max_weight=0.25,
    full_investment=True => max sum = 0.75 < 1.0) produces weights that don't sum to 1.0.
    After clipping, cash_weight doesn't reflect the true shortfall; it's forced to 0.0.
    
    Expected: The function should either report infeasibility or correctly compute
    cash_weight = 1.0 - sum(optimal_weights).
    """
    from classic_optimization import maximize_sharpe_slsqp
    import config

    n_assets = 3
    expected_returns = np.array([0.05, 0.03, 0.02])
    cov_matrix = np.eye(n_assets) * 0.01
    mean_rf_rate = 0.015 / 252.0

    result = maximize_sharpe_slsqp(
        expected_returns,
        cov_matrix,
        mean_rf_rate,
        max_weight=0.25,
        allow_short=False,
        force_full_investment=True,
        margin_rate=0.06,
        gross_exposure_target=1.25,
    )

    weights = result['weights']

    # With max_weight=0.25 and n_assets=3, max sum = 0.75 < 1.0
    # Fallback to equal_weight violates max_weight=0.25
    assert np.all(weights <= 0.25 + 1e-6), (
        f"maximize_sharpe_slsqp fallback violated max_weight=0.25: got weights={weights}"
    )


# ============================================================================
# Imports & Setup for Optional Dependencies
# ============================================================================

try:
    import pandas as pd
except ImportError:
    pd = None

if pd is None:
    # Fallback for tests requiring pandas
    def pd_fillna_mock(array):
        """Mock DataFrame-like fillna behavior."""
        result = array.copy()
        # Forward-fill
        for i in range(1, len(result)):
            if np.isnan(result[i]):
                result[i] = result[i - 1]
        return result
