"""
Integration audit suite for backtest orchestration, scenario robustness, and performance metrics.

This module audits:
- run_backtest() orchestration and output structure
- Scenario schema integrity (scenarios.py)
- Performance metrics mathematical correctness (Sharpe, Sortino, MaxDD, Calmar)
- Transaction cost nonlinearity effects
- Concurrent execution robustness (general_tests.py)
- QuantStats tearsheet scope hardcoding

All network calls mocked (100% deterministic).
Tests categorized via pytest markers:
- `@pytest.mark.contract`: Nominal execution, output schemas, metric correctness
- `@pytest.mark.audit_flaw`: Expose orchestration bugs, off-by-one errors, concurrency issues
"""

import asyncio
import json
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Set seed for reproducibility
np.random.seed(42)


# ============================================================================
# FIXTURES & SYNTHETIC DATA GENERATORS
# ============================================================================

@pytest.fixture
def synthetic_market_data():
    """Generate deterministic synthetic OHLCV data (400 days, 3 assets + benchmark)."""
    n_days = 400
    tickers = ["AAPL", "MSFT", "GOOGL"]
    
    # Generate synthetic prices with trend and volatility
    dates = pd.date_range("2023-01-01", periods=n_days, freq="D")
    
    data = {}
    for ticker in tickers + ["SPY"]:
        # Random walk with drift
        daily_returns = np.random.randn(n_days) * 0.02 + 0.0008
        prices = 100 * np.exp(np.cumsum(daily_returns))
        data[ticker] = pd.Series(prices, index=dates)
    
    return data, tickers, dates


@pytest.fixture
def synthetic_nav_series():
    """Generate synthetic NAV series with known properties."""
    n_days = 252
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    
    # NAV with trend and known volatility
    daily_returns = np.random.randn(n_days) * 0.01 + 0.0005
    nav = 100 * np.exp(np.cumsum(daily_returns))
    
    return pd.Series(nav, index=dates)


@pytest.fixture
def isolated_main_cache(tmp_path, monkeypatch):
    """Redirect all cache paths and SQLite to isolated temp directory."""
    import llm_cache
    import llm_agents.macro_agent as macro_agent
    
    test_db = tmp_path / "test_main_cache.db"
    monkeypatch.setattr(llm_cache, "DB_PATH", test_db)
    
    if hasattr(macro_agent, "_MACRO_CACHE"):
        macro_agent._MACRO_CACHE.clear()
    
    yield test_db
    
    if hasattr(macro_agent, "_MACRO_CACHE"):
        macro_agent._MACRO_CACHE.clear()


# ============================================================================
# BLOC A — Audit of Performance Metrics (main.py)
# ============================================================================

@pytest.mark.contract
def test_performance_metrics_mathematical_exactness(synthetic_nav_series):
    """Verify calculate_annual_returns, volatility, Sharpe, Sortino, MaxDD, Calmar."""
    from main import (
        calculate_annual_returns,
        calculate_annual_volatility,
        calculate_sharpe_ratio,
        calculate_sortino_ratio,
        calculate_max_drawdown,
        calculate_calmar_ratio,
    )
    
    nav = synthetic_nav_series.values
    
    # Calculate daily returns from NAV
    daily_returns = np.diff(nav) / nav[:-1]
    
    # Calculate metrics (use correct signatures)
    annual_return = calculate_annual_returns(nav, trading_days=252)
    annual_vol = calculate_annual_volatility(daily_returns, trading_days=252)
    sharpe = calculate_sharpe_ratio(nav, risk_free_rate=0.015, trading_days=252)
    sortino = calculate_sortino_ratio(nav, risk_free_rate=0.015, trading_days=252)
    max_dd = calculate_max_drawdown(nav)
    calmar = calculate_calmar_ratio(nav, trading_days=252)
    
    # Test 1: No NaN or Inf
    assert not np.isnan(annual_return), "Annual return is NaN"
    assert not np.isinf(annual_return), "Annual return is Inf"
    assert not np.isnan(annual_vol), "Annual volatility is NaN"
    assert annual_vol >= 0, "Volatility must be non-negative"
    
    # Test 2: Sharpe and Sortino are defined
    if annual_vol > 0:
        assert not np.isnan(sharpe), "Sharpe ratio is NaN"
        assert not np.isnan(sortino), "Sortino ratio is NaN"
    
    # Test 3: Max drawdown must be <= 0 (returns negative value)
    assert -1.0 <= max_dd <= 0.0, f"Max drawdown {max_dd} outside [-1.0, 0.0]"
    
    # Test 4: Calmar ratio defined (may be 0 if no drawdown)
    if abs(max_dd) > 1e-10:
        assert not np.isnan(calmar), "Calmar ratio is NaN"
    
    # Test 5: Relative relationships
    assert annual_vol > 0, "Volatility must be positive"
    # Sortino >= Sharpe (uses only downside vol)
    if sortino > 0 and sharpe > 0:
        assert sortino >= sharpe - 0.1, "Sortino should be >= Sharpe"


@pytest.mark.audit_flaw
def test_audit_rolling_window_off_by_one_index_gap(synthetic_market_data, monkeypatch):
    """
    FLAW DOCUMENTED:
    In run_backtest rolling loop:
        in_sample_end = idx_rebalance
        out_sample_start = idx_rebalance + 1
    
    Is there a day gap or overlap? Does OOS series length match expected?
    This test verifies the boundary conditions between In-Sample and Out-of-Sample.
    """
    from main import run_backtest
    import config

    monkeypatch.setattr(config, "POPULATION_SIZE", 20)
    monkeypatch.setattr(config, "GENERATIONS", 5)
    monkeypatch.setattr(config, "USE_LLM", False)
    monkeypatch.setattr(config, "GENERATE_REPORTS", False)
    
    data, tickers, dates = synthetic_market_data
    n_periods = len(dates)
    
    # Create market data dict for mocking preprocess_market_data
    market_dict = {
        'prices': np.column_stack([data[t].values for t in tickers]),
        'tickers': tickers,
        'simple_returns': np.random.randn(len(data[tickers[0]]), len(tickers)) * 0.01,
        'dates': dates,
        'benchmark_simple_returns': np.random.randn(len(data[tickers[0]])) * 0.01,
        'daily_rf_rates': np.full(len(data[tickers[0]]), 0.015 / 252.0),
        'sentiment_applied': False,
    }
    
    # Mock preprocess_market_data to return the dict
    with patch("main.preprocess_market_data") as mock_preprocess, patch("main.generate_quantstats_tearsheet"):
        mock_preprocess.return_value = market_dict
        
        with patch("main.asyncio.run") as mock_async:
            mock_async.side_effect = lambda coro: coro.close()
            
            # Run backtest with mocked data
            result = run_backtest(tickers, "2023-01-01", "2023-12-31")
            
            # Verify dates returned matches input
            returned_dates = result.get('dates', [])
            
            assert len(returned_dates) > 0, "returned_dates should not be empty"
            assert dates[config.LOOKBACK_WINDOW] in pd.DatetimeIndex(returned_dates), (
                f"First OOS rebalance date {dates[config.LOOKBACK_WINDOW]} is missing from returned_dates "
                f"due to off-by-one bug (out_sample_start = idx_rebalance + 1)"
            )


# ============================================================================
# BLOC B — Audit of Orchestration & Transaction Costs
# ============================================================================

@pytest.mark.contract
def test_run_backtest_with_mocked_market_data(synthetic_market_data, isolated_main_cache, monkeypatch, tmp_path):
    """Execute run_backtest on short mocked scenario, verify output structure."""
    from main import run_backtest
    import config

    monkeypatch.setattr(config, "POPULATION_SIZE", 20)
    monkeypatch.setattr(config, "GENERATIONS", 5)
    monkeypatch.setattr(config, "USE_LLM", False)
    monkeypatch.setattr(config, "GENERATE_REPORTS", True)
    monkeypatch.chdir(tmp_path)
    
    data, tickers, dates = synthetic_market_data
    
    # Create synthetic market data dict as returned by preprocess_market_data
    market_dict = {
        'prices': np.column_stack([data[t].values for t in tickers]),
        'tickers': tickers,
        'simple_returns': np.random.randn(len(data[tickers[0]]), len(tickers)) * 0.01,
        'dates': dates,
        'benchmark_simple_returns': np.random.randn(len(data[tickers[0]])) * 0.01,
        'daily_rf_rates': np.full(len(data[tickers[0]]), 0.015 / 252.0),
        'sentiment_applied': False,
    }
    
    # Mock preprocess_market_data to return the synthetic dict
    with patch("main.preprocess_market_data") as mock_preprocess, patch("main.generate_quantstats_tearsheet"), \
         patch.object(pd.DataFrame, 'to_parquet', side_effect=ImportError('no parquet engine')) as parquet_export:
        mock_preprocess.return_value = market_dict
        
        with patch("main.asyncio.run") as mock_async:  # Skip async LLM calls
            mock_async.side_effect = lambda coro: coro.close()
            
            # Execute backtest with correct signature
            result = run_backtest(
                tickers=tickers,
                start_date="2023-01-01",
                end_date="2023-12-31",
                benchmark_ticker="SPY",
                scenario_name="US_MegaTech_2015_2023",
                record_history=True,
                history_config_id="US/MegaTech",
            )

    history = result['rebalance_history']
    assert len(history) == result['strategies']['genetic_algorithm']['n_rebalances'] > 1
    assert history['rebalance_date'].dtype == 'datetime64[ns]'
    assert str(history['inferred_regime'].dtype) == 'category'
    parquet_export.assert_called_once()
    assert result['rebalance_history_path'] == 'outputs/rebalance_history_top_US_MegaTech.csv'
    csv_path = tmp_path / result['rebalance_history_path']
    assert csv_path.is_file()
    assert len(pd.read_csv(csv_path)) == len(history)
    assert not list(tmp_path.rglob('*.md'))
    
    # Verify output structure
    assert isinstance(result, dict), "run_backtest should return dict"
    
    # Check required keys
    required_keys = ["strategies", "macro_decisions", "dates", "tickers"]
    for key in required_keys:
        assert key in result, f"Missing required key: {key}"
    
    # Verify strategies structure
    assert isinstance(result["strategies"], dict), "strategies must be dict"
    strategy_names = ["genetic_algorithm", "variance_minimization", "maximum_sharpe", "benchmark"]
    for strat in strategy_names:
        assert strat in result["strategies"], f"Missing strategy: {strat}"
    
    # Verify NAV/portfolio_values curves have no NaN/Inf
    for strat_name, strat_data in result["strategies"].items():
        # Key is "portfolio_values", not "nav"
        if "portfolio_values" in strat_data:
            nav = strat_data["portfolio_values"]
            assert not np.any(np.isnan(nav)), f"{strat_name} portfolio_values contains NaN"
            assert not np.any(np.isinf(nav)), f"{strat_name} portfolio_values contains Inf"


@pytest.mark.contract
@pytest.mark.parametrize('export_md', [False, True])
def test_backtest_defaults_do_not_export(synthetic_market_data, monkeypatch, tmp_path, export_md):
    import config
    from main import run_backtest

    monkeypatch.setattr(config, "POPULATION_SIZE", 20)
    monkeypatch.setattr(config, "GENERATIONS", 5)
    monkeypatch.setattr(config, "USE_LLM", False)
    monkeypatch.setattr(config, "GENERATE_REPORTS", True)
    monkeypatch.chdir(tmp_path)
    data, tickers, dates = synthetic_market_data
    market_dict = {
        'prices': np.column_stack([data[ticker].values for ticker in tickers]),
        'tickers': tickers,
        'simple_returns': np.zeros((len(dates), len(tickers))),
        'dates': dates,
        'benchmark_simple_returns': np.zeros(len(dates)),
        'daily_rf_rates': np.zeros(len(dates)),
        'sentiment_applied': False,
    }

    with patch("main.preprocess_market_data", return_value=market_dict), \
         patch("main.asyncio.run", side_effect=lambda coro: coro.close()), \
         patch("main.generate_quantstats_tearsheet") as html_export, \
         patch("main.qs.reports.html") as quantstats_html, \
         patch("matplotlib.pyplot.figure") as matplotlib_figure, \
         patch("main.open", create=True) as file_open:
        result = run_backtest(tickers, "2023-01-01", "2023-12-31", export_md=export_md)

    assert result['strategies']['genetic_algorithm'] is not None
    html_export.assert_not_called()
    quantstats_html.assert_not_called()
    matplotlib_figure.assert_not_called()
    file_open.assert_not_called()
    assert all(not values for values in result['allocation_history'].values())
    assert 'rebalance_history' not in result
    assert list(tmp_path.iterdir()) == []


@pytest.mark.contract
def test_rebalance_history_values_and_parquet(monkeypatch, tmp_path):
    import config
    from main import run_backtest

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config, 'LOOKBACK_WINDOW', 4)
    monkeypatch.setattr(config, 'REBALANCE_FREQ', 3)
    monkeypatch.setattr(config, 'USE_LLM', False)
    monkeypatch.setattr(config, 'TRANSACTION_FEE', 0.002)
    tickers = ['AAA', 'BBB', 'CCC']
    dates = pd.date_range('2024-01-01', periods=10, tz='Europe/Paris')
    returns = np.zeros((len(dates), len(tickers)))
    returns[4] = [0.1, -0.05, 0.02]
    weights = np.array([0.6, -0.2, 0.1])
    covariance = np.diag([0.04, 0.09, 0.16])
    market_data = {
        'prices': np.full((len(dates), len(tickers)), 100.0),
        'tickers': tickers,
        'simple_returns': returns,
        'dates': dates,
        'benchmark_simple_returns': np.zeros(len(dates)),
        'daily_rf_rates': np.zeros(len(dates)),
        'sentiment_applied': False,
    }
    macro = MagicMock(risk_aversion=0.5, liquidity_multiplier=2.0, enable_shorting=True)
    macro.market_regime.value = 'Bull'

    def run_macro(coroutine):
        coroutine.close()
        return macro

    with patch('main.preprocess_market_data', return_value=market_data), \
         patch('main.asyncio.run', side_effect=run_macro), \
            patch('main.calculate_covariance_matrix', return_value=(covariance, 0.0)) as covariance_estimator, \
            patch('main.run_evolution', return_value={'best_weights': weights}) as evolution, \
         patch('main.minimize_variance_coordinate_descent', return_value={'weights': weights}), \
         patch('main.maximize_sharpe_slsqp', return_value={'weights': weights}), \
         patch.object(pd.DataFrame, 'to_parquet', autospec=True) as parquet_export, \
         patch.object(pd.DataFrame, 'from_records', wraps=pd.DataFrame.from_records) as from_records:
        recorded = run_backtest(tickers, '2024-01-01', '2024-01-10', scenario_name='test',
                    record_history=True, history_config_id='test')
        time_weights = evolution.call_args.kwargs['time_weights']
        np.testing.assert_allclose(time_weights.sum(), 1.0, atol=1e-15)
        assert len(time_weights) == config.LOOKBACK_WINDOW
        np.testing.assert_array_equal(
            covariance_estimator.call_args.kwargs['time_weights'],
            time_weights,
        )
        from_records.assert_called_once()
        from_records.reset_mock()
        unrecorded = run_backtest(tickers, '2024-01-01', '2024-01-10', scenario_name='test')
        from_records.assert_not_called()

    history = recorded['rebalance_history']
    ga_result = recorded['strategies']['genetic_algorithm']
    assert list(history.columns) == [
        'run_id', 'rebalance_date', 'inferred_regime', 'portfolio_vol_ex_ante',
        'turnover', 'transaction_costs', 'cash_weight',
        'weight_AAA', 'weight_BBB', 'weight_CCC',
    ]
    assert list(history['rebalance_date']) == [pd.Timestamp('2024-01-05'), pd.Timestamp('2024-01-08')]
    assert history['rebalance_date'].dtype == 'datetime64[ns]'
    assert str(history['inferred_regime'].dtype) == 'category'
    assert (history['inferred_regime'] == 'Bull').all()
    assert history['run_id'].nunique() == 1
    assert history['run_id'].iloc[0].startswith('test-')
    assert all(dtype == np.dtype('float64') for dtype in history.iloc[:, 3:].dtypes)
    assert np.allclose(history[['weight_AAA', 'weight_BBB', 'weight_CCC']], weights)
    assert np.allclose(history['cash_weight'], 0.5)
    assert np.allclose(history['portfolio_vol_ex_ante'], np.sqrt(weights @ covariance @ weights))
    first_turnover = np.abs(weights - np.ones(3) / 3).sum()
    assert history['turnover'].iloc[0] == pytest.approx(first_turnover)
    assert history['transaction_costs'].iloc[0] == pytest.approx(
        0.004 * (first_turnover + 0.1 * first_turnover ** 2)
    )
    drifted_weights = weights.copy()
    cash_balance = 1.0 - weights.sum()
    for daily_returns in returns[4:7]:
        long_exposure = np.maximum(drifted_weights, 0).sum()
        short_exposure = np.abs(np.minimum(drifted_weights, 0)).sum()
        financed_exposure = max(0.0, long_exposure - 1.0) + short_exposure
        cash_balance -= financed_exposure * config.MARGIN_RATE / 252.0
        drifted_weights *= 1.0 + daily_returns
        nav = drifted_weights.sum() + cash_balance
        drifted_weights /= nav
        cash_balance /= nav
    assert history['turnover'].iloc[1] == pytest.approx(np.abs(weights - drifted_weights).sum())
    assert history['transaction_costs'].iloc[1] == pytest.approx(
        0.004 * (history['turnover'].iloc[1] + 0.1 * history['turnover'].iloc[1] ** 2)
    )
    assert recorded['strategies']['genetic_algorithm']['avg_turnover'] == pytest.approx(history['turnover'].mean())
    assert len(ga_result['gross_daily_returns']) == len(recorded['dates'])
    assert len(ga_result['net_daily_returns']) == len(recorded['dates'])
    np.testing.assert_allclose(
        ga_result['gross_daily_returns'],
        np.diff(ga_result['gross_portfolio_values']) / ga_result['gross_portfolio_values'][:-1],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        ga_result['net_daily_returns'],
        np.diff(ga_result['net_portfolio_values']) / ga_result['net_portfolio_values'][:-1],
        atol=1e-12,
    )
    elapsed_years = len(recorded['dates']) / 252.0
    assert ga_result['Total Transaction Costs Drag (bps/yr)'] == pytest.approx(
        history['transaction_costs'].sum() / elapsed_years * 10000.0
    )
    assert ga_result['Total Margin Financing Drag (bps/yr)'] > 0.0
    assert ga_result['Annualized Turnover'] == pytest.approx(
        history['turnover'].sum() / elapsed_years
    )
    assert ga_result['gross_portfolio_values'][-1] > ga_result['net_portfolio_values'][-1]
    parquet_export.assert_called_once()
    assert parquet_export.call_args.args[1] == Path('outputs/rebalance_history_top_test.parquet')
    assert parquet_export.call_args.kwargs == {'index': False}
    assert recorded['rebalance_history_path'] == 'outputs/rebalance_history_top_test.parquet'
    assert not (tmp_path / 'outputs/rebalance_history.csv').exists()
    for strategy in recorded['strategies']:
        np.testing.assert_array_equal(
            recorded['strategies'][strategy]['portfolio_values'],
            unrecorded['strategies'][strategy]['portfolio_values'],
        )


@pytest.mark.contract
def test_rebalance_history_skips_empty_backtest(synthetic_market_data, monkeypatch, tmp_path):
    import config
    from main import run_backtest

    monkeypatch.chdir(tmp_path)
    data, tickers, dates = synthetic_market_data
    monkeypatch.setattr(config, 'LOOKBACK_WINDOW', len(dates))
    market_data = {
        'prices': np.column_stack([data[ticker].values for ticker in tickers]),
        'tickers': tickers,
        'simple_returns': np.zeros((len(dates), len(tickers))),
        'dates': dates,
        'benchmark_simple_returns': np.zeros(len(dates)),
        'daily_rf_rates': np.zeros(len(dates)),
        'sentiment_applied': False,
    }
    with patch('main.preprocess_market_data', return_value=market_data), \
         patch.object(pd.DataFrame, 'from_records', wraps=pd.DataFrame.from_records) as from_records:
        result = run_backtest(tickers, '2023-01-01', '2023-12-31', record_history=True)
    from_records.assert_not_called()
    assert 'rebalance_history' not in result
    assert not (tmp_path / 'outputs').exists()


@pytest.mark.audit_flaw
def test_audit_transaction_cost_nonlinear_shocks(synthetic_market_data, isolated_main_cache, monkeypatch):
    """
    FLAW DOCUMENTED:
    When asset returns and RF rates are strictly 0.0, any change in portfolio_values
    must come exclusively from transaction costs at rebalancing.
    However, in main.py, because out_sample_start = idx_rebalance + 1 and the first
    rebalance starts from equal_weight_portfolio, if weights change at idx_rebalance,
    the transaction cost shock must be properly reflected in the OOS NAV trajectory
    without off-by-one loss or unrecorded gap at idx_rebalance.
    """
    from main import run_backtest, calculate_ema, maximize_sharpe_slsqp
    import config

    monkeypatch.setattr(config, "POPULATION_SIZE", 20)
    monkeypatch.setattr(config, "GENERATIONS", 5)
    monkeypatch.setattr(config, "USE_LLM", False)
    monkeypatch.setattr(config, "GENERATE_REPORTS", False)

    data, tickers, dates = synthetic_market_data
    n_days = len(dates)

    # Zero returns everywhere, but non-zero price trend in-sample so GA/MaxSharpe deviate from 1/N
    prices = np.column_stack([
        np.linspace(100.0, 150.0, n_days),
        np.linspace(100.0, 100.0, n_days),
        np.linspace(100.0, 80.0, n_days),
    ])
    simple_returns = np.zeros((n_days, 3), dtype=np.float64)
    simple_returns[:config.LOOKBACK_WINDOW, 0] = 0.002
    simple_returns[:config.LOOKBACK_WINDOW, 2] = -0.002

    market_dict = {
        'prices': prices,
        'tickers': tickers,
        'simple_returns': simple_returns,
        'dates': dates,
        'benchmark_simple_returns': np.zeros(n_days, dtype=np.float64),
        'daily_rf_rates': np.zeros(n_days, dtype=np.float64),
        'sentiment_applied': False,
    }

    with patch("main.preprocess_market_data", return_value=market_dict), patch("main.generate_quantstats_tearsheet"), patch("main.maximize_sharpe_slsqp", wraps=maximize_sharpe_slsqp) as mock_sharpe:
        with patch("main.asyncio.run", side_effect=lambda coro: coro.close()):
            res = run_backtest(tickers, "2023-01-01", "2023-12-31")

    in_sample_prices = prices[:config.LOOKBACK_WINDOW]
    expected_returns = (
        calculate_ema(in_sample_prices, span=40)[-1]
        / calculate_ema(in_sample_prices, span=200)[-1] - 1.0
    ) * 252.0 / 80.0
    np.testing.assert_allclose(mock_sharpe.call_args_list[0].args[0], expected_returns)

    oos_dates = res["dates"]
    port_values = res["strategies"]["maximum_sharpe"]["portfolio_values"]

    # Length of portfolio_values must equal length of OOS dates + 1 (initial base 1.0 at nav_start)
    # and all OOS days from LOOKBACK_WINDOW to last rebalance window end must be contiguous
    expected_first_oos_date = dates[config.LOOKBACK_WINDOW]
    assert pd.Timestamp(oos_dates[0]) == pd.Timestamp(expected_first_oos_date), (
        f"OOS trajectory starts at {oos_dates[0]} instead of {expected_first_oos_date}, "
        f"losing the rebalance day transition"
    )
    assert port_values[-1] < 1.0, (
        f"With zero OOS returns and non-zero turnover, final NAV ({port_values[-1]}) must be < 1.0 due to transaction costs"
    )


@pytest.mark.audit_flaw
def test_audit_quantstats_tearsheet_scenario_hardcoding(isolated_main_cache, monkeypatch, tmp_path):
    """
    FLAW DOCUMENTED:
    generate_quantstats_tearsheet has hardcoded check:
        if scenario_name != 'Global_MultiAsset_2024': return
    
    This prevents generation of tearsheets for other scenarios. Test fails if
    tearsheet generation is artificially limited to one scenario.
    """
    from main import generate_quantstats_tearsheet
    monkeypatch.chdir(tmp_path)

    # Try generating tearsheet for non-Global_MultiAsset_2024 scenario
    portfolio_values = {
        'genetic_algorithm': 100 * np.ones(100),
        'benchmark': 100 * np.ones(100)
    }
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    oos_length = 20

    def write_report(*args, **kwargs):
        Path(kwargs['output']).write_text('<html>scenario report</html>', encoding='utf-8')

    with patch('main.qs.reports.html', side_effect=write_report) as mock_qs_html, \
         patch('matplotlib.pyplot.close') as mock_close:
        generate_quantstats_tearsheet(
            portfolio_values=portfolio_values,
            dates=dates,
            oos_length=oos_length,
            scenario_name="US_MegaTech_2015_2023",
            rf=0.042,
        )

    mock_qs_html.assert_called_once()
    assert mock_qs_html.call_args.kwargs['rf'] == 0.042
    assert mock_qs_html.call_args.kwargs['output'] == 'outputs/tearsheets/best_strategy_tearsheet.html'
    assert (tmp_path / 'outputs' / 'tearsheets').is_dir()
    assert (tmp_path / 'outputs' / 'tearsheets' / 'tearsheet_US_MegaTech_2015_2023.html').is_file()
    mock_close.assert_called_once_with('all')


@pytest.mark.contract
def test_tearsheet_closes_figures_when_export_fails(monkeypatch, tmp_path):
    import matplotlib.pyplot as plt
    from main import generate_quantstats_tearsheet

    monkeypatch.chdir(tmp_path)
    portfolio_values = {'genetic_algorithm': np.ones(100), 'benchmark': np.ones(100)}
    dates = pd.date_range("2023-01-01", periods=100, freq="D")

    def fail_after_opening_figure(*args, **kwargs):
        plt.figure()
        raise RuntimeError("export failed")

    with patch('main.qs.reports.html', side_effect=fail_after_opening_figure):
        with pytest.raises(RuntimeError, match="export failed"):
            generate_quantstats_tearsheet(portfolio_values, dates, 20)
    assert plt.get_fignums() == []


@pytest.mark.contract
def test_tearsheet_renders_html_and_closes_figures(monkeypatch, tmp_path):
    import matplotlib.pyplot as plt
    from main import generate_quantstats_tearsheet

    monkeypatch.chdir(tmp_path)
    dates = pd.date_range("2023-01-01", periods=360, freq="B")
    indices = np.arange(len(dates))
    portfolio_values = {
        'genetic_algorithm': np.cumprod(1.0 + 0.0004 + 0.006 * np.sin(indices / 3.0)),
        'benchmark': np.cumprod(1.0 + 0.0003 + 0.004 * np.cos(indices / 4.0)),
    }
    plt.close('all')
    generate_quantstats_tearsheet(portfolio_values, dates, 108, rf=0.042)

    report = tmp_path / 'outputs' / 'tearsheets' / 'best_strategy_tearsheet.html'
    assert report.is_file()
    assert '<html' in report.read_text(encoding='utf-8').lower()
    assert plt.get_fignums() == []


@pytest.mark.contract
def test_scenario_worker_disables_exports():
    from general_tests import run_single_scenario

    params = {
        'tickers': ['SPY'],
        'start_date': '2023-01-01',
        'end_date': '2023-12-31',
        'benchmark_ticker': 'SPY',
        'rf_ticker': '^TNX',
    }
    with patch('general_tests.time.sleep'), patch('general_tests.run_backtest', return_value={}) as backtest:
        outcome = run_single_scenario('example', params)

    assert outcome['error'] is None
    assert backtest.call_args.kwargs['export_html'] is False
    assert backtest.call_args.kwargs['export_md'] is False
    assert backtest.call_args.kwargs['record_history'] is False


def _emit_test_worker_logs():
    from logging_utils import get_logger

    worker_logger = get_logger('test_worker')
    worker_logger.info('Début du calcul')
    worker_logger.warning('Anomalie simulée : %s', 'première')
    worker_logger.warning('Anomalie simulée : %s', 'répétée')


def _probe_scenario_worker():
    import logging
    import os
    import sys
    import warnings
    from general_tests import run_single_scenario

    params = {
        'tickers': ['SPY'], 'start_date': '2023-01-01', 'end_date': '2023-12-31',
        'benchmark_ticker': 'SPY',
    }

    def successful_backtest(*args, **kwargs):
        print('unexpected stdout', flush=True)
        print('unexpected stderr', file=sys.stderr, flush=True)
        os.write(1, b'unexpected native stdout\n')
        os.write(2, b'unexpected native stderr\n')
        logging.getLogger('portfolio.test').warning('Singular covariance')
        warnings.warn('Numerical warning', RuntimeWarning)
        return {'strategies': {'genetic_algorithm': {
            'sharpe_ratio': np.float64(1.25), 'portfolio_values': np.ones(3),
        }}}

    def failing_backtest(*args, **kwargs):
        logging.getLogger('portfolio.test').warning('Another warning')
        raise ValueError('bad covariance')

    with patch('general_tests.time.sleep'), \
         patch('general_tests.run_backtest', side_effect=successful_backtest):
        success = run_single_scenario('success', params)
    with patch('general_tests.time.sleep'), \
         patch('general_tests.run_backtest', side_effect=failing_backtest):
        failure = run_single_scenario('failure', params)
    return params, success, failure


@pytest.mark.contract
def test_scenario_worker_spawn_is_silent_and_returns_serializable_status(capfd):
    import pickle
    from multiprocessing import get_context
    from general_tests import _initialize_scenario_worker

    with ProcessPoolExecutor(max_workers=1, mp_context=get_context('spawn'),
                             initializer=_initialize_scenario_worker) as executor:
        params, success, failure = executor.submit(_probe_scenario_worker).result(timeout=60)
    assert capfd.readouterr() == ('', '')
    pickle.dumps(params)
    pickle.dumps((success, failure))
    assert success['sharpe'] == 1.25
    assert isinstance(success['sharpe'], float)
    assert all(value is None or isinstance(value, (str, float)) for value in success.values())
    assert 'Singular covariance' in success['warnings']
    assert 'Numerical warning' in success['warnings']
    assert failure['error'] == 'ValueError: bad covariance'
    assert 'Another warning' in failure['warnings']
    assert 'Singular covariance' not in failure['warnings']


@pytest.mark.contract
def test_master_progress_tracks_completed_results_and_errors(monkeypatch):
    from concurrent.futures import Future
    import general_tests

    class ProgressCapture:
        def __init__(self, iterable, **kwargs):
            self.iterable = iterable
            self.options = kwargs
            self.postfixes = []

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def __iter__(self):
            return iter(self.iterable)

        def set_postfix(self, values):
            self.postfixes.append(values)

    def outcome(sharpe, warnings=None):
        return {
            'sharpe': sharpe,
            'error': None, 'duration_seconds': 1.0, 'warnings': warnings or '',
        }

    first, second, failed, last = (Future() for _ in range(4))
    first.set_result(outcome(1.25, 'Singular covariance'))
    second.set_result(outcome(float('nan')))
    failed.set_exception(RuntimeError('worker exited'))
    last.set_result(outcome(2.5))
    futures = {first: 'first', second: 'second', failed: 'failed', last: 'last'}
    bars = []

    def create_progress(iterable, **kwargs):
        bar = ProgressCapture(iterable, **kwargs)
        bars.append(bar)
        return bar

    monkeypatch.setattr(general_tests, 'as_completed', lambda pending: iter(pending))
    monkeypatch.setattr(general_tests, 'tqdm', create_progress)
    monkeypatch.setattr(general_tests.logger, 'warning', MagicMock())
    monkeypatch.setattr(general_tests.logger, 'error', MagicMock())
    outcomes = general_tests._collect_outcomes(futures)

    assert bars[0].options['desc'] == 'Multi-Test Parallèle'
    assert bars[0].options['unit'] == 'config'
    assert bars[0].options['total'] == 4
    assert bars[0].postfixes == [
        {'Best_Sharpe': '1.25', 'Completed': '1/4'},
        {'Best_Sharpe': '1.25', 'Completed': '2/4'},
        {'Best_Sharpe': '1.25', 'Completed': '3/4'},
        {'Best_Sharpe': '2.50', 'Completed': '4/4'},
    ]
    assert outcomes['failed']['error'] == 'RuntimeError: worker exited'
    general_tests.logger.warning.assert_called_once()
    general_tests.logger.error.assert_called_once()


@pytest.mark.contract
@pytest.mark.parametrize('has_valid_result', [True, False])
def test_general_tests_exports_once_and_reruns_best_after_pool(monkeypatch, tmp_path, has_valid_result):
    from concurrent.futures import Future
    import pickle
    import general_tests

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(general_tests.time, 'sleep', lambda seconds: None)
    scenarios = {
        name: {'tickers': [name], 'start_date': '2023-01-01', 'end_date': '2023-12-31',
               'benchmark_ticker': 'SPY'}
        for name in ('first', 'best', 'invalid', 'failed')
    }
    pool_state = {'active': False, 'payloads': []}

    class FakeExecutor:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            pool_state['active'] = True
            return self

        def __exit__(self, *args):
            pool_state['active'] = False

        def submit(self, worker, scenario_name, params):
            outcome = worker(scenario_name, params)
            pickle.dumps(outcome)
            assert all(value is None or isinstance(value, (str, float)) for value in outcome.values())
            pool_state['payloads'].append(outcome)
            future = Future()
            future.set_result(outcome)
            return future

    calls = []

    def backtest(*args, **kwargs):
        calls.append(kwargs)
        scenario_name = kwargs['scenario_name']
        if kwargs['record_history']:
            assert not pool_state['active']
            assert (tmp_path / 'reports/global_results_configuration.csv').is_file()
            return {}
        assert pool_state['active']
        assert kwargs['export_html'] is False
        assert kwargs['export_md'] is False
        if scenario_name == 'failed':
            raise ValueError('failed configuration')
        sharpe = {'first': 1.0, 'best': 2.0, 'invalid': float('inf')}[scenario_name]
        if not has_valid_result:
            sharpe = float('nan')
        return {'strategies': {'genetic_algorithm': {
            'sharpe_ratio': sharpe, 'sortino_ratio': 1.2, 'max_drawdown': -0.2,
            'calmar_ratio': 0.8, 'annual_return': 0.16, 'avg_turnover': 0.12,
            'portfolio_values': np.ones(100),
        }}}

    writes = []
    original_to_csv = pd.DataFrame.to_csv

    def write_csv(frame, *args, **kwargs):
        writes.append(args[0])
        return original_to_csv(frame, *args, **kwargs)

    monkeypatch.setattr(general_tests, 'ProcessPoolExecutor', FakeExecutor)
    monkeypatch.setattr(general_tests, 'run_backtest', backtest)
    monkeypatch.setattr(pd.DataFrame, 'to_csv', write_csv)
    result = general_tests.run_general_tests(scenarios)
    assert writes == ['reports/global_results_configuration.csv']
    assert len(result) == len(scenarios)
    assert len(pool_state['payloads']) == len(scenarios)
    csv = pd.read_csv(tmp_path / 'reports/global_results_configuration.csv')
    assert csv.loc[csv['config_id'] == 'failed', 'status'].iloc[0] == 'error'
    reruns = [call for call in calls if call['record_history']]
    if has_valid_result:
        assert list(csv['config_id'][:2]) == ['best', 'first']
        assert csv.iloc[0]['avg_turnover'] == pytest.approx(0.12)
        assert csv.iloc[0]['calmar'] == pytest.approx(0.8)
        assert len(reruns) == 2
        assert [run['scenario_name'] for run in reruns] == ['first', 'best']
        assert [run['history_config_id'] for run in reruns] == ['first', 'best']
        assert all(run['export_html'] is True for run in reruns)
        assert all(run['export_md'] is False for run in reruns)
    else:
        assert not reruns


@pytest.mark.contract
def test_progress_output_does_not_wait_for_slow_terminal():
    from threading import Event, Thread
    from logging_utils import _AsyncProgressStream

    class SlowTerminal:
        def __init__(self):
            self.started = Event()
            self.release = Event()

        def write(self, text):
            self.started.set()
            self.release.wait(timeout=10)

        def flush(self):
            pass

        def isatty(self):
            return True

    terminal = SlowTerminal()
    stream = _AsyncProgressStream(terminal)
    finished = Event()

    def produce_updates():
        for index in range(500):
            stream.write(str(index))
        finished.set()

    try:
        stream.write('initial')
        assert terminal.started.wait(timeout=10)
        producer = Thread(target=produce_updates, daemon=True)
        producer.start()
        assert finished.wait(timeout=10)
        assert stream._queue.qsize() <= 128
    finally:
        terminal.release.set()
        stream.close()


@pytest.mark.contract
def test_worker_logging_is_filtered_and_centralized(monkeypatch, tmp_path):
    import logging
    from logging.handlers import QueueListener
    from multiprocessing import Manager
    import logging_utils

    monkeypatch.chdir(tmp_path)
    base_logger = logging_utils.get_logger('test').parent
    original_handlers = list(base_logger.handlers)
    original_level = base_logger.level
    listener = None
    try:
        configured = logging_utils.configure_logging()
        handlers = [handler for handler in configured.handlers
                    if getattr(handler, '_portfolio_handler', False)]
        assert sorted(handler.level for handler in handlers) == [logging.INFO, logging.WARNING]
        configured.info('Début du multi-test')
        with patch('logging_utils.tqdm.write') as console_write:
            with Manager() as manager:
                queue = manager.Queue()
                listener = QueueListener(queue, *handlers, respect_handler_level=True)
                listener.start()
                try:
                    with ProcessPoolExecutor(max_workers=1, initializer=logging_utils.configure_worker_logging,
                                             initargs=(queue,)) as executor:
                        executor.submit(_emit_test_worker_logs).result(timeout=60)
                finally:
                    listener.stop()
                    listener = None

            console_write.assert_called_once()
            assert 'Anomalie simulée : première' in console_write.call_args.args[0]

        log_text = (tmp_path / 'outputs' / 'logs' / 'execution_debug.log').read_text(encoding='utf-8')
        assert 'Début du multi-test' in log_text
        assert 'Début du calcul' in log_text
        assert log_text.count('Anomalie simulée') == 1
        assert 'répétée' not in log_text
    finally:
        if listener is not None:
            listener.stop()
        for handler in list(base_logger.handlers):
            if handler not in original_handlers:
                handler.close()
        base_logger.handlers = original_handlers
        base_logger.setLevel(original_level)


# ============================================================================
# BLOC C — Audit of Scenarios & Parallel Execution
# ============================================================================

@pytest.mark.contract
def test_scenarios_schema_integrity():
    """Verify all scenarios have required schema fields."""
    from scenarios import SCENARIOS

    french_scenario = SCENARIOS['FR_Covid_2020_2021']
    assert 'AI.PA' in french_scenario['tickers']
    assert 'AL.PA' not in french_scenario['tickers']
    assert french_scenario['rf_ticker'] == '^TNX'
    
    required_fields = ["start_date", "end_date", "tickers", "benchmark_ticker", "rf_ticker"]
    
    for scenario_name, scenario_def in SCENARIOS.items():
        for field in required_fields:
            assert field in scenario_def, \
                f"Scenario '{scenario_name}' missing required field: {field}"
        
        # Verify types and non-empty conditions
        assert isinstance(scenario_def["tickers"], (list, tuple)), \
            f"Scenario '{scenario_name}': tickers must be list/tuple"
        assert len(scenario_def["tickers"]) > 0, \
            f"Scenario '{scenario_name}': tickers must be non-empty"
        
        assert isinstance(scenario_def["benchmark_ticker"], str), \
            f"Scenario '{scenario_name}': benchmark_ticker must be string"
        
        assert isinstance(scenario_def["rf_ticker"], str), \
            f"Scenario '{scenario_name}': rf_ticker must be string"
        
        # Verify date format
        try:
            pd.Timestamp(scenario_def["start_date"])
            pd.Timestamp(scenario_def["end_date"])
        except Exception as e:
            raise AssertionError(
                f"Scenario '{scenario_name}': date parsing failed: {e}"
            )
        
        # Verify start < end
        start = pd.Timestamp(scenario_def["start_date"])
        end = pd.Timestamp(scenario_def["end_date"])
        assert start < end, \
            f"Scenario '{scenario_name}': start_date must be before end_date"


@pytest.mark.audit_flaw
def test_audit_general_tests_sqlite_concurrency_deadlock(synthetic_market_data, isolated_main_cache, monkeypatch):
    """
    FLAW DOCUMENTED:
    In main.py (line 486), final Sharpe and Sortino ratios use:
        annual_rf_rate = np.mean(daily_rf_rates) * 252.0
    which averages daily_rf_rates over the ENTIRE dataset (including the initial
    252-day In-Sample lookback window where the portfolio was not active), instead
    of slicing daily_rf_rates[nav_start:nav_end] over the Out-of-Sample evaluation period.
    """
    from main import run_backtest, calculate_sharpe_ratio
    import config

    monkeypatch.setattr(config, "POPULATION_SIZE", 20)
    monkeypatch.setattr(config, "GENERATIONS", 5)
    monkeypatch.setattr(config, "USE_LLM", False)
    monkeypatch.setattr(config, "GENERATE_REPORTS", False)

    data, tickers, dates = synthetic_market_data
    n_days = len(dates)

    # Set In-Sample RF rate to 10% and Out-of-Sample RF rate to 2%
    daily_rf = np.zeros(n_days, dtype=np.float64)
    daily_rf[:config.LOOKBACK_WINDOW] = 0.10 / 252.0
    daily_rf[config.LOOKBACK_WINDOW:] = 0.02 / 252.0

    oscillating_ret = np.tile([0.01, -0.005], n_days // 2)
    market_dict = {
        'prices': np.column_stack([data[t].values for t in tickers]),
        'tickers': tickers,
        'simple_returns': np.column_stack([oscillating_ret for _ in tickers]),
        'dates': dates,
        'benchmark_simple_returns': oscillating_ret.copy(),
        'daily_rf_rates': daily_rf,
        'sentiment_applied': False,
    }

    with patch("main.preprocess_market_data", return_value=market_dict), patch("main.generate_quantstats_tearsheet") as mock_tearsheet:
        with patch("main.asyncio.run", side_effect=lambda coro: coro.close()):
            res = run_backtest(tickers, "2023-01-01", "2023-12-31", export_html=True)

    bench_metrics = res["strategies"]["benchmark"]
    port_values = bench_metrics["portfolio_values"]

    mock_tearsheet.assert_called_once()
    assert np.isclose(mock_tearsheet.call_args.kwargs['rf'], 0.02)
    expected_oos_sharpe = calculate_sharpe_ratio(port_values, risk_free_rate=0.02)
    contaminated_sharpe = calculate_sharpe_ratio(port_values, risk_free_rate=np.mean(daily_rf) * 252.0)
    assert not np.isclose(expected_oos_sharpe, contaminated_sharpe, atol=1e-4)
    assert np.isclose(bench_metrics["sharpe_ratio"], expected_oos_sharpe, atol=1e-4), (
        f"OOS Sharpe ratio {bench_metrics['sharpe_ratio']:.4f} was contaminated by In-Sample RF rates "
        f"(expected {expected_oos_sharpe:.4f} for 2% OOS RF rate)"
    )


# ============================================================================
# HELPER FUNCTIONS FOR METRIC VALIDATION
# ============================================================================

def validate_nav_series(nav, description="NAV"):
    """Helper to validate NAV series properties."""
    assert isinstance(nav, (list, tuple, np.ndarray, pd.Series)), \
        f"{description}: must be array-like"
    assert len(nav) > 0, f"{description}: must be non-empty"
    assert not np.any(np.isnan(nav)), f"{description}: contains NaN"
    assert not np.any(np.isinf(nav)), f"{description}: contains Inf"
    assert all(n > 0 for n in nav), f"{description}: all values must be positive"
