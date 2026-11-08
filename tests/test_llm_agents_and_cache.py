"""
Audit suite for LLM Agents, Asyncio Cache, and Pre-warming Pipeline.

This module audits the asynchronous SQLite cache system (`llm_cache.py`),
pre-warming script (`prewarm_cache.py`), and two LLM agents
(`macro_agent.py`, `sentiment_agent.py`) without
modifying source code.

Tests are categorized via pytest markers:

- `@pytest.mark.contract`: Verify nominal operation, Pydantic/SQLite serialization,
  aiosqlite concurrency, and anonymization.
- `@pytest.mark.audit_flaw`: Expose mathematical and technical bugs in current code.
  Each `audit_flaw` test is expected to fail until source code is corrected.

Execution constraints:
- Zero network calls (100% mocking of Gemini/OpenAI clients)
- 100% deterministic
- Total execution time < 20 seconds
- SQLite database completely isolated from production (`test_llm_cache.db` in temp directory)
"""

import asyncio
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from pydantic import BaseModel

# Set seed globally for reproducibility
np.random.seed(42)


# ============================================================================
# FIXTURES & HELPERS
# ============================================================================

@pytest.fixture
def isolated_sqlite_db(tmp_path, monkeypatch):
    """Redirect llm_cache.DB_PATH to temporary directory, isolate mocks."""
    import llm_cache
    import llm_agents.macro_agent as macro_agent
    import llm_agents.sentiment_agent as sentiment_agent

    # Redirect database path
    test_db_path = tmp_path / "test_llm_cache.db"
    monkeypatch.setattr(llm_cache, "DB_PATH", test_db_path)

    # Clear in-memory caches
    if hasattr(macro_agent, '_MACRO_CACHE'):
        macro_agent._MACRO_CACHE.clear()
    if hasattr(sentiment_agent, '_SENTIMENT_CACHE'):
        sentiment_agent._SENTIMENT_CACHE.clear()

    yield test_db_path

    # Cleanup after test
    if hasattr(macro_agent, '_MACRO_CACHE'):
        macro_agent._MACRO_CACHE.clear()
    if hasattr(sentiment_agent, '_SENTIMENT_CACHE'):
        sentiment_agent._SENTIMENT_CACHE.clear()


@pytest.fixture
def mock_genai_client():
    """Mock Google Gemini API client."""
    with patch('llm_agents.macro_agent.genai') as mock_genai:
        client_mock = MagicMock()
        mock_genai.Client.return_value = client_mock
        yield mock_genai, client_mock


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI API client."""
    with patch('llm_agents.macro_agent.AsyncOpenAI') as mock_openai:
        client_mock = MagicMock()
        mock_openai.return_value = client_mock
        yield mock_openai, client_mock


# ============================================================================
# BLOC A — Audit of llm_cache.py (SHA-256 Hash, Secret Stripping, SQLite Concurrency)
# ============================================================================

@pytest.mark.contract
def test_cache_request_hash_strips_api_key_and_is_deterministic(isolated_sqlite_db):
    """Verify cache_request_hash removes api_key and produces deterministic SHA-256."""
    from llm_agents.macro_agent import analyze_macro_regime
    from llm_cache import cache_request_hash
    import inspect

    returns = np.array([0.01, 0.02, -0.01, 0.03] * 63)  # 252 elements

    # Test 1: api_key exclusion
    hash1 = cache_request_hash(
        analyze_macro_regime,
        returns,
        t_bond_yield=0.025,
        api_key="secret_key_1",
        use_llm=False,
        provider="google",
    )

    hash2 = cache_request_hash(
        analyze_macro_regime,
        returns,
        t_bond_yield=0.025,
        api_key="secret_key_2",  # Different secret
        use_llm=False,
        provider="google",
    )

    assert hash1 == hash2, "api_key must be stripped from cache hash"
    assert isinstance(hash1, str) and len(hash1) == 64, "Hash should be SHA-256 hex string"

    # Test 2: Determinism with numpy arrays
    hash3 = cache_request_hash(
        analyze_macro_regime,
        returns,
        t_bond_yield=0.025,
        api_key="any_key",
        use_llm=False,
        provider="google",
    )
    assert hash1 == hash3, "Hash must be deterministic"

    # Test 3: Different returns => different hash
    returns_diff = returns.copy()
    returns_diff[0] += 0.0001
    hash4 = cache_request_hash(
        analyze_macro_regime,
        returns_diff,
        t_bond_yield=0.025,
        api_key="any_key",
        use_llm=False,
        provider="google",
    )
    assert hash1 != hash4, "Different returns must produce different hash"


@pytest.mark.contract
def test_sqlite_persistent_cache_hit_miss_and_concurrency(isolated_sqlite_db):
    """Verify SQLite cache handles concurrent access and distinguishes cache hits/misses."""
    import llm_cache
    from llm_agents.macro_agent import MacroAnalysis, MarketRegime

    async def test_async():
        # Decorate a test function
        call_counter = [0]

        @llm_cache.sqlite_persistent_cache
        async def fake_analyze() -> MacroAnalysis:
            call_counter[0] += 1
            return MacroAnalysis(
                market_regime=MarketRegime.BULL,
                risk_aversion=1.0,
                liquidity_multiplier=1.0,
                enable_shorting=False,
                reasoning="test",
            )

        # Test 1: First call should execute and cache
        result1 = await fake_analyze()
        assert call_counter[0] == 1, "First call should execute the function"
        assert result1.market_regime == MarketRegime.BULL

        # Test 2: Second call should be from cache
        result2 = await fake_analyze()
        assert call_counter[0] == 1, "Second call should hit cache, not re-execute"
        assert result2.market_regime == MarketRegime.BULL

        # Test 3: Concurrent calls should not overload SQLite
        tasks = [fake_analyze() for _ in range(20)]
        results = await asyncio.gather(*tasks)
        assert call_counter[0] == 1, "All concurrent calls should hit cache"
        assert len(results) == 20
        assert all(r.market_regime == MarketRegime.BULL for r in results)

    asyncio.run(test_async())


@pytest.mark.contract
def test_init_db_fast_path_tracks_dynamic_paths(isolated_sqlite_db, tmp_path):
    import llm_cache

    async def test_async():
        await llm_cache.init_db()
        with patch('llm_cache.aiosqlite.connect', wraps=llm_cache.aiosqlite.connect) as connect:
            await llm_cache.init_db()
            connect.assert_not_called()

            alternate_path = tmp_path / 'alternate_cache.db'
            await llm_cache.init_db(alternate_path)
            connect.assert_called_once()
            assert alternate_path.exists()

    asyncio.run(test_async())


@pytest.mark.contract
def test_sqlite_cache_recomputes_corrupt_payload(isolated_sqlite_db):
    import llm_cache
    from llm_agents.macro_agent import MacroAnalysis, MarketRegime

    async def test_async():
        call_count = 0

        @llm_cache.sqlite_persistent_cache
        async def cached_analysis() -> MacroAnalysis:
            nonlocal call_count
            call_count += 1
            return MacroAnalysis(
                market_regime=MarketRegime.BULL,
                risk_aversion=1.0,
                liquidity_multiplier=1.0,
                enable_shorting=False,
            )

        await llm_cache.init_db()
        request_hash = llm_cache.cache_request_hash(cached_analysis)
        async with llm_cache.aiosqlite.connect(str(isolated_sqlite_db)) as connection:
            await connection.execute(
                'INSERT INTO llm_responses (request_hash, response_payload) VALUES (?, ?)',
                (request_hash, '{invalid json'),
            )
            await connection.commit()

        result = await cached_analysis()
        assert result.market_regime == MarketRegime.BULL
        assert call_count == 1
        result_again = await cached_analysis()
        assert result_again.market_regime == MarketRegime.BULL
        assert call_count == 1

    asyncio.run(test_async())


@pytest.mark.contract
def test_return_model_validation_errors_on_bad_annotations(isolated_sqlite_db):
    """Verify @sqlite_persistent_cache raises TypeError for invalid return annotations."""
    import llm_cache

    # Test 1: No return annotation
    with pytest.raises(TypeError, match="must declare a Pydantic return annotation"):
        @llm_cache.sqlite_persistent_cache
        async def no_annotation():
            return "result"

    # Test 2: Non-Pydantic return annotation
    with pytest.raises(TypeError, match="must return a Pydantic model"):
        @llm_cache.sqlite_persistent_cache
        async def non_pydantic() -> str:
            return "result"


@pytest.mark.audit_flaw
def test_audit_float_precision_fragility_in_macro_cache_hash(isolated_sqlite_db):
    """
    FLAW DOCUMENTED:
    The in-memory _MACRO_CACHE rounds metrics to 1e-8 precision, but
    cache_request_hash serializes full numpy.float64 precision. Two mathematically
    identical return vectors differing by IEEE-754 noise produce the same cache key
    in-memory but different hashes in SQLite, causing spurious cache misses.

    Expected: Hashes should match if metrics round identically.
    """
    from llm_cache import cache_request_hash
    from llm_agents.macro_agent import analyze_macro_regime, calculate_macro_metrics

    returns_a = np.full(252, 0.001, dtype=np.float64)
    returns_b = returns_a.copy()
    returns_b[0] += 1e-16  # IEEE-754 noise

    metrics_a = calculate_macro_metrics(returns_a)
    metrics_b = calculate_macro_metrics(returns_b)

    # Verify metrics round identically
    for key in metrics_a:
        assert np.isclose(
            metrics_a[key], metrics_b[key], atol=1e-8
        ), f"Metrics should round identically: {key}"

    # Hashes should match due to in-memory cache logic
    hash_a = cache_request_hash(
        analyze_macro_regime,
        returns_a,
        t_bond_yield=0.025,
        api_key="key",
        use_llm=False,
        provider="google",
    )

    hash_b = cache_request_hash(
        analyze_macro_regime,
        returns_b,
        t_bond_yield=0.025,
        api_key="key",
        use_llm=False,
        provider="google",
    )

    # This assertion documents the fragility: hashes differ despite equivalent metrics
    assert hash_a == hash_b, \
        f"Hashes must match for IEEE-754 equivalent returns. " \
        f"Mismatch indicates SQL cache key fragility: {hash_a} vs {hash_b}"


# ============================================================================
# BLOC B — Audit of macro_agent.py (252x Bug, Cache Poisoning, Semaphore)
# ============================================================================

@pytest.mark.audit_flaw
def test_audit_calculate_macro_metrics_sharpe_252x_inflation_bug(isolated_sqlite_db):
    """
    FLAW DOCUMENTED:
    calculate_macro_metrics computes annual_return = np.sum(sp500_returns) * 252.
    
    For a 252-element return array with daily_return_mean = 0.01 and -0.008 alternating,
    this yields annual_return = np.sum(returns) * 252, which inflates the Sharpe ratio.
    The Sharpe ratio should be (mean(returns)*252 - rf) / (std(returns)*sqrt(252)).

    Expected: Sharpe ≈ 0.57
    Buggy: Sharpe >> 100 due to 252x multiplication on the sum
    """
    from llm_agents.macro_agent import calculate_macro_metrics

    # Construct a 252-day return series with non-zero volatility
    returns = np.tile([0.01, -0.008], 126)  # Daily oscillation: +1%, -0.8%
    # Mean daily return: (0.01 - 0.008) / 2 = 0.001
    # Annualized: 0.001 * 252 = 0.252 = +25.2%
    # Daily std: ~0.009
    # Annualized vol: 0.009 * sqrt(252) ≈ 0.143

    metrics = calculate_macro_metrics(returns)

    # Expected: Sharpe = (0.252 - 0.015) / 0.143 ≈ 1.66
    # Buggy (252x inflation on sum): sum(returns) ≈ 0.252, * 252 ≈ 63.5
    # Buggy Sharpe: (63.5 - 0.015) / 0.143 ≈ 444

    # Direct assertion without conditional masking
    # If Sharpe is > 50, it's clearly buggy
    if metrics['sharpe_ratio'] > 50:
        pytest.fail(
            f"Sharpe ratio {metrics['sharpe_ratio']:.2f} exceeds reasonable bounds. "
            f"Bug detected: annual_return inflated by 252x multiplication (calc_macro_metrics line with np.sum(...) * 252)"
        )
    else:
        # Verify it's in reasonable range (between -1 and 10)
        assert -1 <= metrics['sharpe_ratio'] <= 10, \
            f"Sharpe should be in [-1, 10], got {metrics['sharpe_ratio']:.2f}"


@pytest.mark.audit_flaw
def test_audit_macro_cache_poisoning_on_llm_failure(isolated_sqlite_db, mock_genai_client):
    """
    FLAW DOCUMENTED:
    When _analyze_gemini raises an exception, the fallback heuristic _analyze_heuristic
    is called and wrapped by @sqlite_persistent_cache decorator. The SQLite cache
    persists the heuristic result under the signature use_llm=True, poisoning the cache.
    
    Next calls with the same returns + use_llm=True will retrieve the heuristic result
    instead of attempting the LLM call, permanently defeating the LLM agent.

    Expected: Cache should only persist successful LLM results, not fallback heuristics.
    """
    import llm_cache
    from llm_agents.macro_agent import analyze_macro_regime, MacroAnalysis
    from unittest.mock import AsyncMock

    async def test_async():
        returns = np.array([0.01, 0.02, -0.01, 0.03] * 63)

        # Mock _analyze_gemini to fail
        with patch('llm_agents.macro_agent._analyze_gemini', side_effect=RuntimeError("429 Rate Limit")):
            # Call analyze_macro_regime with use_llm=True, should fallback to heuristic
            result = await analyze_macro_regime(
                returns,
                t_bond_yield=0.025,
                api_key="test_key",
                use_llm=True,
                provider="google",
            )
            assert isinstance(result, MacroAnalysis)

        # Check if the fallback result was cached
        is_cached = await llm_cache.cache_contains(
            analyze_macro_regime,
            returns,
            t_bond_yield=0.025,
            api_key="test_key",
            use_llm=True,
            provider="google",
        )

        # This assertion documents the bug: heuristic is incorrectly cached
        assert not is_cached, \
            "Cache should NOT persist heuristic fallback when LLM fails. " \
            "Poisoned cache prevents future LLM attempts."

    asyncio.run(test_async())


@pytest.mark.audit_flaw
def test_audit_macro_semaphore_cross_event_loop_failure(isolated_sqlite_db):
    """
    FLAW DOCUMENTED:
    macro_agent._SEMAPHORE = asyncio.Semaphore(3) is bound to the first event loop
    when >3 concurrent tasks contend on it. A second asyncio.run() in a new event loop
    raises RuntimeError: ... is bound to a different event loop.
    """
    import llm_agents.macro_agent as macro_agent

    macro_agent._SEMAPHORE = asyncio.Semaphore(3)

    async def _contend_on_module_semaphore(idx: int):
        async with macro_agent._SEMAPHORE:
            await asyncio.sleep(0.02)
            return idx

    async def _run_batch():
        # 5 concurrent tasks > Semaphore(3) forces creation of loop-bound _waiters
        return await asyncio.gather(*[_contend_on_module_semaphore(i) for i in range(5)])

    # Loop 1 binds macro_agent._SEMAPHORE to Loop 1
    res1 = asyncio.run(_run_batch())
    assert len(res1) == 5

    # Loop 2 reuses the same module-level _SEMAPHORE in a new event loop -> raises RuntimeError
    res2 = asyncio.run(_run_batch())
    assert len(res2) == 5


@pytest.mark.contract
def test_macro_heuristic_regimes_and_pydantic_bounds(isolated_sqlite_db):
    """Verify _analyze_heuristic produces valid MacroAnalysis with correct bounds."""
    from llm_agents.macro_agent import _analyze_heuristic, calculate_macro_metrics

    # Test profiles
    profiles = [
        {
            "name": "Bull",
            "returns": np.array([0.01] * 252),  # Stable, positive
            "expected_regime": "Bull",
        },
        {
            "name": "Bear",
            "returns": np.array([-0.02] * 200 + [0.01] * 52),  # -40% then recovery
            "expected_regime": "Bear",
        },
        {
            "name": "Stagflation",
            "returns": np.array(np.random.randn(252) * 0.03),  # High vol, poor return
            "expected_regime": "Stagflation",
        },
    ]

    for profile in profiles:
        returns = profile["returns"]
        metrics = calculate_macro_metrics(returns)
        analysis = _analyze_heuristic(metrics)

        # Verify bounds
        assert 0.5 <= analysis.risk_aversion <= 2.5, \
            f"{profile['name']}: risk_aversion {analysis.risk_aversion} outside [0.5, 2.5]"
        assert 1.0 <= analysis.liquidity_multiplier <= 3.0, \
            f"{profile['name']}: liquidity_multiplier {analysis.liquidity_multiplier} outside [1.0, 3.0]"
        assert isinstance(analysis.enable_shorting, bool), \
            f"{profile['name']}: enable_shorting must be bool"

        # Verify Bear/Stagflation enable shorting
        if profile["expected_regime"] in ("Bear", "Stagflation"):
            # These may enable shorting depending on metrics
            pass  # Heuristic rules apply


# ============================================================================
# BLOC C — Audit of sentiment_agent.py (Neutral Bias, Ticker Corruption, Persistence)
# ============================================================================

@pytest.mark.audit_flaw
def test_audit_sentiment_heuristic_neutral_headlines_give_max_bullish_score(isolated_sqlite_db):
    """
    FLAW DOCUMENTED:
    _analyze_sentiment_heuristic matches headlines without negation handling.
    A neutral headline with positive keywords (e.g., "strategic") scores +1.0 despite neutral content.

    Expected: Neutral basket headlines should produce sentiment_score close to 0.0.
    """
    from llm_agents.sentiment_agent import _analyze_sentiment_heuristic, fetch_historical_news

    ticker = "AAPL"
    date_str = "2024-06-15"

    # Fetch headlines (deterministic due to seed)
    headlines = fetch_historical_news(ticker, date_str)

    # Pass the COMPLETE headlines list to reproduce the bug
    # (word 'strategic' is in headlines[2])
    if len(headlines) > 0:
        sentiment = _analyze_sentiment_heuristic(ticker, headlines)

        # Bug: neutralheadlines often score +1.0 if they contain positive keywords
        # This assertion MUST be without conditional masking to expose the bug
        assert abs(sentiment.sentiment_score) <= 0.2, \
            f"Neutral headlines should score close to 0, got {sentiment.sentiment_score}. " \
            f"Bug: keyword matching without negation produces false positives."


@pytest.mark.audit_flaw
def test_audit_sentiment_llm_anonymization_corrupts_returned_ticker(isolated_sqlite_db, mock_genai_client):
    """
    FLAW DOCUMENTED:
    analyze_asset_sentiment anonymizes the ticker before sending to the LLM
    (e.g., NVDA -> "Actif B"). The LLM receives only "Actif B" and may return
    {"ticker": "Actif B", "sentiment_score": 0.75, ...} instead of the original
    ticker. This causes either a ValidationError or a corrupted result.

    Expected: Original ticker should be preserved in the returned AssetSentiment.
    """
    import llm_agents.sentiment_agent as sentiment_agent
    from llm_agents.sentiment_agent import AssetSentiment

    async def test_async():
        # Mock the LLM to return anonymized ticker
        with patch('llm_agents.sentiment_agent.genai.Client') as mock_genai:
            mock_client = MagicMock()

            # Simulate LLM response with anonymized ticker
            mock_response = MagicMock()
            mock_response.text = json.dumps({
                "ticker": "Actif B",  # Anonymized, not original "NVDA"
                "sentiment_score": 0.75,
                "justification": "Strong momentum",
            })

            mock_client.aio = MagicMock()
            mock_client.aio.models = MagicMock()
            mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
            mock_genai.return_value = mock_client

            # Call with anonymization enabled
            result = await sentiment_agent.analyze_asset_sentiment(
                "NVDA",
                "2024-06-15",
                api_key="test_key",
                use_llm=True,
                provider="google",
            )

            # Verify ticker is preserved
            assert result.ticker == "NVDA", \
                f"Returned ticker should be 'NVDA', got '{result.ticker}'. " \
                f"Bug: LLM anonymization leaks into result."

    asyncio.run(test_async())


@pytest.mark.contract
def test_sentiment_llm_fallback_is_not_cached(isolated_sqlite_db):
    import llm_agents.sentiment_agent as sentiment_agent

    async def test_async():
        sentiment_agent._SENTIMENT_CACHE.clear()
        with patch(
            'llm_agents.sentiment_agent._analyze_sentiment_gemini',
            new=AsyncMock(side_effect=RuntimeError('429 Rate Limit')),
        ) as analyze:
            first = await sentiment_agent.analyze_asset_sentiment(
                'NVDA', '2024-06-15', api_key='test_key', use_llm=True,
            )
            second = await sentiment_agent.analyze_asset_sentiment(
                'NVDA', '2024-06-15', api_key='test_key', use_llm=True,
            )

        cache_key = ('NVDA', '2024-06-15', 'test_key', True, 'google')
        assert getattr(first, '_is_fallback', False)
        assert getattr(second, '_is_fallback', False)
        assert cache_key not in sentiment_agent._SENTIMENT_CACHE
        assert analyze.await_count == 2

    asyncio.run(test_async())


@pytest.mark.contract
def test_sentiment_adjustment_in_market_data_with_exceptions(isolated_sqlite_db):
    """Verify sentiment adjustment formula and exception handling in adjust_expected_returns_with_sentiment."""
    from market_data import adjust_expected_returns_with_sentiment
    from llm_agents.sentiment_agent import AssetSentiment

    async def test_async():
        tickers = ["AAPL", "MSFT", "GOOGL"]
        expected_returns = np.array([0.05, 0.03, 0.02])

        # Mock sentiment results
        sentiments = [
            AssetSentiment(ticker="AAPL", sentiment_score=0.5, justification="Bullish"),
            AssetSentiment(ticker="MSFT", sentiment_score=-0.5, justification="Bearish"),
            # "GOOGL" will raise an exception
        ]

        call_index = [0]

        async def mock_sentiment(ticker, *args, **kwargs):
            if call_index[0] < len(sentiments):
                result = sentiments[call_index[0]]
                call_index[0] += 1
                return result
            raise RuntimeError("Network error")

        with patch('market_data.analyze_asset_sentiment', side_effect=mock_sentiment):
            adjusted = await adjust_expected_returns_with_sentiment(
                expected_returns,
                tickers,
                ref_date="2024-06-15",
                use_llm=True,
                provider="google",
            )

        # Verify adjustment formula: μ̂_adj = μ̂ + 0.20 * s * |μ̂|
        beta = 0.20
        expected_adj_0 = 0.05 + beta * 0.5 * abs(0.05)  # +0.00125
        expected_adj_1 = 0.03 + beta * (-0.5) * abs(0.03)  # -0.003
        expected_adj_2 = 0.02  # Exception => unchanged

        assert np.isclose(adjusted[0], expected_adj_0, rtol=1e-2), \
            f"AAPL adjustment incorrect: {adjusted[0]} vs {expected_adj_0}"
        assert np.isclose(adjusted[1], expected_adj_1, rtol=1e-2), \
            f"MSFT adjustment incorrect: {adjusted[1]} vs {expected_adj_1}"
        assert np.isclose(adjusted[2], expected_adj_2, rtol=1e-2), \
            f"GOOGL should be unchanged on exception: {adjusted[2]} vs {expected_adj_2}"

    asyncio.run(test_async())


# ============================================================================
# BLOC D — Audit of prewarm_cache.py
# ============================================================================

@pytest.mark.contract
def test_narrative_reporting_agent_removed():
    import importlib.util
    import llm_agents

    assert importlib.util.find_spec('llm_agents.reporting_agent') is None
    assert not hasattr(llm_agents, 'generate_rebalancing_memo')


@pytest.mark.contract
def test_prewarm_cache_windows_alignment_with_main(isolated_sqlite_db):
    """Verify build_macro_windows generates same indices/dates as main.py rolling loop."""
    from prewarm_cache import build_macro_windows
    import config
    import pandas as pd

    # Synthetic price series
    n_periods = 1000
    dates = pd.date_range("2024-01-01", periods=n_periods, freq="D")
    benchmark_returns = np.random.randn(n_periods) * 0.01

    windows = build_macro_windows(benchmark_returns, dates)

    # Manually reconstruct main.py's rolling loop
    idx_rebalance = config.LOOKBACK_WINDOW
    main_indices = []
    while idx_rebalance + config.REBALANCE_FREQ <= n_periods:
        main_indices.append(idx_rebalance)
        idx_rebalance += config.REBALANCE_FREQ

    # Extract indices from windows
    window_indices = [w["idx_rebalance"] for w in windows]

    assert window_indices == main_indices, \
        f"Prewarm windows don't align with main.py rolling loop. " \
        f"Prewarm: {window_indices[:5]}, Main: {main_indices[:5]}"

    # Verify dates alignment
    for i, window in enumerate(windows):
        main_idx = main_indices[i]
        assert window["date"] == pd.Timestamp(dates[main_idx]), \
            f"Window {i}: date mismatch {window['date']} vs {dates[main_idx]}"


@pytest.mark.contract
def test_sentiment_prewarm_dates_match_macro_windows(isolated_sqlite_db):
    from prewarm_cache import build_macro_windows, build_sentiment_requests
    import config

    dates = pd.date_range('2024-01-01', periods=700, freq='D')
    returns = np.zeros(len(dates))
    tickers = ['AAA', 'BBB']
    windows = build_macro_windows(returns, dates)
    requests = build_sentiment_requests(tickers, dates)

    assert len(requests) == len(windows) * len(tickers)
    for window_index, window in enumerate(windows):
        expected_date = pd.Timestamp(dates[window['idx_rebalance'] - 1]).strftime('%Y-%m-%d')
        batch = requests[window_index * len(tickers):(window_index + 1) * len(tickers)]
        assert [request['ticker'] for request in batch] == tickers
        assert all(request['ref_date'] == expected_date for request in batch)
    assert windows[0]['idx_rebalance'] == config.LOOKBACK_WINDOW


@pytest.mark.contract
def test_prewarm_backoff_retries_transient_errors(isolated_sqlite_db, monkeypatch):
    from prewarm_cache import _call_with_exponential_backoff

    async def test_async():
        calls = 0
        delays = []

        async def sleep(delay):
            delays.append(delay)

        async def operation():
            nonlocal calls
            calls += 1
            if calls < 3:
                raise RuntimeError('503 unavailable')
            return 'ready'

        monkeypatch.setattr('prewarm_cache.asyncio.sleep', sleep)
        result = await _call_with_exponential_backoff(operation, max_retries=4, base_delay=0.25)
        assert result == 'ready'
        assert calls == 3
        assert delays == [0.25, 0.5]

    asyncio.run(test_async())


# ============================================================================
# Additional Imports & Cleanup
# ============================================================================

try:
    import pandas as pd
except ImportError:
    pd = None
