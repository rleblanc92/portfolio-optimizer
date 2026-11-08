"""Prewarm the persistent macro-agent cache before running a backtest.

This module deliberately performs no portfolio optimization. It downloads the
market inputs, reproduces the rolling benchmark windows used by ``main.py``
and persists one macro analysis per rebalance date.

Examples
--------
    python prewarm_cache.py
    python prewarm_cache.py FR_Covid_2020_2021
    python prewarm_cache.py all
"""

from __future__ import annotations

import argparse
import asyncio
from typing import Any

import numpy as np
import pandas as pd

import config
from llm_agents import (
    analyze_asset_sentiment,
    analyze_macro_regime,
    calculate_macro_metrics,
)
from llm_cache import cache_contains, init_db
from market_data import preprocess_market_data
from scenarios import SCENARIOS


PREWARM_CONCURRENCY = 5
DEFAULT_T_BOND_YIELD = 0.025
DEFAULT_SCENARIO = "Global_MultiAsset_2024"


async def _call_with_exponential_backoff(
    coro_factory,
    max_retries: int = 4,
    base_delay: float = 1.0,
):
    """Retry transient LLM failures, including agent-level fallback results."""
    for attempt in range(max_retries + 1):
        try:
            result = await coro_factory()
            if getattr(result, "_is_fallback", False):
                raise RuntimeError("LLM unavailable: agent returned fallback")
            return result
        except Exception as error:
            message = f"{type(error).__name__}: {error}".lower()
            is_transient = any(
                marker in message
                for marker in ("429", "ratelimit", "rate limit", "503", "timeout", "timed out", "unavailable")
            )
            if not is_transient or attempt == max_retries:
                raise
            await asyncio.sleep(base_delay * (2 ** attempt))


def build_macro_windows(
    benchmark_simple_returns: np.ndarray,
    dates: np.ndarray | pd.DatetimeIndex,
) -> list[dict[str, Any]]:
    """Reproduce the exact rolling windows consumed by ``main.py``."""
    n_periods = len(dates)
    if len(benchmark_simple_returns) != n_periods:
        raise ValueError(
            "Benchmark returns and market dates must have the same length."
        )

    windows = []
    idx_rebalance = config.LOOKBACK_WINDOW
    while idx_rebalance + config.REBALANCE_FREQ <= n_periods:
        # Keep these bounds identical to main.py's rolling backtest loop.
        in_sample_start = idx_rebalance - config.LOOKBACK_WINDOW
        in_sample_end = idx_rebalance
        benchmark_in_sample = benchmark_simple_returns[
            in_sample_start:in_sample_end
        ]
        metrics = calculate_macro_metrics(benchmark_in_sample)
        windows.append(
            {
                "idx_rebalance": idx_rebalance,
                "date": pd.Timestamp(dates[idx_rebalance]),
                "returns": benchmark_in_sample,
                "metrics": metrics,
            }
        )
        idx_rebalance += config.REBALANCE_FREQ

    return windows


def load_benchmark_windows(params: dict[str, Any]) -> list[dict[str, Any]]:
    """Load scenario data without invoking sentiment or portfolio logic."""
    market_data = preprocess_market_data(
        params["tickers"],
        params["start_date"],
        params["end_date"],
        apply_sentiment=False,
        use_llm=False,
        api_key=None,
        provider=config.LLM_PROVIDER,
        benchmark_ticker=params["benchmark_ticker"],
        rf_ticker=params.get("rf_ticker"),
    )
    benchmark_returns = market_data.get("benchmark_simple_returns")
    if benchmark_returns is None:
        raise RuntimeError("The scenario does not provide benchmark returns.")
    return build_macro_windows(benchmark_returns, market_data["dates"])


def build_sentiment_requests(
    tickers: list[str],
    dates: np.ndarray | pd.DatetimeIndex,
) -> list[dict[str, Any]]:
    """Build sentiment requests on the exact in-sample dates used by main.py."""
    n_periods = len(dates)
    requests = []
    idx_rebalance = config.LOOKBACK_WINDOW
    while idx_rebalance + config.REBALANCE_FREQ <= n_periods:
        ref_date = pd.Timestamp(dates[idx_rebalance - 1]).strftime("%Y-%m-%d")
        for ticker in tickers:
            requests.append({"ticker": ticker, "ref_date": ref_date})
        idx_rebalance += config.REBALANCE_FREQ
    return requests


def load_prewarm_inputs(
    params: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load a scenario once and derive macro windows plus sentiment requests."""
    market_data = preprocess_market_data(
        params["tickers"],
        params["start_date"],
        params["end_date"],
        apply_sentiment=False,
        use_llm=False,
        api_key=None,
        provider=config.LLM_PROVIDER,
        benchmark_ticker=params["benchmark_ticker"],
        rf_ticker=params.get("rf_ticker"),
    )
    benchmark_returns = market_data.get("benchmark_simple_returns")
    if benchmark_returns is None:
        raise RuntimeError("The scenario does not provide benchmark returns.")
    windows = build_macro_windows(benchmark_returns, market_data["dates"])
    requests = build_sentiment_requests(market_data["tickers"], market_data["dates"])
    return windows, requests


async def _prewarm_window(
    scenario_name: str,
    window: dict[str, Any],
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    """Populate one cache entry while respecting the global API limit."""
    request_kwargs = {
        "t_bond_yield": DEFAULT_T_BOND_YIELD,
        "api_key": config.LLM_API_KEY,
        "use_llm": config.USE_LLM,
        "provider": config.LLM_PROVIDER,
    }
    returns = window["returns"]

    async with semaphore:
        try:
            was_cached = await cache_contains(
                analyze_macro_regime,
                returns,
                **request_kwargs,
            )
            analysis = await _call_with_exponential_backoff(
                lambda: analyze_macro_regime(returns, **request_kwargs)
            )
            is_cached = await cache_contains(
                analyze_macro_regime,
                returns,
                **request_kwargs,
            )
            if was_cached:
                status = "HIT"
            elif is_cached:
                status = "STORED"
            else:
                status = "MISS"
            if getattr(analysis, "_is_fallback", False):
                status = "FALLBACK"
            return {
                "scenario": scenario_name,
                "date": window["date"],
                "status": status,
                "metrics": window["metrics"],
                "error": None,
            }
        except Exception as error:
            return {
                "scenario": scenario_name,
                "date": window["date"],
                "status": "ERROR",
                "metrics": window["metrics"],
                "error": f"{type(error).__name__}: {error}",
            }


async def prewarm_scenario(
    scenario_name: str,
    windows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Warm every rebalance window concurrently with bounded API pressure."""
    semaphore = asyncio.Semaphore(5)
    tasks = [
        _prewarm_window(scenario_name, window, semaphore)
        for window in windows
    ]
    return await asyncio.gather(*tasks)


async def prewarm_sentiment_scenario(
    scenario_name: str,
    requests: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Warm sentiment responses while bounding concurrent SDK requests."""
    semaphore = asyncio.Semaphore(PREWARM_CONCURRENCY)

    async def prewarm_request(request: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            try:
                result = await _call_with_exponential_backoff(
                    lambda: analyze_asset_sentiment(
                        request["ticker"],
                        request["ref_date"],
                        api_key=config.LLM_API_KEY,
                        use_llm=config.USE_LLM,
                        provider=config.LLM_PROVIDER,
                    )
                )
                return {
                    "scenario": scenario_name,
                    "ticker": request["ticker"],
                    "ref_date": request["ref_date"],
                    "status": "FALLBACK" if getattr(result, "_is_fallback", False) else "WARMED",
                    "sentiment_score": float(result.sentiment_score),
                    "error": None,
                }
            except Exception as error:
                return {
                    "scenario": scenario_name,
                    "ticker": request["ticker"],
                    "ref_date": request["ref_date"],
                    "status": "ERROR",
                    "sentiment_score": None,
                    "error": f"{type(error).__name__}: {error}",
                }

    return await asyncio.gather(*(prewarm_request(request) for request in requests))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Populate llm_cache.db for one or all configured scenarios."
    )
    parser.add_argument(
        "scenario",
        nargs="?",
        default=DEFAULT_SCENARIO,
        choices=(*SCENARIOS, "all"),
        help="Scenario name, or 'all' (default: active main.py scenario).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    scenario_items = (
        list(SCENARIOS.items())
        if args.scenario == "all"
        else [(args.scenario, SCENARIOS[args.scenario])]
    )

    asyncio.run(init_db())
    total_windows = 0
    total_sentiment_requests = 0
    for scenario_name, params in scenario_items:
        print(f"[{scenario_name}] Loading market data...", flush=True)
        windows, sentiment_requests = load_prewarm_inputs(params)
        total_windows += len(windows)
        print(
            f"[{scenario_name}] Prewarming {len(windows)} macro windows "
            f"with concurrency={PREWARM_CONCURRENCY}...",
            flush=True,
        )
        outcomes = asyncio.run(prewarm_scenario(scenario_name, windows))
        for outcome in outcomes:
            metrics = outcome["metrics"]
            suffix = (
                f" error={outcome['error']}" if outcome["error"] else ""
            )
            print(
                f"[{outcome['date'].strftime('%Y-%m-%d')}] "
                f"{outcome['status']} "
                f"vol={metrics['annual_volatility']:.6f} "
                f"dd={metrics['max_drawdown']:.6f}{suffix}",
                flush=True,
            )
        if config.USE_LLM:
            total_sentiment_requests += len(sentiment_requests)
            print(
                f"[{scenario_name}] Prewarming {len(sentiment_requests)} sentiment requests "
                f"with concurrency={PREWARM_CONCURRENCY}...",
                flush=True,
            )
            sentiment_outcomes = asyncio.run(
                prewarm_sentiment_scenario(scenario_name, sentiment_requests)
            )
            for outcome in sentiment_outcomes:
                suffix = f" error={outcome['error']}" if outcome["error"] else ""
                print(
                    f"[{outcome['ref_date']}] {outcome['ticker']} {outcome['status']}"
                    f"{suffix}",
                    flush=True,
                )

    print(
        f"Prewarm complete: {total_windows} macro windows and "
        f"{total_sentiment_requests} sentiment requests processed."
    )


if __name__ == "__main__":
    main()