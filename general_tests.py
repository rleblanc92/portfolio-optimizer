import os
import logging
import math
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

import pandas as pd
from tqdm.auto import tqdm

from logging_utils import configure_logging, get_logger, get_progress_stream
from main import run_backtest
from scenarios import SCENARIOS


STRATEGIES = (
    'genetic_algorithm',
    'variance_minimization',
    'maximum_sharpe',
    'benchmark',
)
METRICS = (
    'annual_return',
    'annual_volatility',
    'sharpe_ratio',
    'sortino_ratio',
    'max_drawdown',
    'calmar_ratio',
    'median_turnover',
    'avg_turnover',
)
GA_METRIC_NAMES = {
    'sharpe': 'sharpe_ratio',
    'sortino': 'sortino_ratio',
    'max_drawdown': 'max_drawdown',
    'calmar': 'calmar_ratio',
    'cagr': 'annual_return',
    'avg_turnover': 'avg_turnover',
}

logger = get_logger(__name__)


class _WorkerStatusHandler(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.messages = []
        self._seen = set()

    def emit(self, record):
        message = f'{record.name}: {record.getMessage()}'[:512]
        if message not in self._seen:
            if len(self.messages) < 10:
                self._seen.add(message)
                self.messages.append(message)
            elif len(self.messages) == 10:
                self.messages.append('Avertissements supplémentaires omis.')


def _initialize_scenario_worker():
    null_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(null_fd, 1)
        os.dup2(null_fd, 2)
    finally:
        if null_fd not in (1, 2):
            os.close(null_fd)
    portfolio_logger = logging.getLogger('portfolio')
    portfolio_logger.handlers = [logging.NullHandler()]
    portfolio_logger.setLevel(logging.WARNING)
    root_logger = logging.getLogger()
    root_logger.handlers = [logging.NullHandler()]
    root_logger.setLevel(logging.WARNING)
    logging.captureWarnings(True)


def run_single_scenario(scenario_name, params):
    """Exécute un scénario isolé et capture toute erreur de backtest."""
    time.sleep(random.uniform(1.0, 3.0))
    start_time = time.perf_counter()
    status_handler = _WorkerStatusHandler()
    portfolio_logger = logging.getLogger('portfolio')
    root_logger = logging.getLogger()
    portfolio_logger.addHandler(status_handler)
    root_logger.addHandler(status_handler)
    try:
        results = run_backtest(
            params['tickers'],
            params['start_date'],
            params['end_date'],
            params['benchmark_ticker'],
            params.get('rf_ticker'),
            scenario_name=scenario_name,
            export_html=False,
            export_md=False,
            record_history=False,
        )
        strategies = results.get('strategies', {})
        metrics = {
            f'{strategy_name}_{metric_name}': (
                float(strategy_metrics[metric_name])
                if strategy_metrics.get(metric_name) is not None else None
            )
            for strategy_name in STRATEGIES
            for strategy_metrics in [strategies.get(strategy_name) or {}]
            for metric_name in METRICS
        }
        return {
            'config_id': scenario_name,
            'error': None,
            'duration_seconds': time.perf_counter() - start_time,
            'warnings': '\n'.join(status_handler.messages),
            **metrics,
            **{name: metrics[f'genetic_algorithm_{metric}']
               for name, metric in GA_METRIC_NAMES.items()},
        }
    except Exception as exc:
        return {
            'config_id': scenario_name,
            'error': f'{type(exc).__name__}: {exc}'[:1024],
            'duration_seconds': time.perf_counter() - start_time,
            'warnings': '\n'.join(status_handler.messages),
        }
    finally:
        portfolio_logger.removeHandler(status_handler)
        root_logger.removeHandler(status_handler)


def _scenario_row(scenario_name, outcome):
    row = {
        'config_id': scenario_name,
        'scenario': scenario_name,
        'status': 'ok' if outcome['error'] is None else 'error',
        'error': outcome['error'],
        'duration_seconds': outcome['duration_seconds'],
    }
    for strategy_name in STRATEGIES:
        for metric_name in METRICS:
            name = f'{strategy_name}_{metric_name}'
            row[name] = outcome.get(name)
    for name in GA_METRIC_NAMES:
        row[name] = outcome.get(name)
    return row


def _collect_outcomes(futures):
    outcomes = {}
    best_sharpe = None
    with tqdm(as_completed(futures), total=len(futures), desc="Multi-Test Parallèle",
              unit="config", leave=True, mininterval=0.5,
              file=get_progress_stream()) as pbar:
        for completed_count, future in enumerate(pbar, start=1):
            scenario_name = futures[future]
            try:
                outcome = future.result()
            except Exception as exc:
                outcome = {
                    'config_id': scenario_name,
                    'error': f'{type(exc).__name__}: {exc}',
                    'duration_seconds': 0.0,
                    'warnings': '',
                }
            outcomes[scenario_name] = outcome
            if outcome['warnings']:
                logger.warning(f"{scenario_name}: {outcome['warnings']}")
            if outcome['error'] is not None:
                logger.error(f"Échec du scénario {scenario_name}: {outcome['error']}")
            sharpe = outcome.get('sharpe')
            if sharpe is not None and math.isfinite(sharpe):
                best_sharpe = sharpe if best_sharpe is None else max(best_sharpe, sharpe)
            pbar.set_postfix({
                'Best_Sharpe': f'{best_sharpe:.2f}' if best_sharpe is not None else 'N/A',
                'Completed': f'{completed_count}/{len(futures)}',
            })
    return outcomes


def run_general_tests(scenarios, max_workers=2):
    os.makedirs('reports', exist_ok=True)
    scenario_items = list(scenarios.items())

    with ProcessPoolExecutor(max_workers=max_workers, mp_context=get_context('spawn'),
                             initializer=_initialize_scenario_worker) as executor:
        futures = {}
        for index, (scenario_name, params) in enumerate(scenario_items):
            futures[executor.submit(run_single_scenario, scenario_name, params)] = scenario_name
            if index < len(scenario_items) - 1:
                time.sleep(1.0)
        outcomes = _collect_outcomes(futures)
        rows = [_scenario_row(scenario_name, outcomes[scenario_name])
                for scenario_name, _ in scenario_items]

    df = pd.DataFrame(rows)
    if not df.empty:
        df.sort_values(
            'sharpe', ascending=False, na_position='last', kind='stable', inplace=True,
            key=lambda values: values.where(values.map(lambda value: pd.notna(value) and math.isfinite(value))),
        )
    df.to_csv('reports/global_results_configuration.csv', index=False)
    candidates = [
        row for row in rows
        if row['status'] == 'ok' and row['sharpe'] is not None and math.isfinite(row['sharpe'])
    ]
    if candidates:
        candidates_sorted = sorted(candidates, key=lambda row: row['sharpe'])
        for candidate in candidates_sorted:
            config_id = candidate['config_id']
            params = scenarios[config_id]
            logger.info("Génération des rapports (TearSheet & Historique) pour : %s", config_id)
            run_backtest(
                params['tickers'], params['start_date'], params['end_date'],
                params['benchmark_ticker'], params.get('rf_ticker'),
                scenario_name=config_id, export_html=True, export_md=False,
                record_history=True, history_config_id=config_id,
            )
    else:
        logger.warning("Aucune configuration avec un Sharpe AG fini ; réexécution ignorée.")
    return df


if __name__ == '__main__':
    configure_logging()
    logger.info("Début du multi-test")
    df = run_general_tests(SCENARIOS)
    logger.info("Fin du multi-test : %d scénarios", len(df))
    print(df.to_string(index=False))
