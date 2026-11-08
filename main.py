"""
Orchestrateur de backtest et calcul des métriques de performance.
Stratégie de fenêtre glissante (Rolling Window) avec in-sample/out-of-sample.
Intégration de l'agent macro-économique pour ajustement dynamique du risque.
"""
# Force la résolution synchrone des modules dans sys.modules avant thread pool
import sklearn.covariance
import openai
import asyncio
import numpy as np
import pandas as pd
import os
import re
import shutil
import quantstats as qs
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

qs.extend_pandas()

import config
from market_data import (
    preprocess_market_data,
    calculate_covariance_matrix,
    calculate_ema,
    calculate_exponential_time_weights,
    adjust_expected_returns_with_sentiment,
)
from genetic_algorithm import (
    run_evolution,
    repair_chromosome,
    evaluate_fitness,
    calculate_turnover,
    calculate_portfolio_returns,
)
from classic_optimization import (
    minimize_variance_coordinate_descent,
    maximize_sharpe_slsqp,
    equal_weight_portfolio,
)
from llm_agents import analyze_macro_regime
from scenarios import SCENARIOS
from logging_utils import configure_logging, get_logger

logger = get_logger(__name__)

def calculate_annual_returns(portfolio_values, trading_days=252):
    """
    Calcule le rendement annualisé composé (CAGR) à partir des valeurs du portefeuille.
    
    Parameters
    ----------
    portfolio_values : np.ndarray
        Série des valeurs du portefeuille.
    trading_days : int
        Nombre de jours de trading par an (défaut: 252).
    
    Returns
    -------
    float
        Rendement annualisé composé.
    """
    if (
        len(portfolio_values) < 2
        or portfolio_values[0] <= 1e-12
        or portfolio_values[-1] <= 0.0
    ):
        return 0.0
        
    total_return = portfolio_values[-1] / portfolio_values[0]
    years = (len(portfolio_values) - 1) / trading_days
    return (total_return ** (1 / years)) - 1.0


def calculate_annual_volatility(daily_returns, trading_days=252):
    """
    Calcule la volatilité annualisée à partir des rendements journaliers.
    
    Parameters
    ----------
    daily_returns : np.ndarray
        Série des rendements journaliers.
    trading_days : int
        Nombre de jours de trading par an (défaut: 252).
    
    Returns
    -------
    float
        Volatilité annualisée.
    """
    return np.std(daily_returns) * np.sqrt(trading_days)


def calculate_sharpe_ratio(portfolio_values, risk_free_rate=0.0, trading_days=252):
    """
    Calcule le ratio de Sharpe.
    
    Parameters
    ----------
    portfolio_values : np.ndarray
        Série des valeurs du portefeuille.
    risk_free_rate : float
        Taux sans risque annualisé (défaut: 0.015).
    trading_days : int
        Nombre de jours de trading par an (défaut: 252).
    
    Returns
    -------
    float
        Ratio de Sharpe.
    """
    annual_return = calculate_annual_returns(portfolio_values, trading_days)
    
    # Calcul des rendements journaliers pour la volatilité
    daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
    annual_volatility = calculate_annual_volatility(daily_returns, trading_days)
    
    if annual_volatility < 1e-10:
        return 0.0
    
    return (annual_return - risk_free_rate) / annual_volatility


def calculate_sortino_ratio(portfolio_values, risk_free_rate=0.0, trading_days=252):
    """
    Calcule le ratio de Sortino (utilise la downside deviation).
    
    Parameters
    ----------
    portfolio_values : np.ndarray
        Série des valeurs du portefeuille.
    risk_free_rate : float
        Taux sans risque annualisé (défaut: 0.015).
    trading_days : int
        Nombre de jours de trading par an (défaut: 252).
    
    Returns
    -------
    float
        Ratio de Sortino.
    """
    # Calcul des rendements journaliers pour la downside deviation
    daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
    
    # Downside deviation relative au taux sans risque journalier
    negative_returns = np.minimum(daily_returns - (risk_free_rate / 252.0), 0.0)
    downside_variance = np.mean(negative_returns ** 2)
    downside_deviation = np.sqrt(downside_variance) * np.sqrt(trading_days)
    
    if downside_deviation < 1e-10:
        return 0.0
    
    annual_return = calculate_annual_returns(portfolio_values, trading_days)
    return (annual_return - risk_free_rate) / downside_deviation


def calculate_max_drawdown(portfolio_values):
    """
    Calcule le Max Drawdown (perte maximale) de la courbe de valeur.
    
    Parameters
    ----------
    portfolio_values : np.ndarray
        Série des valeurs du portefeuille.
    
    Returns
    -------
    float
        Max Drawdown (en valeur absolute négative).
    """
    cumulative_max = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - cumulative_max) / cumulative_max
    max_drawdown = np.min(drawdown)
    
    return max_drawdown


def calculate_calmar_ratio(portfolio_values, trading_days=252):
    """
    Calcule le ratio de Calmar (rendement annualisé / |Max Drawdown|).
    
    Parameters
    ----------
    portfolio_values : np.ndarray
        Série des valeurs du portefeuille.
    trading_days : int
        Nombre de jours de trading par an (défaut: 252).
    
    Returns
    -------
    float
        Ratio de Calmar.
    """
    annual_return = calculate_annual_returns(portfolio_values, trading_days)
    max_dd = calculate_max_drawdown(portfolio_values)
    
    if abs(max_dd) < 1e-10:
        return 0.0
    
    return annual_return / abs(max_dd)


def generate_quantstats_tearsheet(
    portfolio_values,
    dates,
    oos_length,
    scenario_name=None,
    rf=0.0,
    output_filename: str | None = None,
):
    """Generate a QuantStats tear sheet for the out-of-sample period."""
    if oos_length <= 0:
        return

    if len(dates) > config.LOOKBACK_WINDOW:
        nav_start = config.LOOKBACK_WINDOW - 1
        nav_end = min(config.LOOKBACK_WINDOW + oos_length, len(dates))
    else:
        nav_end = len(dates)
        nav_start = max(0, nav_end - oos_length - 1)
    nav_dates = pd.DatetimeIndex(dates[nav_start:nav_end])

    if len(nav_dates) < 2:
        return

    nav_ga = pd.Series(
        portfolio_values['genetic_algorithm'][nav_start:nav_end],
        index=nav_dates,
        name='genetic_algorithm',
    )
    nav_benchmark = pd.Series(
        portfolio_values['benchmark'][nav_start:nav_end],
        index=nav_dates,
        name='benchmark',
    )
    returns_ga = nav_ga.pct_change().dropna()
    returns_benchmark = nav_benchmark.pct_change().dropna()

    if returns_ga.empty or returns_benchmark.empty:
        return

    if output_filename is not None:
        safe_name = re.sub(r'[^A-Za-z0-9_.-]', '_', str(output_filename))
        if not safe_name.endswith('.html'):
            safe_name = f'{safe_name}.html'
        output = Path('outputs/tearsheets') / safe_name
    else:
        output = Path('outputs/tearsheets/best_strategy_tearsheet.html')
    output.parent.mkdir(parents=True, exist_ok=True)
    if scenario_name == 'Global_MultiAsset_2024':
        title = 'Global MultiAsset 2024 - Out-of-Sample'
    else:
        label = scenario_name or 'Portfolio'
        title = f'{label} - Out-of-Sample'
    import matplotlib.pyplot as plt
    try:
        qs.reports.html(
            returns_ga,
            benchmark=returns_benchmark,
            rf=rf,
            output=str(output),
            title=title,
        )
    finally:
        plt.close('all')

    if scenario_name is not None and output.is_file():
        safe_scenario_name = re.sub(r'[^A-Za-z0-9_.-]', '_', str(scenario_name))
        scenario_output = Path('outputs/tearsheets') / f'tearsheet_{safe_scenario_name}.html'
        if output != scenario_output:
            shutil.copyfile(output, scenario_output)


def run_backtest(
    tickers,
    start_date,
    end_date,
    benchmark_ticker='SPY',
    rf_ticker=None,
    scenario_name=None,
    export_html: bool = False,
    export_md: bool = False,
    record_history: bool = False,
    history_config_id: str | None = None,
):
    """
    Exécute le backtest complet avec fenêtre glissante.
    
    Parameters
    ----------
    tickers : list of str
        Liste des symboles boursiers.
    start_date : str
        Date de début au format 'YYYY-MM-DD'.
    end_date : str
        Date de fin au format 'YYYY-MM-DD'.
    export_html : bool
        Génère le rapport QuantStats après le calcul du backtest.
    export_md : bool
        Argument conservé pour compatibilité ; l'export Markdown est désactivé.
    record_history : bool
        Collecte et exporte les rééquilibrages de la stratégie génétique.
    history_config_id : str, optional
        Identifiant utilisé pour nommer l'historique d'une configuration sélectionnée.
    
    Returns
    -------
    dict
        Résultats du backtest contenant les métriques de performance
        pour chaque stratégie (AG, Variance Min, Max Sharpe, benchmark).
    """
    # Prétraitement complet des données
    logger.info("Début du backtest : %s", scenario_name or "Portfolio")
    if export_md:
        logger.warning("L'export Markdown est désactivé ; utilisez record_history=True.")
    market_data = preprocess_market_data(
        tickers,
        start_date,
        end_date,
        apply_sentiment=config.USE_LLM,
        use_llm=config.USE_LLM,
        api_key=config.LLM_API_KEY,
        provider=config.LLM_PROVIDER,
        benchmark_ticker=benchmark_ticker,
        rf_ticker=rf_ticker,
    )
    
    prices = market_data['prices']
    tickers = market_data['tickers']
    simple_returns = market_data['simple_returns']
    dates = market_data['dates']
    benchmark_simple_returns = market_data['benchmark_simple_returns']
    n_assets = len(tickers)
    n_periods = len(dates)
    daily_rf_rates = market_data['daily_rf_rates']
    if daily_rf_rates is None:
        daily_rf_rates = np.zeros(n_periods)
    time_weights = calculate_exponential_time_weights(
        config.LOOKBACK_WINDOW,
        config.TIME_WEIGHT_HALFLIFE,
    )
    
    # Initialisation des résultats par stratégie
    strategies = {
        'genetic_algorithm': [],
        'variance_minimization': [],
        'maximum_sharpe': [],
        'benchmark': []
    }
    
    portfolio_values = {
        'genetic_algorithm': np.ones(n_periods),
        'variance_minimization': np.ones(n_periods),
        'maximum_sharpe': np.ones(n_periods),
        'benchmark': np.ones(n_periods)
    }

    gross_portfolio_values = {
        strategy_name: np.ones(n_periods)
        for strategy_name in portfolio_values
    }
    gross_daily_returns = {
        strategy_name: np.zeros(n_periods)
        for strategy_name in portfolio_values
    }
    net_daily_returns = {
        strategy_name: np.zeros(n_periods)
        for strategy_name in portfolio_values
    }
    
    turnover_history = {
        'genetic_algorithm': [],
        'variance_minimization': [],
        'maximum_sharpe': [],
        'benchmark': []
    }
    transaction_cost_totals = {strategy_name: 0.0 for strategy_name in strategies}
    margin_financing_cost_totals = {strategy_name: 0.0 for strategy_name in strategies}

    allocation_history = {
        'genetic_algorithm': [],
        'variance_minimization': [],
        'maximum_sharpe': []
    }
    out_of_sample_dates = []
    
    previous_weights = {
        'genetic_algorithm': np.ones(n_assets) / n_assets,
        'variance_minimization': np.ones(n_assets) / n_assets,
        'maximum_sharpe': np.ones(n_assets) / n_assets
    }
    
    # Historique des décisions macro
    macro_decisions = []
    rebalance_records = [] if record_history else None
    run_id = f"{scenario_name or 'Portfolio'}-{uuid4().hex}" if record_history else None
    
    # Boucle de fenêtre glissante
    idx_rebalance = config.LOOKBACK_WINDOW
    
    while idx_rebalance + config.REBALANCE_FREQ <= n_periods:
        # Fenêtre In-Sample : historique de LOOKBACK_WINDOW jours
        in_sample_start = idx_rebalance - config.LOOKBACK_WINDOW
        in_sample_end = idx_rebalance
        
        in_sample_simple_returns = simple_returns[in_sample_start:in_sample_end, :]

        # Définition du sur-historique pour convergence asymptotique de l'EMA
        burn_in_start = max(0, in_sample_end - (config.LOOKBACK_WINDOW + 3 * 200))
        momentum_prices = prices[burn_in_start:in_sample_end, :]

        ema_40 = calculate_ema(momentum_prices, span=40)
        ema_200 = calculate_ema(momentum_prices, span=200)

        # Extraction stricte de l'estimation convergée à la clôture de la fenêtre IS
        ema_40_last = ema_40[-1, :]
        ema_200_last = ema_200[-1, :]
        effective_ema_lag = (200.0 - 40.0) / 2.0
        with np.errstate(divide='ignore', invalid='ignore'):
            mu_in_daily = np.where(
                ema_200_last > 0.0,
                ((ema_40_last / ema_200_last) - 1.0) / effective_ema_lag,
                0.0,
            )
        mu_in = np.nan_to_num(
            mu_in_daily * 252.0,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        current_date_str = pd.Timestamp(
            dates[in_sample_end - 1]
        ).strftime('%Y-%m-%d')
        if getattr(config, 'USE_LLM', False):
            mu_in = asyncio.run(
                adjust_expected_returns_with_sentiment(
                    mu_in,
                    tickers,
                    ref_date=current_date_str,
                    api_key=config.LLM_API_KEY,
                    use_llm=True,
                    provider=config.LLM_PROVIDER,
                )
            )
        mean_rf_rate = np.mean(daily_rf_rates[in_sample_start:in_sample_end])
        
        # Recalculer sigma_in dynamiquement depuis les rendements simples In-Sample
        sigma_in, _ = calculate_covariance_matrix(
            in_sample_simple_returns,
            time_weights=time_weights,
        )
        simple_ret_in = np.ascontiguousarray(in_sample_simple_returns)
        
        # Agent macro-économique : analyse du régime de marché via le benchmark
        benchmark_in_sample = benchmark_simple_returns[in_sample_start:in_sample_end]
        
        try:
            # PHASE 2 (backtest): ce chemin doit etre un Cache Hit SQLite exact,
            # prechauffe par prewarm_cache.py avant le lancement du calcul.
            # Aucun retry/backoff reseau n'est autorise ici: la resilience reseau
            # appartient exclusivement a la phase de pre-warming.
            macro_analysis = asyncio.run(
                analyze_macro_regime(
                    benchmark_in_sample,
                    t_bond_yield=0.025,  # 2.5% comme défaut
                    api_key=config.LLM_API_KEY,
                    use_llm=config.USE_LLM,
                    provider=config.LLM_PROVIDER
                )
            )
            risk_aversion_adjusted = macro_analysis.risk_aversion
            liq_mult = macro_analysis.liquidity_multiplier
            dynamic_allow_short = macro_analysis.enable_shorting
            market_regime = macro_analysis.market_regime.value
            
            macro_decisions.append({
                'date': pd.Timestamp(dates[idx_rebalance]).to_pydatetime(),
                'regime': market_regime,
                'risk_aversion': risk_aversion_adjusted
            })
            
        except Exception as e:
            logger.warning("Échec de l'analyse macro, paramètres par défaut : %s", e)
            risk_aversion_adjusted = config.RISK_AVERSION
            liq_mult = 1.0
            dynamic_allow_short = False
            market_regime = "Unknown"

        dynamic_fee = config.TRANSACTION_FEE * liq_mult
        dynamic_lambda_turnover = dynamic_fee * (252 / config.REBALANCE_FREQ)
        
        # ────────────────────────────────────────────────────────────────
        # PHASE 3 : Stratégie 1 - ALGORITHME GÉNÉTIQUE (Moteur Réel)
        # ────────────────────────────────────────────────────────────────
        # Autorise la borne minimale configurée si le short est activé
        w_min_val = config.MIN_WEIGHT if dynamic_allow_short else 0.0
        try:
            result_ga = run_evolution(
                expected_returns=mu_in,
                simple_returns=simple_ret_in,
                previous_weights=previous_weights['genetic_algorithm'],
                population_size=config.POPULATION_SIZE,
                generations=config.GENERATIONS,
                mutation_rate=config.MUTATION_RATE,
                lambda_risk=risk_aversion_adjusted,
                lambda_skew=config.SKEWNESS_PREFERENCE,
                lambda_kurt=config.KURTOSIS_AVERSION,
                lambda_turnover=dynamic_lambda_turnover,
                max_weight=config.MAX_WEIGHT,
                w_min=w_min_val,
                elite_ratio=0.05,
                mean_rf_rate=mean_rf_rate,
                margin_rate=config.MARGIN_RATE,
                gross_exposure_target=config.GROSS_EXPOSURE,
                cov_matrix=sigma_in,
                time_weights=time_weights,
                seed=None
            )
            w_ga = result_ga['best_weights']
        except Exception as e:
            logger.warning("Échec de l'optimisation génétique : %s", e)
            w_ga = equal_weight_portfolio(n_assets)

        cleaned_w_ga = np.where(np.abs(w_ga) < 1e-12, 0.0, w_ga)
        if (
            np.all(np.isfinite(cleaned_w_ga))
            and np.all(cleaned_w_ga >= w_min_val)
            and np.all(cleaned_w_ga <= config.MAX_WEIGHT)
            and np.sum(np.abs(cleaned_w_ga)) <= config.GROSS_EXPOSURE
            and (w_min_val < 0.0 or np.sum(cleaned_w_ga) <= 1.0)
        ):
            w_ga = cleaned_w_ga
        
        # ────────────────────────────────────────────────────────────────
        # PHASE 4 : Stratégies 2-4 - BENCHMARKS CLASSIQUES
        # ────────────────────────────────────────────────────────────────
        
        # Variance Minimale (descente de coordonnées)
        try:
            result_var = minimize_variance_coordinate_descent(
                sigma_in,
                allow_short=dynamic_allow_short,
                force_full_investment=True,
                max_weight=config.MAX_WEIGHT,
                gross_exposure_target=config.GROSS_EXPOSURE,
            )
            w_var = result_var['weights']
        except Exception as e:
            logger.warning("Échec de l'optimisation de variance : %s", e)
            w_var = equal_weight_portfolio(n_assets)
        
        # Maximum Sharpe (SLSQP)
        try:
            result_sharpe = maximize_sharpe_slsqp(
                mu_in,
                sigma_in,
                mean_rf_rate,
                config.MAX_WEIGHT,
                allow_short=dynamic_allow_short,
                force_full_investment=True,
                margin_rate=config.MARGIN_RATE,
                gross_exposure_target=config.GROSS_EXPOSURE,
                mean_rf_rate=mean_rf_rate,
            )
            w_sharpe = result_sharpe['weights']
        except Exception as e:
            logger.warning("Échec de l'optimisation de Sharpe : %s", e)
            w_sharpe = equal_weight_portfolio(n_assets)
        
        # ────────────────────────────────────────────────────────────────
        # PHASE 5 : Fenêtre Out-of-Sample (21 jours) - Évaluation
        # ────────────────────────────────────────────────────────────────
        out_sample_start = idx_rebalance
        out_sample_end = min(idx_rebalance + config.REBALANCE_FREQ, n_periods)
        out_sample_simple_returns = simple_returns[out_sample_start:out_sample_end, :]
        out_of_sample_dates.extend(dates[out_sample_start:out_sample_end].tolist())
        
        # Application des trois stratégies sur l'out-of-sample
        for strategy_name, weights in [
            ('genetic_algorithm', w_ga),
            ('variance_minimization', w_var),
            ('maximum_sharpe', w_sharpe)
        ]:
            n_days_oos = len(out_sample_simple_returns)
            port_returns = np.zeros(n_days_oos)
            gross_port_returns = np.zeros(n_days_oos)

            # Keep gross and net holdings separate because margin costs affect net drift.
            gross_current_weights = weights.copy()
            gross_cash_weight = 1.0 - np.sum(weights)
            current_weights = weights.copy()
            cash_weight = 1.0 - np.sum(weights)

            for t in range(n_days_oos):
                daily_ret = out_sample_simple_returns[t]
                daily_cash_ret = daily_rf_rates[out_sample_start + t]

                gross_long_exposure = np.sum(np.maximum(gross_current_weights, 0.0))
                gross_free_cash_weight = max(0.0, 1.0 - gross_long_exposure)
                gross_port_returns[t] = (
                    np.dot(gross_current_weights, daily_ret)
                    + (gross_free_cash_weight * daily_cash_ret)
                )

                long_exposure = np.sum(np.maximum(current_weights, 0.0))
                short_exposure = np.sum(np.abs(np.minimum(current_weights, 0.0)))
                financed_exposure = (
                    max(0.0, long_exposure - 1.0) + short_exposure
                )
                free_cash_weight = max(0.0, 1.0 - long_exposure)
                daily_margin_cost = financed_exposure * (
                    config.MARGIN_RATE / 252.0
                )

                # Rendement P&L exact (poids de début de journée)
                asset_return = np.dot(current_weights, daily_ret)
                total_return = (
                    asset_return
                    + (free_cash_weight * daily_cash_ret)
                    - daily_margin_cost
                )
                port_returns[t] = total_return
                margin_financing_cost_totals[strategy_name] += daily_margin_cost

                gross_current_weights = gross_current_weights * (1.0 + daily_ret)
                gross_cash_weight += gross_free_cash_weight * daily_cash_ret
                gross_total_value = np.sum(gross_current_weights) + gross_cash_weight
                if gross_total_value > 1e-10:
                    gross_current_weights /= gross_total_value
                    gross_cash_weight /= gross_total_value

                # Dérive des poids (Weight Drift Mark-to-Market) pour t+1
                current_weights = current_weights * (1.0 + daily_ret)
                cash_weight = (
                    cash_weight + (free_cash_weight * daily_cash_ret)
                    - daily_margin_cost
                )

                # Renormalisation sur la nouvelle NAV théorique
                total_value = np.sum(current_weights) + cash_weight
                if total_value > 1e-10:
                    current_weights /= total_value
                    cash_weight /= total_value

            # Calcul du turnover (coûts de transaction)
            turnover = calculate_turnover(
                weights,
                previous_weights[strategy_name],
            )
            turnover_history[strategy_name].append(turnover)
            
            # Permanent market impact: linear friction plus a quadratic
            # liquidity shock for institutional-size reallocations.
            transaction_cost = (
                (turnover * dynamic_fee)
                + (0.1 * dynamic_fee * (turnover ** 2))
            )
            transaction_cost_totals[strategy_name] += transaction_cost
            if rebalance_records is not None and strategy_name == 'genetic_algorithm':
                rebalance_records.append({
                    'run_id': run_id,
                    'rebalance_date': pd.Timestamp(dates[idx_rebalance]),
                    'inferred_regime': market_regime,
                    'portfolio_vol_ex_ante': float(np.sqrt(max(0.0, np.dot(weights, np.dot(sigma_in, weights))))),
                    'turnover': float(turnover),
                    'transaction_costs': float(transaction_cost),
                    'cash_weight': float(1.0 - np.sum(weights)),
                    **{f'weight_{asset}': float(weight) for asset, weight in zip(tickers, weights)},
                })
            current_nav = portfolio_values[strategy_name][out_sample_start - 1]
            current_nav -= current_nav * transaction_cost
            current_gross_nav = gross_portfolio_values[strategy_name][out_sample_start - 1]

            if record_history:
                allocation_history[strategy_name].extend(
                    [weights.copy() for _ in range(len(port_returns))]
                )
            
            # Mise à jour de la valeur du portefeuille (valeur accumulée)
            for i, ret in enumerate(port_returns):
                idx_out = out_sample_start + i
                if idx_out < n_periods:
                    current_nav *= (1.0 + ret)
                    portfolio_values[strategy_name][idx_out] = current_nav
                    current_gross_nav *= (1.0 + gross_port_returns[i])
                    gross_portfolio_values[strategy_name][idx_out] = current_gross_nav
                    gross_daily_returns[strategy_name][idx_out] = gross_port_returns[i]
                    net_daily_returns[strategy_name][idx_out] = (
                        (1.0 - transaction_cost) * (1.0 + ret) - 1.0
                        if i == 0 else ret
                    )

            # Capture des poids dérivés pour le turnover du rééquilibrage suivant
            previous_weights[strategy_name] = current_weights.copy()
            
        # Le benchmark utilise directement les rendements de l'indice téléchargé.
        benchmark_returns = benchmark_simple_returns[out_sample_start:out_sample_end]
        turnover_history['benchmark'].append(0.0)
        for i, ret in enumerate(benchmark_returns):
            idx_out = out_sample_start + i
            if idx_out < n_periods:
                portfolio_values['benchmark'][idx_out] = \
                    portfolio_values['benchmark'][idx_out - 1] * (1 + ret)
                gross_portfolio_values['benchmark'][idx_out] = \
                    gross_portfolio_values['benchmark'][idx_out - 1] * (1 + ret)
                gross_daily_returns['benchmark'][idx_out] = ret
                net_daily_returns['benchmark'][idx_out] = ret
        
        # ────────────────────────────────────────────────────────────────
        # Progression vers la fenêtre suivante
        # ────────────────────────────────────────────────────────────────
        idx_rebalance += config.REBALANCE_FREQ
    
    # Calcul des métriques finales
    results = {
        'strategies': {},
        'macro_decisions': macro_decisions,
        'dates': np.asarray(out_of_sample_dates),
        'tickers': list(tickers),
        'allocation_history': allocation_history,
    }

    oos_rf_slice = daily_rf_rates[
        config.LOOKBACK_WINDOW:config.LOOKBACK_WINDOW + len(out_of_sample_dates)
    ]
    annual_rf_rate = float(np.mean(oos_rf_slice) * 252.0) if len(oos_rf_slice) > 0 else 0.0
    
    for strategy_name in strategies.keys():
        if len(out_of_sample_dates) > 0:
            nav_start = config.LOOKBACK_WINDOW - 1
            nav_end = min(config.LOOKBACK_WINDOW + len(out_of_sample_dates), n_periods)
            port_values = portfolio_values[strategy_name][nav_start:nav_end]
            
            # Extraction des valeurs pour le calcul
            annual_ret = calculate_annual_returns(port_values)
            
            # Calcul des rendements journaliers pour la volatilité et Sortino
            daily_returns = np.diff(port_values) / port_values[:-1]
            annual_vol = calculate_annual_volatility(daily_returns)
            
            sharpe = calculate_sharpe_ratio(port_values, annual_rf_rate)
            sortino = calculate_sortino_ratio(port_values, annual_rf_rate)
            max_dd = calculate_max_drawdown(port_values)
            calmar = calculate_calmar_ratio(port_values)
            median_turnover = np.median(turnover_history[strategy_name])
            elapsed_years = len(out_of_sample_dates) / 252.0
            total_transaction_cost_drag = (
                transaction_cost_totals[strategy_name] / elapsed_years * 10000.0
                if elapsed_years > 0.0 else 0.0
            )
            total_margin_financing_drag = (
                margin_financing_cost_totals[strategy_name] / elapsed_years * 10000.0
                if elapsed_years > 0.0 else 0.0
            )
            annualized_turnover = (
                np.sum(turnover_history[strategy_name]) / elapsed_years
                if elapsed_years > 0.0 else 0.0
            )
            strategy_daily_start = config.LOOKBACK_WINDOW
            strategy_daily_end = strategy_daily_start + len(out_of_sample_dates)
            gross_nav = gross_portfolio_values[strategy_name][nav_start:nav_end]
            gross_returns = gross_daily_returns[strategy_name][strategy_daily_start:strategy_daily_end]
            net_returns = net_daily_returns[strategy_name][strategy_daily_start:strategy_daily_end]
            
            results['strategies'][strategy_name] = {
                'annual_return': annual_ret,
                'annual_volatility': annual_vol,
                'sharpe_ratio': sharpe,
                'sortino_ratio': sortino,
                'max_drawdown': max_dd,
                'calmar_ratio': calmar,
                'median_turnover': median_turnover,
                'avg_turnover': float(np.mean(turnover_history[strategy_name])),
                'Total Transaction Costs Drag (bps/yr)': total_transaction_cost_drag,
                'Total Margin Financing Drag (bps/yr)': total_margin_financing_drag,
                'Annualized Turnover': annualized_turnover,
                'n_rebalances': len(turnover_history[strategy_name]),
                'portfolio_values': port_values,
                'net_portfolio_values': port_values,
                'gross_portfolio_values': gross_nav,
                'gross_daily_returns': gross_returns,
                'net_daily_returns': net_returns,
            }
        else:
            results['strategies'][strategy_name] = None

    if export_html:
        generate_quantstats_tearsheet(
            portfolio_values,
            dates,
            len(out_of_sample_dates),
            scenario_name=scenario_name,
            rf=annual_rf_rate,
        )

    if rebalance_records:
        df_history = pd.DataFrame.from_records(rebalance_records)
        rebalance_dates = pd.to_datetime(df_history['rebalance_date'])
        if rebalance_dates.dt.tz is not None:
            rebalance_dates = rebalance_dates.dt.tz_localize(None)
        df_history['rebalance_date'] = rebalance_dates.astype('datetime64[ns]')
        df_history['inferred_regime'] = df_history['inferred_regime'].astype('category')
        numeric_columns = df_history.columns.difference(['run_id', 'rebalance_date', 'inferred_regime'])
        df_history[numeric_columns] = df_history[numeric_columns].astype('float64')
        output_dir = Path('outputs')
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_config_id = re.sub(r'[^A-Za-z0-9_.-]', '_', str(history_config_id)) if history_config_id is not None else None
        history_name = f'rebalance_history_top_{safe_config_id}' if safe_config_id is not None else 'rebalance_history'
        history_path = output_dir / f'{history_name}.parquet'
        try:
            df_history.to_parquet(history_path, index=False)
        except ImportError:
            history_path = output_dir / f'{history_name}.csv'
            df_history.to_csv(history_path, index=False)
        results['rebalance_history'] = df_history
        results['rebalance_history_path'] = str(history_path)
        logger.info("Historique des rééquilibrages exporté : %s", history_path)

    logger.info("Fin du backtest : %s", scenario_name or "Portfolio")
    
    return results


def print_results(results):
    """
    Affiche les résultats du backtest de manière lisible.
    
    Parameters
    ----------
    results : dict
        Résultats du backtest par stratégie et décisions macro.
    """
    lines = ["", "="*80, "RÉSULTATS DU BACKTEST - OPTIMISATION DE PORTEFEUILLE", "="*80, ""]
    
    strategies = results.get('strategies', {})
    macro_decisions = results.get('macro_decisions', [])
    
    # Affichage des résultats par stratégie
    for strategy_name, metrics in strategies.items():
        if metrics is None:
            lines.extend([f"[{strategy_name.upper()}] : Données insuffisantes", ""])
            continue

        lines.extend([
            f"[{strategy_name.upper()}]",
            f"  Rendement annualisé     : {metrics['annual_return']*100:>8.2f}%",
            f"  Volatilité annualisée   : {metrics['annual_volatility']*100:>8.2f}%",
            f"  Ratio de Sharpe         : {metrics['sharpe_ratio']:>8.3f}",
            f"  Ratio de Sortino        : {metrics['sortino_ratio']:>8.3f}",
            f"  Max Drawdown            : {metrics['max_drawdown']*100:>8.2f}%",
            f"  Ratio de Calmar         : {metrics['calmar_ratio']:>8.3f}",
            f"  Turnover médian         : {metrics['median_turnover']:>8.3f}",
            f"  Nombre de rééquilibrages: {metrics['n_rebalances']:>8d}",
            "",
        ])
    
    # Affichage des décisions macro
    lines.extend(["", "-"*80, "DÉCISIONS MACRO-ÉCONOMIQUES", "-"*80, ""])
    
    if macro_decisions:
        for decision in macro_decisions[:10]:  # Afficher les 10 premières
            lines.append(f"  {decision['date'].strftime('%Y-%m-%d')} | "
                     f"Régime: {decision['regime']:<20} | "
                     f"λ_risk = {decision['risk_aversion']:.2f}")
        
        if len(macro_decisions) > 10:
            lines.append(f"\n  ... et {len(macro_decisions) - 10} autres décisions")
    else:
        lines.append("  Aucune décision macro enregistrée")
    
    lines.extend(["", "="*80, ""])
    print("\n".join(lines))

if __name__ == "__main__":
    configure_logging()
    # Configuration du backtest
    ACTIVE_SCENARIO = 'Global_MultiAsset_2024'
    scenario = SCENARIOS[ACTIVE_SCENARIO]
    tickers = scenario['tickers']
    start_date = scenario['start_date']
    end_date = scenario['end_date']
    benchmark_ticker = scenario['benchmark_ticker']
    rf_ticker = scenario['rf_ticker']
    
    print(f"\nLancement du backtest...", flush=True)
    print(f"  Tickers : {', '.join(tickers)}", flush=True)
    print(f"  Période : {start_date} à {end_date}", flush=True)
    print(f"  Stratégies : AG, Variance Min, Max Sharpe, Benchmark", flush=True)
    print(f"  Benchmark : {benchmark_ticker}", flush=True)
    print(f"  Taux sans risque : {rf_ticker}", flush=True)
    mode = "LLM" if config.USE_LLM else "Heuristique déterministe"
    print(f"  Agent Macro : {mode}", flush=True)
    
    # Exécution du backtest
    results = run_backtest(
        tickers,
        start_date,
        end_date,
        benchmark_ticker,
        rf_ticker,
        scenario_name=ACTIVE_SCENARIO,
        export_html=True,
        export_md=False,
        record_history=True,
    )
    
    # Affichage des résultats
    print_results(results)
