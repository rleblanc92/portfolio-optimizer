"""
Module de gestion des données de marché pour l'optimisation de portefeuille.
Acquisition et prétraitement vectorisé via NumPy et yfinance.
Intégration de l'analyse de sentiment pour ajustement des rendements espérés.
"""

import asyncio
import threading
import time
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.covariance import LedoitWolf
from logging_utils import get_logger

logger = get_logger(__name__)

# Optionnel : import de l'agent de sentiment (peut être désactivé)
try:
    from llm_agents import analyze_asset_sentiment
    HAS_SENTIMENT_AGENT = True
except ImportError:
    HAS_SENTIMENT_AGENT = False

# Constante de sensibilité sémantique
SENTIMENT_BETA = 0.20
DOWNLOAD_RETRIES = 3
_YF_DOWNLOAD_LOCK = threading.Lock()


def fetch_historical_data(tickers, start_date, end_date):
    """
    Récupère les données historiques des cours de clôture via yfinance.
    
    Parameters
    ----------
    tickers : list of str
        Liste des symboles boursiers.
    start_date : str
        Date de début au format 'YYYY-MM-DD'.
    end_date : str
        Date de fin au format 'YYYY-MM-DD'.
    
    Returns
    -------
    np.ndarray
        Matrice des cours historiques (n_dates, n_assets).
    list of str
        Liste des tickers (pour traçabilité).
    np.ndarray
        Dates correspondantes.
    """
    if not tickers:
        raise ValueError("La liste des tickers ne peut pas être vide.")

    data = None
    last_error = None
    for attempt in range(DOWNLOAD_RETRIES):
        try:
            # yfinance partage un cache SQLite de cookies/crumbs entre appels.
            # Le verrou évite que les tâches asyncio.to_thread le sollicitent
            # simultanément et déclenchent "database is locked".
            with _YF_DOWNLOAD_LOCK:
                data = yf.download(
                    tickers,
                    start=start_date,
                    end=end_date,
                    progress=False,
                    auto_adjust=False,
                    threads=False,
                )
        except Exception as exc:
            last_error = exc
            data = None

        if data is not None and not data.empty:
            break
        if attempt < DOWNLOAD_RETRIES - 1:
            time.sleep(1.0 * (attempt + 1))

    if data is None or data.empty:
        detail = f" ({last_error})" if last_error is not None else ""
        raise RuntimeError(
            f"Aucune donnée Yahoo Finance pour {tickers} entre "
            f"{start_date} et {end_date}{detail}"
        )

    if 'Close' not in data:
        raise RuntimeError(
            f"La réponse Yahoo Finance ne contient pas de clôture pour {tickers}."
        )
    
    close_data = data['Close']
    if isinstance(close_data, pd.Series):
        close_data = close_data.to_frame(name=tickers[0])

    valid_tickers = [
        ticker for ticker in tickers
        if ticker in close_data.columns and not close_data[ticker].isna().all()
    ]
    if not valid_tickers:
        raise RuntimeError(
            f"Aucune clôture exploitable pour {tickers} entre "
            f"{start_date} et {end_date}."
        )

    close_data = close_data[valid_tickers].ffill().fillna(0.0)
    prices = close_data.values
    
    dates = pd.DatetimeIndex(pd.to_datetime(close_data.index))
    
    return prices, valid_tickers, dates


def fetch_benchmark_data(benchmark_ticker, start_date, end_date):
    """Récupère les cours historiques d'un indice de référence."""
    prices, _, dates = fetch_historical_data(
        [benchmark_ticker], start_date, end_date
    )
    if prices.ndim != 2 or prices.shape[1] != 1:
        raise RuntimeError(
            f"Données benchmark invalides pour {benchmark_ticker}: "
            f"shape={prices.shape}."
        )
    return prices[:, 0], dates


def fetch_rf_data(rf_ticker, start_date, end_date):
    """Récupère les rendements souverains cotés en pourcentage annuel."""
    prices, _, dates = fetch_historical_data([rf_ticker], start_date, end_date)
    if prices.ndim != 2 or prices.shape[1] != 1:
        raise RuntimeError(
            f"Données taux sans risque invalides pour {rf_ticker}: "
            f"shape={prices.shape}."
        )
    return prices[:, 0], dates


def calculate_simple_returns(prices):
    """
    Calcule les rendements simples à partir des prix.
    
    Rendements simples additifs transversalement (pas de problème de non-additivité).
    
    Parameters
    ----------
    prices : np.ndarray
        Matrice des prix (n_dates, n_assets).
    
    Returns
    -------
    np.ndarray
        Matrice des rendements simples (n_dates, n_assets).
    """
    prev_prices = prices[:-1, :]
    diff_prices = np.diff(prices, axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        step_returns = np.where(prev_prices > 0.0, diff_prices / prev_prices, 0.0)
    returns = np.vstack([np.zeros((1, prices.shape[1])), step_returns])
    return np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)


def calculate_ema(prices, span):
    """
    Calcule la Moyenne Mobile Exponentielle (EMA) de manière vectorisée.
    
    Parameters
    ----------
    prices : np.ndarray
        Matrice des prix (n_dates, n_assets).
    span : int
        Nombre de périodes pour le calcul de l'EMA.
    
    Returns
    -------
    np.ndarray
        Matrice des EMA (n_dates, n_assets).
    """
    alpha = 2.0 / (span + 1.0)
    n_dates, n_assets = prices.shape
    ema = np.full((n_dates, n_assets), np.nan, dtype=np.float64)
    ema[0, :] = prices[0, :]

    for t in range(1, n_dates):
        prev_ema = ema[t - 1, :]
        curr_price = prices[t, :]
        newly_listed = (prev_ema <= 0.0) & (curr_price > 0.0)
        ema[t, :] = np.where(
            newly_listed,
            curr_price,
            alpha * curr_price + (1.0 - alpha) * prev_ema,
        )
    
    return ema


def calculate_exponential_time_weights(n_days, halflife):
    """Return normalized exponential weights with the newest day weighted most."""
    if n_days <= 0:
        raise ValueError("n_days must be positive")
    if not np.isfinite(halflife) or halflife <= 0.0:
        raise ValueError("halflife must be a positive finite number")

    ages = np.arange(n_days - 1, -1, -1, dtype=np.float64)
    time_weights = np.exp2(-ages / float(halflife))
    time_weights /= np.sum(time_weights)
    return time_weights


def calculate_covariance_matrix(simple_returns, shrinkage='ledoit_wolf', time_weights=None):
    """
    Calcule la matrice de covariance régularisée via l'estimateur de Ledoit-Wolf.

    WARNING
    -------
    L'annualisation linéaire de rendements simples repose sur une approximation
    qui ne respecte pas exactement l'extrapolation temporelle des rendements
    arithmétiques. Elle suppose implicitement une dynamique longitudinale
    compatible avec des rendements logarithmiques i.i.d., ce qui peut introduire
    un biais d'estimation de second ordre en haute volatilité, notamment pour
    les cryptomonnaies. Une calibration exacte nécessiterait
    ultérieurement une transformation stochastique d'Itô pour modéliser cette
    extrapolation temporelle.
    
    Parameters
    ----------
    simple_returns : np.ndarray
        Matrice des rendements simples (n_dates, n_assets).
    shrinkage : str
        Type de régularisation. Défaut: 'ledoit_wolf'.
    time_weights : np.ndarray, optional
        Poids temporels non négatifs (n_dates,), normalisés avant estimation.
    
    Returns
    -------
    np.ndarray
        Matrice de covariance régularisée (n_assets, n_assets).
    float
        Paramètre de shrinkage appliqué.
    """
    annualized_returns = simple_returns * np.sqrt(252.0)
    normalized_time_weights = None
    n_dates = simple_returns.shape[0]
    if time_weights is not None:
        normalized_time_weights = np.asarray(time_weights, dtype=np.float64)
        if normalized_time_weights.ndim != 1 or len(normalized_time_weights) != n_dates:
            raise ValueError("time_weights must have one value per return date")
        if not np.all(np.isfinite(normalized_time_weights)) or np.any(normalized_time_weights < 0.0):
            raise ValueError("time_weights must be finite and non-negative")
        total_weight = np.sum(normalized_time_weights)
        if total_weight <= 0.0:
            raise ValueError("time_weights must have a positive sum")
        normalized_time_weights = normalized_time_weights / total_weight
        if np.allclose(
            normalized_time_weights,
            1.0 / n_dates,
            rtol=0.0,
            atol=1e-15,
        ):
            normalized_time_weights = None

    if shrinkage == 'ledoit_wolf':
        if normalized_time_weights is None:
            lw = LedoitWolf()
            lw.fit(annualized_returns)
        else:
            weighted_mean = normalized_time_weights @ annualized_returns
            centered_returns = annualized_returns - weighted_mean
            weighted_centered_returns = centered_returns * np.sqrt(
                normalized_time_weights[:, np.newaxis] * n_dates
            )
            lw = LedoitWolf(assume_centered=True)
            lw.fit(weighted_centered_returns)
        cov_matrix, shrinkage_coeff = lw.covariance_, lw.shrinkage_
    else:
        if normalized_time_weights is None:
            cov_matrix = np.cov(annualized_returns.T)
        else:
            weighted_mean = normalized_time_weights @ annualized_returns
            centered_returns = annualized_returns - weighted_mean
            cov_matrix = centered_returns.T @ (
                centered_returns * normalized_time_weights[:, np.newaxis]
            )
        shrinkage_coeff = 0.0

    variances = np.diag(cov_matrix)
    if normalized_time_weights is None:
        zero_var_mask = variances < 1e-8
    else:
        # A recent low-volatility recovery must not be mistaken for a flat asset.
        zero_var_mask = np.var(annualized_returns, axis=0) < 1e-8
    if np.any(zero_var_mask):
        np.fill_diagonal(
            cov_matrix,
            np.where(zero_var_mask, 1.0, variances),
        )
    
    return cov_matrix, shrinkage_coeff


async def adjust_expected_returns_with_sentiment(
    expected_returns, tickers, ref_date=None, api_key=None, use_llm=False,
    provider='google'
):
    """
    Ajuste les rendements espérés en fonction de l'analyse de sentiment.
    
    Formule : μ̂_adjusted = μ̂ + (SENTIMENT_BETA × sentiment_score × |μ̂|)
    
    Fallback automatique à μ̂ si analyse échoue (sentiment_score = 0.0).
    
    Parameters
    ----------
    expected_returns : np.ndarray
        Vecteur des rendements espérés (n_assets,) calculés via EMA.
    tickers : list of str
        Liste des symboles boursiers.
    ref_date : str, optional
        Date de référence pour l'analyse (défaut: aujourd'hui).
    use_llm : bool, optional
        Si True, utilise LLM pour analyse. Si False (défaut), heuristique.
    
    Returns
    -------
    np.ndarray
        Vecteur des rendements ajustés (n_assets,).
    """
    if not HAS_SENTIMENT_AGENT:
        # Si l'agent de sentiment n'est pas disponible, retourner les rendements inchangés
        return expected_returns.copy()
    
    if ref_date is None:
        from datetime import datetime
        ref_date = datetime.now().strftime('%Y-%m-%d')
    
    adjusted_returns = expected_returns.copy()

    semaphore = asyncio.Semaphore(3)

    async def bounded_sentiment_analysis(ticker):
        async with semaphore:
            return await analyze_asset_sentiment(
                ticker,
                ref_date,
                api_key=api_key,
                use_llm=use_llm,
                provider=provider,
            )

    tasks = [bounded_sentiment_analysis(ticker) for ticker in tickers]
    sentiment_results = await asyncio.gather(*tasks, return_exceptions=True)

    for idx, (ticker, sentiment_analysis) in enumerate(zip(tickers, sentiment_results)):
        try:
            if isinstance(sentiment_analysis, Exception):
                raise sentiment_analysis
            sentiment_score = sentiment_analysis.sentiment_score
        except Exception as e:
            # Fallback : sentiment neutre (0.0)
            logger.warning("Échec de l'analyse de sentiment, sentiment neutre : %s", e)
            sentiment_score = 0.0

        mu_original = expected_returns[idx]
        sentiment_adjustment = SENTIMENT_BETA * sentiment_score * abs(mu_original)
        adjusted_returns[idx] = mu_original + sentiment_adjustment
    
    return adjusted_returns


def preprocess_market_data(
    tickers, start_date, end_date, fast_span=40, slow_span=200,
    apply_sentiment=False, use_llm=False, api_key=None, provider='google',
    benchmark_ticker=None, rf_ticker=None
):
    """
    Pipeline complet de prétraitement des données de marché.
    
    Optionnellement, ajuste les rendements espérés via analyse de sentiment.
    
    Parameters
    ----------
    tickers : list of str
        Liste des symboles boursiers.
    start_date : str
        Date de début au format 'YYYY-MM-DD'.
    end_date : str
        Date de fin au format 'YYYY-MM-DD'.
    fast_span : int
        Fenêtre rapide pour l'EMA (défaut: 40).
    slow_span : int
        Fenêtre lente pour l'EMA (défaut: 200).
    apply_sentiment : bool, optional
        Si True, applique l'ajustement de sentiment aux rendements espérés.
    use_llm : bool, optional
        Si True et apply_sentiment=True, utilise LLM pour analyse.
    api_key : str, optional
        Clé API du fournisseur LLM.
    provider : str, optional
        Fournisseur LLM: 'google' ou 'openai'.
    
    Returns
    -------
    dict
        Dictionnaire contenant:
        - 'simple_returns' : np.ndarray (n_dates, n_assets)
        - 'prices' : np.ndarray (n_dates, n_assets)
        - 'tickers' : list of str
        - 'dates' : np.ndarray
        - 'sentiment_applied' : bool
    """
    # 1. Récupération via boucle d'événements asynchrones déléguée aux threads
    fetch_specs = [
        (fetch_historical_data, (tickers, start_date, end_date))
    ]
    if benchmark_ticker:
        fetch_specs.append(
            (fetch_benchmark_data, (benchmark_ticker, start_date, end_date))
        )
    if rf_ticker:
        fetch_specs.append(
            (fetch_rf_data, (rf_ticker, start_date, end_date))
        )

    async def fetch_all_data():
        tasks = [
            asyncio.to_thread(fetch_function, *args)
            for fetch_function, args in fetch_specs
        ]
        return await asyncio.gather(*tasks)

    fetched_data = asyncio.run(fetch_all_data())
    prices, tickers_list, dates = fetched_data[0]
    fetched_index = 1
    if benchmark_ticker:
        benchmark_prices, benchmark_dates = fetched_data[fetched_index]
        fetched_index += 1
    else:
        benchmark_prices = None
        benchmark_dates = None
    if rf_ticker:
        rf_yields, rf_dates = fetched_data[fetched_index]
    else:
        rf_yields = None
        rf_dates = None

    benchmark_simple_returns = None
    if benchmark_prices is not None:
        asset_index = pd.to_datetime(dates)
        benchmark_series = pd.Series(
            benchmark_prices,
            index=pd.to_datetime(benchmark_dates),
        ).reindex(asset_index).ffill().bfill()
        benchmark_prices = benchmark_series.to_numpy()
        benchmark_simple_returns = calculate_simple_returns(
            benchmark_prices.reshape(-1, 1)
        ).ravel()

    daily_rf_rates = None
    if rf_yields is not None:
        asset_index = pd.to_datetime(dates)
        yield_series = pd.Series(
            rf_yields,
            index=pd.to_datetime(rf_dates),
        ).reindex(asset_index).ffill().bfill()
        daily_rf_rates = (
            (1.0 + (yield_series.to_numpy() / 100.0)) ** (1.0 / 252.0)
        ) - 1.0
        daily_rf_rates = np.nan_to_num(daily_rf_rates, nan=0.0)
    
    # 2. Calcul des rendements simples
    simple_returns = calculate_simple_returns(prices)
    
    return {
        'simple_returns': simple_returns,
        'prices': prices,
        'tickers': tickers_list,
        'dates': dates,
        'benchmark_ticker': benchmark_ticker,
        'benchmark_prices': benchmark_prices,
        'benchmark_simple_returns': benchmark_simple_returns,
        'rf_ticker': rf_ticker,
        'daily_rf_rates': daily_rf_rates,
        'sentiment_applied': apply_sentiment
    }