"""
Évaluation multi-objectifs vectorisée de la population.
Vectorisation via NumPy et compilation C via Numba.
"""

import numpy as np
from numba import njit


@njit(cache=True)
def calculate_portfolio_returns(weights, simple_returns, mean_rf_rate=0.0, margin_rate=0.0):
    """
    Calcule la série des rendements du portefeuille (avec rendement du cash).
    
    Parameters
    ----------
    weights : np.ndarray
        Vecteur de poids (n_assets,).
    simple_returns : np.ndarray
        Matrice des rendements simples (n_periods, n_assets).
    mean_rf_rate : float
        Taux sans risque journalier moyen sur la fenêtre In-Sample.
    margin_rate : float
        Coût annuel du financement de la marge.
    
    Returns
    -------
    np.ndarray
        Série des rendements du portefeuille (n_periods,).
    """
    long_exposure = np.sum(np.maximum(weights, 0.0))
    short_exposure = np.sum(np.abs(np.minimum(weights, 0.0)))
    financed_exposure = max(0.0, long_exposure - 1.0) + short_exposure
    free_cash_weight = max(0.0, 1.0 - long_exposure)
    daily_constant = (free_cash_weight * mean_rf_rate) - (
        financed_exposure * (margin_rate / 252.0)
    )
    return np.dot(simple_returns, weights) + daily_constant


@njit(cache=True)
def calculate_skewness(returns):
    """
    Calcule le skewness (asymétrie) d'une série de rendements.
    
    Parameters
    ----------
    returns : np.ndarray
        Série des rendements (n_periods,).
    
    Returns
    -------
    float
        Skewness.
    """
    n = len(returns)
    mean = 0.0
    for r in returns:
        mean += r
    mean /= n
    
    # Variance
    variance = 0.0
    for r in returns:
        variance += (r - mean) ** 2
    variance /= n
    
    # Déviation standard
    std_dev = np.sqrt(variance)
    
    if std_dev < 1e-10:
        return 0.0
    
    # Moment centré d'ordre 3
    third_moment = 0.0
    for r in returns:
        third_moment += (r - mean) ** 3
    third_moment /= n
    
    # Skewness = M3 / std^3
    skewness = third_moment / (std_dev ** 3)
    
    return skewness


@njit(cache=True)
def calculate_kurtosis_excess(returns):
    """
    Calcule le kurtosis excessif (excess kurtosis) d'une série de rendements.
    
    Parameters
    ----------
    returns : np.ndarray
        Série des rendements (n_periods,).
    
    Returns
    -------
    float
        Kurtosis excessif (kurtosis - 3).
    """
    n = len(returns)
    mean = 0.0
    for r in returns:
        mean += r
    mean /= n
    
    # Variance
    variance = 0.0
    for r in returns:
        variance += (r - mean) ** 2
    variance /= n
    
    # Déviation standard
    std_dev = np.sqrt(variance)
    
    if std_dev < 1e-10:
        return 0.0
    
    # Moment centré d'ordre 4
    fourth_moment = 0.0
    for r in returns:
        fourth_moment += (r - mean) ** 4
    fourth_moment /= n
    
    # Kurtosis excessif = (M4 / std^4) - 3
    kurtosis_excess = (fourth_moment / (std_dev ** 4)) - 3.0
    
    return kurtosis_excess


@njit(cache=True)
def _calculate_skew_and_kurt(returns):
    n = len(returns)
    mean = 0.0
    for r in returns:
        mean += r
    mean /= n

    m2 = 0.0
    m3 = 0.0
    m4 = 0.0
    for r in returns:
        diff = r - mean
        diff_sq = diff * diff
        m2 += diff_sq
        m3 += diff_sq * diff
        m4 += diff_sq * diff_sq

    variance = m2 / n
    std_dev = np.sqrt(variance)
    if std_dev < 1e-10:
        return 0.0, 0.0

    return (m3 / n) / (std_dev ** 3), ((m4 / n) / (std_dev ** 4)) - 3.0


@njit(cache=True)
def _is_uniform_time_weights(time_weights):
    if len(time_weights) == 0:
        return True
    total_weight = 0.0
    for weight in time_weights:
        total_weight += weight
    if total_weight <= 0.0:
        return False
    expected_weight = total_weight / len(time_weights)
    for weight in time_weights:
        if abs(weight - expected_weight) > 1e-15:
            return False
    return True


@njit(cache=True)
def _calculate_weighted_moments(returns, time_weights):
    n = len(returns)
    total_weight = 0.0
    weighted_mean = 0.0
    for i in range(n):
        weight = time_weights[i]
        total_weight += weight
        weighted_mean += weight * returns[i]
    if total_weight <= 0.0:
        raise ValueError("time_weights must have a positive sum")
    weighted_mean /= total_weight

    m2 = 0.0
    m3 = 0.0
    m4 = 0.0
    for i in range(n):
        diff = returns[i] - weighted_mean
        diff_sq = diff * diff
        weight = time_weights[i]
        m2 += weight * diff_sq
        m3 += weight * diff_sq * diff
        m4 += weight * diff_sq * diff_sq

    weighted_variance = m2 / total_weight
    std_dev = np.sqrt(weighted_variance)
    if std_dev < 1e-10:
        return weighted_mean, weighted_variance, 0.0, 0.0

    weighted_skewness = (m3 / total_weight) / (std_dev ** 3)
    weighted_kurtosis = ((m4 / total_weight) / (std_dev ** 4)) - 3.0
    return weighted_mean, weighted_variance, weighted_skewness, weighted_kurtosis


@njit(cache=True)
def calculate_turnover(current_weights, previous_weights):
    """Calculate the L1 turnover used by the nonlinear impact model.
    
    Parameters
    ----------
    current_weights : np.ndarray
        Poids actuels (n_assets,).
    previous_weights : np.ndarray
        Poids précédents (n_assets,).
    
    Returns
    -------
    float
        ``sum(abs(w_i^(t) - w_i^(t-1)))``.
    """
    turnover = 0.0
    for i in range(len(current_weights)):
        turnover += abs(current_weights[i] - previous_weights[i])

    return turnover


@njit(cache=True)
def evaluate_fitness(weights, expected_returns, simple_returns, previous_weights,
                     lambda_risk, lambda_skew, lambda_kurt, lambda_turnover,
                     mean_rf_rate, margin_rate=0.0,
                     cov_matrix=np.empty((0, 0), dtype=np.float64),
                     time_weights=np.empty(0, dtype=np.float64)):
    """
    Évaluation multi-objectifs du fitness d'un portefeuille.
    
    Fonction objectif :
    Fitness(w) = w^T μ - λ_risk Var_emp(w) + λ_skew S(w)
                 - λ_kurt K(w) - (λ_TO T + 0.1 λ_TO T²)
    
    Parameters
    ----------
    weights : np.ndarray
        Vecteur de poids du portefeuille (n_assets,).
    expected_returns : np.ndarray
        Vecteur des rendements espérés (n_assets,).
    simple_returns : np.ndarray
        Matrice des rendements simples historiques (n_periods, n_assets).
    previous_weights : np.ndarray
        Poids du portefeuille au rééquilibrage précédent (n_assets,).
    lambda_risk : float
        Poids de la pénalité de risque.
    lambda_skew : float
        Poids de la préférence de skewness.
    lambda_kurt : float
        Poids de l'aversion au kurtosis.
    lambda_turnover : float
        Poids de l'aversion au turnover.
    mean_rf_rate : float
        Taux sans risque journalier moyen sur la fenêtre In-Sample.
    margin_rate : float
        Coût annuel du financement de la marge.
    cov_matrix : np.ndarray, optional
        Matrice de covariance régularisée, annualisée par Ledoit-Wolf.
    time_weights : np.ndarray, optional
        Poids temporels normalisés des observations, les plus récents étant les
        plus pondérés. Un vecteur uniforme conserve le chemin historique.
    
    Returns
    -------
    float
        Valeur du fitness.
    """
    n_assets = len(weights)
    
    long_exposure = np.sum(np.maximum(weights, 0.0))
    short_exposure = np.sum(np.abs(np.minimum(weights, 0.0)))
    risky_exposure = min(1.0, long_exposure + short_exposure)
    financed_exposure = max(0.0, long_exposure - 1.0) + short_exposure

    free_cash_weight = max(0.0, 1.0 - long_exposure)
    annual_rf_rate = mean_rf_rate * 252.0
    expected_return = (
        np.dot(weights, expected_returns)
        + (free_cash_weight * annual_rf_rate)
        - (financed_exposure * margin_rate)
    )
    portfolio_returns = (
        np.dot(simple_returns, weights)
        + (free_cash_weight * mean_rf_rate)
        - (financed_exposure * (margin_rate / 252.0))
    )

    has_time_weights = time_weights.size > 0
    if has_time_weights and (
        time_weights.ndim != 1 or len(time_weights) != len(portfolio_returns)
    ):
        raise ValueError("time_weights must have one value per return date")
    use_weighted_moments = has_time_weights and not _is_uniform_time_weights(time_weights)
    if use_weighted_moments:
        weighted_mean, weighted_variance, raw_skew, raw_kurt = _calculate_weighted_moments(
            portfolio_returns,
            time_weights,
        )
    else:
        raw_skew, raw_kurt = _calculate_skew_and_kurt(portfolio_returns)

    if cov_matrix.size == 0:
        if use_weighted_moments:
            variance = weighted_variance * 252.0
        else:
            # Compatibilité avec les anciens appels ne fournissant pas la covariance.
            mean_ret = np.mean(portfolio_returns)
            variance = np.sum((portfolio_returns - mean_ret) ** 2) / len(portfolio_returns)
            variance = variance * 252.0
    else:
        # Variance analytique annualisée issue de Ledoit-Wolf.
        variance = np.dot(weights, np.dot(cov_matrix, weights))
    
    skewness = raw_skew * risky_exposure
    kurtosis = raw_kurt * risky_exposure
    
    # Terme 5: L1 turnover plus quadratic permanent market impact.
    turnover = calculate_turnover(weights, previous_weights)
    turnover_penalty = (
        (turnover * lambda_turnover)
        + (0.1 * lambda_turnover * (turnover ** 2))
    )
    
    # Fonction objectif complète
    fitness = (expected_return 
               - lambda_risk * variance 
               + lambda_skew * skewness 
               - lambda_kurt * kurtosis 
               - turnover_penalty)
    
    return fitness


@njit(cache=True)
def evaluate_population(population, expected_returns, simple_returns, previous_weights,
                        lambda_risk, lambda_skew, lambda_kurt, lambda_turnover,
                        mean_rf_rate, margin_rate=0.0,
                        cov_matrix=np.empty((0, 0), dtype=np.float64),
                        time_weights=np.empty(0, dtype=np.float64)):
    """
    Évalue le fitness de toute une population.
    Vectorisée et compilée avec @njit pour éliminer les context switches Python.
    
    Parameters
    ----------
    population : np.ndarray
        Matrice de population (population_size, n_assets).
    expected_returns : np.ndarray
        Vecteur des rendements espérés (n_assets,).
    simple_returns : np.ndarray
        Matrice des rendements simples (n_periods, n_assets).
    previous_weights : np.ndarray
        Poids précédents (n_assets,).
    lambda_risk : float
        Poids du risque.
    lambda_skew : float
        Poids du skewness.
    lambda_kurt : float
        Poids du kurtosis.
    lambda_turnover : float
        Poids du turnover.
    mean_rf_rate : float
        Taux sans risque journalier moyen sur la fenêtre In-Sample.
    margin_rate : float
        Coût annuel du financement de la marge.
    cov_matrix : np.ndarray, optional
        Matrice de covariance régularisée, annualisée par Ledoit-Wolf.
    time_weights : np.ndarray, optional
        Poids temporels des observations, partagés entre tous les individus.
    
    Returns
    -------
    np.ndarray
        Vecteur des fitness values (population_size,).
    """
    population_size = population.shape[0]
    if time_weights.size > 0 and (
        time_weights.ndim != 1 or time_weights.shape[0] != simple_returns.shape[0]
    ):
        raise ValueError("time_weights must have one value per return date")
    fitness_values = np.empty(population_size)
    
    for i in range(population_size):
        fitness_values[i] = evaluate_fitness(
            population[i, :],
            expected_returns,
            simple_returns,
            previous_weights,
            lambda_risk,
            lambda_skew,
            lambda_kurt,
            lambda_turnover,
            mean_rf_rate,
            margin_rate,
            cov_matrix,
            time_weights,
        )
    
    return fitness_values
