"""
Modèles d'optimisation classiques de référence (benchmarks).
Vectorisation via NumPy et compilation Numba où applicable.
"""

import numpy as np
from numba import njit
from scipy.optimize import minimize
from scipy.linalg import solve


@njit(cache=True)
def minimize_variance_coordinate_descent_numba(cov_matrix, max_iterations=20, tolerance=1e-5, allow_short=False, max_weight=1.0):
    """
    Minimisation de la variance via descente de coordonnées.
    
    Résout : min_w [1/2 * w^T Σ w + 1/2 * (sum(w) - 1)^2]
    sous bornes et projection sur la somme faisable.
    
    Pénalité quadratique :
    f̄(w) = 1/2 * w^T Σ w + 1/2 * (sum(w) - 1)^2
    
    Gradient :
    ∇f̄(w) = Σw + (sum(w) - 1) * 1
    
    Hessienne :
    H = Σ + 1*1^T
    
    Parameters
    ----------
    cov_matrix : np.ndarray
        Matrice de covariance (n_assets, n_assets).
    max_iterations : int
        Nombre maximal d'itérations (défaut: 20).
    tolerance : float
        Tolérance pour convergence (défaut: 1e-5).
    allow_short : bool
        Si True, autorise les positions courtes dans [-max_weight, max_weight].
        Si False (défaut), utilise les bornes [0, max_weight].
    max_weight : float
        Borne absolue des poids individuels.
    
    Returns
    -------
    np.ndarray
        Vecteur de poids optimal (n_assets,).
    """
    n_assets = cov_matrix.shape[0]
    
    # Descente de coordonnées projetée sur les bornes individuelles.
    w_min = -max_weight if allow_short else 0.0
    weights = np.clip(np.ones(n_assets) / n_assets, w_min, max_weight)

    # Hessienne statique H = Σ + 1*1^T, régularisée pour garantir
    # la stricte convexité et la stabilité numérique.
    hessian = cov_matrix + np.ones((n_assets, n_assets))
    hessian = hessian + np.eye(n_assets) * 1e-4
    
    for iteration in range(max_iterations * 10):
        previous_weights = weights.copy()

        for i in range(n_assets):
            partial_gradient = -1.0
            for j in range(n_assets):
                partial_gradient += hessian[i, j] * weights[j]

            weights[i] -= partial_gradient / hessian[i, i]
            if weights[i] < w_min:
                weights[i] = w_min
            elif weights[i] > max_weight:
                weights[i] = max_weight

        norm_diff = 0.0
        for i in range(n_assets):
            norm_diff += abs(weights[i] - previous_weights[i])

        if norm_diff < tolerance:
            break

    target_sum = min(1.0, n_assets * max_weight)
    lower = np.min(weights) - max_weight
    upper = np.max(weights) - w_min
    for _ in range(60):
        midpoint = (lower + upper) / 2.0
        projected_sum = 0.0
        for i in range(n_assets):
            projected_sum += min(max(weights[i] - midpoint, w_min), max_weight)
        if projected_sum > target_sum:
            lower = midpoint
        else:
            upper = midpoint
    for i in range(n_assets):
        weights[i] = min(max(weights[i] - upper, w_min), max_weight)
    
    return weights


def minimize_variance_coordinate_descent(
    cov_matrix,
    max_iterations=20,
    tolerance=1e-5,
    allow_short=False,
    force_full_investment=False,
    max_weight=0.25,
    gross_exposure_target=1.0,
):
    """
    Minimisation de la variance via descente de coordonnées.
    
    Wrapper autour de la fonction Numba pour interface publique.
    
    Parameters
    ----------
    cov_matrix : np.ndarray
        Matrice de covariance (n_assets, n_assets).
    max_iterations : int
        Nombre maximal d'itérations.
    tolerance : float
        Tolérance pour convergence.
    allow_short : bool
        Si True, autorise les positions courtes (long/short).
    
    Returns
    -------
    dict
        Dictionnaire contenant les infos d'optimisation:
        - 'weights': vecteur de poids optimal (n_assets,)
        - 'portfolio_variance': variance du portefeuille
        - 'convergence_criterion': norme de convergence finale
    """
    if allow_short:
        n_assets = cov_matrix.shape[0]
        target_sum = min(1.0, n_assets * max_weight)
        bounds = tuple((-max_weight, max_weight) for _ in range(n_assets))
        constraints = [{
            'type': 'ineq',
            'fun': lambda weights: gross_exposure_target - np.sum(
                np.sqrt(weights**2 + 1e-16)
            ),
        }]

        if force_full_investment:
            constraints.append({
                'type': 'eq',
                'fun': lambda weights: np.sum(weights) - target_sum,
            })

        result = minimize(
            lambda weights: np.dot(weights, np.dot(cov_matrix, weights)),
            np.ones(n_assets) / n_assets,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-9, 'maxiter': 1000},
        )
        if not result.success or not np.all(np.isfinite(result.x)):
            weights = np.minimum(equal_weight_portfolio(n_assets), max_weight)
        else:
            weights = result.x
        weights = np.clip(weights, -max_weight, max_weight)
    else:
        weights = minimize_variance_coordinate_descent_numba(
            cov_matrix, max_iterations, tolerance, allow_short, max_weight
        )

    # Calcul de la variance finale
    portfolio_variance = 0.0
    for i in range(len(weights)):
        for j in range(len(weights)):
            portfolio_variance += weights[i] * cov_matrix[i, j] * weights[j]
    
    result = {
        'weights': weights,
        'portfolio_variance': portfolio_variance,
    }
    
    return result


def maximize_sharpe_slsqp(
    expected_returns,
    cov_matrix,
    risk_free_rate=0.0,
    max_weight=0.25,
    allow_short=False,
    method='SLSQP',
    force_full_investment=False,
    margin_rate=0.0,
    gross_exposure_target=1.0,
    mean_rf_rate=0.0,
):
    """
    Maximisation du ratio de Sharpe via SLSQP (Sequential Least Squares Programming).
    
    Minimise : -Sharpe_Ratio = -(w^T μ - Rf) / sqrt(w^T Σ w)
    
    Sécurité numérique :
    - Volatilité minimale : 1e-8 pour éviter division par zéro
    - Rf fixé à 0.0 par défaut (peut être modifié)
    
    Parameters
    ----------
    expected_returns : np.ndarray
        Vecteur des rendements espérés (n_assets,).
    cov_matrix : np.ndarray
        Matrice de covariance (n_assets, n_assets).
    risk_free_rate : float
        Taux sans risque (défaut: 0.0).
    max_weight : float
        Poids maximum par actif (défaut: 0.25).
    allow_short : bool
        Si True, autorise les positions courtes (bornes: -max_weight à max_weight).
        Si False (défaut), long-only (bornes: 0 à max_weight).
    method : str
        Méthode de scipy.optimize.minimize (défaut: 'SLSQP').
    mean_rf_rate : float
        Rendement journalier moyen de la poche Cash.
    
    Returns
    -------
    dict
        Dictionnaire contenant:
        - 'weights': vecteur de poids optimal (n_assets,)
        - 'sharpe_ratio': ratio de Sharpe du portefeuille
        - 'portfolio_return': rendement du portefeuille
        - 'portfolio_volatility': volatilité du portefeuille
        - 'success': booléen indiquant la convergence
    """
    n_assets = len(expected_returns)
    
    # Fonction objectif : négatif du ratio de Sharpe
    def objective(weights):
        long_exposure = np.sum(np.maximum(weights, 0.0))
        short_exposure = np.sum(np.abs(np.minimum(weights, 0.0)))
        financed_exposure = max(0.0, long_exposure - 1.0) + short_exposure
        free_cash_weight = max(0.0, 1.0 - long_exposure)
        annual_rf_rate = risk_free_rate * 252.0
        annual_cash_rate = mean_rf_rate * 252.0
        portfolio_return = (
            np.dot(weights, expected_returns)
            + (free_cash_weight * annual_cash_rate)
            - (financed_exposure * margin_rate)
        )
        
        # Volatilité du portefeuille (covariance déjà annualisée)
        portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)  # déjà en unités annualisées
        
        # Sécurité numérique : volatilité minimale
        if portfolio_volatility < 1e-8:
            portfolio_volatility = 1e-8
        
        # Ratio de Sharpe (négatif pour minimisation)
        sharpe_ratio = -(portfolio_return - annual_rf_rate) / portfolio_volatility
        
        return sharpe_ratio
    
    # Bornes : Conditionnelles selon allow_short
    if allow_short:
        bounds = tuple((-max_weight, max_weight) for _ in range(n_assets))
    else:
        bounds = tuple((0.0, max_weight) for _ in range(n_assets))
    
    # Contraintes : exposition brute et, si nécessaire, investissement total
    constraints = []
    if allow_short:
        constraint_gross_exposure = {
            'type': 'ineq',
            'fun': lambda w: gross_exposure_target - np.sum(
                np.sqrt(w**2 + 1e-16)
            )
        }
        constraints.append(constraint_gross_exposure)

    target_sum = min(1.0, n_assets * max_weight)
    if force_full_investment or not allow_short:
        constraint_sum = {'type': 'eq', 'fun': lambda w: np.sum(w) - target_sum}
        constraints.append(constraint_sum)
    
    # Initialisation : poids égaux
    w0 = np.minimum(np.ones(n_assets) / n_assets, max_weight)
    
    # Résolution
    result = minimize(
        objective,
        w0,
        method=method,
        bounds=bounds,
        constraints=constraints,
        options={'ftol': 1e-9, 'maxiter': 1000}
    )

    success = bool(result.success and np.all(np.isfinite(result.x)))
    if not success:
        optimal_weights = np.minimum(equal_weight_portfolio(n_assets), max_weight)
    else:
        optimal_weights = result.x

    optimal_weights = np.clip(
        optimal_weights,
        -max_weight if allow_short else 0.0,
        max_weight,
    )
    
    # Calcul des statistiques finales
    long_exposure = np.sum(np.maximum(optimal_weights, 0.0))
    short_exposure = np.sum(np.abs(np.minimum(optimal_weights, 0.0)))
    financed_exposure = max(0.0, long_exposure - 1.0) + short_exposure
    free_cash_weight = max(0.0, 1.0 - long_exposure)
    annual_rf_rate = risk_free_rate * 252.0
    annual_cash_rate = mean_rf_rate * 252.0
    portfolio_return = (
        np.dot(optimal_weights, expected_returns)
        + (free_cash_weight * annual_cash_rate)
        - (financed_exposure * margin_rate)
    )
    portfolio_variance = np.dot(optimal_weights, np.dot(cov_matrix, optimal_weights))
    portfolio_volatility = np.sqrt(portfolio_variance)
    
    if portfolio_volatility > 1e-8:
        sharpe_ratio = (portfolio_return - annual_rf_rate) / portfolio_volatility
    else:
        sharpe_ratio = 0.0
    
    return {
        'weights': optimal_weights,
        'sharpe_ratio': sharpe_ratio,
        'portfolio_return': portfolio_return,
        'portfolio_volatility': portfolio_volatility,
        'success': success,
        'message': result.message
    }


def equal_weight_portfolio(n_assets):
    """
    Benchmark simple : portefeuille de poids égaux (1/N).
    
    Parameters
    ----------
    n_assets : int
        Nombre d'actifs.
    
    Returns
    -------
    np.ndarray
        Vecteur de poids égaux (n_assets,).
    """
    return np.ones(n_assets) / n_assets
