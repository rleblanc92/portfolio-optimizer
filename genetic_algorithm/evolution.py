"""
Noyau d'orchestre d'évolution génétique pour l'optimisation de portefeuille.

Orchestrate complète de l'algorithme génétique :
- Initialisation de population
- Boucle générationnelle
- Sélection par tournoi (vectorisée Numba)
- Croisement (BLX-α)
- Mutation (Bump Mutation)
- Réparation des contraintes
- Élitisme (5% meilleurs portefeuilles)
- Tracking du meilleur portefeuille absolu

Auteur et Développeur Exclusif: Robin Leblanc
Architecture Propriétaire d'Optimisation de Portefeuille
"""

import numpy as np
from multiprocessing import current_process
from numba import njit
from tqdm.auto import tqdm
from logging_utils import get_progress_stream

# Imports des modules génétiques existants
from .crossover import blx_alpha_crossover, crossover_population
from .mutation import bump_mutation, mutate_population
from .repair import repair_chromosome, repair_population
from .fitness import evaluate_fitness, evaluate_population


@njit(cache=True)
def _tournament_selection(fitness_scores, tournament_size=3, n_selected=None):
    """
    Sélection par tournoi vectorisée 100% NumPy/Numba.
    
    Logique :
    - Chaque tournoi: tirage aléatoire de 'tournament_size' indices
    - Sélection du meilleur fitness parmi les candidats du tournoi
    - Répète pour générer n_selected individus
    
    Architecture mémoire :
    - Réutilise un tableau d'indices et un historique des échanges
    - Exécute un Fisher-Yates partiel limité à 'tournament_size' échanges
    - Restaure l'ordre initial après chaque tournoi
    
    Parameters
    ----------
    fitness_scores : np.ndarray
        Vecteur des scores de fitness (population_size,).
        Scores plus élevés = meilleurs portefeuilles.
    tournament_size : int
        Nombre d'individus par tournoi (défaut: 3).
    n_selected : int, optional
        Nombre d'individus à sélectionner (défaut: len(fitness_scores)).
    
    Returns
    -------
    np.ndarray
        Vecteur des indices des individus sélectionnés (n_selected,).
    """
    population_size = len(fitness_scores)
    
    if n_selected is None:
        n_selected = population_size
    
    selected_indices = np.empty(n_selected, dtype=np.int64)
    indices = np.arange(population_size, dtype=np.int64)
    swap_history = np.empty(tournament_size, dtype=np.int64)
    tournament_indices = np.empty(tournament_size, dtype=np.int64)
    
    # Sélection par tournoi : répète n_selected fois
    for i in range(n_selected):
        # Fisher-Yates partiel : mélange uniquement les k premières positions.
        for j in range(tournament_size):
            swap_index = np.random.randint(j, population_size)
            swap_history[j] = swap_index
            indices[j], indices[swap_index] = indices[swap_index], indices[j]
            tournament_indices[j] = indices[j]
        
        # Trouver le meilleur fitness dans ce tournoi
        best_in_tournament = tournament_indices[0]
        best_fitness = fitness_scores[best_in_tournament]
        
        for j in range(1, tournament_size):
            idx = tournament_indices[j]
            if fitness_scores[idx] > best_fitness:
                best_fitness = fitness_scores[idx]
                best_in_tournament = idx
        
        selected_indices[i] = best_in_tournament

        # Restaurer l'ordre initial pour le tournoi suivant.
        for j in range(tournament_size - 1, -1, -1):
            swap_index = swap_history[j]
            indices[j], indices[swap_index] = indices[swap_index], indices[j]
    
    return selected_indices


def run_evolution(expected_returns, simple_returns, previous_weights, 
                  population_size=800, generations=1000, mutation_rate=0.10,
                  lambda_risk=0.5, lambda_skew=0.05, lambda_kurt=0.01,
                  lambda_turnover=0.024, max_weight=0.25, w_min=0.0,
                  elite_ratio=0.05, mean_rf_rate=0.0, seed=None,
                  margin_rate=0.0, gross_exposure_target=1.0,
                  risk_free_rate=None,
                  cov_matrix=np.empty((0, 0), dtype=np.float64),
                  time_weights=np.empty(0, dtype=np.float64)):
    """
    Orchestre complète du noyau d'évolution génétique.
    
    Architecture :
    1. Initialisation : population aléatoire, réparée
    2. Boucle générationnelle :
       a. Évaluation de fitness (5 composantes)
       b. Sélection par tournoi (taille 3)
       c. Croisement (BLX-α, α=0.5)
       d. Mutation (Bump Mutation)
       e. Réparation itérative
       f. Élitisme (5% meilleurs)
    3. Tracking du meilleur absolu (aucune régression)
    4. Retour du meilleur portefeuille trouvé
    
    Parameters
    ----------
    expected_returns : np.ndarray
        Vecteur des rendements espérés (n_assets,).
    simple_returns : np.ndarray
        Matrice des rendements simples historiques (n_periods, n_assets).
    previous_weights : np.ndarray
        Poids du portefeuille au rééquilibrage précédent (n_assets,).
    population_size : int
        Taille de la population (défaut: 800).
    generations : int
        Nombre de générations (défaut: 1000).
    mutation_rate : float
        Probabilité de mutation (défaut: 0.10).
    lambda_risk : float
        Poids de la pénalité de risque (défaut: 0.5).
    lambda_skew : float
        Poids de la préférence de skewness (défaut: 0.05).
    lambda_kurt : float
        Poids de l'aversion au kurtosis (défaut: 0.01).
    lambda_turnover : float
        Poids de l'aversion au turnover (défaut: 0.024).
    max_weight : float
        Poids maximum par actif (défaut: 0.25).
    w_min : float
        Poids minimum par actif (défaut: 0.0 pour long-only).
    elite_ratio : float
        Ratio d'élitisme (défaut: 0.05 = 5%).
    seed : int, optional
        Graine aléatoire pour reproductibilité.
    mean_rf_rate : float
        Taux sans risque journalier moyen sur la fenêtre In-Sample.
    margin_rate : float
        Coût annuel du financement de la marge.
    gross_exposure_target : float
        Exposition brute maximale autorisée.
    cov_matrix : np.ndarray, optional
        Matrice de covariance régularisée, annualisée par Ledoit-Wolf.
    time_weights : np.ndarray, optional
        Poids temporels réutilisés pour toutes les évaluations de population.
    risk_free_rate : float, optional
        Alias de compatibilité pour les anciens appels.
    
    Returns
    -------
    dict
        Dictionnaire contenant :
        - 'best_weights': np.ndarray - Meilleur portefeuille trouvé
        - 'best_fitness': float - Meilleur score de fitness
        - 'fitness_history': np.ndarray - Fitness max par génération
        - 'generations_count': int - Nombre de générations exécutées
        - 'population_size': int - Taille de population utilisée
    """
    if risk_free_rate is not None:
        mean_rf_rate = risk_free_rate

    # Fixation de la graine pour reproductibilité
    if seed is not None:
        np.random.seed(seed)
    
    n_assets = len(expected_returns)
    n_elite = max(1, int(population_size * elite_ratio))  # Au moins 1
    
    # Validation des dimensions
    assert simple_returns.shape[1] == n_assets, \
        f"Mismatch simple_returns: expected n_assets={n_assets}, got {simple_returns.shape[1]}"
    assert len(previous_weights) == n_assets, \
        f"Mismatch previous_weights: expected {n_assets}, got {len(previous_weights)}"
    
    # ========================================================================
    # Phase 1: Initialisation de la population
    # ========================================================================
    # Génération uniforme respectant les bornes
    population = np.random.uniform(w_min, max_weight, size=(population_size, n_assets))
    if n_assets > 4:
        half = population_size // 2
        sparse_mask = np.random.rand(population_size - half, n_assets) < 0.60
        population[half:][sparse_mask] = 0.0
    
    # Réparation vectorisée via @njit
    population = repair_population(population, w_min, max_weight,
                                   gross_exposure_target=gross_exposure_target,
                                   max_iterations=10, tolerance=1e-10)
    
    # Évaluation initiale
    fitness_scores = evaluate_population(
        population, expected_returns, simple_returns, previous_weights,
        lambda_risk, lambda_skew, lambda_kurt, lambda_turnover,
        mean_rf_rate, margin_rate, cov_matrix, time_weights
    )
    
    # Tracking du meilleur portefeuille absolu
    best_idx = np.argmax(fitness_scores)
    best_weights = population[best_idx].copy()
    best_fitness = fitness_scores[best_idx]
    
    # Historique du fitness
    fitness_history = np.zeros(generations)
    fitness_history[0] = best_fitness
    stagnation_counter = 0
    early_stopping_patience = 200
    min_delta = 1e-7
    last_significant_fitness = best_fitness
    generations_count = generations
    
    # ========================================================================
    # Phase 2: Boucle générationnelle
    # ========================================================================
    generation_range = range(1, generations)
    pbar = None
    if current_process().name == 'MainProcess':
        pbar = tqdm(generation_range, desc="Optimisation GA", unit="gen", leave=True,
                    disable=None, mininterval=0.5, file=get_progress_stream())
        generation_range = pbar
    for gen in generation_range:
        # --- 2a. Sélection par tournoi ---
        selected_indices = _tournament_selection(
            fitness_scores, tournament_size=3, n_selected=population_size
        )
        selected_population = population[selected_indices].copy()
        
        # --- 2b. Croisement (BLX-α) vectorisé via @njit ---
        offspring = crossover_population(selected_population)
        
        # --- 2c. Mutation (Bump Mutation) vectorisée via @njit ---
        effective_mutation_rate = min(0.50, mutation_rate * 2.5) if stagnation_counter >= 50 else mutation_rate
        offspring = mutate_population(offspring, effective_mutation_rate, w_min, max_weight)
        
        # --- 2d. Réparation itérative vectorisée via @njit ---
        offspring = repair_population(offspring, w_min, max_weight, 
                                     gross_exposure_target=gross_exposure_target, 
                                     max_iterations=10, tolerance=1e-10)
        
        # --- 2e. Évaluation de la nouvelle population ---
        offspring_fitness = evaluate_population(
            offspring, expected_returns, simple_returns, previous_weights,
            lambda_risk, lambda_skew, lambda_kurt, lambda_turnover,
            mean_rf_rate, margin_rate, cov_matrix, time_weights
        )
        
        # --- 2f. Élitisme & remplacement standard ---
        elite_indices = np.argsort(fitness_scores)[-n_elite:]
        worst_offspring_indices = np.argsort(offspring_fitness)[:n_elite]
        for i in range(n_elite):
            worst_idx = worst_offspring_indices[i]
            elite_idx = elite_indices[i]
            offspring[worst_idx] = population[elite_idx]
            offspring_fitness[worst_idx] = fitness_scores[elite_idx]

        population = offspring
        fitness_scores = offspring_fitness
        
        # --- 2g. Tracking du meilleur portefeuille absolu (aucune régression) ---
        current_best_idx = np.argmax(fitness_scores)
        current_best_fitness = fitness_scores[current_best_idx]
        
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_weights = population[current_best_idx].copy()

        if best_fitness - last_significant_fitness > min_delta:
            last_significant_fitness = best_fitness
            stagnation_counter = 0
        else:
            stagnation_counter += 1

        fitness_history[gen] = best_fitness

        update_progress_metrics = (
            generations < 20
            or gen % 10 == 0
            or gen == generations - 1
            or (stagnation_counter >= early_stopping_patience and gen >= 200)
        )
        if pbar is not None and not pbar.disable and update_progress_metrics:
            long_exposure = np.sum(np.maximum(best_weights, 0.0))
            short_exposure = np.sum(np.abs(np.minimum(best_weights, 0.0)))
            financed_exposure = max(0.0, long_exposure - 1.0) + short_exposure
            portfolio_returns = (
                np.dot(simple_returns, best_weights)
                + max(0.0, 1.0 - long_exposure) * mean_rf_rate
                - financed_exposure * margin_rate / 252.0
            )
            volatility = np.std(portfolio_returns) * np.sqrt(252.0)
            nav = np.concatenate(([1.0], np.cumprod(1.0 + portfolio_returns)))
            if volatility > 1e-10 and len(portfolio_returns) > 0 and nav[-1] > 0.0:
                annual_return = nav[-1] ** (252.0 / len(portfolio_returns)) - 1.0
                best_sharpe = (annual_return - mean_rf_rate * 252.0) / volatility
            else:
                best_sharpe = 0.0
            peaks = np.maximum.accumulate(nav)
            best_mdd = np.min(nav / peaks - 1.0)
            avg_turnover = np.mean(np.sum(np.abs(population - previous_weights), axis=1))
            pbar.set_postfix({"Best_Sharpe": f"{best_sharpe:.2f}",
                              "Best_MaxDD": f"{best_mdd:.2%}",
                              "Avg_Turnover": f"{avg_turnover:.2%}"}, refresh=False)

        if stagnation_counter >= early_stopping_patience and gen >= 200:
            fitness_history[gen:] = best_fitness
            generations_count = gen + 1
            break

    if pbar is not None:
        pbar.close()
    
    # ========================================================================
    # Phase 3: Résultats finaux
    # ========================================================================
    results = {
        'best_weights': best_weights,
        'best_fitness': best_fitness,
        'fitness_history': fitness_history,
        'generations_count': generations_count,
        'population_size': population_size,
    }
    
    return results
