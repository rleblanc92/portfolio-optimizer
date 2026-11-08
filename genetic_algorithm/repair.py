"""
Opérateur de réparation itératif garantissant les contraintes.
Vectorisation via NumPy et compilation C via Numba.
"""

import numpy as np
from numba import njit


@njit(cache=True)
def repair_chromosome(chromosome, w_min, w_max, gross_exposure_target=1.0, max_iterations=10, tolerance=1e-8):
    """
    Répare un chromosome en deux étapes itératives.
    
    Étape 1 : Clip les poids à [w_min, w_max]
    Étape 2 : Réduit proportionnellement les poids si l'exposition dépasse
              la borne, et initialise un portefeuille long/short quasi nul.
    
    Parameters
    ----------
    chromosome : np.ndarray
        Vecteur de poids (n_assets,).
    w_min : float
        Poids minimum.
    w_max : float
        Poids maximum.
    gross_exposure_target : float
        Exposition brute cible (défaut: 1.0).
    max_iterations : int
        Nombre maximal d'itérations (défaut: 10).
    tolerance : float
        Tolérance pour convergence (défaut: 1e-8).
    
    Returns
    -------
    np.ndarray
        Chromosome réparé.
    """
    repaired = chromosome.copy()
    n_assets = len(repaired)

    # Une cible supérieure à la somme des bornes individuelles est impossible.
    max_gross_exposure = n_assets * max(abs(w_min), abs(w_max))
    target_exposure = gross_exposure_target
    if target_exposure < 0.0:
        target_exposure = 0.0
    if target_exposure > max_gross_exposure:
        target_exposure = max_gross_exposure
    if w_min >= 0.0 and target_exposure > 1.0:
        target_exposure = 1.0

    # Étape 1 : Clip à [w_min, w_max]
    for i in range(n_assets):
        if repaired[i] < w_min:
            repaired[i] = w_min
        elif repaired[i] > w_max:
            repaired[i] = w_max

    gross_exposure = 0.0
    for i in range(n_assets):
        gross_exposure += abs(repaired[i])

    # Une réduction homothétique conserve les bornes déjà respectées.
    if gross_exposure > target_exposure:
        scaling_factor = target_exposure / gross_exposure
        for i in range(n_assets):
            repaired[i] *= scaling_factor
        gross_exposure = target_exposure

    elif w_min < 0.0 and gross_exposure < 1e-8 and gross_exposure < target_exposure - tolerance:
        remaining_exposure = target_exposure - gross_exposure
        capacity_total = 0.0
        for i in range(n_assets):
            if repaired[i] > 0.0:
                capacity_total += max(0.0, w_max - repaired[i])
            elif repaired[i] < 0.0:
                capacity_total += max(0.0, repaired[i] - w_min)
            elif w_max == abs(w_min):
                capacity_total += max(0.0, w_max)
            elif w_max > abs(w_min):
                capacity_total += max(0.0, w_max)
            else:
                capacity_total += max(0.0, -w_min)

        if capacity_total > 1e-10:
            if remaining_exposure > capacity_total:
                remaining_exposure = capacity_total
            for i in range(n_assets):
                if repaired[i] > 0.0:
                    capacity = max(0.0, w_max - repaired[i])
                    repaired[i] += remaining_exposure * capacity / capacity_total
                elif repaired[i] < 0.0:
                    capacity = max(0.0, repaired[i] - w_min)
                    repaired[i] -= remaining_exposure * capacity / capacity_total
                elif w_max == abs(w_min):
                    capacity = max(0.0, w_max)
                    weight = remaining_exposure * capacity / capacity_total
                    if i % 2 == 0:
                        repaired[i] = weight
                    else:
                        repaired[i] = -weight
                elif w_max > abs(w_min):
                    capacity = max(0.0, w_max)
                    repaired[i] += remaining_exposure * capacity / capacity_total
                else:
                    capacity = max(0.0, -w_min)
                    repaired[i] -= remaining_exposure * capacity / capacity_total

    # Garde numérique finale : l'expansion ne peut jamais franchir les bornes.
    for i in range(n_assets):
        if repaired[i] < w_min:
            repaired[i] = w_min
        elif repaired[i] > w_max:
            repaired[i] = w_max

    final_gross = 0.0
    for i in range(n_assets):
        final_gross += abs(repaired[i])
    if final_gross > target_exposure and final_gross > 1e-12:
        factor = target_exposure / final_gross
        for i in range(n_assets):
            repaired[i] *= factor
    
    return repaired


@njit(cache=True)
def repair_population(population, w_min, w_max, gross_exposure_target=1.0, max_iterations=10, tolerance=1e-8):
    """
    Applique la réparation à une population entière.
    Vectorisée et compilée avec @njit pour éliminer les context switches Python.
    
    Parameters
    ----------
    population : np.ndarray
        Matrice de population (population_size, n_assets).
    w_min : float
        Poids minimum.
    w_max : float
        Poids maximum.
    gross_exposure_target : float
        Exposition brute cible.
    max_iterations : int
        Nombre maximal d'itérations.
    tolerance : float
        Tolérance numérique.
    
    Returns
    -------
    np.ndarray
        Population réparée.
    """
    population_size = population.shape[0]
    repaired_population = np.empty_like(population)
    
    for i in range(population_size):
        repaired_population[i, :] = repair_chromosome(
            population[i, :],
            w_min,
            w_max,
            gross_exposure_target,
            max_iterations,
            tolerance
        )
    
    return repaired_population
