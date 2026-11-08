"""
Opérateur de croisement BLX-α (Blend Crossover with alpha).
Vectorisation via NumPy et compilation C via Numba.
"""

import numpy as np
from numba import njit


# Constante d'exploration du croisement BLX-α
ALPHA_BLX = 0.5


@njit(cache=True)
def blx_alpha_crossover(parent1, parent2, seed=None):
    """
    Opérateur de croisement BLX-α pour deux parents continus.
    
    Logique vectorisée :
    - Extrait les bornes locales Cmin = min(p1, p2) et Cmax = max(p1, p2)
    - Calcule l'écart Δ = Cmax - Cmin
    - Génère l'enfant en tirant uniformément chaque gène dans 
      l'intervalle [Cmin - α*Δ, Cmax + α*Δ]
    
    Parameters
    ----------
    parent1 : np.ndarray
        Vecteur du parent 1 (n_assets,).
    parent2 : np.ndarray
        Vecteur du parent 2 (n_assets,).
    seed : int, optional
        Graine aléatoire (non utilisée en @njit mais conservée pour interface).
    
    Returns
    -------
    np.ndarray
        Chromosome enfant (n_assets,).
    """
    n_genes = len(parent1)
    child = np.empty(n_genes)
    
    for i in range(n_genes):
        # Bornes locales
        c_min = min(parent1[i], parent2[i])
        c_max = max(parent1[i], parent2[i])
        
        # Écart
        delta = c_max - c_min
        
        # Intervalle élargi
        lower_bound = c_min - ALPHA_BLX * delta
        upper_bound = c_max + ALPHA_BLX * delta
        
        # Tirage uniforme
        child[i] = lower_bound + np.random.rand() * (upper_bound - lower_bound)
    
    return child


@njit(cache=True)
def crossover_population(selected_population):
    """
    Croisement BLX-α à l'échelle de la population avec brassage aléatoire.
    Vectorisé et compilé avec @njit pour éliminer les context switches Python.

    Deux permutations indépendantes fournissent les parents de chaque enfant,
    ce qui évite les biais liés à la position dans la population sélectionnée.
    
    Parameters
    ----------
    selected_population : np.ndarray
        Matrice des parents sélectionnés (n_parents, n_assets).
    
    Returns
    -------
    np.ndarray
        Population d'enfants (n_parents, n_assets).
    """
    n_parents = selected_population.shape[0]
    n_assets = selected_population.shape[1]

    if n_parents < 2:
        return selected_population.copy()
    
    offspring = np.empty((n_parents, n_assets))

    indices_p1 = np.random.permutation(n_parents)
    indices_p2 = np.random.permutation(n_parents)

    for i in range(n_parents):
        p1 = indices_p1[i]
        p2 = indices_p2[i]
        while p1 == p2:
            p2 = np.random.randint(0, n_parents)

        for g in range(n_assets):
            c_min = min(selected_population[p1, g], selected_population[p2, g])
            c_max = max(selected_population[p1, g], selected_population[p2, g])
            delta = c_max - c_min
            lower_bound = c_min - ALPHA_BLX * delta
            upper_bound = c_max + ALPHA_BLX * delta
            offspring[i, g] = lower_bound + np.random.rand() * (upper_bound - lower_bound)
    
    return offspring


def crossover_batch(parents, population_size, seed=None):
    """
    Croisement BLX-α appliqué à un batch de couples de parents (legacy).
    
    Parameters
    ----------
    parents : np.ndarray
        Matrice des parents sélectionnés (population_size, n_assets).
    population_size : int
        Taille de la population (nombre d'individus à générer).
    seed : int, optional
        Graine aléatoire pour la reproductibilité.
    
    Returns
    -------
    np.ndarray
        Population d'enfants (population_size, n_assets).
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_assets = parents.shape[1]
    offspring = np.empty((population_size, n_assets))
    
    for k in range(population_size):
        # Sélection aléatoire de deux parents distincts
        idx1, idx2 = np.random.choice(len(parents), size=2, replace=False)
        offspring[k, :] = blx_alpha_crossover(parents[idx1, :], parents[idx2, :])
    
    return offspring
