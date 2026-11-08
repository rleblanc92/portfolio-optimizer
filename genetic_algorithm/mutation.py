"""
Opérateur de mutation Bump Mutation.
Vectorisation via NumPy et compilation C via Numba.
"""

import numpy as np
from numba import njit


@njit(cache=True)
def bump_mutation(chromosome, mutation_rate, w_min, w_max, seed=None):
    """
    Opérateur Bump Mutation pour portefeuille continu.
    
    Logique :
    - Sélectionne aléatoirement deux indices d'actifs distincts i et j
    - Tire un incrément δ ~ U(0.002, 0.05)
    - Applique un transfert de masse : wi += δ, wj -= δ
    - Valide les contraintes : wi + δ ≤ w_max et wj - δ ≥ w_min
    - Si contraintes violées, la mutation ne s'applique pas
    
    Parameters
    ----------
    chromosome : np.ndarray
        Chromosome (portefeuille de poids), shape (n_assets,).
    mutation_rate : float
        Probabilité d'appliquer la mutation (entre 0 et 1).
    w_min : float
        Poids minimum autorisé pour chaque actif.
    w_max : float
        Poids maximum autorisé pour chaque actif.
    seed : int, optional
        Graine aléatoire (non utilisée en @njit mais conservée pour interface).
    
    Returns
    -------
    np.ndarray
        Chromosome muté (copy, pas in-place par défaut en @njit).
    """
    mutated = chromosome.copy()
    n_assets = len(chromosome)

    if n_assets < 2:
        return mutated
    
    # Appliquer la mutation avec probabilité mutation_rate
    if np.random.rand() < mutation_rate:
        # Sélectionner deux indices distincts
        # Tirage de deux actifs distincts
        idx_i = np.random.randint(0, n_assets)
        idx_j = np.random.randint(0, n_assets)
        
        # Assurer que i et j sont distincts
        while idx_j == idx_i:
            idx_j = np.random.randint(0, n_assets)
        
        # Incrément aléatoire dans [0.002, 0.05]
        delta = 0.002 + np.random.rand() * (0.05 - 0.002)
        
        # Vérifier les contraintes avant d'appliquer
        if mutated[idx_i] + delta <= w_max and mutated[idx_j] - delta >= w_min:
            # Appliquer le transfert de masse
            mutated[idx_i] += delta
            mutated[idx_j] -= delta
    
    return mutated


@njit(cache=True)
def bump_mutation_inplace(chromosome, mutation_rate, w_min, w_max):
    """
    Opérateur Bump Mutation in-place (modifie le chromosome directement).
    
    Parameters
    ----------
    chromosome : np.ndarray
        Chromosome (portefeuille de poids), shape (n_assets,).
    mutation_rate : float
        Probabilité d'appliquer la mutation (entre 0 et 1).
    w_min : float
        Poids minimum autorisé pour chaque actif.
    w_max : float
        Poids maximum autorisé pour chaque actif.
    """
    n_assets = len(chromosome)

    if n_assets < 2:
        return
    
    if np.random.rand() < mutation_rate:
        # Sélectionner deux indices distincts
        idx_i = np.random.randint(0, n_assets)
        idx_j = np.random.randint(0, n_assets)
        
        while idx_j == idx_i:
            idx_j = np.random.randint(0, n_assets)
        
        # Incrément aléatoire dans [0.002, 0.05]
        delta = 0.002 + np.random.rand() * (0.05 - 0.002)
        
        # Vérifier et appliquer les contraintes
        if chromosome[idx_i] + delta <= w_max and chromosome[idx_j] - delta >= w_min:
            chromosome[idx_i] += delta
            chromosome[idx_j] -= delta


@njit(cache=True)
def mutate_population(population, mutation_rate, w_min, w_max):
    """
    Applique la mutation Bump Mutation à une population entière.
    Vectorisée et compilée avec @njit pour éliminer les context switches Python.
    
    Parameters
    ----------
    population : np.ndarray
        Matrice de population (population_size, n_assets).
    mutation_rate : float
        Probabilité de mutation par individu.
    w_min : float
        Poids minimum.
    w_max : float
        Poids maximum.
    
    Returns
    -------
    np.ndarray
        Population mutée (population_size, n_assets).
    """
    population_size = population.shape[0]
    mutated_population = population.copy()
    
    for i in range(population_size):
        bump_mutation_inplace(mutated_population[i, :], mutation_rate, w_min, w_max)
    
    return mutated_population
