"""
Moteur d'évolution génétique vectorisé avec compilation Numba.

Module principal exposant les opérateurs et évaluations du système
d'optimisation de portefeuille multi-objectifs.
"""

# Opérateurs d'évolution
from .crossover import blx_alpha_crossover, crossover_batch, ALPHA_BLX
from .mutation import bump_mutation, bump_mutation_inplace, mutate_population
from .repair import repair_chromosome, repair_population

# Évaluation multi-objectifs
from .fitness import (
    evaluate_fitness,
    evaluate_population,
    calculate_portfolio_returns,
    calculate_skewness,
    calculate_kurtosis_excess,
    calculate_turnover,
)

# Noyau d'évolution (orchestration complète)
from .evolution import run_evolution

__version__ = '1.0.0'
__author__ = 'Robin Leblanc'

__all__ = [
    # Métadonnées
    '__version__',
    '__author__',
    
    # Croisement BLX-α
    'blx_alpha_crossover',
    'crossover_batch',
    'ALPHA_BLX',
    
    # Mutation Bump
    'bump_mutation',
    'bump_mutation_inplace',
    'mutate_population',
    
    # Réparation des contraintes
    'repair_chromosome',
    'repair_population',
    
    # Évaluation du fitness
    'evaluate_fitness',
    'evaluate_population',
    'calculate_portfolio_returns',
    'calculate_skewness',
    'calculate_kurtosis_excess',
    'calculate_turnover',
    
    # Noyau d'évolution
    'run_evolution',
]
