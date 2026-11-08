"""
Configuration centralisée pour l'algorithme génétique d'optimisation de portefeuille.
"""

import os

from dotenv import load_dotenv


load_dotenv(override=True)

LLM_API_KEY = os.getenv("LLM_API_KEY")
USE_LLM = os.getenv("USE_LLM", "False").lower() in ("true", "1", "yes")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "google")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
GENERATE_REPORTS = os.getenv("GENERATE_REPORTS", "False").lower() in ("true", "1", "yes")

# Paramètres de l'algorithme génétique
POPULATION_SIZE = 800
GENERATIONS = 1000
MUTATION_RATE = 0.10

# Paramètres de préférence de risque et rendement
RISK_AVERSION = 0.5
RISK_FREE_RATE = 0.015
MARGIN_RATE = RISK_FREE_RATE + 0.02
# Five assets at MAX_WEIGHT=0.25 cap feasible gross exposure at 1.25.
GROSS_EXPOSURE = 1.25
ALLOW_SHORT = True
MIN_WEIGHT = -0.25

# Paramètres pour les moments d'ordre supérieur
SKEWNESS_PREFERENCE = 0.005
KURTOSIS_AVERSION = 0.001

# Contraintes de poids du portefeuille
MAX_WEIGHT = 0.25

# Paramètres de coûts de transaction
TRANSACTION_FEE = 0.002
TURNOVER_AVERSION = TRANSACTION_FEE * (252 / 21)

# Paramètres temporels
LOOKBACK_WINDOW = 252
REBALANCE_FREQ = 21
TIME_WEIGHT_HALFLIFE = float(os.getenv("TIME_WEIGHT_HALFLIFE", "63.0"))
