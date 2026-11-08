"""
Initialisation du module d'agents LLM pour l'optimisation de portefeuille.
"""

from .macro_agent import (
    analyze_macro_regime,
    MacroAnalysis,
    MarketRegime,
    calculate_macro_metrics,
)

from .sentiment_agent import (
    analyze_asset_sentiment,
    AssetSentiment,
    fetch_historical_news,
)

__all__ = [
    # Macro-économique
    'analyze_macro_regime',
    'MacroAnalysis',
    'MarketRegime',
    'calculate_macro_metrics',
    # Sentiment/Alternative Data
    'analyze_asset_sentiment',
    'AssetSentiment',
    'fetch_historical_news',
]
