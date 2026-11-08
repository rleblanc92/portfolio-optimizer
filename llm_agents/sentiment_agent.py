"""
Agent d'analyse de sentiment pour alternative data.
Ajustement dynamique des rendements espérés via analyse fondamentale.
Cache LLM activé pour éviter appels réseau redondants lors du backtest.
"""

import asyncio
from contextvars import ContextVar
import hashlib
import inspect
import json
import re
import sys
from threading import Lock
from types import ModuleType
from typing import List, Tuple
from weakref import WeakKeyDictionary, ref
from pydantic import BaseModel, Field
import config
from llm_cache import sqlite_persistent_cache
from logging_utils import get_logger

logger = get_logger(__name__)

try:
    from google import genai
    from google.genai import types
    from openai import AsyncOpenAI
except ImportError:
    pass


STOCHASTIC_CONTAINMENT_DIRECTIVE = (
    "Vous agissez en tant qu'évaluateur d'un espace d'état stochastique abstrait. "
    "Il vous est formellement interdit de tenter d'associer ces métriques à un "
    "événement macroéconomique ou financier historique réel. Concentrez-vous "
    "exclusivement sur la géométrie du risque, la convexité des pertes maximales "
    "et la distribution de la volatilité pour en déduire de manière probabiliste "
    "le multiplicateur de liquidité, le régime de marché et l'aversion au risque."
)


class _LoopAwareSemaphore:
    def __init__(self, value):
        self._value = value
        self._semaphores = WeakKeyDictionary()
        self._lock = Lock()
        self._acquired = ContextVar("sentiment_acquired_semaphores", default=())

    async def __aenter__(self):
        loop = asyncio.get_running_loop()
        with self._lock:
            semaphore_ref = self._semaphores.get(loop)
            semaphore = semaphore_ref() if semaphore_ref is not None else None
            if semaphore is None:
                semaphore = asyncio.Semaphore(self._value)
                self._semaphores[loop] = ref(semaphore)
        await semaphore.acquire()
        self._acquired.set(self._acquired.get() + (semaphore,))

    async def __aexit__(self, exc_type, exc_value, traceback):
        acquired = self._acquired.get()
        self._acquired.set(acquired[:-1])
        acquired[-1].release()


class _SentimentAgentModule(ModuleType):
    def __setattr__(self, name, value):
        if name == "_SEMAPHORE" and isinstance(value, asyncio.Semaphore):
            value = _LoopAwareSemaphore(value._value)
        super().__setattr__(name, value)


sys.modules[__name__].__class__ = _SentimentAgentModule
_SEMAPHORE = _LoopAwareSemaphore(3)


class AssetSentiment(BaseModel):
    """Schéma de sortie structuré de l'analyse de sentiment."""
    ticker: str = Field(..., description="Symbole de l'actif")
    sentiment_score: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Score de sentiment ∈ [-1.0, 1.0]. -1.0 = krach imminent, 1.0 = très haussier, 0.0 = neutre"
    )
    justification: str = Field(
        default="",
        description="Explication succincte de la note de sentiment"
    )


def fetch_historical_news(ticker: str, date: str, max_articles: int = 3) -> List[str]:
    """Génère des actualités pseudo-dynamiques et reproductibles par ticker-date."""
    seed_str = f"{ticker}_{date}"
    seed_val = int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % 100

    if seed_val < 25:
        headlines = [
            f"{ticker} faces regulatory scrutiny and declining margins",
            f"Analysts downgrade {ticker} citing weak demand",
            f"{ticker} misses Q3 earnings consensus significantly",
        ]
    elif seed_val > 75:
        headlines = [
            f"{ticker} crushes earnings expectations, raises guidance",
            f"New strategic product launch drives {ticker} shares higher",
            f"Bullish momentum accelerates for {ticker} across all sectors",
        ]
    else:
        headlines = [
            f"{ticker} Q3 earnings meet analyst expectations",
            f"{ticker} maintains market share in competitive sector",
            f"{ticker} CFO discusses strategic initiatives for the upcoming year",
        ]

    return headlines[:max_articles]


_SENTIMENT_CACHE = {}
_SENTIMENT_SYSTEM_PROMPT = STOCHASTIC_CONTAINMENT_DIRECTIVE + """

Tu es un analyste quantitatif spécialisé en Alternative Data.
Tu dois évaluer uniquement le contenu sémantique abstrait fourni pour Actif B.

Règles strictes d'étalonnage du score de sentiment dans [-1.0, 1.0] :
- Résultats supérieurs au consensus, relèvement de guidance, forte accélération haussière : score ∈ [0.5, 1.0]
- Résultats simplement conformes aux attentes ("meet expectations"), maintien de part de marché ("maintains market share") ou discussions stratégiques courantes sans surprise chiffrée : score STRICTEMENT NEUTRE ∈ [-0.1, 0.1] (cible 0.0)
- Pression réglementaire, baisse des marges, dégradation d'analystes ("downgrade") ou résultats inférieurs aux attentes ("misses consensus") : score ∈ [-1.0, -0.5]

Réponds uniquement en JSON structuré valide :
{
  "ticker": "Actif B",
  "sentiment_score": <flottant entre -1.0 et 1.0>,
  "justification": "<explication brève>"
}
"""


def _parse_sentiment_json(raw_text: str, ticker: str) -> AssetSentiment:
    if not raw_text or not raw_text.strip():
        raise ValueError("Réponse LLM sentiment vide.")
    start_idx = raw_text.find("{")
    end_idx = raw_text.rfind("}")
    if start_idx == -1 or end_idx < start_idx:
        raise ValueError("Réponse LLM sentiment sans objet JSON.")
    payload = json.loads(raw_text[start_idx:end_idx + 1])
    if not isinstance(payload, dict):
        raise ValueError("La réponse LLM sentiment doit être un objet JSON.")
    payload["sentiment_score"] = max(
        -1.0, min(1.0, float(payload.get("sentiment_score", 0.0)))
    )
    payload["ticker"] = ticker
    return AssetSentiment.model_validate(payload)


async def analyze_asset_sentiment(ticker: str, date: str, api_key=None,
                                  use_llm: bool = False,
                                  provider: str = 'google') -> AssetSentiment:
    """
    Analyse le sentiment d'un actif à une date donnée.
    
    CACHE LLM: Mémorise résultats pour tuple (ticker, date) identique.
    Bloque requêtes réseau redondantes et évite RateLimitError lors du backtest.
    
    Peut utiliser soit une heuristique déterministe (défaut), soit un LLM.
    
    Parameters
    ----------
    ticker : str
        Symbole boursier.
    date : str
        Date au format 'YYYY-MM-DD'.
    api_key : str, optional
        Clé API pour le fournisseur LLM.
    use_llm : bool, optional
        Si True, utilise un LLM. Si False (défaut), heuristique déterministe.
    
    Returns
    -------
    AssetSentiment
        Objet contenant le score de sentiment et la justification.
    """
    cache_key = (ticker, date, api_key, use_llm, provider)
    if cache_key in _SENTIMENT_CACHE:
        return _SENTIMENT_CACHE[cache_key].model_copy(update={"ticker": ticker})

    # Récupération des titres
    headlines = fetch_historical_news(ticker, date)
    
    if use_llm and api_key:
        result = _analyze_sentiment_with_llm(ticker, headlines, api_key, provider)
        if inspect.isawaitable(result):
            result = await result
    else:
        result = _analyze_sentiment_heuristic(ticker, headlines)

    is_fallback = getattr(result, "_is_fallback", False)
    result = result.model_copy(update={"ticker": ticker})
    if is_fallback:
        object.__setattr__(result, "_is_fallback", True)
    else:
        _SENTIMENT_CACHE[cache_key] = result
    return result


def _clear_sentiment_cache():
    _SENTIMENT_CACHE.clear()


analyze_asset_sentiment.cache_clear = _clear_sentiment_cache


def _analyze_sentiment_heuristic(ticker: str, headlines: List[str]) -> AssetSentiment:
    """
    Analyse heuristique déterministe du sentiment (fallback).
    
    Basée sur mots-clés simples dans les titres.
    """
    text = " ".join(headlines).lower()
    
    # Mots-clés positifs
    positive_keywords = ['beat', 'growth', 'strong', 'raised', 'bullish', 'surge', 'crushes', 'raises', 'higher', 'accelerates']
    # Mots-clés négatifs
    negative_keywords = ['miss', 'decline', 'weakness', 'crash', 'bearish', 'plunge', 'downgrade', 'declining', 'weak', 'misses', 'scrutiny']
    
    positive_count = sum(1 for kw in positive_keywords if kw in text)
    negative_count = sum(1 for kw in negative_keywords if kw in text)
    
    # Calcul du score (-1 à +1)
    if positive_count + negative_count == 0:
        sentiment_score = 0.0
        justification = "Neutral sentiment from news"
    else:
        sentiment_score = (positive_count - negative_count) / (positive_count + negative_count)
        sentiment_score = max(-1.0, min(1.0, sentiment_score))  # Clipping strict
        
        if sentiment_score > 0.3:
            justification = "Positive sentiment detected"
        elif sentiment_score < -0.3:
            justification = "Negative sentiment detected"
        else:
            justification = "Mixed or neutral sentiment"
    
    return AssetSentiment(
        ticker=ticker,
        sentiment_score=sentiment_score,
        justification=justification
    )


async def _analyze_sentiment_with_llm(ticker: str, headlines: List[str], api_key: str,
                                      provider: str = 'google') -> AssetSentiment:
    """
    Analyse de sentiment via LLM (Google Gemini ou OpenAI).
    
    Parameters
    ----------
    ticker : str
        Symbole boursier.
    headlines : list of str
        Titres à analyser.
    api_key : str
        Clé API.
    provider : str
        'google' ou 'openai' (défaut: 'google').
    
    Returns
    -------
    AssetSentiment
        Résultat de l'analyse.
    """
    try:
        if provider.lower() == 'google':
            return await _analyze_sentiment_gemini(ticker, headlines, api_key)
        elif provider.lower() == 'openai':
            return await _analyze_sentiment_openai(ticker, headlines, api_key)
        else:
            raise ValueError(f"Fournisseur LLM inconnu: {provider}")
    except Exception as e:
        # Fallback sur heuristique en cas d'erreur
        logger.warning("Échec LLM sentiment, utilisation de l'heuristique : %s", e)
        fallback_result = _analyze_sentiment_heuristic(ticker, headlines)
        object.__setattr__(fallback_result, "_is_fallback", True)
        return fallback_result


async def _analyze_sentiment_gemini(ticker: str, headlines: List[str], api_key: str) -> AssetSentiment:
    result = await _analyze_sentiment_gemini_cached(ticker, headlines, api_key)
    return result.model_copy(update={"ticker": ticker})


@sqlite_persistent_cache
async def _analyze_sentiment_gemini_cached(ticker: str, headlines: List[str], api_key: str) -> AssetSentiment:
    """
    Analyse via Google Gemini API.
    
    Nécessite: pip install google-genai
    """
    client = genai.Client(api_key=api_key)
    
    anonymous_ticker = "Actif B"
    anonymous_headlines = [
        re.sub(re.escape(ticker), anonymous_ticker, headline, flags=re.IGNORECASE)
        for headline in headlines
    ]
    headlines_text = "\n".join([f"- {h}" for h in anonymous_headlines])
    
    system_prompt = _SENTIMENT_SYSTEM_PROMPT
    
    user_prompt = f"""Analyse ces titres pour {anonymous_ticker}:

{headlines_text}

Donne un score de sentiment entre -1.0 et 1.0."""
    
    async with _SEMAPHORE:
        response = await client.aio.models.generate_content(
            model=getattr(config, "GEMINI_MODEL", "gemini-1.5-flash"),
            contents=f"{system_prompt}\n\n{user_prompt}",
            config=types.GenerateContentConfig(
                temperature=0.3,
                top_p=0.8,
                max_output_tokens=300,
                response_mime_type="application/json",
                response_schema=AssetSentiment,
            )
        )

    return _parse_sentiment_json(response.text, ticker)


async def _analyze_sentiment_openai(ticker: str, headlines: List[str], api_key: str) -> AssetSentiment:
    result = await _analyze_sentiment_openai_cached(ticker, headlines, api_key)
    return result.model_copy(update={"ticker": ticker})


@sqlite_persistent_cache
async def _analyze_sentiment_openai_cached(ticker: str, headlines: List[str], api_key: str) -> AssetSentiment:
    """
    Analyse via OpenAI GPT API.
    
    Nécessite: pip install openai
    """
    anonymous_ticker = "Actif B"
    anonymous_headlines = [
        re.sub(re.escape(ticker), anonymous_ticker, headline, flags=re.IGNORECASE)
        for headline in headlines
    ]
    headlines_text = "\n".join([f"- {h}" for h in anonymous_headlines])
    
    system_prompt = _SENTIMENT_SYSTEM_PROMPT
    
    user_prompt = f"""Analyse ces titres pour {anonymous_ticker}:

{headlines_text}

Donne un score de sentiment entre -1.0 et 1.0."""
    
    async with _SEMAPHORE:
        async with AsyncOpenAI(api_key=api_key) as client:
            response = await client.beta.chat.completions.parse(
                model=config.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                response_format=AssetSentiment
            )

    message = response.choices[0].message
    parsed = message.parsed
    if parsed is None:
        return _parse_sentiment_json(message.content, ticker)
    sentiment_score = max(-1.0, min(1.0, float(parsed.sentiment_score)))
    return parsed.model_copy(update={"ticker": ticker, "sentiment_score": sentiment_score})
