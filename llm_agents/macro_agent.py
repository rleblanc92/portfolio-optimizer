"""
Agent macro-économique pour ajustement dynamique du risque.
Analyse comportement de marché et recommande un niveau d'aversion au risque.
"""

from enum import Enum
from typing import Optional
import asyncio
from contextvars import ContextVar
import json
import re
import sys
from threading import Lock
from types import ModuleType
from weakref import WeakKeyDictionary, ref
import numpy as np
from pydantic import BaseModel, Field
import config
from llm_cache import sqlite_persistent_cache
from logging_utils import get_logger

logger = get_logger(__name__)

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None


class _LoopAwareSemaphore:
    def __init__(self, value):
        self._value = value
        self._semaphores = WeakKeyDictionary()
        self._lock = Lock()
        self._acquired = ContextVar("macro_acquired_semaphores", default=())

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


class _MacroAgentModule(ModuleType):
    def __setattr__(self, name, value):
        if name == "_SEMAPHORE" and isinstance(value, asyncio.Semaphore):
            value = _LoopAwareSemaphore(value._value)
        super().__setattr__(name, value)


_MACRO_CACHE = {}
_SEMAPHORE = _LoopAwareSemaphore(3)
sys.modules[__name__].__class__ = _MacroAgentModule
STOCHASTIC_CONTAINMENT_DIRECTIVE = (
    "Vous agissez en tant qu'évaluateur d'un espace d'état stochastique abstrait. "
    "Il vous est formellement interdit de tenter d'associer ces métriques à un "
    "événement macroéconomique ou financier historique réel. Concentrez-vous "
    "exclusivement sur la géométrie du risque, la convexité des pertes maximales "
    "et la distribution de la volatilité pour en déduire de manière probabiliste "
    "le multiplicateur de liquidité, le régime de marché et l'aversion au risque."
)


class MarketRegime(str, Enum):
    """Régimes de marché détectés par l'agent macro."""
    BULL = "Bull"
    BEAR = "Bear"
    STAGFLATION = "Stagflation"
    HIGH_VOL_RECOVERY = "High_Vol_Recovery"


class MacroAnalysis(BaseModel):
    """Schéma de sortie structuré de l'analyse macro."""
    market_regime: MarketRegime = Field(
        ...,
        description="Régime de marché détecté (Bull, Bear, Stagflation, High_Vol_Recovery)"
    )
    risk_aversion: float = Field(
        ...,
        ge=0.5,
        le=2.5,
        description="Coefficient d'aversion au risque λ_risk ∈ [0.5, 2.5]"
    )
    liquidity_multiplier: float = Field(
        1.0,
        ge=1.0,
        le=3.0,
        description=(
            "Multiplicateur d'écartement des spreads bid-ask. "
            "1.0 = liquidité normale, 3.0 = liquidité asséchée (krach)."
        )
    )
    enable_shorting: bool = Field(
        False,
        description=(
            "True si le régime justifie la vente à découvert "
            "(Bear/Stagflation), False pour rester Long-Only."
        )
    )
    reasoning: str = Field(
        default="",
        description="Explication de la décision prise par le CIO"
    )


def _parse_and_clamp_macro_json(raw_text: str) -> MacroAnalysis:
    if not raw_text or not raw_text.strip():
        raise ValueError("Réponse LLM macro vide.")
    cleaned = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        raw_text.strip(),
        flags=re.MULTILINE,
    ).strip()
    start_idx = cleaned.find("{")
    end_idx = cleaned.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx >= start_idx:
        cleaned = cleaned[start_idx:end_idx + 1]
    data = json.loads(cleaned)
    if not isinstance(data, dict):
        raise ValueError("La réponse LLM macro doit être un objet JSON.")
    if data.get("risk_aversion") is not None:
        data["risk_aversion"] = max(0.5, min(2.5, float(data["risk_aversion"])))
    if data.get("liquidity_multiplier") is not None:
        data["liquidity_multiplier"] = max(
            1.0, min(3.0, float(data["liquidity_multiplier"]))
        )
    return MacroAnalysis.model_validate(data)


def calculate_macro_metrics(sp500_returns):
    """
    Calcule les métriques macro-économiques sur une fenêtre d'historique.
    
    Parameters
    ----------
    sp500_returns : np.ndarray
        Série des rendements journaliers de l'Indice Actions Global de Référence.
    
    Returns
    -------
    dict
        Dictionnaire contenant:
        - 'annual_volatility': Volatilité annualisée
        - 'cumulative_return': Rendement cumulé
        - 'max_drawdown': Perte maximale
        - 'sharpe_ratio': Ratio de Sharpe (Rf=1.5%)
        - 'n_days': Nombre de jours
    """
    n_days = len(sp500_returns)
    
    # Volatilité annualisée
    annual_vol = np.std(sp500_returns) * np.sqrt(252)
    
    # Rendement cumulé
    cumulative_return = np.prod(1 + sp500_returns) - 1
    
    # Max Drawdown
    cumsum = np.cumprod(1 + sp500_returns)
    cummax = np.maximum.accumulate(cumsum)
    drawdown = (cumsum - cummax) / cummax
    max_drawdown = np.min(drawdown)
    
    # Sharpe Ratio
    annual_return = np.mean(sp500_returns) * 252.0
    rf = 0.015
    sharpe = (annual_return - rf) / annual_vol if annual_vol > 1e-10 else 0.0
    
    return {
        'annual_volatility': annual_vol,
        'cumulative_return': cumulative_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe,
        'n_days': n_days
    }


@sqlite_persistent_cache
async def analyze_macro_regime(sp500_returns, t_bond_yield=None, api_key=None,
                               use_llm=False, provider='google') -> MacroAnalysis:
    """
    Analyse le régime macro-économique et recommande un ajustement du risque.
    
    Utilise soit une heuristique déterministe (défaut), soit un LLM si use_llm=True.
    
    Parameters
    ----------
    sp500_returns : np.ndarray
        Série des rendements journaliers de l'Indice Actions Global de Référence.
    t_bond_yield : float, optional
        Taux de l'Actif Sans Risque (non utilisé en version déterministe).
    api_key : str, optional
        Clé API pour le fournisseur LLM.
    use_llm : bool, optional
        Si True, utilise un LLM pour l'analyse. Si False (défaut), heuristique.
    provider : str, optional
        Fournisseur LLM: 'google' (Gemini) ou 'openai' (défaut: 'google').
    
    Returns
    -------
    MacroAnalysis
        Objet contenant régime détecté et coefficient d'aversion au risque.
    """
    # Calcul des métriques
    metrics = calculate_macro_metrics(sp500_returns)
    metrics_key = tuple(
        (key, round(float(value), 8))
        for key, value in sorted(metrics.items())
    )
    cache_key = (
        metrics_key,
        None if t_bond_yield is None else round(float(t_bond_yield), 8),
        bool(use_llm),
        provider.lower(),
        api_key or '',
    )

    if cache_key in _MACRO_CACHE:
        return _MACRO_CACHE[cache_key]
    
    if use_llm and api_key:
        analysis = await _analyze_with_llm(
            metrics, t_bond_yield, api_key, provider
        )
    else:
        analysis = _analyze_heuristic(metrics)

    if not getattr(analysis, "_is_fallback", False):
        _MACRO_CACHE[cache_key] = analysis
    return analysis


def _analyze_heuristic(metrics):
    """
    Analyse heuristique déterministe (fallback).
    
    Règles :
    - Marché haussier calme : volatilité < 15%, drawdown > -10% → λ ∈ [0.5, 1.0]
    - Incertitude : volatilité 15-25%, Sharpe négatif → λ ∈ [1.0, 1.5]
    - Krach : drawdown < -15%, volatilité > 25% → λ ∈ [1.5, 2.5]
    - Recovery : volatilité élevée mais rendement positif → λ ∈ [1.2, 1.8]
    """
    annual_vol = metrics['annual_volatility']
    cumulative_ret = metrics['cumulative_return']
    max_dd = metrics['max_drawdown']
    sharpe = metrics['sharpe_ratio']
    
    # Logique de classification
    if max_dd > -0.10 and annual_vol < 0.15 and cumulative_ret > 0.05:
        # Marché haussier calme
        regime = MarketRegime.BULL
        risk_aversion = 0.75
        reasoning = "Bull market: low volatility, positive returns, minimal drawdown"
    
    elif max_dd < -0.15 and annual_vol > 0.25:
        # Krach/marché baissier
        regime = MarketRegime.BEAR
        risk_aversion = 2.0
        reasoning = "Bear market: high volatility, significant drawdown (>15%)"
    
    elif annual_vol > 0.20 and sharpe < 0.0:
        # Stagnation/faible rendement avec volatilité
        regime = MarketRegime.STAGFLATION
        risk_aversion = 1.8
        reasoning = "Stagflation regime: high vol but poor risk-adjusted returns"
    
    elif annual_vol > 0.20 and cumulative_ret > 0.10:
        # Volatilité élevée mais recovery
        regime = MarketRegime.HIGH_VOL_RECOVERY
        risk_aversion = 1.3
        reasoning = "High volatility recovery: elevated risk but positive momentum"
    
    elif 0.15 <= annual_vol < 0.25:
        # Incertitude modérée
        regime = MarketRegime.BULL
        risk_aversion = 1.2
        reasoning = "Moderate uncertainty: balanced risk approach"
    
    else:
        # Défaut : neutralité
        regime = MarketRegime.BULL
        risk_aversion = 1.0
        reasoning = "Default regime: neutral risk posture"

    enable_shorting = regime in (MarketRegime.BEAR, MarketRegime.STAGFLATION)

    if regime == MarketRegime.BULL:
        liquidity_multiplier = 1.0
    elif regime in (MarketRegime.STAGFLATION, MarketRegime.HIGH_VOL_RECOVERY):
        liquidity_multiplier = 1.5
    else:
        liquidity_multiplier = 2.5
    
    return MacroAnalysis(
        market_regime=regime,
        risk_aversion=risk_aversion,
        liquidity_multiplier=liquidity_multiplier,
        enable_shorting=enable_shorting,
        reasoning=reasoning
    )


async def _analyze_with_llm(metrics, t_bond_yield, api_key, provider='google'):
    """
    Analyse via LLM (Google Gemini ou OpenAI).
    
    Parameters
    ----------
    metrics : dict
        Métriques calculées.
    t_bond_yield : float
        Rendement des obligations.
    api_key : str
        Clé API.
    provider : str
        'google' ou 'openai'.
    
    Returns
    -------
    MacroAnalysis
        Résultat de l'analyse LLM.
    """
    try:
        if provider.lower() == 'google':
            return await _analyze_gemini(metrics, t_bond_yield, api_key)
        elif provider.lower() == 'openai':
            return await _analyze_openai(metrics, t_bond_yield, api_key)
        else:
            raise ValueError(f"Fournisseur LLM inconnu: {provider}")
    except Exception as e:
        # Fallback sur heuristique en cas d'erreur
        logger.warning("Échec LLM macro, utilisation de l'heuristique : %s", e)
        result = _analyze_heuristic(metrics)
        object.__setattr__(result, "_is_fallback", True)
        return result


async def _analyze_gemini(metrics, t_bond_yield, api_key) -> MacroAnalysis:
    """
    Analyse via Google Gemini API.
    
    Nécessite: pip install google-genai
    """
    client = genai.Client(api_key=api_key)
    
    # Formatage des métriques
    metrics_text = f"""
Période d'observation T (252 observations):
- Volatilité annualisée: {metrics['annual_volatility']*100:.2f}%
- Rendement cumulé: {metrics['cumulative_return']*100:.2f}%
- Max Drawdown: {metrics['max_drawdown']*100:.2f}%
- Ratio de Sharpe: {metrics['sharpe_ratio']:.3f}
- Taux de l'Actif Sans Risque: {(t_bond_yield or 2.5):.2f}%
"""
    
    prompt_system = STOCHASTIC_CONTAINMENT_DIRECTIVE + """

Tu es le Chief Investment Officer quantitatif d'un hedge fund multi-stratégies.
Tu analyses les métriques de risque de la Période d'observation T.

Règles strictes d'ajustement de λ_risk:
1. Marché haussier calme (vol<15%, DD>-10%): λ_risk ∈ [0.5, 1.0]
2. Incertitude modérée (vol 15-25%, Sharpe faible): λ_risk ∈ [1.0, 1.5]
3. Krach/Drawdown > 15% (vol>25%): λ_risk ∈ [1.5, 2.5]
4. Recovery (vol élevée, rendement positif): λ_risk ∈ [1.2, 1.8]

Évalue l'état de la liquidité du marché. Si la volatilité dépasse 20%, les spreads bid-ask s'écartent. Retourne un `liquidity_multiplier` proportionnel à cette friction (entre 1.0 et 3.0).

Réponds en JSON structuré:
{
  "market_regime": "Bull|Bear|Stagflation|High_Vol_Recovery",
  "risk_aversion": <flottant entre 0.5 et 2.5>,
    "liquidity_multiplier": <flottant entre 1.0 et 3.0>,
    "enable_shorting": <booléen: true uniquement en krach ou forte correction baissière>,
  "reasoning": "<explication brève>"
}

Détermine `enable_shorting` (booléen). Retourne `true` uniquement si le marché est en krach ou en forte correction baissière pour autoriser l'algorithme à shorter les actifs, sinon `false`.
"""
    
    prompt_user = f"Analyse ces métriques de marché et recommande λ_risk:\n{metrics_text}"
    
    async with _SEMAPHORE:
        response = await client.aio.models.generate_content(
            model=getattr(config, "GEMINI_MODEL", "gemini-1.5-flash"),
            contents=f"{prompt_system}\n\n{prompt_user}",
            config=types.GenerateContentConfig(
                temperature=0.3,
                top_p=0.8,
                max_output_tokens=500,
                response_mime_type="application/json",
                response_schema=MacroAnalysis,
            )
        )

    return _parse_and_clamp_macro_json(response.text)


async def _analyze_openai(metrics, t_bond_yield, api_key) -> MacroAnalysis:
    """
    Analyse via OpenAI GPT API.
    
    Nécessite: pip install openai
    """
    # Formatage des métriques
    metrics_text = f"""
Période d'observation T (252 observations):
- Volatilité annualisée: {metrics['annual_volatility']*100:.2f}%
- Rendement cumulé: {metrics['cumulative_return']*100:.2f}%
- Max Drawdown: {metrics['max_drawdown']*100:.2f}%
- Ratio de Sharpe: {metrics['sharpe_ratio']:.3f}
- Taux de l'Actif Sans Risque: {(t_bond_yield or 2.5):.2f}%
"""
    
    system_prompt = STOCHASTIC_CONTAINMENT_DIRECTIVE + """

Tu es le Chief Investment Officer quantitatif d'un hedge fund multi-stratégies.
Tu analyses les métriques de risque de la Période d'observation T.

Règles strictes d'ajustement de λ_risk:
1. Marché haussier calme (vol<15%, DD>-10%): λ_risk ∈ [0.5, 1.0]
2. Incertitude modérée (vol 15-25%, Sharpe faible): λ_risk ∈ [1.0, 1.5]
3. Krach/Drawdown > 15% (vol>25%): λ_risk ∈ [1.5, 2.5]
4. Recovery (vol élevée, rendement positif): λ_risk ∈ [1.2, 1.8]

Évalue l'état de la liquidité du marché. Si la volatilité dépasse 20%, les spreads bid-ask s'écartent. Retourne un `liquidity_multiplier` proportionnel à cette friction (entre 1.0 et 3.0).

Réponds en JSON structuré:
{
  "market_regime": "Bull|Bear|Stagflation|High_Vol_Recovery",
  "risk_aversion": <flottant entre 0.5 et 2.5>,
    "liquidity_multiplier": <flottant entre 1.0 et 3.0>,
    "enable_shorting": <booléen: true uniquement en krach ou forte correction baissière>,
  "reasoning": "<explication brève>"
}

Détermine `enable_shorting` (booléen). Retourne `true` uniquement si le marché est en krach ou en forte correction baissière pour autoriser l'algorithme à shorter les actifs, sinon `false`.
"""
    
    user_prompt = f"Analyse ces métriques de marché et recommande λ_risk:\n{metrics_text}"
    
    async with _SEMAPHORE:
        async with AsyncOpenAI(api_key=api_key) as client:
            response = await client.beta.chat.completions.parse(
                model=config.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                response_format=MacroAnalysis
            )

    message = response.choices[0].message
    if message.parsed is not None:
        return message.parsed
    if message.content:
        return _parse_and_clamp_macro_json(message.content)
    raise ValueError("Réponse LLM macro OpenAI vide.")
