> **Langue / Language :** **Français** | [Switch to English version (README.md)](README.md)

---

# Moteur d'Optimisation de Portefeuille Multi-Objectifs & Backtesting Systématique

**Auteur :** Robin Leblanc

**Stack :** Python 3.8+ | NumPy | Numba JIT | SciPy | Scikit-Learn | Pandas | QuantStats | Pytest | Pydantic

---

## 1. Vue d'Ensemble & Architecture

Ce projet implémente un moteur quantitatif d'allocation d'actifs et de backtesting *Out-of-Sample* combinant une métaheuristique génétique multi-objectifs compilée en code machine via **Numba LLVM (`@njit`)**, des méthodes d'optimisation convexe classiques (Newton-Raphson, SLSQP) et une couche d'agents LLM asynchrones pour la détection de régimes macroéconomiques et l'analyse de sentiment.

## 📊 Rapports de Backtest (Tearsheets QuantStats)

Accès direct aux rapports interactifs complets hébergés sur GitHub Pages :

| Scénario / Univers | Horizon | Objectif d'analyse | Rapport interactif |
| :--- | :--- | :--- | :--- |
| **US MegaTech** | 2015 – 2023 | Régime haussier et concentration technologique | [Ouvrir le Tearsheet ↗](https://<username>.github.io/<repo>/tearsheet_US_MegaTech_2015_2023.html) |
| **Crash Covid & Rebond** | 2020 – 2021 | Choc de volatilité et récupération rapide | [Ouvrir le Tearsheet ↗](https://<username>.github.io/<repo>/tearsheet_Covid_2020_2021.html) |
| **Crise Financière (GFC)** | 2007 – 2009 | Résistance en marché baissier sévère et risque de queue | [Ouvrir le Tearsheet ↗](https://<username>.github.io/<repo>/tearsheet_GFC_2007_2009.html) |
| **Multi-Asset Global** | 2024 | Diversification multi-classes d'actifs et régimes macro | [Ouvrir le Tearsheet ↗](https://<username>.github.io/<repo>/tearsheet_MultiAsset_2024.html) |


Conçu pour répliquer les contraintes d'exécution réelles en salle des marchés, le moteur modélise explicitement :

* La **dérive quotidienne des poids** (*weight drift*) entre deux dates de rééquilibrage.
* Les **coûts de transaction linéaires (L1) et l'impact de marché quadratique permanent (L2)**.


* La **rémunération du cash libre** au taux sans risque évolutif et le **coût de financement sur marge** (*margin financing drag*) pour les positions courtes et l'effet de levier.


* Le contrôle des **moments d'ordre supérieur** (*skewness* et *excess kurtosis*) et la régularisation de la matrice de covariance par rétrécissement de **Ledoit-Wolf**.



```text
Optimisation de portefeuille AG/
├── config.py                        # Hyperparamètres centralisés et variables d'environnement
├── market_data.py                   # Pipeline de données, rendements arithmétiques, EMA 40/200 & Ledoit-Wolf
├── classic_optimization.py          # Benchmarks convexes (Minimum Variance Newton-Raphson & Max Sharpe SLSQP)
├── main.py                          # Orchestrateur de backtest unitaire (Rolling Window IS/OOS)
├── general_tests.py                 # Orchestrateur multi-scénarios parallélisé (spawn) & reporting QuantStats
├── genetic_algorithm/               # Noyau d'évolution vectorisé et compilé JIT (Numba @njit)
│   ├── __init__.py
│   ├── evolution.py                 # Boucle évolutionnaire, tournoi Fisher-Yates in-place & hyper-mutation
│   ├── crossover.py                 # Croisement continu BLX-α vectorisé
│   ├── mutation.py                  # Bump Mutation par transfert stochastique de masse en place
│   ├── repair.py                    # Projection sur bornes et contrôle d'exposition brute (Gross Exposure)
│   └── fitness.py                   # Fonction objectif d'ordre 4 (Variance, Skewness, Kurtosis, Coûts L1/L2)
├── llm_agents/                      # Couche sémantique et macroéconomique avec validation Pydantic
│   ├── __init__.py
│   ├── macro_agent.py               # Classification de régime de marché et pilotage dynamique de lambda_risk
│   └── sentiment_agent.py           # Scoring sémantique d'actualités et modulation bayésienne des rendements
├── tests/
│   └── test_backtest_integration.py # Suite d'audit mathématique et d'intégration système (Pytest)
├── outputs/
│   ├── tearsheets/                  # Rapports HTML QuantStats Out-of-Sample par scénario
│   ├── logs/                        # Journaux d'exécution dédupliqués (execution_debug.log)
│   └── rebalance_history.parquet    # Historique détaillé des poids, régimes, frais et turnovers
├── requirements.txt
├── setup.sh / setup.bat
└── README.md

```

---

## 2. Modélisation Mathématique & Ingénierie Quantitative

### 2.1 Estimateurs de Rendement et de Covariance (`market_data.py`)

* **Additivité transversale :** Utilisation exclusive des rendements arithmétiques simples $R_{i,t} = \frac{P_{i,t} - P_{i,t-1}}{P_{i,t-1}}$ pour garantir l'exactitude linéaire du rendement du portefeuille $R_{p,t} = \sum_{i=1}^N w_i R_{i,t}$.


* **Espérance de rendement ($\hat{\mu}$) par filtrage tendanciel EMA 40/200 :**
* Facteur de lissage : $\alpha = \frac{2}{N+1}$ avec une fenêtre rapide de $40$ jours et une fenêtre lente de $200$ jours (retard de phase effectif : $\frac{200 - 40}{2} = 80$ jours).


* Estimateur journalier et annualisé :



$$\hat{\mu}_{i,t}^{\text{daily}} = \frac{1}{80}\left(\frac{\text{EMA}_{40}(P_{i,t})}{\text{EMA}_{200}(P_{i,t})} - 1\right), \quad \hat{\mu}_{i,t}^{\text{ann}} = 252 \cdot \hat{\mu}_{i,t}^{\text{daily}}$$




* **Matrice de Covariance régularisée ($\Sigma_{\text{LW}}$) :**
* Estimation par rétrécissement analytique de **Ledoit-Wolf** (`scikit-learn`) pour éliminer le mauvais conditionnement spectral en échantillon fini ($T = 252$), annualisée par $\Sigma_{\text{ann}} = 252 \cdot \Sigma_{\text{daily}}$.


* Pondération temporelle exponentielle partagée avec les moments de la fonction de fitness via une demi-vie configurable `TIME_WEIGHT_HALFLIFE` (défaut : $63$ jours de bourse, soit un trimestre).





### 2.2 Fonction Objectif Multi-Moments & Frictions de Marché (`fitness.py`)

Pour un vecteur de poids $w \in \mathbb{R}^N$, soient :

* $L = \sum_{i=1}^N \max(w_i, 0)$ l'exposition longue et $S = \sum_{i=1}^N \max(-w_i, 0)$ l'exposition courte.


* $c = \max(0, 1 - L)$ la fraction de cash libre rémunérée au taux sans risque annualisé $R_f$.


* $F = \max(0, L - 1) + S$ l'exposition financée soumise au taux de marge $r_{\text{margin}}$.


* $E = \min(1, L + S)$ le facteur d'échelle d'exposition risquée.


* $T = \sum_{i=1}^N \vert{}w_i^{(t)} - w_i^{(t-1)}\vert{}$ le *turnover* bilatéral par rapport au portefeuille déflaté de la période précédente.



L'utilité exacte évaluée pour chaque individu de la population s'écrit :

$$\text{Fitness}(w) = w^\top \hat{\mu} + c R_f - F \, r_{\text{margin}} - \lambda_{\text{risk}} \, w^\top \Sigma_{\text{LW}} w + \lambda_{\text{skew}} E \, S_3(w) - \lambda_{\text{kurt}} E \, K_4(w) - \lambda_{\text{TO}} \left(T + 0.1 T^2\right)$$

où :

* **Variance annualisée :** $w^\top \Sigma_{\text{LW}} w$ (ou variance empirique des rendements journaliers du portefeuille multipliée par $252$ si aucune matrice externe n'est passée).


* **Asymétrie standardisée (*Skewness*) :** $S_3(w) = \frac{M_3(r_p)}{\sigma_p^3}$ pondérée par $\lambda_{\text{skew}} = 0.05$ pour favoriser la convexité haussière.


* **Kurtosis en excès (*Excess Kurtosis*) :** $K_4(w) = \frac{M_4(r_p)}{\sigma_p^4} - 3$ pondérée par $\lambda_{\text{kurt}} = 0.01$ pour pénaliser le risque de queue (*tail risk*).


* **Pénalité de Turnover L1 + Impact Quadratique L2 :** $\lambda_{\text{TO}} = \text{TRANSACTION\_FEE} \times \frac{252}{\text{REBALANCE\_FREQ}}$ modélise à la fois les frais proportionnels de courtage/spread ($T$) et l'impact de marché non-linéaire ($0.1 T^2$).



---

## 3. Ingénierie Système Bas Niveau & Noyau Numba (`genetic_algorithm/`)

L'intégralité de la boucle évolutionnaire est compilée en code natif via `@njit(cache=True)` en éliminant les allocations dynamiques dans les boucles critiques :

1. **Initialisation Hybride & Masque de Parcimonie (`evolution.py`) :**
* Génération uniforme sur $[w_{\min}, w_{\max}]$ (`POPULATION_SIZE = 800`).


* Si $N > 4$, application d'un masque de parcimonie (*sparsity*) mettant à zéro $60\,\%$ des gènes sur la seconde moitié de la population initiale afin d'accélérer la découverte de portefeuilles concentrés.


2. **Sélection par Tournoi Sans Allocation Mémoire (`_tournament_selection`) :**
* Implémentation d'un mélange de **Fisher-Yates partiel en place** ($k = 3$) sur un tableau d'indices pré-alloué, suivie de l'annulation symétrique des permutations (*swap reversal*) à la fin de chaque tirage. Complexité $\mathcal{O}(k)$ en temps et $\mathcal{O}(1)$ en mémoire additionnelle.


3. **Croisement Continu BLX-$\alpha$ (`crossover.py`) :**
* Appariement aléatoire via deux permutations indépendantes des parents. Pour chaque actif, tirage uniforme dans $[C_{\min} - \alpha \Delta, \, C_{\max} + \alpha \Delta]$ avec $\alpha = 0.5$ et $\Delta = \vert{}w_i^{(1)} - w_i^{(2)}\vert{}$.




4. **Mutation par Transfert de Masse en Place (`mutation.py`) :**
* Opérateur `bump_mutation_inplace` transférant une masse $\delta \sim \mathcal{U}(0.002, 0.05)$ d'un actif $j$ vers un actif $i$ ($w_i \leftarrow w_i + \delta$, $w_j \leftarrow w_j - \delta$) sous réserve du strict respect des bornes $[w_{\min}, w_{\max}]$, préservant ainsi la somme des poids sans renormalisation destructive.




5. **Opérateur de Réparation & Contrainte d'Exposition Brute (`repair.py`) :**
* Écrêtage strict (*clipping*) sur $[w_{\min}, w_{\max}]$.


* Contrôle d'exposition brute : si $\sum_i \vert{}w_i\vert{} > \text{Gross Exposure Target}$ ($1.0$ par défaut), réduction homothétique $w_i \leftarrow w_i \cdot \frac{\text{Target}}{\sum_j \vert{}w_j\vert{}}$.


* En configuration Long-Only, si $\sum_i w_i < 1.0$, la poche résiduelle $1 - \sum_i w_i$ est automatiquement affectée au taux sans risque $R_f$ (convergence garantie en moins de $5$ itérations).




6. **Convergence, Élitisme & Hyper-Mutation Adaptative (`evolution.py`) :**
* Conservation directe des $5\,\%$ meilleurs individus (`elite_ratio = 0.05`).
* *Early stopping* si le gain de fitness reste inférieur à $\Delta_{\min} = 10^{-7}$ sur `patience = 200` générations.
* Déclenchement d'un régime d'**hyper-mutation** ($\text{taux} \times 2.5$, plafonné à $0.50$) dès $50$ générations de stagnation pour extraire la population des minima locaux non-convexes.



---

## 4. Couche Multi-Agents LLM & Validation de Schéma (`llm_agents/`)

En amont de l'optimisation numérique, deux agents interrogent des modèles de langage (Gemini 1.5 Flash / GPT-4o-mini) avec validation stricte des sorties via **Pydantic** et repli déterministe automatique en cas d'erreur réseau ou de *rate limit* :

* **Agent Macro-Économique (`macro_agent.py`) :** Analyse les métriques *In-Sample* (volatilité réalisée, drawdown, rendement cumulé) pour classifier l'état du marché parmi 4 régimes (`Bull`, `Bear`, `Stagflation`, `High Vol Recovery`) et ajuster dynamiquement le coefficient d'aversion au risque $\lambda_{\text{risk}} \in [0.5, 2.5]$.


* **Agent de Sentiment (`sentiment_agent.py`) :** Évalue le flux de nouvelles financières par actif pour extraire un score sémantique $s_i \in [-1.0, 1.0]$ modulant le vecteur des rendements espérés :



$$\hat{\mu}_{i,\text{adjusted}} = \hat{\mu}_i + \left(\beta_{\text{sentiment}} \cdot s_i \cdot \vert{}\hat{\mu}_i\vert{}\right)$$



Un cache persistant SQLite doublé d'un cache mémoire `_SENTIMENT_CACHE` indexé par `(ticker, date, api_key, use_llm, provider)` garantit l'idempotence des backtests et élimine les appels API redondants.



---

## 5. Protocole de Backtesting, Orchestration Parallèle & Suite d'Audit

### 5.1 Fenêtre Glissante (*Rolling Window*) & Réalisme Comptable

* **In-Sample (`LOOKBACK_WINDOW = 252` jours) :** Estimation de $\hat{\mu}$, $\Sigma_{\text{LW}}$, détection du régime macro et optimisation des portefeuilles (GA Multi-Objectifs, Minimum Variance par Newton-Raphson, Maximum Sharpe par SLSQP, et benchmark équipondéré $1/N$).


* **Out-of-Sample (`REBALANCE_FREQ = 21` jours) :** Application des poids optimisés sur la fenêtre de test suivante. Entre deux dates de rééquilibrage, les poids dérivent quotidiennement selon les rendements réalisés des sous-jacents (*weight drift*), avec imputation quotidienne du taux sans risque et des coûts de financement sur marge. À chaque rééquilibrage, les frais de transaction impactent directement la NAV.


* **Traçabilité & Reporting (`general_tests.py`) :** Exécution multi-processus (`multiprocessing` en mode `spawn`) avec agrégation asynchrone des journaux via `QueueListener` vers `outputs/logs/execution_debug.log`, export de la trajectoire complète dans `outputs/rebalance_history.parquet` (avec repli automatique vers `.csv`) et génération des rapports HTML **QuantStats** dans `outputs/tearsheets/`.



### 5.2 Suite d'Audit Mathématique & Système (`tests/test_backtest_integration.py`)

Le dépôt intègre une suite de tests Pytest divisée en contrats d'interface (`@pytest.mark.contract`) et audits de failles quantitatives (`@pytest.mark.audit_flaw`) entièrement isolés du réseau :

* **Absence d'effet *Off-by-One* et de *Look-Ahead Bias* :** Vérifie que le premier jour *Out-of-Sample* coïncide exactement avec `dates[config.LOOKBACK_WINDOW]` et qu'aucune donnée future ne fuite dans la fenêtre d'estimation.
* **Isolation stricte du Taux Sans Risque OOS :** Garantit que les ratios de Sharpe et Sortino finaux ainsi que le paramètre `rf` transmis à QuantStats utilisent exclusivement la moyenne du taux sans risque sur la période *Out-of-Sample* (aucune contamination par la période *In-Sample*).
* **Audit des Frictions Non-Linéaires :** Vérifie qu'en présence de rendements d'actifs et d'un taux sans risque strictement nuls en OOS, tout écart par rapport à l'allocation initiale dégrade mécaniquement la NAV finale ($V_{\text{final}} < 1.0$) sous l'effet exact des coûts de transaction et de marge.
* **Prévention des fuites mémoire Graphiques :** Contrôle la fermeture systématique des figures Matplotlib (`plt.close('all')`) lors de la génération séquentielle des Tearsheets HTML QuantStats.

---

## 6. Résultats Empiriques Out-of-Sample & Post-Mortem Quantitatif

Évaluation *Out-of-Sample* (base 252 jours/an, nette de frais de transaction et de financement) extraite des rapports QuantStats générés sur 4 régimes de marché distincts :

| Scénario (Période OOS) | Taux Sans Risque ($R_f$ moy. OOS) | Rendement Cumulé (GA vs Bench $1/N$) | CAGR (GA vs Bench) | Ratio de Sharpe (GA vs Bench) | Ratio de Sortino (GA vs Bench) | Max Drawdown (GA vs Bench) | Volatilité Ann. (GA vs Bench) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **`Global_MultiAsset_2024`** *(Jan 2025 – Sep 2026)* | 4,20 % | **+31,58 %** vs +25,95 % | **17,91 %** vs 14,91 % | **1,36** vs 1,03 | **2,04** vs 1,46 | **-7,83 %** vs -10,59 % | **10,08 %** vs 10,51 % |
| **`Global_GFC_2007_2009`** *(Juil 2007 – Déc 2009)* | 3,60 % | **+13,00 %** vs -28,74 % | **5,19 %** vs -13,09 % | **0,20** vs -0,43 | **0,29** vs -0,60 | **-22,22 %** vs -54,99 % | **13,57 %** vs 30,52 % |
| **`FR_Covid_2020_2021`** *(Déc 2019 – Déc 2021)* | 1,20 % | +7,75 % vs **+40,77 %** | 3,89 % vs **19,12 %** | 0,22 vs **0,79** | 0,29 vs **1,08** | **-36,03 %** vs -38,59 % | **16,58 %** vs 25,21 % |
| **`US_MegaTech_2015_2023`** *(Jan 2016 – Déc 2023)* | 2,30 % | +312,02 % vs **+760,09 %** | 19,58 % vs **31,19 %** | 0,72 vs **1,02** | 1,03 vs **1,50** | **-27,76 %** vs -43,72 % | **26,95 %** vs 29,28 % |

### Analyse Critique des Régimes

1. **Convexité défensive en crise systémique (`Global_GFC_2007_2009`) :** Lors de l'effondrement progressif des marchés en 2008, la détection du régime baissier et la pénalisation des queues épaisses (*kurtosis*) permettent au moteur d'inverser son bêta (autour de $-0,45$) et de générer **+17,15 %** sur l'année 2008 contre **-38,14 %** pour le benchmark, réduisant le Max Drawdown de plus de $32$ points de pourcentage ($-22,22\,\%$ contre $-54,99\,\%$).
2. **Efficience Multi-Actifs (`Global_MultiAsset_2024`) :** Sur un univers diversifié récent avec des taux d'intérêt positifs ($R_f = 4,2\,\%$), la combinaison de la covariance Ledoit-Wolf et du contrôle du *turnover* domine le benchmark sur l'ensemble des métriques ajustées du risque (Sharpe $1,36$ vs $1,03$, Calmar $2,29$ vs $1,41$).
3. **Inertie d'estimation face aux chocs exogènes en V (`FR_Covid_2020_2021`) :** Le krach pandémique de mars 2020 intervient de manière trop abrupte pour être anticipé par la fenêtre glissante pré-choc. Une fois la chute intégrée dans l'échantillon d'estimation, l'aversion à la variance et à la kurtosis réduit drastiquement l'exposition risquée (volatilité ramenée à $16,58\,\%$ contre $25,21\,\%$), ce qui pénalise la capture du rebond symétrique en V du second trimestre 2020.
4. **Arbitrage Rendement / Drawdown en marché ultra-directionnel (`US_MegaTech_2015_2023`) :** Sur un panier concentré de valeurs technologiques américaines en forte tendance haussière séculaire, le plafonnement des poids (`MAX_WEIGHT`) et la pénalisation de la variance brident mécaniquement le *momentum* brut par rapport à un portefeuille $1/N$, mais protègent efficacement le capital lors du marché baissier technologique de 2022 (perte limitée à **-22,72 %** contre **-40,86 %** pour le benchmark, et Max Drawdown global réduit de $-43,72\,\%$ à $-27,76\,\%$).

---

## 7. Installation & Utilisation

### Installation Automatisée (Linux / macOS / Windows)

```bash
# Linux / macOS
chmod +x setup.sh
./setup.sh

# Windows (CMD)
setup.bat

```

### Installation Manuelle

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows : .venv\Scripts\activate.bat
pip install --upgrade pip
pip install -r requirements.txt

```

### Exécution des Backtests et de la Suite d'Audit

```bash
# 1. Lancer le backtest principal avec génération du Tearsheet HTML et de l'historique Parquet
python main.py

# 2. Lancer la campagne de stress-tests multi-scénarios parallélisée
python general_tests.py

# 3. Exécuter la suite complète de tests unitaires et d'audit quantitatif
pytest tests/test_backtest_integration.py -v

```

### Paramétrage (`config.py`)

```python
# Paramètres de l'algorithme génétique (Numba JIT)
POPULATION_SIZE = 800          # Taille de la population
GENERATIONS = 1000             # Nombre maximal de générations
MUTATION_RATE = 0.10           # Taux de mutation nominal (avant hyper-mutation adaptative)

# Préférences de risque et moments d'ordre supérieur
RISK_AVERSION = 0.5            # Pondération nominale de la variance Ledoit-Wolf
SKEWNESS_PREFERENCE = 0.05     # Valorisation de l'asymétrie positive (ordre 3)
KURTOSIS_AVERSION = 0.01       # Pénalisation des queues épaisses (ordre 4)
TIME_WEIGHT_HALFLIFE = 63      # Demi-vie (jours) de la pondération exponentielle temporelle

# Contraintes d'exposition et frictions de marché
ALLOW_SHORT = True             # Autorisation des ventes à découvert
MAX_WEIGHT = 0.25              # Borne supérieure de concentration par actif
TRANSACTION_FEE = 0.002        # Coût de transaction L1 (20 bps) + impact quadratique L2

# Fenêtres temporelles du Rolling Backtest
LOOKBACK_WINDOW = 252          # Fenêtre d'estimation In-Sample (1 an de bourse)
REBALANCE_FREQ = 21            # Fréquence de rééquilibrage Out-of-Sample (1 mois)

```

---

## 8. Références Académiques

1. **Ledoit, O., & Wolf, M. (2004).** *A well-conditioned estimator for large-dimensional covariance matrices.* Journal of Multivariate Analysis, 88(2), 365–411.


2. **Eshelman, L. J., & Schaffer, J. D. (1993) / Spears, W. M., & Anand, V. (1991).** *Real-Coded Genetic Algorithms and Interval-Schemata (BLX-$\alpha$ Crossover)*.


3. **Harvey, C. R., Liechty, J. C., Liechty, M. W., & Müller, P. (2010).** *Portfolio selection with higher moments.* Quantitative Finance, 10(5), 469–485.


4. **Almgren, R., & Chriss, N. (2001).** *Optimal execution of portfolio transactions.* Journal of Risk, 3, 5–40.

---

## Auteur & Licence

**Auteur :** Robin Leblanc — Élève-Ingénieur à l'École Centrale de Lyon & Licencié en Sciences Économiques et Gestion (Université Lumière Lyon 2).

**Licence :** Propriétaire — Tous droits réservés.