"""Régimes de stress pour l'évaluation robuste du moteur de portefeuille.

Chaque scénario déclare explicitement ``rf_ticker``. Le pipeline transforme
ce prix ou rendement souverain en rendement quotidien constant de la poche
sans risque, adossé aux bons du Trésor. Cette poche permet à l'algorithme
génétique d'effectuer une désertion tactique du marché (Market Flight), au
lieu de rester mécaniquement contraint sur une frontière efficiente de
Markowitz lorsque la structure de covariance devient pathologique.
"""


_FRENCH_TICKERS = [
    'MC.PA', 'TTE.PA', 'AI.PA', 'SAN.PA', 'OR.PA',
    'BNP.PA', 'DG.PA', 'CS.PA', 'CAP.PA',
]


SCENARIOS = {
    'US_MegaTech_2015_2023': {
        'start_date': '2015-01-01',
        'end_date': '2023-12-31',
        'tickers': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA'],
        'benchmark_ticker': 'SPY',
        'rf_ticker': '^TNX',
        'objective': (
            "Mesurer la capture du momentum et de l'asymétrie positive "
            "dans un régime de croissance spéculative."
        ),
    },
    'FR_Covid_2020_2021': {
        'start_date': '2019-01-01',
        'end_date': '2021-12-31',
        'tickers': _FRENCH_TICKERS.copy(),
        'benchmark_ticker': '^FCHI',
        'rf_ticker': '^TNX',
        'objective': (
            "Mesurer l'adaptation du régime macro au choc Covid et la "
            "réallocation vers la poche sans risque lors du V-Shape."
        ),
    },
    'Global_GFC_2007_2009': {
        'start_date': '2006-08-01',
        'end_date': '2009-12-31',
        'tickers': [
            '^GSPC', '^FTSE', '^GDAXI', '^N225',
            'GC=F', 'EURUSD=X', 'JPY=X',
        ],
        'benchmark_ticker': '^GSPC',
        'rf_ticker': '^TNX',
        'objective': (
            "Tester la résilience du Sortino, l'épaisseur des queues et la "
            "concentration des variances croisées pendant une crise de liquidité."
        ),
    },
    'Global_MultiAsset_2024': {
        'start_date': '2024-01-01',
        'end_date': '2026-09-16',
        'tickers': [
            'SPY', 'QQQ', 'IWM', 'VEA', 'EEM',
            'TLT', 'IEF', 'LQD', 'HYG',
            'GLD', 'DBC', 'VNQ', 'IBIT',
        ],
        'benchmark_ticker': 'SPY',
        'rf_ticker': '^TNX',
    },
}