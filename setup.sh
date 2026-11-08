#!/bin/bash

set -e

echo "=========================================="
echo "Configuration de l'environnement virtuel"
echo "=========================================="

echo ""
echo "[1/5] Création de l'environnement virtuel .venv..."
python3 -m venv .venv
echo "✓ Environnement virtuel créé avec succès"

echo ""
echo "[2/5] Activation de l'environnement virtuel..."
source .venv/bin/activate
echo "✓ Environnement virtuel activé"

echo ""
echo "[3/5] Mise à jour de pip..."
pip install --upgrade pip
echo "✓ pip mis à jour avec succès"

echo ""
echo "[4/5] Installation des dépendances..."
pip install -r requirements.txt
echo "✓ Dépendances installées avec succès"

echo ""
echo "[5/5] Vérification des installations..."
python -c "import numpy, scipy, sklearn, numba, yfinance, pandas; print('✓ Tous les packages sont accessibles')"
echo ""

echo "=========================================="
echo "Configuration terminée avec succès !"
echo "=========================================="
echo ""
echo "Pour activer l'environnement à l'avenir, exécutez:"
echo "  source .venv/bin/activate"
echo ""
echo "Pour désactiver l'environnement, exécutez:"
echo "  deactivate"
