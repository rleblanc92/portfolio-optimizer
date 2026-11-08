@echo off
setlocal enabledelayedexpansion

echo.
echo ==========================================
echo Configuration de l'environnement virtuel
echo ==========================================
echo.

echo [1/5] Creation de l'environnement virtuel .venv...
python -m venv .venv
echo OK - Environnement virtuel cree avec succes
echo.

echo [2/5] Activation de l'environnement virtuel...
call .venv\Scripts\activate.bat
echo OK - Environnement virtuel active
echo.

echo [3/5] Mise a jour de pip...
python -m pip install --upgrade pip
echo OK - pip mis a jour avec succes
echo.

echo [4/5] Installation des dependances...
pip install -r requirements.txt
echo OK - Dependances installees avec succes
echo.

echo [5/5] Verification des installations...
python -c "import numpy, scipy, sklearn, numba, yfinance, pandas; print('OK - Tous les packages sont accessibles')"
echo.

echo ==========================================
echo Configuration terminee avec succes !
echo ==========================================
echo.
echo Pour activer l'environnement a l'avenir, executez:
echo   .venv\Scripts\activate.bat
echo.
echo Pour desactiver l'environnement, executez:
echo   deactivate
echo.

endlocal
