@echo off
chcp 65001 > nul
title Launcher Geral - Stand ADS Unimar Aberta
color 0A

cd /d "%~dp0"
echo ===============================================================================
echo       INICIANDO LAUNCHER DO STAND ADS UNIMAR ABERTA...
echo ===============================================================================
python menu_stand_unimar.py
if %errorlevel% neq 0 (
    echo.
    echo O programa foi encerrado ou encontrou um erro.
    pause
)
