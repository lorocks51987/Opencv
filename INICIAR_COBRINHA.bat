@echo off
chcp 65001 > nul
title Snake Donuts Arcade - Stand ADS Unimar Aberta
color 0E

cd /d "%~dp0"
echo ===============================================================================
echo       INICIANDO SNAKE DONUTS ARCADE (ADS UNIMAR ABERTA)...
echo ===============================================================================
python Snake_Donut\main.py
if %errorlevel% neq 0 (
    echo.
    echo O jogo foi encerrado ou encontrou um erro.
    pause
)
