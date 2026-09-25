@echo off
chcp 65001 > nul
title Snake Donuts - Arcade Edition
color 0A

cd /d "%~dp0"
echo Iniciando Snake Donuts - Arcade Edition...
if exist ".venv\Scripts\python.exe" (
    .venv\Scripts\python.exe Snake_Donut\Snake_Arcade_Edition.py
) else (
    python Snake_Donut\Snake_Arcade_Edition.py
)
if %errorlevel% neq 0 (
    echo.
    echo O programa foi encerrado ou encontrou um erro.
    pause
)
