@echo off
chcp 65001 > nul
title IA Educacional de LIBRAS - ADS Unimar
color 0A

cd /d "%~dp0"
echo Iniciando IA Educacional de LIBRAS...
if exist ".venv\Scripts\python.exe" (
    .venv\Scripts\python.exe libras\libras_stand_edition.py
) else (
    python libras\libras_stand_edition.py
)
if %errorlevel% neq 0 (
    echo.
    echo O programa foi encerrado ou encontrou um erro.
    pause
)
