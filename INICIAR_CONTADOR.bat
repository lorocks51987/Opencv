@echo off
chcp 65001 > nul
title Math Blitz - Desafio dos Dedos
color 0A

cd /d "%~dp0"
echo Iniciando Math Blitz - Desafio dos Dedos...
if exist ".venv\Scripts\python.exe" (
    .venv\Scripts\python.exe finger_arcade\Finger_Arcade.py
) else (
    python finger_arcade\Finger_Arcade.py
)
if %errorlevel% neq 0 (
    echo.
    echo O programa foi encerrado ou encontrou um erro.
    pause
)
