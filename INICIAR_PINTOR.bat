@echo off
chcp 65001 > nul
title Virtual Painter - ADS Unimar
color 0A

cd /d "%~dp0"
echo Iniciando Virtual Painter com QR Code...
if exist ".venv\Scripts\python.exe" (
    .venv\Scripts\python.exe virtual_painter\VirtualPainter_Arcade.py
) else (
    python virtual_painter\VirtualPainter_Arcade.py
)
if %errorlevel% neq 0 (
    echo.
    echo O programa foi encerrado ou encontrou um erro.
    pause
)
