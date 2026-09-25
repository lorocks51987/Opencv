@echo off
chcp 65001 > nul
title Pintor Virtual com QR Code - Stand ADS Unimar Aberta
color 0D

cd /d "%~dp0"
echo ===============================================================================
echo       INICIANDO PINTOR VIRTUAL COM QR CODE (ADS UNIMAR ABERTA)...
echo ===============================================================================
python virtual_painter\VirtualPainter.py
if %errorlevel% neq 0 (
    echo.
    echo O programa foi encerrado ou encontrou um erro.
    pause
)
