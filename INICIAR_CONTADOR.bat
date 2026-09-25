@echo off
chcp 65001 > nul
title Finger Arcade & Gesture Arena - Stand ADS Unimar Aberta
color 0B

cd /d "%~dp0"
echo ===============================================================================
echo       INICIANDO FINGER ARCADE & GESTURE ARENA (ADS UNIMAR ABERTA)...
echo ===============================================================================
python finger_arcade\Finger_Arcade.py
if %errorlevel% neq 0 (
    echo.
    echo O programa foi encerrado ou encontrou um erro.
    pause
)
