@echo off
chcp 65001 > nul
title IA Educacional de LIBRAS - Stand ADS Unimar Aberta
color 0B

cd /d "%~dp0"
echo ===============================================================================
echo       INICIANDO IA EDUCACIONAL DE LIBRAS (ADS UNIMAR ABERTA)...
echo ===============================================================================
python libras\libras_classifier.py
if %errorlevel% neq 0 (
    echo.
    echo O programa foi encerrado ou encontrou um erro.
    pause
)
