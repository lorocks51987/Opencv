@echo off
chcp 65001 > nul
title Instalador de Dependências - Stand ADS Unimar Aberta
color 0B

echo ===============================================================================
echo       ADS UNIMAR - INSTALADOR DE DEPENDENCIAS DO STAND (OPENCV + IA)
echo ===============================================================================
echo.

where python >nul 2>nul
if %errorlevel% neq 0 (
    color 0C
    echo [ERRO CRITICO] O Python nao foi encontrado no sistema!
    echo Certifique-se de instalar o Python 3.10 ou superior e marcar a opcao:
    echo "Add Python to PATH" durante a instalacao.
    echo.
    pause
    exit /b 1
)

echo [1/3] Verificando versao do Python...
python --version
echo.

echo [2/3] Atualizando o gerenciador pip...
python -m pip install --upgrade pip
echo.

echo [3/3] Instalando dependencias do requirements.txt...
python -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    color 0C
    echo.
    echo [ERRO] Falha ao instalar algumas dependencias.
    echo Verifique sua conexao com a internet e tente novamente.
    echo.
    pause
    exit /b 1
)

echo.
color 0A
echo ===============================================================================
echo      TODAS AS DEPENDENCIAS FORAM INSTALADAS COM SUCESSO!
echo      Agora voce ja pode iniciar qualquer jogo com 2 cliques nos arquivos .bat
echo ===============================================================================
echo.
pause
