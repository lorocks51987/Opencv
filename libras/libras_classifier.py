"""
Ponto de entrada do classificador de LIBRAS.
Executa a versão moderna Stand Edition com Dashboard Único e Desafio das Vogais.
"""
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from libras_stand_edition import main

if __name__ == "__main__":
    main()
