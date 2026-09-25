"""
Ponto de entrada principal para o Snake Donuts.
Executa a versão definitiva: Snake Donuts - Arcade Edition (Unimar Aberta).
"""
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from Snake_Arcade_Edition import main

if __name__ == "__main__":
    main()
